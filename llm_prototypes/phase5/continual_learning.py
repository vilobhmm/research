"""
Phase 5.2 — Continual Learning in Agent Systems
=================================================
Implements continual learning strategies for agents that acquire
new capabilities without catastrophic forgetting of old ones.

Strategies:
  • Elastic Weight Consolidation (EWC)
  • Experience Replay
  • Mixture of Experts routing
  • Progressive phase: Math → Coding → Writing

Run:
    python phase5/continual_learning.py

Requirements: torch, numpy, rich
"""

import copy
import math
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from rich import print
    from rich.console import Console
    console = Console()
except ImportError:
    console = None

# ── Task Definitions ──────────────────────────────────────────────────────────

# Each task is a function mapping input vector → output vector
# We simulate 3 domains: Math, Coding, Writing

def generate_math_task(n: int = 500) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """
    Math task: predict [sin(x), cos(x), x^2] from [x, x^2, x^3]
    Represents learning mathematical functions.
    """
    data = []
    for _ in range(n):
        x = random.uniform(-3, 3)
        inp = torch.tensor([x, x**2, x**3, abs(x), math.sqrt(abs(x))], dtype=torch.float32)
        out = torch.tensor([math.sin(x), math.cos(x), x**2 / 9.0], dtype=torch.float32)
        data.append((inp, out))
    return data


def generate_coding_task(n: int = 500) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """
    Coding task: predict code quality features from token statistics.
    Simulates learning to evaluate code.
    """
    data = []
    for _ in range(n):
        # Simulated code stats: [length, indent_depth, func_count, line_count, complexity]
        length    = random.uniform(0.1, 1.0)
        indent    = random.uniform(0, 0.5)
        funcs     = random.uniform(0, 0.8)
        lines     = random.uniform(0.1, 1.0)
        cyclo     = random.uniform(0, 1.0)
        inp = torch.tensor([length, indent, funcs, lines, cyclo], dtype=torch.float32)
        # Predict [readability, maintainability, test_coverage]
        readability    = max(0, min(1, 1 - 0.3 * indent - 0.2 * cyclo + 0.2 * funcs))
        maintainability = max(0, min(1, 0.5 + 0.3 * (1 - cyclo) - 0.2 * length))
        test_coverage  = max(0, min(1, 0.3 * funcs + 0.1 * lines + random.uniform(-0.1, 0.1)))
        out = torch.tensor([readability, maintainability, test_coverage], dtype=torch.float32)
        data.append((inp, out))
    return data


def generate_writing_task(n: int = 500) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """
    Writing task: predict writing quality from text statistics.
    Simulates learning to evaluate text quality.
    """
    data = []
    for _ in range(n):
        # Text stats: [vocab_diversity, avg_sentence_len, paragraph_count, passive_ratio, clarity]
        vocab_div  = random.uniform(0.3, 1.0)
        sent_len   = random.uniform(0.1, 0.9)
        para_count = random.uniform(0.1, 0.8)
        passive    = random.uniform(0, 0.6)
        clarity    = random.uniform(0.3, 1.0)
        inp = torch.tensor([vocab_div, sent_len, para_count, passive, clarity], dtype=torch.float32)
        # Predict [coherence, engagement, grammar_score]
        coherence   = max(0, min(1, 0.4 * clarity + 0.3 * vocab_div + 0.2 * para_count - 0.1 * passive))
        engagement  = max(0, min(1, 0.5 * vocab_div + 0.3 * clarity - 0.2 * passive))
        grammar     = max(0, min(1, clarity + 0.1 * (1 - passive) + random.uniform(-0.1, 0.1)))
        out = torch.tensor([coherence, engagement, grammar], dtype=torch.float32)
        data.append((inp, out))
    return data


TASKS = {
    "math":   (generate_math_task,   "Math functions"),
    "coding": (generate_coding_task,  "Code quality"),
    "writing": (generate_writing_task, "Writing quality"),
}


# ── Dataset ───────────────────────────────────────────────────────────────────

class TaskDataset(Dataset):
    def __init__(self, data: list[tuple[torch.Tensor, torch.Tensor]]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ── Base Agent Network ────────────────────────────────────────────────────────

class AgentNetwork(nn.Module):
    """Shared backbone with task-specific heads."""

    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = 128,
        output_dim: int = 3,
        num_shared_layers: int = 3,
    ):
        super().__init__()
        # Shared layers (domain-agnostic features)
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1)]
        for _ in range(num_shared_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1)]
        self.shared = nn.Sequential(*layers)

        # Single task head (or domain-specific in MoE variant)
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.shared(x))

    def get_shared_params(self) -> list[nn.Parameter]:
        return list(self.shared.parameters())


# ── Elastic Weight Consolidation (EWC) ───────────────────────────────────────

class EWC:
    """
    EWC (Kirkpatrick et al., 2017): protect important parameters
    from previous tasks by adding a quadratic penalty.
    Importance estimated via Fisher information.
    """

    def __init__(self, model: AgentNetwork, dataset: TaskDataset, lambda_ewc: float = 1000.0):
        self.lambda_ewc = lambda_ewc
        self.params: dict[str, torch.Tensor] = {}        # θ*: optimal params from previous task
        self.fisher: dict[str, torch.Tensor] = {}        # F: Fisher information (importance)
        self._compute_fisher(model, dataset)

    def _compute_fisher(self, model: AgentNetwork, dataset: TaskDataset):
        """Estimate Fisher information via squared gradients on task data."""
        model.eval()
        # Store current optimal parameters
        for name, param in model.named_parameters():
            self.params[name] = param.data.clone()
            self.fisher[name] = torch.zeros_like(param.data)

        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        model.train()
        for x, y in list(loader)[:20]:  # Use subset for efficiency
            model.zero_grad()
            output = model(x)
            loss = F.mse_loss(output, y)
            loss.backward()
            for name, param in model.named_parameters():
                if param.grad is not None:
                    self.fisher[name] += param.grad.data.clone() ** 2

        # Normalise
        n = min(20, len(loader))
        for name in self.fisher:
            self.fisher[name] /= max(n, 1)

    def penalty(self, model: AgentNetwork) -> torch.Tensor:
        """EWC penalty: λ/2 * Σ F_i (θ_i - θ*_i)^2"""
        loss = torch.tensor(0.0)
        for name, param in model.named_parameters():
            if name in self.fisher:
                loss += (self.fisher[name] * (param - self.params[name]).pow(2)).sum()
        return 0.5 * self.lambda_ewc * loss


# ── Mixture of Experts ────────────────────────────────────────────────────────

class MixtureOfExperts(nn.Module):
    """
    Each domain has its own expert head.
    A gating network learns to route inputs to the right expert.
    New domains add new expert heads without modifying existing ones.
    """

    def __init__(self, input_dim: int = 5, hidden_dim: int = 128, output_dim: int = 3):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.experts: nn.ModuleList = nn.ModuleList()
        self.domain_heads: dict[str, int] = {}  # domain → expert index
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.gate: Optional[nn.Linear] = None

    def add_expert(self, domain: str):
        """Add a new expert head for a new domain."""
        expert = nn.Linear(self.hidden_dim, self.output_dim)
        nn.init.xavier_uniform_(expert.weight)
        self.experts.append(expert)
        self.domain_heads[domain] = len(self.experts) - 1
        # Update gating network
        n = len(self.experts)
        self.gate = nn.Linear(self.hidden_dim, n)

    def forward(self, x: torch.Tensor, domain: Optional[str] = None) -> torch.Tensor:
        features = self.shared(x)

        if domain is not None and domain in self.domain_heads:
            # Supervised routing: use the known expert
            expert_idx = self.domain_heads[domain]
            return self.experts[expert_idx](features)

        if len(self.experts) == 0:
            raise ValueError("No experts added yet")

        if self.gate is not None:
            # Soft routing: weighted combination of experts
            gate_scores = F.softmax(self.gate(features), dim=-1)  # (B, n_experts)
            expert_outs = torch.stack([e(features) for e in self.experts], dim=-1)  # (B, D, n)
            return (expert_outs * gate_scores.unsqueeze(1)).sum(-1)

        # Single expert
        return self.experts[0](features)


# ── Experience Replay Buffer ──────────────────────────────────────────────────

class ReplayBuffer:
    """Stores examples from past tasks for interleaved replay."""

    def __init__(self, capacity: int = 200):
        self.capacity = capacity
        self.buffer: list[tuple[torch.Tensor, torch.Tensor, str]] = []

    def add(self, data: list[tuple[torch.Tensor, torch.Tensor]], domain: str, n: int = 50):
        """Add n random examples from data to the buffer."""
        sample = random.sample(data, min(n, len(data)))
        for x, y in sample:
            self.buffer.append((x, y, domain))
            if len(self.buffer) > self.capacity:
                self.buffer.pop(0)

    def sample(self, k: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Sample k examples from the replay buffer."""
        selected = random.sample(self.buffer, min(k, len(self.buffer)))
        return [(x, y) for x, y, _ in selected]

    def __len__(self):
        return len(self.buffer)


# ── Training Utilities ────────────────────────────────────────────────────────

def train_epoch(
    model: nn.Module,
    data: list[tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    ewc: Optional[EWC] = None,
    replay_buffer: Optional[ReplayBuffer] = None,
    replay_ratio: float = 0.3,
    domain: Optional[str] = None,  # for MoE
) -> float:
    """Train for one epoch on the given task data."""
    model.train()
    loader = DataLoader(TaskDataset(data), batch_size=32, shuffle=True)
    total_loss = 0.0

    for x, y in loader:
        # Task loss
        if isinstance(model, MixtureOfExperts):
            pred = model(x, domain=domain)
        else:
            pred = model(x)
        task_loss = F.mse_loss(pred, y)

        # EWC penalty (prevents forgetting important weights)
        ewc_loss = ewc.penalty(model) if ewc is not None else torch.tensor(0.0)

        # Replay loss (interleave old task examples)
        replay_loss = torch.tensor(0.0)
        if replay_buffer and len(replay_buffer) > 0:
            replay_batch = replay_buffer.sample(max(4, int(32 * replay_ratio)))
            if replay_batch:
                rx = torch.stack([b[0] for b in replay_batch])
                ry = torch.stack([b[1] for b in replay_batch])
                if isinstance(model, MixtureOfExperts):
                    rp = model(rx)  # soft routing for replay
                else:
                    rp = model(rx)
                replay_loss = F.mse_loss(rp, ry)

        loss = task_loss + ewc_loss + 0.5 * replay_loss
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += task_loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate_domain(
    model: nn.Module,
    data: list[tuple[torch.Tensor, torch.Tensor]],
    domain: Optional[str] = None,
) -> float:
    """Compute MSE loss on a domain's test set."""
    model.eval()
    loader = DataLoader(TaskDataset(data), batch_size=64)
    total_loss = 0.0
    for x, y in loader:
        if isinstance(model, MixtureOfExperts):
            pred = model(x, domain=domain)
        else:
            pred = model(x)
        total_loss += F.mse_loss(pred, y).item()
    return total_loss / len(loader)


# ── Continual Learning Experiment ─────────────────────────────────────────────

@dataclass
class PhaseResult:
    phase: int
    domain_trained: str
    scores_after: dict[str, float]  # domain → MSE after this phase


def run_continual_learning_experiment(
    strategy: str = "replay",   # "naive", "ewc", "replay", "moe"
    n_epochs: int = 30,
    verbose: bool = True,
) -> list[PhaseResult]:
    """
    Train on three domains sequentially, measuring forgetting.
    """
    torch.manual_seed(42)
    random.seed(42)

    # Generate data
    domains_seq = ["math", "coding", "writing"]
    train_data  = {d: TASKS[d][0](n=400) for d in domains_seq}
    test_data   = {d: TASKS[d][0](n=100) for d in domains_seq}

    # Initialise model
    if strategy == "moe":
        model = MixtureOfExperts(input_dim=5, hidden_dim=64, output_dim=3)
    else:
        model = AgentNetwork(input_dim=5, hidden_dim=64, output_dim=3, num_shared_layers=2)

    replay_buffer = ReplayBuffer(capacity=150) if strategy in ("replay",) else None
    ewc_regulariser: Optional[EWC] = None
    results: list[PhaseResult] = []

    for phase_idx, domain in enumerate(domains_seq, 1):
        if verbose:
            print(f"\n  Phase {phase_idx}: Training on [bold]{TASKS[domain][1]}[/bold] ({domain})")

        # For MoE: add a new expert for this domain
        if strategy == "moe":
            model.add_expert(domain)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

        for epoch in range(n_epochs):
            train_epoch(
                model,
                train_data[domain],
                optimizer,
                ewc=ewc_regulariser if strategy == "ewc" else None,
                replay_buffer=replay_buffer,
                replay_ratio=0.3,
                domain=domain if strategy == "moe" else None,
            )

        # After training: update EWC importance weights
        if strategy == "ewc":
            ewc_regulariser = EWC(model, TaskDataset(train_data[domain]))

        # Add to replay buffer
        if replay_buffer is not None:
            replay_buffer.add(train_data[domain], domain, n=50)

        # Evaluate on ALL domains so far
        scores = {}
        for eval_domain in domains_seq[:phase_idx]:
            score = evaluate_domain(
                model, test_data[eval_domain],
                domain=eval_domain if strategy == "moe" else None,
            )
            scores[eval_domain] = score

        results.append(PhaseResult(
            phase=phase_idx,
            domain_trained=domain,
            scores_after=scores,
        ))

        if verbose:
            for d, s in scores.items():
                bar = "█" * max(0, int((0.15 - s) / 0.15 * 20))
                status = "current" if d == domain else "old"
                print(f"    [{status}] {d:<8}: MSE={s:.4f} {bar}")

    return results


# ── Main Experiment ───────────────────────────────────────────────────────────

def main():
    print("\n[bold]Phase 5.2 — Continual Learning in Agent Systems[/bold]\n")
    print("Problem: Learn Math → Coding → Writing without forgetting Math\n")

    strategies = ["naive", "ewc", "replay", "moe"]
    strategy_names = {
        "naive": "Naive (no protection)",
        "ewc":   "EWC (Elastic Weight Consolidation)",
        "replay": "Experience Replay",
        "moe":   "Mixture of Experts",
    }

    all_results: dict[str, list[PhaseResult]] = {}

    for strategy in strategies:
        print(f"\n{'─'*50}")
        print(f"[bold]Strategy: {strategy_names[strategy]}[/bold]")
        results = run_continual_learning_experiment(strategy=strategy, n_epochs=25, verbose=True)
        all_results[strategy] = results

    # ── Compare strategies
    print("\n\n[bold]Strategy Comparison: Forgetting Analysis[/bold]")
    print("\nFinal performance on Math task after learning all 3 domains:")
    print(f"\n  {'Strategy':<40} {'Math MSE':>10} {'Coding MSE':>12} {'Writing MSE':>13} {'Avg':>8}")
    print("  " + "─" * 88)

    for strategy, results in all_results.items():
        final = results[-1]  # After training on all 3 domains
        math_score   = final.scores_after.get("math", float("nan"))
        coding_score = final.scores_after.get("coding", float("nan"))
        writing_score = final.scores_after.get("writing", float("nan"))
        avg = np.mean([s for s in [math_score, coding_score, writing_score] if not math.isnan(s)])
        name = strategy_names[strategy]
        print(f"  {name:<40} {math_score:>10.4f} {coding_score:>12.4f} {writing_score:>13.4f} {avg:>8.4f}")

    # ── Forgetting analysis
    print("\n[bold]Catastrophic Forgetting Analysis:[/bold]")
    print("(Math performance after learning each new domain)")
    print()

    for strategy, results in all_results.items():
        print(f"  {strategy_names[strategy]}:")
        math_scores = []
        for r in results:
            if "math" in r.scores_after:
                math_scores.append(r.scores_after["math"])
                phase_label = f"after learning {r.domain_trained}"
                bar = "█" * max(0, int((0.15 - r.scores_after["math"]) / 0.15 * 20))
                print(f"    {phase_label:<30}: {r.scores_after['math']:.4f} {bar}")

        if len(math_scores) >= 2:
            forgetting = math_scores[-1] - math_scores[0]
            direction = "↑ worse" if forgetting > 0 else "↓ better (no forgetting!)"
            print(f"    Forgetting: {forgetting:+.4f} ({direction})")
        print()

    # ── Conceptual explanation
    print("[bold]How Each Strategy Prevents Forgetting:[/bold]")
    print("""
  Naive:
    • No protection — simply overwrites weights for new task
    • Catastrophic forgetting: Math performance degrades severely
    • Fast to train, zero overhead

  EWC (Elastic Weight Consolidation):
    • Estimates parameter importance via Fisher information
    • Adds quadratic penalty: λ/2 * Σ F_i (θ_i - θ*_i)²
    • Important parameters for old tasks are protected
    • Tradeoff: less flexible for new tasks if λ too high

  Experience Replay:
    • Stores examples from previous tasks in a buffer
    • Mixes old and new examples during training
    • Simple and effective; requires extra memory
    • Best results: reservoir sampling for diverse buffer

  Mixture of Experts:
    • Each domain gets its own expert head
    • Shared backbone learns general features
    • New domains add parameters, don't modify old ones
    • Most scalable but most memory-intensive
    """)

    print("[bold]Agent Continual Learning in Practice:[/bold]")
    print("  • Start with strong base model (pre-training)")
    print("  • Add domain-specific heads for new tools (MoE)")
    print("  • Maintain replay buffer of past interactions")
    print("  • Periodic evaluation on all domains to detect forgetting")
    print("  • Fine-tune carefully with small learning rates")
    print("  • Parameter-efficient methods (LoRA) reduce forgetting by design")

    print("\n[bold green]Continual Learning experiment complete![/bold green]")


if __name__ == "__main__":
    main()
