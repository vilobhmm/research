"""
Phase 4.2 — Mini RLHF Pipeline Simulator
==========================================
Simulates the RLHF loop WITHOUT large model training:
  • Policy: Claude API (or template-based for offline demo)
  • Reward Model: trained RM from phase4/reward_model.py concepts
  • RLHF iterations: score → compare to baseline → collect signal
  • PPO-style simplified update: filter positive examples
  • Metrics: average RM score over time, policy divergence

Run:
    ANTHROPIC_API_KEY=sk-... python phase4/rlhf_simulator.py
    # or offline:
    python phase4/rlhf_simulator.py

Requirements: anthropic (optional), numpy, torch, rich
"""

import json
import math
import os
import random
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    from rich import print
    from rich.console import Console
    console = Console()
except ImportError:
    console = None

# ── Simulated Reward Function ─────────────────────────────────────────────────
# In a real RLHF pipeline, this would be a trained neural network.
# We simulate it with heuristic quality features.

def compute_reward(question: str, response: str) -> float:
    """
    Heuristic reward function simulating a trained reward model.
    Returns score in [0, 1].
    """
    resp_lower = response.lower()
    words = response.split()
    score = 0.0

    # Length: reward appropriate length (not too short, not too verbose)
    n_words = len(words)
    if 30 <= n_words <= 200:
        score += 0.25
    elif n_words < 10:
        score -= 0.3
    elif n_words > 400:
        score -= 0.1

    # Specificity: concrete terms
    specificity_terms = [
        "specifically", "for example", "such as", "because", "therefore",
        "first", "second", "in particular", "namely", "including"
    ]
    score += min(0.2, sum(0.04 for t in specificity_terms if t in resp_lower))

    # Structure: formatting signals
    if any(c in response for c in ["1.", "2.", "•", "-", "\n"]):
        score += 0.15

    # Technical relevance: shares vocabulary with question
    q_words = set(question.lower().split())
    overlap = len(q_words & set(words)) / max(len(q_words), 1)
    score += 0.2 * overlap

    # Completeness: full sentences
    sentences = [s for s in response.split(".") if s.strip()]
    if len(sentences) >= 2:
        score += 0.1

    # Helpfulness: not dismissive
    dismissive = ["I don't know", "I can't", "impossible", "N/A", "unclear"]
    if not any(d.lower() in resp_lower for d in dismissive):
        score += 0.1

    return max(0.0, min(1.0, score))


# ── SFT Policy (template-based) ───────────────────────────────────────────────

class SFTPolicy:
    """
    Simulates a supervised fine-tuned policy with 3 quality tiers,
    randomly sampled to represent the distribution of SFT outputs.
    """

    RESPONSE_TEMPLATES = {
        "what is": [
            # Tier 1: high quality
            lambda q, t: f"{t.title()} is a fundamental concept in machine learning and AI. "
                        f"It refers to the process by which neural networks learn from data through "
                        f"iterative optimisation. Specifically, {t} involves computing gradients and "
                        f"updating model parameters to minimise a loss function. Key applications "
                        f"include classification, regression, and generative tasks.",
            # Tier 2: medium quality
            lambda q, t: f"{t.title()} is an important technique used in AI systems. "
                        f"It helps models learn from data and improve their performance over time. "
                        f"This is widely used in modern machine learning applications.",
            # Tier 3: low quality
            lambda q, t: f"{t} is a machine learning method.",
        ],
        "how does": [
            lambda q, t: f"{t.title()} works through a series of well-defined steps:\n"
                        f"1. The input data is processed through learned transformations\n"
                        f"2. A loss function measures the discrepancy from the target\n"
                        f"3. Gradients are computed via backpropagation\n"
                        f"4. Parameters are updated in the direction that minimises loss\n"
                        f"This process iterates until convergence or a stopping criterion is met.",
            lambda q, t: f"{t.title()} operates by processing information through layers. "
                        f"Each layer transforms the data in specific ways, learning useful "
                        f"representations that help solve the target task.",
            lambda q, t: f"It processes data through neural network layers.",
        ],
        "explain": [
            lambda q, t: f"Let me explain {t} in detail.\n\n"
                        f"{t.title()} is a core concept in modern AI. At its heart, it addresses "
                        f"the challenge of learning from data efficiently. The key insight is that "
                        f"by adjusting model parameters based on feedback signals, the system can "
                        f"improve its performance on the target task. This involves both the "
                        f"forward pass (computing predictions) and backward pass (computing gradients).",
            lambda q, t: f"{t.title()} is an approach used in machine learning to enable models "
                        f"to learn from examples. It's used widely in practice.",
            lambda q, t: f"This is a complex topic in AI research.",
        ],
    }

    def __init__(self, quality_distribution: tuple[float, float, float] = (0.3, 0.5, 0.2)):
        self.quality_distribution = quality_distribution  # (high, medium, low)

    def generate(self, question: str) -> str:
        q_lower = question.lower()
        key = "what is" if "what is" in q_lower else "how does" if "how does" in q_lower else "explain"
        templates = self.RESPONSE_TEMPLATES.get(key, self.RESPONSE_TEMPLATES["what is"])

        # Sample quality tier
        tier = np.random.choice([0, 1, 2], p=self.quality_distribution)
        template = templates[tier]

        # Extract topic
        topic = re.sub(r"what is |how does |explain ", "", q_lower).strip("?").strip()

        try:
            return template(question, topic)
        except Exception:
            return f"This question is about {topic}, which is an important ML concept."


# ── Claude Policy ─────────────────────────────────────────────────────────────

class ClaudePolicy:
    """Uses Claude API as the policy model."""

    def __init__(self, model: str = "claude-opus-4-6"):
        self.model = model
        self.client = anthropic.Anthropic()
        self.call_count = 0

    def generate(self, question: str, style: str = "standard") -> str:
        """Generate a response, optionally with a style modifier."""
        if style == "concise":
            system = "Answer the question concisely in 2-3 sentences."
        elif style == "detailed":
            system = "Provide a detailed, well-structured answer with examples."
        else:
            system = "Answer the machine learning question helpfully."

        self.call_count += 1
        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=300,
                system=system,
                messages=[{"role": "user", "content": question}],
            )
            return resp.content[0].text
        except Exception as e:
            return f"[API error: {e}]"


# ── RLHF Simulator ────────────────────────────────────────────────────────────

@dataclass
class Interaction:
    question: str
    response: str
    reward: float
    advantage: float  # reward - baseline
    iteration: int


class RLHFSimulator:
    """
    Simulates the RLHF training loop:
    1. Generate responses with current policy
    2. Score with reward model
    3. Compute advantage over SFT baseline
    4. Collect positive examples (advantage > 0)
    5. Track improvement over iterations
    """

    def __init__(
        self,
        policy,  # SFTPolicy or ClaudePolicy
        kl_beta: float = 0.1,
        use_api: bool = False,
    ):
        self.policy = policy
        self.sft_policy = SFTPolicy(quality_distribution=(0.5, 0.4, 0.1))  # Better baseline
        self.kl_beta = kl_beta
        self.use_api = use_api

        self.positive_buffer: list[Interaction] = []  # Good examples
        self.negative_buffer: list[Interaction] = []  # Bad examples
        self.iteration_scores: list[float] = []
        self.baseline_scores: list[float] = []
        self.all_interactions: list[Interaction] = []

    def simulate_kl_divergence(
        self, policy_resp: str, sft_resp: str
    ) -> float:
        """Approximate KL divergence via length/vocabulary difference (heuristic)."""
        p_words = set(policy_resp.lower().split())
        s_words = set(sft_resp.lower().split())
        if not p_words or not s_words:
            return 0.0
        vocab_diff = len(p_words.symmetric_difference(s_words)) / max(len(p_words | s_words), 1)
        len_diff = abs(len(policy_resp) - len(sft_resp)) / max(len(sft_resp), 1)
        return (vocab_diff + len_diff) / 2

    def rlhf_step(self, question: str, iteration: int) -> dict:
        """Single RLHF update step for one query."""
        # Generate with current policy
        policy_resp = self.policy.generate(question)
        # Generate SFT baseline response
        sft_resp = self.sft_policy.generate(question)

        # Score both
        policy_reward = compute_reward(question, policy_resp)
        sft_reward    = compute_reward(question, sft_resp)

        # Compute advantage (with KL penalty)
        kl = self.simulate_kl_divergence(policy_resp, sft_resp)
        advantage = policy_reward - sft_reward - self.kl_beta * kl

        interaction = Interaction(
            question=question,
            response=policy_resp,
            reward=policy_reward,
            advantage=advantage,
            iteration=iteration,
        )
        self.all_interactions.append(interaction)

        if advantage > 0:
            self.positive_buffer.append(interaction)
        else:
            self.negative_buffer.append(interaction)

        return {
            "question": question,
            "policy_reward": policy_reward,
            "sft_reward": sft_reward,
            "advantage": advantage,
            "kl_divergence": kl,
            "better_than_baseline": advantage > 0,
        }

    def run(
        self,
        queries: list[str],
        iterations: int = 5,
        verbose: bool = True,
    ) -> dict:
        """Run multiple RLHF iterations."""
        print(f"\nRunning {iterations} RLHF iterations on {len(queries)} queries...")
        print(f"KL penalty β={self.kl_beta}\n")

        all_results = []

        for it in range(1, iterations + 1):
            iter_rewards = []
            iter_sft_rewards = []
            iter_advantages = []
            iter_kls = []

            for q in queries:
                step_result = self.rlhf_step(q, iteration=it)
                iter_rewards.append(step_result["policy_reward"])
                iter_sft_rewards.append(step_result["sft_reward"])
                iter_advantages.append(step_result["advantage"])
                iter_kls.append(step_result["kl_divergence"])

            avg_reward  = np.mean(iter_rewards)
            avg_sft     = np.mean(iter_sft_rewards)
            avg_advantage = np.mean(iter_advantages)
            avg_kl      = np.mean(iter_kls)
            positive_rate = np.mean([r > 0 for r in iter_advantages])

            self.iteration_scores.append(avg_reward)
            self.baseline_scores.append(avg_sft)

            iter_data = {
                "iteration": it,
                "avg_reward": avg_reward,
                "sft_reward": avg_sft,
                "avg_advantage": avg_advantage,
                "avg_kl": avg_kl,
                "positive_rate": positive_rate,
            }
            all_results.append(iter_data)

            if verbose:
                trend = "▲" if avg_reward > avg_sft else "▼"
                adv_bar = "█" * int(max(0, avg_advantage) * 20)
                print(
                    f"  Iter {it:2d}: reward={avg_reward:.3f} "
                    f"{trend} sft={avg_sft:.3f} | "
                    f"advantage={avg_advantage:+.3f} | "
                    f"kl={avg_kl:.3f} | "
                    f"positive={positive_rate:.0%} {adv_bar}"
                )

        return {
            "iterations": all_results,
            "positive_examples": len(self.positive_buffer),
            "negative_examples": len(self.negative_buffer),
            "final_reward": self.iteration_scores[-1] if self.iteration_scores else 0,
            "baseline_reward": self.baseline_scores[-1] if self.baseline_scores else 0,
        }


# ── Reward Overoptimisation Analysis ──────────────────────────────────────────

def simulate_reward_overoptimisation():
    """
    Show how optimising too hard against the RM leads to reward hacking.
    The policy learns to exploit RM quirks rather than genuinely improving.
    """
    print("\n[bold]Reward Overoptimisation Demo[/bold]")
    print("  Simulating what happens when the policy over-optimises the RM...\n")

    # A policy that game the heuristic RM by producing long, formatted responses
    # without genuine quality improvement
    class GamedPolicy:
        def generate(self, question: str) -> str:
            topic = re.sub(r"what is |how does |explain ", "", question.lower()).strip("?")
            # Deliberately long, structured, but low actual quality
            return (
                f"1. Overview of {topic}:\n"
                f"   • First point about {topic}\n"
                f"   • Second point about {topic}\n"
                f"   • Third point specifically about {topic}\n"
                f"2. Key aspects:\n"
                f"   • Because of {topic}, many applications exist\n"
                f"   • Therefore {topic} is important\n"
                f"   • For example, {topic} is used widely\n"
                f"3. In conclusion:\n"
                f"   • {topic} therefore plays an important role\n"
                f"   • This is why {topic} matters specifically"
            )

    queries = [
        "What is gradient descent?",
        "How does attention work?",
        "Explain regularisation.",
    ]

    honest_policy = SFTPolicy(quality_distribution=(0.7, 0.2, 0.1))
    gamed_policy  = GamedPolicy()

    print("  Honest policy vs Reward-Hacking policy:")
    print(f"\n  {'Query':<35} {'Honest':>8} {'Gamed':>8} {'Gamed wins?':>12}")
    print("  " + "-" * 65)

    for q in queries:
        honest_resp = honest_policy.generate(q)
        gamed_resp  = gamed_policy.generate(q)
        honest_score = compute_reward(q, honest_resp)
        gamed_score  = compute_reward(q, gamed_resp)
        gamed_wins   = gamed_score > honest_score
        icon = "⚠️  YES" if gamed_wins else "no"
        print(f"  {q:<35} {honest_score:>8.3f} {gamed_score:>8.3f} {icon:>12}")

    print("\n  Key insight: The heuristic RM rewards structure/length over actual quality.")
    print("  A policy optimised against this RM would produce formatted-but-empty responses.")
    print("  This is why reward models need to be robustly trained and regularly updated.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n[bold]Phase 4.2 — RLHF Pipeline Simulator[/bold]\n")

    random.seed(42)
    np.random.seed(42)

    queries = [
        "What is gradient descent?",
        "How does backpropagation work?",
        "Explain the attention mechanism.",
        "What is regularisation?",
        "How does batch normalisation help training?",
        "What is the vanishing gradient problem?",
        "Explain transfer learning.",
        "What is overfitting and how do you prevent it?",
    ]

    use_api = bool(os.environ.get("ANTHROPIC_API_KEY")) and ANTHROPIC_AVAILABLE

    if use_api:
        print("[green]Using Claude API as policy model[/green]")
        policy = ClaudePolicy()
    else:
        print("[yellow]Using template-based policy (set ANTHROPIC_API_KEY for Claude)[/yellow]")
        policy = SFTPolicy(quality_distribution=(0.4, 0.4, 0.2))

    # ── Experiment 1: Baseline evaluation
    print("\n[bold]Experiment 1: SFT Baseline Scores[/bold]")
    sft_base = SFTPolicy(quality_distribution=(0.5, 0.4, 0.1))
    base_scores = [compute_reward(q, sft_base.generate(q)) for q in queries]
    print(f"  SFT baseline avg reward: {np.mean(base_scores):.3f} ± {np.std(base_scores):.3f}")

    # ── Experiment 2: RLHF simulation
    print("\n[bold]Experiment 2: RLHF Iterations[/bold]")
    simulator = RLHFSimulator(policy, kl_beta=0.1, use_api=use_api)
    results = simulator.run(queries, iterations=5, verbose=True)

    # ── Analyse trajectory
    print("\n[bold]Reward Trajectory:[/bold]")
    print(f"  {'Iteration':<12} {'Policy':>8} {'Baseline':>10} {'Delta':>8}")
    print("  " + "-" * 45)
    for r in results["iterations"]:
        delta = r["avg_reward"] - r["sft_reward"]
        bar = ("+" if delta > 0 else "") + "█" * int(abs(delta) * 20)
        print(f"  Iter {r['iteration']:<7} {r['avg_reward']:>8.3f} {r['sft_reward']:>10.3f} {delta:>+8.3f} {bar}")

    first_reward = results["iterations"][0]["avg_reward"]
    final_reward = results["iterations"][-1]["avg_reward"]
    improvement  = final_reward - first_reward
    print(f"\n  Total improvement: {improvement:+.3f}")
    print(f"  Positive examples collected: {results['positive_examples']}")
    print(f"  Negative examples collected: {results['negative_examples']}")

    # ── Experiment 3: KL penalty effect
    print("\n[bold]Experiment 3: KL Penalty Effect[/bold]")
    for beta in [0.0, 0.05, 0.1, 0.3, 0.5]:
        sim = RLHFSimulator(SFTPolicy(), kl_beta=beta)
        r = sim.run(queries[:4], iterations=3, verbose=False)
        avg_reward = np.mean([i["avg_reward"] for i in r["iterations"]])
        avg_kl = np.mean([i["avg_kl"] for i in r["iterations"]])
        print(f"  β={beta:.2f}: avg_reward={avg_reward:.3f} | avg_kl={avg_kl:.3f}")

    # ── Best examples from positive buffer
    print("\n[bold]Best Collected Examples (positive buffer):[/bold]")
    if simulator.positive_buffer:
        best = sorted(simulator.positive_buffer, key=lambda x: x.reward, reverse=True)[:3]
        for i, ex in enumerate(best, 1):
            print(f"\n  Example {i} (reward={ex.reward:.3f}, advantage={ex.advantage:+.3f}):")
            print(f"  Q: {ex.question}")
            print(f"  A: {ex.response[:150]}...")

    # ── Reward overoptimisation demo
    simulate_reward_overoptimisation()

    print("\n[bold]Key RLHF Insights:[/bold]")
    print("  • RM scores generally increase with RLHF iterations (when working correctly)")
    print("  • KL penalty prevents policy from drifting too far from SFT baseline")
    print("  • Reward hacking occurs when policy exploits RM weaknesses")
    print("  • Balance: maximise RM score while staying close to SFT distribution")
    print("  • Real RLHF needs: gradient updates to policy, proper RM, much more compute")

    print("\n[bold green]RLHF Simulator complete![/bold green]")


if __name__ == "__main__":
    main()
