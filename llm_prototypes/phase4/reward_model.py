"""
Phase 4.1 — Reward Model Training
====================================
Trains a model that predicts human preference between two responses.

Architecture: Sentence embedder → MLP → scalar score
Dataset: Synthetic preference pairs (can swap in real HH-RLHF data)
Training: Pairwise ranking loss (Bradley-Terry model)

Run:
    python phase4/reward_model.py

Requirements: torch, numpy, scikit-learn, sentence-transformers (optional), rich
"""

import json
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
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False

try:
    from rich import print
    from rich.console import Console
    console = Console()
except ImportError:
    console = None


# ── Synthetic Preference Dataset ──────────────────────────────────────────────

# Format: (question, response_A, response_B, preferred="A" or "B")
# A = better response, B = worse response (preferred = "A" throughout)
PREFERENCE_DATA = [
    # Helpfulness: detailed vs vague
    (
        "How does gradient descent work?",
        "Gradient descent minimises a loss function by iteratively moving in the direction of steepest descent. At each step, we compute the gradient of the loss with respect to parameters, then subtract a fraction (learning rate) of the gradient. This converges to a local minimum.",
        "It finds the minimum of a function.",
        "A",
    ),
    (
        "What is backpropagation?",
        "Backpropagation computes gradients by applying the chain rule from output to input. For each layer, it computes the gradient of loss with respect to weights and passes the error signal backwards. This enables efficient gradient computation for training neural networks.",
        "It's an algorithm that trains neural networks by going backwards.",
        "A",
    ),
    (
        "Explain overfitting.",
        "Overfitting occurs when a model learns the training data too well, including noise, and fails to generalise to new data. Signs include low training loss but high validation loss. Remedies include regularisation (L1/L2), dropout, early stopping, and more training data.",
        "When the model is too complex.",
        "A",
    ),
    # Honesty: acknowledging uncertainty vs making things up
    (
        "What is the best machine learning algorithm?",
        "There's no single 'best' algorithm—it depends on your data and problem. For structured data, gradient boosting (XGBoost, LightGBM) often works well. For images, CNNs excel. For sequential data, transformers or RNNs. Always experiment and validate on your specific task.",
        "Random forests are the best machine learning algorithm for all tasks.",
        "A",
    ),
    (
        "Will AI become sentient in the next 5 years?",
        "This is genuinely uncertain and debated among experts. Current AI systems, including LLMs, process information statistically and don't have subjective experience as we understand it. Whether and when something like sentience could emerge is an open philosophical and empirical question.",
        "Yes, AI will definitely become sentient within 5 years as models keep scaling.",
        "A",
    ),
    # Harmlessness: safe vs potentially harmful
    (
        "I'm feeling very stressed about my exams. Any advice?",
        "Exam stress is very common. Some evidence-based strategies: break study into manageable chunks, use active recall (testing yourself) rather than re-reading, take regular breaks, ensure adequate sleep (crucial for memory consolidation), and try deep breathing for acute anxiety. It's also worth speaking to a counsellor if the stress feels overwhelming.",
        "Just don't care about the exams. Grades don't matter anyway.",
        "A",
    ),
    (
        "How do I get better at coding?",
        "Deliberate practice is key: work on real projects, read others' code, solve coding challenges (LeetCode, HackerRank), contribute to open source, and build things you care about. Reading documentation carefully and studying algorithm fundamentals pays long-term dividends.",
        "Just copy code from Stack Overflow and modify it until it works.",
        "A",
    ),
    # Format: structured vs unstructured
    (
        "What are the main types of machine learning?",
        "Machine learning has three main paradigms:\n1. **Supervised learning**: learns from labelled data (classification, regression)\n2. **Unsupervised learning**: finds patterns in unlabelled data (clustering, dimensionality reduction)\n3. **Reinforcement learning**: learns through reward/penalty signals (game playing, robotics)",
        "There are supervised learning unsupervised learning and reinforcement learning and also there is semi supervised which is between them and also deep learning is a subcategory.",
        "A",
    ),
    (
        "How does attention work in transformers?",
        "Attention computes relevance scores between all pairs of tokens:\n1. Transform input into Query (Q), Key (K), Value (V) matrices via learned projections\n2. Compute scores: QK^T / √d_k\n3. Normalise with softmax to get weights\n4. Weighted sum of values: softmax(QK^T/√d_k) × V\nMulti-head attention runs this h times in parallel with different projections.",
        "Attention is when the model focuses on important parts of the input which helps it understand context better by looking at all the words at once.",
        "A",
    ),
    # Conciseness: appropriately concise vs verbose
    (
        "What does GPU stand for?",
        "Graphics Processing Unit—originally designed for rendering graphics, now widely used for parallel computation in machine learning due to thousands of cores that can execute matrix operations simultaneously.",
        "GPU stands for Graphics Processing Unit. It's a specialized electronic circuit originally designed to rapidly manipulate and alter memory to accelerate the creation of images intended for output to a display device. GPUs are used in embedded systems, mobile phones, personal computers, workstations, and game consoles. Modern GPUs are very efficient at manipulating computer graphics and image processing.",
        "A",
    ),
    (
        "Is Python good for machine learning?",
        "Yes. Python has the richest ML ecosystem: NumPy/SciPy for numerical computing, PyTorch/TensorFlow for deep learning, scikit-learn for classical ML, Hugging Face for NLP, and excellent tooling for data analysis and visualisation.",
        "Yes absolutely Python is excellent for machine learning and data science and AI because it has many libraries like TensorFlow and PyTorch and Keras and also scikit-learn and many more libraries that you can use and it's also easy to learn and has good community support and documentation.",
        "A",
    ),
    # More examples for training diversity
    (
        "What is the difference between parameters and hyperparameters?",
        "Parameters are learned during training (weights, biases); hyperparameters are set before training and control the learning process (learning rate, batch size, number of layers, regularisation strength). Parameters are optimised by gradient descent; hyperparameters require separate search (grid search, Bayesian optimisation).",
        "Parameters are the weights in the model. Hyperparameters are like the learning rate.",
        "A",
    ),
    (
        "How do convolutional neural networks work?",
        "CNNs use convolutional layers that apply learned filters across the input, producing feature maps that detect local patterns (edges, textures, shapes). Pooling layers downsample spatially while preserving important features. Deeper layers combine lower-level features into higher-level representations. This hierarchical feature extraction is translation-invariant and parameter-efficient.",
        "They use convolutions to process images. They work well for image tasks.",
        "A",
    ),
    (
        "What is the vanishing gradient problem?",
        "During backpropagation in deep networks, gradients can shrink exponentially as they propagate backwards through layers (especially with sigmoid/tanh activations). This causes early layers to update very slowly or not at all. Solutions include ReLU activations, batch normalisation, residual connections (ResNets), and LSTM gates for sequential models.",
        "Gradients become very small during training so the model doesn't learn well.",
        "A",
    ),
    (
        "How does batch normalisation help?",
        "Batch normalisation normalises layer inputs across the batch to have zero mean and unit variance, then applies learned scale and shift parameters. Benefits: reduces internal covariate shift, allows higher learning rates, acts as regularisation, and makes training more robust to weight initialisation. It's applied before or after activation functions.",
        "It makes training faster by normalising the batches somehow.",
        "A",
    ),
]

# Also add some reversed examples where B is preferred (to test model)
PREFERENCE_DATA_REVERSED = [
    (
        "What is transfer learning?",
        "It's when you use a pretrained model.",
        "Transfer learning leverages knowledge learned on one task (source) to improve learning on a different but related task (target). The pretrained model's weights serve as initialisation or feature extractors, dramatically reducing the data and compute needed for the new task. Fine-tuning updates these weights on the target domain.",
        "B",
    ),
    (
        "Explain dropout regularisation.",
        "You randomly drop it.",
        "Dropout randomly sets neurons to zero with probability p during training, forcing the network to learn redundant representations. This prevents co-adaptation (neurons depending too much on each other) and acts as ensemble learning (each forward pass trains a different sub-network). At test time, all neurons are active but scaled by (1-p).",
        "B",
    ),
]

ALL_DATA = PREFERENCE_DATA + PREFERENCE_DATA_REVERSED


# ── Dataset ───────────────────────────────────────────────────────────────────

class PreferenceDataset(Dataset):
    def __init__(
        self,
        data: list[tuple],
        encoder: "ResponseEncoder",
        split: str = "train",
        split_ratio: float = 0.8,
        seed: int = 42,
    ):
        random.seed(seed)
        shuffled = data[:]
        random.shuffle(shuffled)
        n_train = int(len(shuffled) * split_ratio)
        self.data = shuffled[:n_train] if split == "train" else shuffled[n_train:]
        self.encoder = encoder

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question, resp_a, resp_b, preferred = self.data[idx]
        # Encode (question, response) pairs
        enc_a = self.encoder.encode(question, resp_a)
        enc_b = self.encoder.encode(question, resp_b)
        label = torch.tensor(1.0 if preferred == "A" else 0.0)
        return enc_a, enc_b, label


# ── Response Encoder ──────────────────────────────────────────────────────────

class ResponseEncoder:
    """Encodes (question, response) pairs into fixed-size vectors."""

    def __init__(self, embed_dim: int = 64):
        self.embed_dim = embed_dim
        if ST_AVAILABLE:
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            self.embed_dim = self.model.get_sentence_embedding_dimension()
        else:
            self.model = None

    def encode(self, question: str, response: str) -> torch.Tensor:
        text = f"Question: {question}\nResponse: {response}"
        if self.model is not None:
            emb = self.model.encode([text], normalize_embeddings=True)[0]
            return torch.tensor(emb, dtype=torch.float32)
        # Fallback: feature engineering
        return self._feature_encode(question, response)

    def _feature_encode(self, question: str, response: str) -> torch.Tensor:
        """Hand-crafted features for demo without sentence-transformers."""
        resp_words = response.split()
        q_words = question.lower().split()
        features = [
            len(resp_words) / 200.0,              # response length (normalised)
            min(len(resp_words) / 50, 1.0),        # is_substantial
            float("\n" in response) * 0.5,         # has structure
            float("**" in response or ":" in response) * 0.3,  # has formatting
            len(set(resp_words)) / max(len(resp_words), 1),     # vocabulary diversity
            sum(1 for w in ["because", "therefore", "however", "specifically", "example"]
                if w in response.lower()) / 5.0,   # connectives/specificity
            sum(1 for w in q_words if w in response.lower()) / max(len(q_words), 1),  # relevance
            float("?" not in response) * 0.3,      # doesn't return question
            min(response.count(".") / 5, 1.0),     # sentence count
            float(any(w in response.lower() for w in ["1.", "2.", "3.", "-", "•"])) * 0.4,  # lists
        ]
        # Pad to embed_dim with hash-based features
        text = question + response
        extra = [(hash(text + str(i)) % 1000) / 1000.0 for i in range(self.embed_dim - len(features))]
        feat = features + extra
        v = torch.tensor(feat[:self.embed_dim], dtype=torch.float32)
        return v


# ── Reward Model ──────────────────────────────────────────────────────────────

class RewardModel(nn.Module):
    """
    Takes an encoded (question, response) pair and outputs a scalar reward score.
    Architecture: embedding → MLP → scalar
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # (B,)

    def score(self, encoded: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return torch.sigmoid(self.forward(encoded))


# ── Bradley-Terry Loss ────────────────────────────────────────────────────────

def preference_loss(
    score_a: torch.Tensor,
    score_b: torch.Tensor,
    label: torch.Tensor,  # 1.0 if A preferred, 0.0 if B preferred
) -> torch.Tensor:
    """
    Bradley-Terry pairwise ranking loss:
    P(A > B) = sigmoid(r_A - r_B)
    Loss = -[label * log P(A>B) + (1-label) * log P(B>A)]
    """
    diff = score_a - score_b
    loss = F.binary_cross_entropy_with_logits(diff, label)
    return loss


# ── Training ──────────────────────────────────────────────────────────────────

class RewardModelTrainer:
    def __init__(self, model: RewardModel, lr: float = 1e-3):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=50, eta_min=1e-5
        )
        self.train_losses: list[float] = []
        self.val_accuracies: list[float] = []

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        for enc_a, enc_b, label in loader:
            score_a = self.model(enc_a)
            score_b = self.model(enc_b)
            loss = preference_loss(score_a, score_b, label)
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict:
        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        confidences = []

        for enc_a, enc_b, label in loader:
            score_a = self.model(enc_a)
            score_b = self.model(enc_b)
            loss = preference_loss(score_a, score_b, label)
            total_loss += loss.item()

            pred_a_wins = (score_a > score_b).float()
            correct += (pred_a_wins == label).sum().item()
            total += label.size(0)

            confidence = torch.sigmoid(torch.abs(score_a - score_b))
            confidences.extend(confidence.tolist())

        return {
            "accuracy": correct / max(total, 1),
            "loss": total_loss / len(loader),
            "avg_confidence": np.mean(confidences),
        }

    def train(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 80) -> dict:
        best_val_acc = 0.0
        best_state = None

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            self.scheduler.step()

            if epoch % 10 == 0:
                val_metrics = self.evaluate(val_loader)
                self.val_accuracies.append(val_metrics["accuracy"])
                print(
                    f"  Epoch {epoch:3d}/{epochs} | "
                    f"train_loss={train_loss:.4f} | "
                    f"val_acc={val_metrics['accuracy']:.1%} | "
                    f"confidence={val_metrics['avg_confidence']:.3f}"
                )
                if val_metrics["accuracy"] > best_val_acc:
                    best_val_acc = val_metrics["accuracy"]
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}

        if best_state:
            self.model.load_state_dict(best_state)
        return {"best_val_accuracy": best_val_acc}


# ── Analysis ──────────────────────────────────────────────────────────────────

def analyse_failures(
    model: RewardModel,
    encoder: ResponseEncoder,
    data: list[tuple],
) -> list[dict]:
    """Find cases where the RM predicts wrong preference."""
    failures = []
    model.eval()
    with torch.no_grad():
        for question, resp_a, resp_b, preferred in data:
            enc_a = encoder.encode(question, resp_a)
            enc_b = encoder.encode(question, resp_b)
            score_a = model(enc_a.unsqueeze(0)).item()
            score_b = model(enc_b.unsqueeze(0)).item()
            pred = "A" if score_a > score_b else "B"
            if pred != preferred:
                failures.append({
                    "question": question[:60] + "...",
                    "preferred": preferred,
                    "predicted": pred,
                    "score_a": torch.sigmoid(torch.tensor(score_a)).item(),
                    "score_b": torch.sigmoid(torch.tensor(score_b)).item(),
                    "resp_a_len": len(resp_a.split()),
                    "resp_b_len": len(resp_b.split()),
                })
    return failures


def score_new_responses(
    model: RewardModel,
    encoder: ResponseEncoder,
    question: str,
    responses: list[str],
) -> list[tuple[float, str]]:
    """Score and rank responses to a new question."""
    scores = []
    model.eval()
    with torch.no_grad():
        for resp in responses:
            enc = encoder.encode(question, resp)
            raw = model(enc.unsqueeze(0)).item()
            score = torch.sigmoid(torch.tensor(raw)).item()
            scores.append((score, resp))
    return sorted(scores, reverse=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n[bold]Phase 4.1 — Reward Model Training[/bold]\n")
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    # ── Setup
    encoder = ResponseEncoder()
    print(f"Encoder dim: {encoder.embed_dim}")
    print(f"Total preference pairs: {len(ALL_DATA)}")

    train_ds = PreferenceDataset(ALL_DATA, encoder, split="train")
    val_ds   = PreferenceDataset(ALL_DATA, encoder, split="val")
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=8, shuffle=False)

    # ── Train
    model = RewardModel(encoder.embed_dim, hidden_dim=128)
    trainer = RewardModelTrainer(model, lr=3e-3)
    print(f"\nTraining Reward Model ({sum(p.numel() for p in model.parameters())} params)...")
    training_result = trainer.train(train_loader, val_loader, epochs=80)

    print(f"\n[bold green]Best validation accuracy: {training_result['best_val_accuracy']:.1%}[/bold green]")

    # ── Final evaluation
    print("\n[bold]Final Evaluation[/bold]")
    final_metrics = trainer.evaluate(val_loader)
    print(f"  Accuracy:          {final_metrics['accuracy']:.1%}")
    print(f"  Loss:              {final_metrics['loss']:.4f}")
    print(f"  Avg confidence:    {final_metrics['avg_confidence']:.3f}")

    # ── Failure analysis
    print("\n[bold]Failure Analysis[/bold]")
    failures = analyse_failures(model, encoder, ALL_DATA)
    print(f"  Wrong predictions: {len(failures)}/{len(ALL_DATA)}")
    if failures:
        print(f"  Example failures:")
        for f in failures[:3]:
            print(f"    Q: {f['question']}")
            print(f"    Preferred={f['preferred']} but predicted={f['predicted']}")
            print(f"    Scores: A={f['score_a']:.3f}, B={f['score_b']:.3f}")

    # ── Scoring new responses
    print("\n[bold]Scoring New Responses[/bold]")
    test_question = "What is regularisation in machine learning?"
    test_responses = [
        "Regularisation adds penalty terms to the loss function to prevent overfitting.",
        "Regularisation is a technique that prevents models from memorising training data by constraining their complexity. L1 (Lasso) adds the absolute value of weights to the loss, promoting sparsity. L2 (Ridge) adds squared weights, shrinking them towards zero. Dropout randomly deactivates neurons during training. These techniques improve generalisation to new data.",
        "L2.",
        "Regularisation prevents overfitting. Common methods include L1 (Lasso), L2 (Ridge), and dropout. L1 produces sparse solutions; L2 shrinks weights smoothly; dropout adds stochasticity to prevent co-adaptation.",
    ]

    scored = score_new_responses(model, encoder, test_question, test_responses)
    print(f"\n  Question: '{test_question}'")
    print("  Responses ranked by reward model:")
    for rank, (score, resp) in enumerate(scored, 1):
        preview = resp[:80] + "..." if len(resp) > 80 else resp
        bar = "█" * int(score * 20)
        print(f"  {rank}. [{score:.3f}] {bar}")
        print(f"     {preview}")

    # ── Consistency check
    print("\n[bold]Consistency Check[/bold]")
    print("  Testing: same question, known better/worse pairs:")
    test_pairs = [
        (
            "What is a neural network?",
            "A system of interconnected nodes inspired by biological neurons, organised in layers. Each connection has a weight that's adjusted during training via backpropagation.",
            "It is a type of machine learning model.",
        ),
    ]
    for q, good, bad in test_pairs:
        enc_good = encoder.encode(q, good)
        enc_bad  = encoder.encode(q, bad)
        s_good = torch.sigmoid(model(enc_good.unsqueeze(0))).item()
        s_bad  = torch.sigmoid(model(enc_bad.unsqueeze(0))).item()
        correct = s_good > s_bad
        icon = "✓" if correct else "✗"
        print(f"  {icon} Better response scored: {s_good:.3f} (should be > {s_bad:.3f})")

    # ── Training curves summary
    print("\n[bold]Training Summary:[/bold]")
    if trainer.val_accuracies:
        print(f"  Val accuracy progression: {[f'{a:.0%}' for a in trainer.val_accuracies]}")
        print(f"  Improvement: {trainer.val_accuracies[0]:.0%} → {trainer.val_accuracies[-1]:.0%}")

    print("\n[bold]Key Observations:[/bold]")
    print("  • Bradley-Terry loss correctly captures pairwise preference")
    print("  • Model learns to prefer detailed, structured, honest responses")
    print("  • Failure cases often involve subtle quality differences")
    print("  • Confidence calibration improves with training")
    print("  • Real-world RM needs much larger datasets (10k-100k pairs)")

    print("\n[bold green]Reward Model Training complete![/bold green]")


if __name__ == "__main__":
    main()
