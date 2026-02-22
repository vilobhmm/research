"""
Phase 1.2 — Embeddings Playground
===================================
Explores how embeddings capture semantic meaning:
  • Trains a word2vec-style skip-gram model from scratch
  • Tests semantic analogies: king - man + woman ≈ queen
  • Visualises embeddings via t-SNE
  • Builds a document similarity search index

Run:
    python phase1/embeddings_playground.py

Requirements: numpy, torch, scikit-learn, matplotlib, sentence-transformers
"""

import math
import re
from collections import Counter
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from rich import print
    from rich.console import Console
    from rich.table import Table
    console = Console()
except ImportError:
    console = None


# ── 1. Vocabulary & Corpus ───────────────────────────────────────────────────

MINI_CORPUS = """
king rules the kingdom with power and authority
queen rules the kingdom with grace and wisdom
man walks in the street with confidence
woman walks in the city with grace
prince is the son of king and will rule
princess is the daughter of queen and will lead
doctor heals the sick patients in hospital
nurse cares for patients and assists doctors
programmer writes code and builds software applications
engineer designs systems and solves technical problems
paris is the capital of france in europe
london is the capital of england in europe
berlin is the capital of germany in europe
rome is the capital of italy in europe
cat is a small furry pet animal
dog is a loyal friendly pet animal
fish swims in the ocean water
bird flies high in the sky
python is a programming language used by developers
java is a programming language used in enterprise
machine learning uses algorithms to learn from data
deep learning uses neural networks with many layers
transformer architecture uses attention mechanisms for language
embedding represents words as vectors in space
""".strip()


class Vocabulary:
    def __init__(self, min_freq: int = 1):
        self.min_freq = min_freq
        self.word2idx: dict[str, int] = {}
        self.idx2word: dict[int, str] = {}
        self.word_freq: dict[str, int] = {}

    def build(self, tokens: list[str]) -> "Vocabulary":
        freq = Counter(tokens)
        self.word_freq = {w: c for w, c in freq.items() if c >= self.min_freq}
        self.word2idx = {"<UNK>": 0}
        self.idx2word = {0: "<UNK>"}
        for i, w in enumerate(sorted(self.word_freq), start=1):
            self.word2idx[w] = i
            self.idx2word[i] = w
        return self

    def encode(self, word: str) -> int:
        return self.word2idx.get(word, 0)

    def __len__(self):
        return len(self.word2idx)


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


# ── 2. Skip-Gram Dataset ──────────────────────────────────────────────────────

class SkipGramDataset(Dataset):
    """Generate (center, context) pairs with negative sampling."""

    def __init__(self, tokens: list[str], vocab: Vocabulary, window: int = 3):
        self.vocab = vocab
        encoded = [vocab.encode(t) for t in tokens]
        self.pairs = []
        for i, center in enumerate(encoded):
            for j in range(max(0, i - window), min(len(encoded), i + window + 1)):
                if i != j:
                    self.pairs.append((center, encoded[j]))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        center, context = self.pairs[idx]
        return torch.tensor(center), torch.tensor(context)


# ── 3. Skip-Gram Model ────────────────────────────────────────────────────────

class SkipGram(nn.Module):
    """
    Two embedding tables:
      - input  embeddings: center word → vector
      - output embeddings: context word → vector
    Training objective: maximise dot-product for positive pairs,
    minimise for negative samples (NCE / negative sampling).
    """

    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.input_emb  = nn.Embedding(vocab_size, embed_dim)
        self.output_emb = nn.Embedding(vocab_size, embed_dim)
        nn.init.uniform_(self.input_emb.weight,  -0.5 / embed_dim, 0.5 / embed_dim)
        nn.init.zeros_(self.output_emb.weight)

    def forward(
        self,
        center: torch.Tensor,
        context: torch.Tensor,
        negatives: torch.Tensor,
    ) -> torch.Tensor:
        center_emb  = self.input_emb(center)                   # (B, D)
        context_emb = self.output_emb(context)                 # (B, D)
        neg_emb     = self.output_emb(negatives)               # (B, K, D)

        # Positive score
        pos_score = (center_emb * context_emb).sum(dim=-1)    # (B,)
        pos_loss  = F.logsigmoid(pos_score)

        # Negative scores
        neg_score = torch.bmm(neg_emb, center_emb.unsqueeze(-1)).squeeze(-1)  # (B, K)
        neg_loss  = F.logsigmoid(-neg_score).sum(dim=-1)

        return -(pos_loss + neg_loss).mean()

    @property
    def embeddings(self) -> np.ndarray:
        """Return normalised input embeddings as numpy array."""
        w = self.input_emb.weight.detach().numpy()
        return w / (np.linalg.norm(w, axis=1, keepdims=True) + 1e-8)


# ── 4. Training ───────────────────────────────────────────────────────────────

def train_skip_gram(
    corpus: str = MINI_CORPUS,
    embed_dim: int = 64,
    epochs: int = 200,
    neg_samples: int = 5,
) -> tuple[SkipGram, Vocabulary]:
    tokens = tokenize(corpus)
    vocab = Vocabulary().build(tokens)
    print(f"Vocabulary size: {len(vocab)}")

    dataset = SkipGramDataset(tokens, vocab, window=3)
    loader  = DataLoader(dataset, batch_size=64, shuffle=True)

    model = SkipGram(len(vocab), embed_dim)
    optim = torch.optim.Adam(model.parameters(), lr=0.01)

    # Noise distribution for negative sampling (unigram^0.75)
    freq = np.array([
        vocab.word_freq.get(vocab.idx2word.get(i, ""), 1)
        for i in range(len(vocab))
    ], dtype=float)
    noise_probs = freq ** 0.75
    noise_probs /= noise_probs.sum()

    print(f"Training Skip-Gram for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for center, context in loader:
            B = center.size(0)
            negatives = torch.from_numpy(
                np.random.choice(len(vocab), size=(B, neg_samples), p=noise_probs)
            ).long()
            loss = model(center, context, negatives)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item()
        if epoch % 50 == 0:
            print(f"  Epoch {epoch:3d}/{epochs} | loss={total_loss / len(loader):.4f}")

    return model, vocab


# ── 5. Semantic Operations ───────────────────────────────────────────────────

class EmbeddingAnalyser:
    def __init__(self, embeddings: np.ndarray, vocab: Vocabulary):
        self.emb   = embeddings          # (V, D) normalised
        self.vocab = vocab

    def vec(self, word: str) -> Optional[np.ndarray]:
        idx = self.vocab.encode(word)
        return self.emb[idx] if idx != 0 else None

    def most_similar(self, word: str, topk: int = 5) -> list[tuple[str, float]]:
        v = self.vec(word)
        if v is None:
            return []
        sims = self.emb @ v
        top  = np.argsort(-sims)[1 : topk + 1]
        return [(self.vocab.idx2word[i], float(sims[i])) for i in top]

    def analogy(self, a: str, b: str, c: str, topk: int = 3) -> list[tuple[str, float]]:
        """a:b :: c:? — answer = b - a + c"""
        va, vb, vc = self.vec(a), self.vec(b), self.vec(c)
        if any(v is None for v in [va, vb, vc]):
            return []
        query = vb - va + vc
        query = query / (np.linalg.norm(query) + 1e-8)
        sims  = self.emb @ query
        # Exclude input words
        exclude = {self.vocab.encode(w) for w in [a, b, c]}
        results = []
        for idx in np.argsort(-sims):
            if idx not in exclude:
                results.append((self.vocab.idx2word[idx], float(sims[idx])))
            if len(results) == topk:
                break
        return results

    def cosine_sim(self, w1: str, w2: str) -> float:
        v1, v2 = self.vec(w1), self.vec(w2)
        if v1 is None or v2 is None:
            return 0.0
        return float(v1 @ v2)


# ── 6. t-SNE Visualisation ───────────────────────────────────────────────────

def plot_tsne(embeddings: np.ndarray, vocab: Vocabulary, save_path: str = "/tmp/embeddings_tsne.png"):
    from sklearn.manifold import TSNE

    # Select interesting words
    words_of_interest = [
        "king", "queen", "man", "woman", "prince", "princess",
        "doctor", "nurse", "programmer", "engineer",
        "paris", "london", "berlin", "rome",
        "cat", "dog", "bird", "fish",
        "python", "java", "machine", "deep",
    ]
    idxs  = [vocab.encode(w) for w in words_of_interest if vocab.encode(w) != 0]
    words = [w for w in words_of_interest if vocab.encode(w) != 0]
    emb_sub = embeddings[idxs]

    tsne = TSNE(n_components=2, perplexity=min(5, len(words) - 1), random_state=42, n_iter=1000)
    coords = tsne.fit_transform(emb_sub)

    # Color groups
    color_map = {
        "royalty": ["king", "queen", "prince", "princess"],
        "professions": ["doctor", "nurse", "programmer", "engineer", "man", "woman"],
        "capitals": ["paris", "london", "berlin", "rome"],
        "animals": ["cat", "dog", "bird", "fish"],
        "tech": ["python", "java", "machine", "deep"],
    }
    colors = {"royalty": "#e74c3c", "professions": "#3498db",
              "capitals": "#2ecc71", "animals": "#f39c12", "tech": "#9b59b6"}
    word_color = {}
    for group, wds in color_map.items():
        for w in wds:
            word_color[w] = colors[group]

    fig, ax = plt.subplots(figsize=(12, 8))
    for i, (word, (x, y)) in enumerate(zip(words, coords)):
        c = word_color.get(word, "#95a5a6")
        ax.scatter(x, y, color=c, s=80, zorder=3)
        ax.annotate(word, (x, y), fontsize=9, ha="center", va="bottom",
                    xytext=(0, 5), textcoords="offset points")

    # Legend
    for group, color in colors.items():
        ax.scatter([], [], color=color, label=group, s=80)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title("t-SNE of Word Embeddings\n(semantically related words cluster together)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"t-SNE plot saved → {save_path}")


# ── 7. Document Similarity Search ────────────────────────────────────────────

class DocumentSearch:
    """Simple bag-of-embeddings document retriever."""

    def __init__(self, analyser: EmbeddingAnalyser):
        self.analyser = analyser
        self.docs: list[str] = []
        self.doc_vecs: list[np.ndarray] = []

    def _doc_vector(self, text: str) -> np.ndarray:
        tokens = tokenize(text)
        vecs = [self.analyser.vec(t) for t in tokens if self.analyser.vec(t) is not None]
        if not vecs:
            return np.zeros(self.analyser.emb.shape[1])
        v = np.mean(vecs, axis=0)
        return v / (np.linalg.norm(v) + 1e-8)

    def index(self, docs: list[str]) -> "DocumentSearch":
        self.docs = docs
        self.doc_vecs = [self._doc_vector(d) for d in docs]
        return self

    def search(self, query: str, topk: int = 3) -> list[tuple[float, str]]:
        qv = self._doc_vector(query)
        sims = [float(qv @ dv) for dv in self.doc_vecs]
        ranked = sorted(enumerate(sims), key=lambda x: -x[1])
        return [(sims[i], self.docs[i]) for i, _ in ranked[:topk]]


# ── 8. Main Demo ──────────────────────────────────────────────────────────────

def main():
    print("\n[bold]Phase 1.2 — Embeddings Playground[/bold]\n")

    # Train
    model, vocab = train_skip_gram(embed_dim=64, epochs=200)
    analyser = EmbeddingAnalyser(model.embeddings, vocab)

    # ── Semantic similarity
    print("\n[bold]1. Semantic Similarity[/bold]")
    pairs = [("king", "queen"), ("man", "woman"), ("paris", "london"),
             ("doctor", "nurse"), ("python", "java"), ("cat", "dog")]
    for w1, w2 in pairs:
        sim = analyser.cosine_sim(w1, w2)
        bar = "█" * int(sim * 20)
        print(f"  {w1:12s} ↔ {w2:12s} : {sim:.3f}  {bar}")

    # ── Analogies
    print("\n[bold]2. Analogies  (a:b :: c:?)[/bold]")
    analogies = [
        ("man", "king",   "woman"),    # → queen
        ("man", "doctor", "woman"),    # → nurse
        ("paris", "france", "london"), # → england
        ("king", "man",   "queen"),    # → woman
    ]
    for a, b, c in analogies:
        results = analyser.analogy(a, b, c, topk=3)
        top = results[0] if results else ("?", 0.0)
        print(f"  {a} : {b} :: {c} : {top[0]}  (sim={top[1]:.3f})")
        if len(results) > 1:
            others = ", ".join(f"{w}({s:.2f})" for w, s in results[1:])
            print(f"    also: {others}")

    # ── Most similar
    print("\n[bold]3. Most Similar Words[/bold]")
    for word in ["king", "programmer", "paris", "cat"]:
        sims = analyser.most_similar(word, topk=4)
        sim_str = ", ".join(f"{w}({s:.2f})" for w, s in sims)
        print(f"  {word:14s} → {sim_str}")

    # ── Dimension analysis
    print("\n[bold]4. Embedding Dimensions Analysis[/bold]")
    emb = model.embeddings
    print(f"  Embedding matrix: {emb.shape}")
    print(f"  Mean norm (should be ~1.0): {np.linalg.norm(emb, axis=1).mean():.3f}")
    print(f"  Avg pairwise cosine: {(emb @ emb.T).mean():.4f}")

    # Gender direction (king-queen, man-woman)
    gender_vecs = []
    for w1, w2 in [("king", "queen"), ("man", "woman"), ("prince", "princess")]:
        v1, v2 = analyser.vec(w1), analyser.vec(w2)
        if v1 is not None and v2 is not None:
            gender_vecs.append(v2 - v1)
    if gender_vecs:
        gender_dir = np.mean(gender_vecs, axis=0)
        gender_dir /= np.linalg.norm(gender_dir) + 1e-8
        print(f"\n  Gender direction (king→queen / man→woman / prince→princess):")
        for w in ["doctor", "nurse", "engineer", "programmer", "king", "queen"]:
            v = analyser.vec(w)
            if v is not None:
                proj = float(v @ gender_dir)
                bar = ("♀" if proj > 0 else "♂") + " " + "█" * int(abs(proj * 15))
                print(f"    {w:14s} : {proj:+.3f}  {bar}")

    # ── Document similarity
    print("\n[bold]5. Document Similarity Search[/bold]")
    documents = [
        "The king and queen rule the kingdom together.",
        "Python and Java are popular programming languages.",
        "Doctors and nurses care for patients in hospitals.",
        "Paris and London are great European capital cities.",
        "Cats and dogs are common household pets.",
        "Machine learning uses deep neural networks for AI.",
    ]
    searcher = DocumentSearch(analyser).index(documents)
    queries = ["royal family and monarchy", "medical professionals healthcare",
               "software development coding"]
    for q in queries:
        results = searcher.search(q, topk=2)
        print(f"\n  Query: '{q}'")
        for sim, doc in results:
            print(f"    [{sim:.3f}] {doc}")

    # ── t-SNE visualisation
    print("\n[bold]6. t-SNE Visualisation[/bold]")
    try:
        plot_tsne(model.embeddings, vocab)
    except Exception as e:
        print(f"  t-SNE failed: {e}")

    print("\n[bold green]Embeddings Playground complete![/bold green]")
    print("Key observations:")
    print("  • Semantically related words cluster together in embedding space")
    print("  • Linear arithmetic captures gender, geography, profession analogies")
    print("  • Bag-of-embeddings enables simple document retrieval")


if __name__ == "__main__":
    main()
