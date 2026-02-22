"""
Phase 2.2 — Retrieval Quality Analysis & Optimisation
========================================================
Deep dive into what makes retrieval work:
  • Multiple chunking strategies (fixed, sentence, sliding window)
  • Multiple embedding models (if available)
  • Hybrid retrieval (BM25 + dense)
  • Metrics: Recall@k, MRR, Precision@k

Run:
    python phase2/retrieval_optimizer.py

Requirements: numpy, scikit-learn, sentence-transformers (optional), rich
"""

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False

try:
    from rich import print
    from rich.console import Console
    from rich.table import Table
    console = Console()
except ImportError:
    console = None

# ── Sample Corpus & Ground Truth ─────────────────────────────────────────────

DOCUMENTS = {
    "attention_paper": """
The attention mechanism in transformers computes scaled dot-product attention.
Given queries Q, keys K, and values V, the output is: softmax(QK^T/sqrt(d_k))V.
Multi-head attention runs h parallel attention functions, projecting queries, keys and values
h times with different linear projections to dimensions d_k, d_k and d_v respectively.
The outputs are concatenated and projected, resulting in a final output.
This allows the model to jointly attend to information from different representation subspaces.
Positional encodings are added to token embeddings using sine and cosine functions of different
frequencies to provide sequence order information to the model. The encoder-decoder architecture
uses self-attention, cross-attention, and feed-forward layers with residual connections.
The feed-forward network applies two linear transformations with a ReLU activation in between.
""",
    "rlhf_paper": """
Reinforcement learning from human feedback trains language models using three stages.
First, supervised fine-tuning creates an initial policy from human demonstrations.
Second, a reward model is trained on human preference comparisons between model outputs.
Third, the policy is optimised using PPO to maximise reward model scores while maintaining
proximity to the original SFT policy through a KL divergence penalty.
The KL penalty prevents reward hacking and mode collapse during optimisation.
Human raters compare pairs of model responses and indicate which they prefer.
These preference pairs form the training data for the reward model.
The reward model learns to predict which responses humans would prefer.
PPO clips the policy gradient to prevent destructively large updates.
""",
    "rag_paper": """
Retrieval-augmented generation combines parametric and non-parametric memory.
The retrieval component uses a dense passage retrieval model based on BERT encoders.
A document index stores pre-computed embeddings for all passages in the knowledge base.
At inference time, the query is encoded and the most similar passages are retrieved.
Retrieved passages are concatenated with the query and fed to the generator.
RAG-Sequence retrieves once and conditions the entire sequence on those documents.
RAG-Token retrieves documents for each generated token separately.
The model is trained end-to-end with gradients flowing through the generator.
Dense retrieval outperforms BM25 sparse retrieval on most knowledge-intensive tasks.
The model learns to retrieve relevant passages that support accurate generation.
""",
}

GROUND_TRUTH = [
    {"query": "How does scaled dot-product attention work?",
     "relevant_docs": ["attention_paper"], "relevant_chunks": []},
    {"query": "What is the KL divergence penalty in RLHF?",
     "relevant_docs": ["rlhf_paper"], "relevant_chunks": []},
    {"query": "How does dense retrieval compare to BM25?",
     "relevant_docs": ["rag_paper"], "relevant_chunks": []},
    {"query": "What are positional encodings?",
     "relevant_docs": ["attention_paper"], "relevant_chunks": []},
    {"query": "How is the reward model trained?",
     "relevant_docs": ["rlhf_paper"], "relevant_chunks": []},
    {"query": "What is RAG-Sequence?",
     "relevant_docs": ["rag_paper"], "relevant_chunks": []},
]


# ── Chunking Strategies ───────────────────────────────────────────────────────

@dataclass
class Chunk:
    text: str
    doc_id: str
    chunk_id: str


class FixedSizeChunker:
    """Split by word count with optional overlap."""
    def __init__(self, size: int = 50, overlap: int = 10):
        self.size = size
        self.overlap = overlap
        self.name = f"Fixed({size}, overlap={overlap})"

    def chunk(self, documents: dict[str, str]) -> list[Chunk]:
        chunks = []
        for doc_id, text in documents.items():
            words = text.split()
            i, n = 0, 0
            while i < len(words):
                end = min(i + self.size, len(words))
                chunks.append(Chunk(
                    text=" ".join(words[i:end]),
                    doc_id=doc_id,
                    chunk_id=f"{doc_id}_{n}",
                ))
                n += 1
                if end == len(words):
                    break
                i += self.size - self.overlap
        return chunks


class SentenceChunker:
    """Split on sentence boundaries."""
    def __init__(self, max_sentences: int = 3):
        self.max_sentences = max_sentences
        self.name = f"Sentence(max={max_sentences})"

    def _split_sentences(self, text: str) -> list[str]:
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    def chunk(self, documents: dict[str, str]) -> list[Chunk]:
        chunks = []
        for doc_id, text in documents.items():
            sentences = self._split_sentences(text)
            i, n = 0, 0
            while i < len(sentences):
                group = sentences[i : i + self.max_sentences]
                chunks.append(Chunk(
                    text=" ".join(group),
                    doc_id=doc_id,
                    chunk_id=f"{doc_id}_{n}",
                ))
                n += 1
                i += self.max_sentences
        return chunks


class SlidingWindowChunker:
    """Sliding window with 50% overlap between consecutive chunks."""
    def __init__(self, size: int = 40):
        self.size = size
        self.name = f"SlidingWindow(size={size}, stride={size//2})"

    def chunk(self, documents: dict[str, str]) -> list[Chunk]:
        chunks = []
        stride = self.size // 2
        for doc_id, text in documents.items():
            words = text.split()
            i, n = 0, 0
            while i < len(words):
                end = min(i + self.size, len(words))
                chunks.append(Chunk(
                    text=" ".join(words[i:end]),
                    doc_id=doc_id,
                    chunk_id=f"{doc_id}_{n}",
                ))
                n += 1
                if end == len(words):
                    break
                i += stride
        return chunks


# ── Embedding Models ──────────────────────────────────────────────────────────

class TFIDFEmbedder:
    """Classic sparse TF-IDF embeddings (always available)."""
    name = "TF-IDF"

    def __init__(self):
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray = np.array([])
        self._fitted = False

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"[a-z]+", text.lower())

    def fit(self, corpus: list[str]) -> "TFIDFEmbedder":
        n = len(corpus)
        df: dict[str, int] = defaultdict(int)
        all_tokens = set()
        for doc in corpus:
            tokens = set(self._tokenize(doc))
            all_tokens.update(tokens)
            for t in tokens:
                df[t] += 1
        self.vocab = {w: i for i, w in enumerate(sorted(all_tokens))}
        self.idf = np.array([
            math.log((n + 1) / (df[w] + 1)) + 1
            for w in sorted(all_tokens)
        ])
        self._fitted = True
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            self.fit(texts)
        vecs = np.zeros((len(texts), len(self.vocab)), dtype=np.float32)
        for i, text in enumerate(texts):
            tokens = self._tokenize(text)
            tf = Counter(tokens)
            total = len(tokens) or 1
            for w, count in tf.items():
                if w in self.vocab:
                    j = self.vocab[w]
                    vecs[i, j] = (count / total) * self.idf[j]
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / (norms + 1e-8)


class DenseEmbedder:
    """Sentence-transformers dense embeddings."""
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.name = f"Dense({model_name})"
        if ST_AVAILABLE:
            self.model = SentenceTransformer(model_name)
        else:
            self.model = None

    def fit(self, corpus: list[str]) -> "DenseEmbedder":
        return self  # No fitting needed

    def encode(self, texts: list[str]) -> np.ndarray:
        if self.model is not None:
            return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        # Demo fallback: hash-based pseudo-embeddings
        dim = 64
        vecs = []
        for t in texts:
            rng = np.random.RandomState(abs(hash(t)) % (2**31))
            v = rng.randn(dim).astype(np.float32)
            v /= np.linalg.norm(v) + 1e-8
            vecs.append(v)
        return np.array(vecs)


# ── BM25 ─────────────────────────────────────────────────────────────────────

class BM25:
    """Okapi BM25 for sparse retrieval."""
    name = "BM25"

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_tokens: list[list[str]] = []
        self.idf: dict[str, float] = {}
        self.avgdl: float = 0.0

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"[a-z]+", text.lower())

    def fit(self, corpus: list[str]) -> "BM25":
        self.corpus_tokens = [self._tokenize(d) for d in corpus]
        n = len(corpus)
        self.avgdl = np.mean([len(t) for t in self.corpus_tokens])
        df: dict[str, int] = defaultdict(int)
        for tokens in self.corpus_tokens:
            for w in set(tokens):
                df[w] += 1
        self.idf = {
            w: math.log((n - f + 0.5) / (f + 0.5) + 1)
            for w, f in df.items()
        }
        return self

    def score(self, query: str, doc_tokens: list[str]) -> float:
        q_tokens = self._tokenize(query)
        tf = Counter(doc_tokens)
        dl = len(doc_tokens)
        score = 0.0
        for w in q_tokens:
            if w not in self.idf:
                continue
            f = tf[w]
            score += self.idf[w] * (
                f * (self.k1 + 1)
                / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            )
        return score

    def search(self, query: str, k: int = 5) -> list[tuple[float, int]]:
        scores = [(self.score(query, tokens), i) for i, tokens in enumerate(self.corpus_tokens)]
        return sorted(scores, reverse=True)[:k]


# ── Retrieval System ──────────────────────────────────────────────────────────

class RetrievalSystem:
    def __init__(self, embedder, bm25_weight: float = 0.0):
        self.embedder = embedder
        self.bm25_weight = bm25_weight  # 0 = pure dense, 1 = pure sparse
        self.chunks: list[Chunk] = []
        self.embeddings: np.ndarray = np.array([])
        self.bm25 = BM25()

    def index(self, chunks: list[Chunk]) -> "RetrievalSystem":
        self.chunks = chunks
        texts = [c.text for c in chunks]
        self.embedder.fit(texts)
        self.embeddings = self.embedder.encode(texts)
        self.bm25.fit(texts)
        return self

    def search(self, query: str, k: int = 5) -> list[tuple[float, Chunk]]:
        texts = [c.text for c in self.chunks]
        # Dense score
        q_emb = self.embedder.encode([query])[0]
        dense_scores = self.embeddings @ q_emb

        if self.bm25_weight > 0:
            # BM25 scores normalised to [0,1]
            bm25_raw = [self.bm25.score(query, self.bm25.corpus_tokens[i])
                        for i in range(len(self.chunks))]
            bm25_max = max(bm25_raw) if max(bm25_raw) > 0 else 1
            bm25_scores = np.array(bm25_raw) / bm25_max

            w = self.bm25_weight
            scores = (1 - w) * dense_scores + w * bm25_scores
        else:
            scores = dense_scores

        idxs = np.argsort(-scores)[:k]
        return [(float(scores[i]), self.chunks[i]) for i in idxs]


# ── Metrics ───────────────────────────────────────────────────────────────────

def recall_at_k(retrieved_docs: list[str], relevant_docs: list[str], k: int) -> float:
    top_k = set(retrieved_docs[:k])
    return len(top_k & set(relevant_docs)) / max(len(relevant_docs), 1)


def precision_at_k(retrieved_docs: list[str], relevant_docs: list[str], k: int) -> float:
    top_k = retrieved_docs[:k]
    hits = sum(1 for d in top_k if d in relevant_docs)
    return hits / max(k, 1)


def mrr(retrieved_docs: list[str], relevant_docs: list[str]) -> float:
    for rank, doc in enumerate(retrieved_docs, 1):
        if doc in relevant_docs:
            return 1.0 / rank
    return 0.0


def evaluate_retrieval(
    system: RetrievalSystem,
    ground_truth: list[dict],
    k_values: list[int] = [1, 3, 5],
) -> dict:
    metrics: dict[str, list[float]] = {f"recall@{k}": [] for k in k_values}
    metrics.update({f"precision@{k}": [] for k in k_values})
    metrics["mrr"] = []

    for item in ground_truth:
        results = system.search(item["query"], k=max(k_values))
        retrieved_doc_ids = [c.doc_id for _, c in results]
        relevant = item["relevant_docs"]

        for k in k_values:
            metrics[f"recall@{k}"].append(recall_at_k(retrieved_doc_ids, relevant, k))
            metrics[f"precision@{k}"].append(precision_at_k(retrieved_doc_ids, relevant, k))
        metrics["mrr"].append(mrr(retrieved_doc_ids, relevant))

    return {k: float(np.mean(v)) for k, v in metrics.items()}


# ── Main Experiment ───────────────────────────────────────────────────────────

def main():
    print("\n[bold]Phase 2.2 — Retrieval Quality Analysis & Optimisation[/bold]\n")

    # ── Experiment 1: Chunking strategies
    print("[bold]Experiment 1: Chunking Strategies[/bold]")
    chunkers = [
        FixedSizeChunker(size=40, overlap=5),
        FixedSizeChunker(size=40, overlap=20),
        SentenceChunker(max_sentences=2),
        SentenceChunker(max_sentences=4),
        SlidingWindowChunker(size=40),
    ]
    embedder = TFIDFEmbedder()

    results_chunking = []
    for chunker in chunkers:
        chunks = chunker.chunk(DOCUMENTS)
        system = RetrievalSystem(embedder)
        system.index(chunks)
        metrics = evaluate_retrieval(system, GROUND_TRUTH)
        results_chunking.append({
            "strategy": chunker.name,
            "num_chunks": len(chunks),
            "avg_chunk_len": np.mean([len(c.text.split()) for c in chunks]),
            **metrics,
        })
        print(f"  {chunker.name:40s} | chunks={len(chunks):3d} | "
              f"Recall@3={metrics['recall@3']:.2f} | MRR={metrics['mrr']:.2f}")

    best_chunker = max(results_chunking, key=lambda x: x["mrr"])
    print(f"\n  Best chunking: {best_chunker['strategy']} (MRR={best_chunker['mrr']:.3f})")

    # ── Experiment 2: Embedding models
    print("\n[bold]Experiment 2: Embedding Models[/bold]")
    best_chunks = SentenceChunker(max_sentences=3).chunk(DOCUMENTS)

    embedders_to_test = [TFIDFEmbedder()]
    if ST_AVAILABLE:
        embedders_to_test.extend([
            DenseEmbedder("all-MiniLM-L6-v2"),
            DenseEmbedder("all-mpnet-base-v2"),
        ])
    else:
        embedders_to_test.append(DenseEmbedder())  # fallback demo

    results_emb = []
    for emb in embedders_to_test:
        system = RetrievalSystem(emb)
        system.index(best_chunks)
        metrics = evaluate_retrieval(system, GROUND_TRUTH)
        results_emb.append({"embedder": emb.name, **metrics})
        print(f"  {emb.name:30s} | Recall@1={metrics['recall@1']:.2f} "
              f"Recall@3={metrics['recall@3']:.2f} | MRR={metrics['mrr']:.2f}")

    best_emb_result = max(results_emb, key=lambda x: x["mrr"])
    print(f"\n  Best embedder: {best_emb_result['embedder']} (MRR={best_emb_result['mrr']:.3f})")

    # ── Experiment 3: Hybrid retrieval (BM25 + dense)
    print("\n[bold]Experiment 3: Hybrid Retrieval (BM25 + Dense)[/bold]")
    weights = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
    results_hybrid = []
    for w in weights:
        system = RetrievalSystem(TFIDFEmbedder(), bm25_weight=w)
        system.index(best_chunks)
        metrics = evaluate_retrieval(system, GROUND_TRUTH)
        label = f"BM25={w:.1f} Dense={1-w:.1f}"
        results_hybrid.append({"weight": w, "label": label, **metrics})
        bar = "█" * int(metrics["mrr"] * 20)
        print(f"  {label:25s} | MRR={metrics['mrr']:.3f} {bar}")

    best_hybrid = max(results_hybrid, key=lambda x: x["mrr"])
    print(f"\n  Best mix: {best_hybrid['label']} (MRR={best_hybrid['mrr']:.3f})")

    # ── Experiment 4: Failure analysis
    print("\n[bold]Experiment 4: Retrieval Failure Analysis[/bold]")
    system = RetrievalSystem(TFIDFEmbedder(), bm25_weight=0.3)
    system.index(SentenceChunker(max_sentences=3).chunk(DOCUMENTS))

    print("\n  Per-query analysis:")
    for item in GROUND_TRUTH:
        results = system.search(item["query"], k=3)
        retrieved_docs = [c.doc_id for _, c in results]
        hit = any(d in item["relevant_docs"] for d in retrieved_docs[:1])
        icon = "✓" if hit else "✗"
        top_doc = retrieved_docs[0] if retrieved_docs else "none"
        print(f"  {icon} '{item['query'][:40]}...'")
        print(f"    Expected: {item['relevant_docs'][0]} | Got: {top_doc}")
        if not hit:
            print(f"    [red]MISS[/red] — Likely cause: query vocabulary mismatch with chunk text")

    # ── Summary table
    print("\n[bold]Summary: All Experiments[/bold]")
    print(f"\n  {'Configuration':<45} {'Recall@1':>9} {'Recall@3':>9} {'MRR':>7}")
    print("  " + "-" * 75)
    for r in results_chunking[:3]:
        print(f"  {r['strategy']:<45} {r['recall@1']:>9.3f} {r['recall@3']:>9.3f} {r['mrr']:>7.3f}")
    print("  " + "-" * 75)
    for r in results_emb:
        label = f"Emb: {r['embedder']}"
        print(f"  {label:<45} {r['recall@1']:>9.3f} {r['recall@3']:>9.3f} {r['mrr']:>7.3f}")
    print("  " + "-" * 75)
    for r in [results_hybrid[0], results_hybrid[3], results_hybrid[-1]]:
        label = f"Hybrid {r['label']}"
        print(f"  {label:<45} {r['recall@1']:>9.3f} {r['recall@3']:>9.3f} {r['mrr']:>7.3f}")

    print("\n[bold]Key Findings:[/bold]")
    print("  • Smaller chunks with overlap improve granularity but increase noise")
    print("  • Dense embeddings outperform TF-IDF on semantic similarity")
    print("  • Hybrid retrieval (BM25 + dense) often beats either alone")
    print("  • Vocabulary mismatch (query words ≠ chunk words) is the main failure mode")
    print("  • Chunk size tradeoff: too small loses context, too large dilutes relevance signal")

    print("\n[bold green]Retrieval Optimisation complete![/bold green]")


if __name__ == "__main__":
    main()
