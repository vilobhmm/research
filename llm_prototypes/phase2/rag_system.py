"""
Phase 2.1 — RAG System from Scratch
======================================
End-to-end Retrieval-Augmented Generation:
  • Chunking pipeline (fixed-size with overlap)
  • Embedding with sentence-transformers
  • FAISS vector store
  • Claude API for generation
  • Evaluation: grounding rate, answer correctness

Run:
    ANTHROPIC_API_KEY=sk-... python phase2/rag_system.py

Requirements: anthropic, sentence-transformers, faiss-cpu, numpy, rich
"""

import json
import os
import re
import textwrap
import time
from dataclasses import dataclass, field
from typing import Optional

import anthropic
import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("faiss-cpu not installed; using brute-force cosine search")

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False
    print("sentence-transformers not installed; using random embeddings (demo mode)")

try:
    from rich import print
    from rich.console import Console
    from rich.panel import Panel
    console = Console()
except ImportError:
    console = None

# ── Demo Knowledge Base ───────────────────────────────────────────────────────

KNOWLEDGE_BASE = {
    "transformer_architecture": """
The Transformer architecture, introduced in 'Attention Is All You Need' (Vaswani et al., 2017),
revolutionised natural language processing by replacing recurrent networks with self-attention.
The key innovation is the scaled dot-product attention mechanism: Attention(Q,K,V) = softmax(QK^T/√d_k)V.
Multi-head attention runs h parallel attention heads, concatenating their outputs.
The encoder uses full bidirectional attention; the decoder uses causal (masked) attention.
Positional encodings using sine and cosine functions inject sequence order information.
The architecture stacks N=6 layers of encoder/decoder blocks, each with attention + feed-forward sublayers
and residual connections followed by layer normalisation.
""",
    "rlhf_method": """
Reinforcement Learning from Human Feedback (RLHF), described in 'Training language models to follow
instructions with human feedback' (Ouyang et al., 2022), aligns LLMs with human preferences through
three stages: supervised fine-tuning (SFT), reward model (RM) training, and PPO optimisation.
In the SFT stage, the model is fine-tuned on human-written demonstrations.
The RM is trained on pairs of model outputs rated by humans, learning to predict preference scores.
Finally, PPO optimises the policy model to maximise RM scores while maintaining a KL-divergence
penalty from the SFT model to prevent reward hacking and mode collapse.
The KL penalty coefficient β controls the tradeoff between reward maximisation and staying on-distribution.
""",
    "rag_method": """
Retrieval-Augmented Generation (RAG), from Lewis et al. (2020), combines parametric (LLM) and
non-parametric (retrieval) memory. The model retrieves top-k relevant documents for a query using
a dual-encoder: a question encoder and a document encoder, both typically based on BERT.
Retrieved documents are prepended to the prompt before generation.
RAG has two variants: RAG-Sequence (retrieves once for the entire output) and RAG-Token
(can retrieve different documents for each output token).
RAG significantly improves performance on knowledge-intensive NLP tasks like open-domain QA,
fact verification, and knowledge-grounded generation, reducing hallucination rates.
""",
    "constitutional_ai": """
Constitutional AI (CAI), from Anthropic (2022), is a method for training harmless AI assistants
using a set of principles (a 'constitution') rather than relying entirely on human feedback labels.
CAI has two phases: supervised learning (SL-CAI) and reinforcement learning (RL-CAI).
In SL-CAI, the model critiques and revises its own harmful outputs according to the constitution.
In RL-CAI, an AI-generated preference model (instead of human raters) provides reward signals.
The constitution includes principles like 'be helpful, harmless, and honest' and specific rules
about avoiding manipulation, deception, or harmful content.
This reduces the need for human red-team labelling while maintaining safety properties.
""",
    "dpo_method": """
Direct Preference Optimisation (DPO), from Rafailov et al. (2023), provides a simpler alternative
to RLHF by directly optimising the language model on preference data without a separate reward model.
DPO derives an implicit reward from the policy itself: r(x,y) = β log(π_θ(y|x)/π_ref(y|x)).
The DPO loss is: L_DPO = -E[log σ(β log(π/π_ref)(y_w) - β log(π/π_ref)(y_l))].
Where y_w is the preferred response and y_l is the less-preferred response.
DPO is more stable than PPO, requires less compute, and avoids reward hacking by design.
It has been shown to match or exceed RLHF performance on alignment benchmarks like TruthfulQA.
""",
    "scaling_laws": """
Scaling Laws for Neural Language Models (Kaplan et al., 2020) showed that LLM performance follows
predictable power laws with respect to three factors: model parameters N, dataset size D, and
compute budget C. The compute-optimal frontier follows: N_opt ∝ C^0.5, D_opt ∝ C^0.5.
The Chinchilla paper (Hoffmann et al., 2022) revised these estimates, finding N_opt ∝ C^0.5 and
D_opt ∝ C^0.5 with a coefficient closer to 20 tokens per parameter.
These laws hold over many orders of magnitude and enable prediction of performance before training.
Key insight: most models are significantly undertrained relative to the compute-optimal frontier.
""",
    "chain_of_thought": """
Chain-of-Thought (CoT) prompting, from Wei et al. (2022), significantly improves LLM reasoning by
asking models to output intermediate reasoning steps before the final answer.
CoT prompting works by providing few-shot examples that include step-by-step reasoning.
It emerges primarily in large models (>100B parameters) and provides the greatest gains on
multi-step arithmetic, commonsense reasoning, and symbolic reasoning tasks.
Zero-shot CoT ('Let's think step by step') also works, eliminating the need for hand-crafted examples.
CoT is an instance of in-context learning where the demonstrations teach the model a reasoning format.
The mechanism likely works by decomposing complex problems into simpler sub-steps the model can solve.
""",
    "ppo_algorithm": """
Proximal Policy Optimisation (PPO), from Schulman et al. (2017), is the dominant RL algorithm for
RLHF fine-tuning of language models. PPO addresses the instability of earlier policy gradient methods
by clipping the probability ratio: L^CLIP = E[min(r_t A_t, clip(r_t, 1-ε, 1+ε) A_t)].
The clipping prevents excessively large policy updates that could destabilise training.
PPO uses a value function (critic) to estimate advantages and a shared actor-critic architecture.
For LLMs, PPO is applied token-by-token: each token is an action, and the reward is the RM score
for the complete sequence (with a per-token KL penalty).
PPO requires careful hyperparameter tuning and is sensitive to reward model quality.
""",
}

# ── Test Set ──────────────────────────────────────────────────────────────────

TEST_QA = [
    {
        "question": "What is the scaled dot-product attention formula?",
        "expected_keywords": ["softmax", "QK", "sqrt", "d_k"],
        "relevant_doc": "transformer_architecture",
    },
    {
        "question": "What are the three stages of RLHF training?",
        "expected_keywords": ["supervised", "reward model", "PPO"],
        "relevant_doc": "rlhf_method",
    },
    {
        "question": "How does RAG reduce hallucination?",
        "expected_keywords": ["retriev", "document", "non-parametric"],
        "relevant_doc": "rag_method",
    },
    {
        "question": "What is the DPO loss function?",
        "expected_keywords": ["log", "sigma", "preferred", "policy"],
        "relevant_doc": "dpo_method",
    },
    {
        "question": "What does Constitutional AI use instead of human red-team labels?",
        "expected_keywords": ["AI", "constitution", "principles"],
        "relevant_doc": "constitutional_ai",
    },
    {
        "question": "What is the PPO clipping mechanism?",
        "expected_keywords": ["clip", "ratio", "epsilon", "1-ε"],
        "relevant_doc": "ppo_algorithm",
    },
]


# ── 1. Chunking ───────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    text: str
    doc_id: str
    chunk_idx: int
    start_char: int
    end_char: int


def chunk_document(
    text: str,
    doc_id: str,
    chunk_size: int = 200,     # tokens (approx words)
    overlap: int = 50,
) -> list[Chunk]:
    """Fixed-size chunking with overlap."""
    words = text.split()
    chunks = []
    idx = 0
    while idx < len(words):
        end = min(idx + chunk_size, len(words))
        chunk_words = words[idx:end]
        chunk_text = " ".join(chunk_words)
        char_start = len(" ".join(words[:idx])) + (1 if idx > 0 else 0)
        chunks.append(Chunk(
            text=chunk_text,
            doc_id=doc_id,
            chunk_idx=len(chunks),
            start_char=char_start,
            end_char=char_start + len(chunk_text),
        ))
        if end == len(words):
            break
        idx += chunk_size - overlap
    return chunks


# ── 2. Embedding Model ────────────────────────────────────────────────────────

class EmbeddingModel:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if ST_AVAILABLE:
            print(f"Loading embedding model: {model_name}")
            self.model = SentenceTransformer(model_name)
            self.dim = self.model.get_sentence_embedding_dimension()
        else:
            self.model = None
            self.dim = 128
            print(f"Using random {self.dim}-dim embeddings (demo mode)")

    def encode(self, texts: list[str]) -> np.ndarray:
        if self.model is not None:
            return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        # Demo mode: reproducible random embeddings (same text → same vector)
        vecs = []
        for t in texts:
            rng = np.random.RandomState(hash(t) % (2**31))
            v = rng.randn(self.dim).astype(np.float32)
            v /= np.linalg.norm(v) + 1e-8
            vecs.append(v)
        return np.array(vecs)


# ── 3. Vector Store ───────────────────────────────────────────────────────────

class VectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.chunks: list[Chunk] = []
        self.embeddings: Optional[np.ndarray] = None
        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatIP(dim)  # Inner product (cosine on normalised vecs)
        else:
            self.index = None

    def add(self, chunks: list[Chunk], embeddings: np.ndarray):
        self.chunks.extend(chunks)
        emb = embeddings.astype(np.float32)
        if self.index is not None:
            self.index.add(emb)
        else:
            self.embeddings = (
                emb if self.embeddings is None
                else np.concatenate([self.embeddings, emb])
            )

    def search(self, query_emb: np.ndarray, k: int = 5) -> list[tuple[float, Chunk]]:
        q = query_emb.reshape(1, -1).astype(np.float32)
        if self.index is not None:
            scores, idxs = self.index.search(q, k)
            return [(float(scores[0][i]), self.chunks[idxs[0][i]])
                    for i in range(len(idxs[0])) if idxs[0][i] >= 0]
        else:
            sims = self.embeddings @ q.T
            idxs = np.argsort(-sims.flatten())[:k]
            return [(float(sims[i]), self.chunks[i]) for i in idxs]


# ── 4. RAG System ─────────────────────────────────────────────────────────────

class RAGSystem:
    def __init__(
        self,
        embedding_model: Optional[EmbeddingModel] = None,
        chunk_size: int = 200,
        chunk_overlap: int = 50,
        top_k: int = 3,
        model: str = "claude-opus-4-6",
    ):
        self.emb_model = embedding_model or EmbeddingModel()
        self.store = VectorStore(dim=self.emb_model.dim)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.model = model
        self._client: Optional[anthropic.Anthropic] = None
        self.indexed_docs: list[str] = []

    @property
    def client(self) -> anthropic.Anthropic:
        if self._client is None:
            self._client = anthropic.Anthropic()
        return self._client

    def index(self, documents: dict[str, str]) -> "RAGSystem":
        """Chunk and embed a dict of {doc_id: text}."""
        print(f"Indexing {len(documents)} documents...")
        all_chunks = []
        for doc_id, text in documents.items():
            chunks = chunk_document(text, doc_id, self.chunk_size, self.chunk_overlap)
            all_chunks.extend(chunks)
            self.indexed_docs.append(doc_id)

        texts = [c.text for c in all_chunks]
        embeddings = self.emb_model.encode(texts)
        self.store.add(all_chunks, embeddings)
        print(f"Indexed {len(all_chunks)} chunks from {len(documents)} docs")
        return self

    def retrieve(self, query: str, k: Optional[int] = None) -> list[tuple[float, Chunk]]:
        k = k or self.top_k
        q_emb = self.emb_model.encode([query])[0]
        return self.store.search(q_emb, k)

    def generate(self, query: str, use_retrieval: bool = True) -> dict:
        """Generate an answer, with or without retrieval."""
        retrieved = []
        if use_retrieval:
            retrieved = self.retrieve(query)

        if retrieved:
            context_parts = []
            for i, (score, chunk) in enumerate(retrieved, 1):
                context_parts.append(
                    f"[Source {i} — {chunk.doc_id}, similarity={score:.3f}]\n{chunk.text.strip()}"
                )
            context = "\n\n".join(context_parts)
            prompt = (
                f"You are a helpful AI research assistant. Answer the question using ONLY "
                f"the provided context. If the context doesn't contain enough information, "
                f"say so. Cite which source(s) you used.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {query}\n\nAnswer:"
            )
        else:
            prompt = f"Answer this question about AI/ML: {query}\n\nAnswer:"

        try:
            with self.client.messages.stream(
                model=self.model,
                max_tokens=512,
                thinking={"type": "adaptive"},
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                answer = stream.get_final_message().content[-1].text
        except Exception as e:
            answer = f"[API error: {e}]"

        return {
            "question": query,
            "answer": answer,
            "retrieved_chunks": retrieved,
            "used_retrieval": use_retrieval,
        }

    def evaluate(self, test_set: list[dict]) -> dict:
        """Evaluate RAG vs direct generation."""
        results = {
            "rag": {"correct": 0, "grounded": 0, "total": 0},
            "direct": {"correct": 0, "total": 0},
        }
        detailed = []

        for item in test_set:
            q = item["question"]
            keywords = [k.lower() for k in item["expected_keywords"]]
            relevant_doc = item.get("relevant_doc", "")

            # RAG answer
            rag_result = self.generate(q, use_retrieval=True)
            rag_ans = rag_result["answer"].lower()

            # Direct answer (no retrieval)
            direct_result = self.generate(q, use_retrieval=False)
            direct_ans = direct_result["answer"].lower()

            # Correctness: how many keywords appear in the answer
            rag_hits    = sum(1 for kw in keywords if kw in rag_ans)
            direct_hits = sum(1 for kw in keywords if kw in direct_ans)
            rag_correct    = rag_hits >= len(keywords) // 2 + 1
            direct_correct = direct_hits >= len(keywords) // 2 + 1

            # Grounding: check if retrieved docs include the relevant source
            retrieved_docs = {c.doc_id for _, c in rag_result["retrieved_chunks"]}
            grounded = relevant_doc in retrieved_docs

            results["rag"]["correct"]  += int(rag_correct)
            results["rag"]["grounded"] += int(grounded)
            results["rag"]["total"]    += 1
            results["direct"]["correct"] += int(direct_correct)
            results["direct"]["total"]   += 1

            detailed.append({
                "question": q[:50] + "...",
                "rag_correct": rag_correct,
                "direct_correct": direct_correct,
                "grounded": grounded,
                "retrieved_docs": list(retrieved_docs),
            })

        n = results["rag"]["total"]
        summary = {
            "rag_accuracy": results["rag"]["correct"] / max(n, 1),
            "direct_accuracy": results["direct"]["correct"] / max(n, 1),
            "grounding_rate": results["rag"]["grounded"] / max(n, 1),
            "num_questions": n,
            "detailed": detailed,
        }
        return summary


# ── 5. Demo ───────────────────────────────────────────────────────────────────

def main():
    print("\n[bold]Phase 2.1 — RAG System from Scratch[/bold]\n")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("[yellow]ANTHROPIC_API_KEY not set — running in retrieval-only demo mode[/yellow]")

    # Build RAG system
    rag = RAGSystem(chunk_size=150, chunk_overlap=30, top_k=3)
    rag.index(KNOWLEDGE_BASE)

    # ── Retrieval demo
    print("\n[bold]1. Retrieval Demo[/bold]")
    test_queries = [
        "How does attention work in transformers?",
        "What is the KL divergence penalty in RLHF?",
        "How does Constitutional AI avoid human labelling?",
    ]
    for q in test_queries:
        retrieved = rag.retrieve(q, k=2)
        print(f"\n  Query: '{q}'")
        for score, chunk in retrieved:
            print(f"  [{score:.3f}] {chunk.doc_id}: {chunk.text[:80].strip()}...")

    # ── Full RAG generation (requires API key)
    if api_key:
        print("\n[bold]2. Full RAG Generation[/bold]")
        demo_questions = [
            "What is scaled dot-product attention?",
            "How does DPO differ from RLHF?",
        ]
        for q in demo_questions:
            print(f"\n  [cyan]Q: {q}[/cyan]")
            result = rag.generate(q, use_retrieval=True)
            print(f"  A: {result['answer'][:300]}...")
            print(f"  Retrieved: {[c.doc_id for _, c in result['retrieved_chunks']]}")

        print("\n[bold]3. Evaluation: RAG vs Direct[/bold]")
        print("Evaluating on test set (this may take a minute)...")
        summary = rag.evaluate(TEST_QA[:3])  # use 3 questions for demo speed

        print(f"\n  RAG accuracy:    {summary['rag_accuracy']:.1%}")
        print(f"  Direct accuracy: {summary['direct_accuracy']:.1%}")
        print(f"  Grounding rate:  {summary['grounding_rate']:.1%}")
        print(f"\n  Question-level results:")
        for d in summary["detailed"]:
            rag_icon    = "✓" if d["rag_correct"] else "✗"
            direct_icon = "✓" if d["direct_correct"] else "✗"
            grnd_icon   = "✓" if d["grounded"] else "✗"
            print(f"    RAG{rag_icon} Direct{direct_icon} Grounded{grnd_icon} | {d['question']}")
    else:
        print("\n[yellow]Skipping generation/evaluation (no API key)[/yellow]")
        print("Set ANTHROPIC_API_KEY to enable Claude-powered generation")

    # ── Index stats
    print("\n[bold]4. Index Statistics[/bold]")
    print(f"  Documents indexed: {len(rag.indexed_docs)}")
    print(f"  Total chunks: {len(rag.store.chunks)}")
    print(f"  Embedding dimension: {rag.emb_model.dim}")
    print(f"  Vector store type: {'FAISS' if FAISS_AVAILABLE else 'NumPy brute-force'}")

    chunk_lengths = [len(c.text.split()) for c in rag.store.chunks]
    print(f"  Avg chunk length: {np.mean(chunk_lengths):.0f} words")
    print(f"  Min/Max chunk: {min(chunk_lengths)} / {max(chunk_lengths)} words")

    print("\n[bold green]RAG System demo complete![/bold green]")


if __name__ == "__main__":
    main()
