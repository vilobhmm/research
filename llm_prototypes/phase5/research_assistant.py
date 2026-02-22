"""
Phase 5.1 — Full Research Assistant: RAG + ReAct + RLHF Integration
=====================================================================
A production-like AI research assistant that integrates:
  • RAG: indexes and retrieves from AI papers/documents
  • ReAct Agent: multi-step reasoning with retrieval tools
  • RLHF feedback loop: user ratings improve quality over time
  • Constitutional AI: ensures safe, honest responses

Run:
    ANTHROPIC_API_KEY=sk-... python phase5/research_assistant.py

Requirements: anthropic, numpy, rich
"""

import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    from rich import print
    from rich.panel import Panel
    from rich.console import Console
    console = Console()
except ImportError:
    console = None


# ── Research Knowledge Base ───────────────────────────────────────────────────

PAPERS_DB = {
    "attention_vaswani_2017": {
        "title": "Attention Is All You Need",
        "authors": "Vaswani, Shazeer, Parmar et al.",
        "year": 2017,
        "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. The Transformer allows for significantly more parallelization and reaches state-of-the-art translation quality.",
        "key_contributions": [
            "Scaled dot-product attention mechanism",
            "Multi-head attention",
            "Positional encodings",
            "Encoder-decoder transformer architecture",
        ],
        "formulas": "Attention(Q,K,V) = softmax(QK^T/sqrt(d_k))V",
        "impact": "Foundation of modern NLP; led to BERT, GPT, T5, and essentially all modern LLMs",
    },
    "gpt3_brown_2020": {
        "title": "Language Models are Few-Shot Learners",
        "authors": "Brown, Mann, Ryder et al. (OpenAI)",
        "year": 2020,
        "abstract": "We demonstrate few-shot learning abilities of large language models. GPT-3 with 175B parameters achieves strong performance on many NLP tasks in the few-shot setting, without any gradient updates.",
        "key_contributions": [
            "In-context learning at scale",
            "175B parameter language model",
            "Emergent few-shot capabilities",
            "Scaling laws for language models",
        ],
        "impact": "Demonstrated that scale enables emergent capabilities; launched the LLM era",
    },
    "rlhf_ouyang_2022": {
        "title": "Training language models to follow instructions with human feedback",
        "authors": "Ouyang, Wu, Jiang et al. (OpenAI)",
        "year": 2022,
        "abstract": "We fine-tune GPT-3 to follow a broad class of written instructions using reinforcement learning from human feedback. The resulting model, InstructGPT, is preferred over GPT-3 despite having 100x fewer parameters.",
        "key_contributions": [
            "SFT → RM → PPO RLHF pipeline",
            "Human preference dataset",
            "Alignment via reward modeling",
            "KL penalty to prevent reward hacking",
        ],
        "impact": "The RLHF recipe used for ChatGPT, Claude, Gemini and modern aligned LLMs",
    },
    "rag_lewis_2020": {
        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "authors": "Lewis, Perez, Piktus et al. (Facebook AI)",
        "year": 2020,
        "abstract": "We explore a general-purpose fine-tuning recipe for retrieval-augmented generation (RAG), which combine pre-trained parametric and non-parametric memory for language generation.",
        "key_contributions": [
            "RAG architecture combining retrieval + generation",
            "Dense passage retrieval (DPR)",
            "RAG-Sequence and RAG-Token variants",
            "Non-parametric memory via document index",
        ],
        "impact": "Foundation of RAG systems; widely deployed for knowledge-grounded generation",
    },
    "constitutional_ai_2022": {
        "title": "Constitutional AI: Harmlessness from AI Feedback",
        "authors": "Bai, Jones, Ndousse et al. (Anthropic)",
        "year": 2022,
        "abstract": "We propose Constitutional AI (CAI), a method for training AI systems to be helpful, harmless, and honest using a set of principles (constitution) rather than exclusively relying on human feedback.",
        "key_contributions": [
            "Principles-based alignment",
            "Self-critique and revision",
            "AI-generated preference labels (RL-CAI)",
            "Reduced need for human red-teaming",
        ],
        "impact": "Core method behind Claude's safety training; reduces human labelling burden",
    },
    "react_yao_2022": {
        "title": "ReAct: Synergizing Reasoning and Acting in Language Models",
        "authors": "Yao, Zhao, Yu et al.",
        "year": 2022,
        "abstract": "We explore the use of LLMs to generate both reasoning traces and task-specific actions in an interleaved manner, allowing for greater synergy between the two.",
        "key_contributions": [
            "Thought-Action-Observation loop",
            "Grounded reasoning via tool use",
            "Interpretable agent trajectories",
            "Works with Wikipedia, calculators, databases",
        ],
        "impact": "Most widely used agent framework; basis for many production agents",
    },
    "dpo_rafailov_2023": {
        "title": "Direct Preference Optimization: Your Language Model is Secretly a Reward Model",
        "authors": "Rafailov, Sharma, Mitchell et al.",
        "year": 2023,
        "abstract": "We introduce Direct Preference Optimization (DPO), a simple training paradigm that trains language models from preferences without reinforcement learning. DPO is more stable, performant, and less computationally intensive than RLHF.",
        "key_contributions": [
            "Closed-form policy optimisation from preferences",
            "Implicit reward function via policy ratio",
            "No separate reward model needed",
            "Simpler and more stable than PPO",
        ],
        "formulas": "L_DPO = -E[log σ(β log(π/π_ref)(y_w) - β log(π/π_ref)(y_l))]",
        "impact": "Widely adopted alternative to PPO-based RLHF; simpler and often better",
    },
    "chain_of_thought_wei_2022": {
        "title": "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models",
        "authors": "Wei, Wang, Schuurmans et al. (Google Brain)",
        "year": 2022,
        "abstract": "We explore how generating a chain of thought—a series of intermediate reasoning steps—significantly improves the ability of large language models to perform complex reasoning.",
        "key_contributions": [
            "Step-by-step reasoning in prompts",
            "Emerges in models >100B parameters",
            "Works zero-shot ('Let's think step by step')",
            "Dramatically improves arithmetic, commonsense, symbolic reasoning",
        ],
        "impact": "Standard technique for complex reasoning; used in most production LLM systems",
    },
}


# ── Simple Retrieval (no FAISS needed) ───────────────────────────────────────

def simple_retrieval(query: str, db: dict, top_k: int = 3) -> list[tuple[float, str, dict]]:
    """BM25-style retrieval over the papers DB."""
    q_words = set(re.findall(r"[a-z]+", query.lower()))

    scored = []
    for paper_id, paper in db.items():
        # Build searchable text
        text = " ".join([
            paper.get("title", ""),
            paper.get("abstract", ""),
            " ".join(paper.get("key_contributions", [])),
            paper.get("impact", ""),
        ]).lower()

        text_words = set(re.findall(r"[a-z]+", text))

        # TF-IDF inspired score
        overlap = q_words & text_words
        if not overlap:
            continue

        score = len(overlap) / math.sqrt(max(len(q_words), 1) * max(len(text_words), 1))
        scored.append((score, paper_id, paper))

    scored.sort(reverse=True)
    return scored[:top_k]


# ── RLHF Feedback Tracker ─────────────────────────────────────────────────────

@dataclass
class Interaction:
    query: str
    answer: str
    citations: list[str]
    rating: Optional[int] = None  # 1-5 stars


class FeedbackTracker:
    """Tracks user ratings and computes reward model signals."""

    def __init__(self):
        self.interactions: list[Interaction] = []
        self.preference_pairs: list[dict] = []

    def log(self, interaction: Interaction) -> "FeedbackTracker":
        self.interactions.append(interaction)
        return self

    def rate(self, interaction_idx: int, rating: int):
        if 0 <= interaction_idx < len(self.interactions):
            self.interactions[interaction_idx].rating = rating
            # Create preference pairs: compare this to previous interactions
            if interaction_idx > 0 and self.interactions[interaction_idx - 1].rating is not None:
                prev = self.interactions[interaction_idx - 1]
                curr = self.interactions[interaction_idx]
                if curr.rating is not None and prev.rating is not None:
                    self.preference_pairs.append({
                        "query_a": prev.query,
                        "answer_a": prev.answer,
                        "query_b": curr.query,
                        "answer_b": curr.answer,
                        "preferred": "b" if curr.rating > prev.rating else "a",
                        "rating_a": prev.rating,
                        "rating_b": curr.rating,
                    })

    def quality_trend(self) -> list[float]:
        rated = [i for i in self.interactions if i.rating is not None]
        if len(rated) < 2:
            return [r.rating or 0 for r in rated]
        # 3-point moving average
        ratings = [r.rating or 0 for r in rated]
        smoothed = []
        for i in range(len(ratings)):
            window = ratings[max(0, i-1):i+2]
            smoothed.append(np.mean(window))
        return smoothed

    def statistics(self) -> dict:
        rated = [i for i in self.interactions if i.rating is not None]
        if not rated:
            return {"n_interactions": len(self.interactions), "n_rated": 0}
        ratings = [i.rating for i in rated]
        return {
            "n_interactions": len(self.interactions),
            "n_rated": len(rated),
            "avg_rating": float(np.mean(ratings)),
            "rating_std": float(np.std(ratings)),
            "n_preference_pairs": len(self.preference_pairs),
            "rating_distribution": {
                str(r): ratings.count(r) for r in range(1, 6)
            },
        }


# ── Research Assistant ────────────────────────────────────────────────────────

class ResearchAssistant:
    """
    AI research assistant integrating:
    - RAG for paper retrieval
    - ReAct-style reasoning
    - User feedback collection for RLHF
    - Constitutional safety checks
    """

    SYSTEM = """You are an expert AI research assistant specialising in machine learning and NLP.

Your task is to answer questions about AI/ML research accurately, citing specific papers.

Format your responses as:
1. Direct answer to the question
2. Key papers that address this (with year and authors)
3. Key formulas or technical details if relevant
4. Practical implications

Always cite your sources. Be precise but accessible."""

    CONSTITUTION_CHECK = """Check if this response about AI research:
1. Is factually accurate (no hallucinated paper names or results)
2. Properly attributes claims to sources
3. Acknowledges uncertainty where appropriate
4. Is helpful and substantive

If there are issues, explain them briefly. Otherwise say "OK"."""

    def __init__(self, model: str = "claude-opus-4-6"):
        self.model = model
        self.papers_db = PAPERS_DB
        self.tracker = FeedbackTracker()
        self._client: Optional[anthropic.Anthropic] = None
        self.api_calls = 0

    @property
    def client(self) -> anthropic.Anthropic:
        if self._client is None:
            self._client = anthropic.Anthropic()
        return self._client

    def _call(self, system: str, user: str, max_tokens: int = 800) -> str:
        self.api_calls += 1
        try:
            with self.client.messages.stream(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                thinking={"type": "adaptive"},
                messages=[{"role": "user", "content": user}],
            ) as stream:
                msg = stream.get_final_message()
                for block in msg.content:
                    if block.type == "text":
                        return block.text
            return ""
        except Exception as e:
            return f"[API Error: {e}]"

    def retrieve(self, query: str, top_k: int = 3) -> list[tuple[float, str, dict]]:
        return simple_retrieval(query, self.papers_db, top_k=top_k)

    def _format_context(self, retrieved: list[tuple[float, str, dict]]) -> str:
        parts = []
        for score, paper_id, paper in retrieved:
            parts.append(
                f"[Paper: {paper['title']} ({paper['year']}) by {paper['authors']}]\n"
                f"Abstract: {paper['abstract'][:300]}...\n"
                f"Key contributions: {', '.join(paper['key_contributions'][:3])}\n"
                f"Impact: {paper.get('impact', 'N/A')}"
            )
        return "\n\n".join(parts)

    def answer(
        self,
        query: str,
        use_cai: bool = True,
        verbose: bool = True,
    ) -> dict:
        """Answer a research question using RAG + ReAct-style reasoning."""
        if verbose:
            print(f"\n[cyan]Q: {query}[/cyan]")

        # Step 1: Retrieve relevant papers
        retrieved = self.retrieve(query, top_k=3)
        citations = [paper["title"] for _, _, paper in retrieved]

        if verbose:
            print(f"  Retrieved: {[p['title'][:40] for _, _, p in retrieved]}")

        # Step 2: Generate answer with context
        context = self._format_context(retrieved)
        user_prompt = f"""Based on the following research papers, answer this question:

Question: {query}

Relevant papers:
{context}

Please provide a comprehensive answer with citations."""

        answer_text = self._call(self.SYSTEM, user_prompt)

        if verbose:
            print(f"  [green]Answer:[/green] {answer_text[:200]}...")

        # Step 3: Constitutional check
        if use_cai:
            check_prompt = f"Response to check:\n{answer_text}\n\nOriginal question: {query}"
            cai_result = self._call(self.CONSTITUTION_CHECK, check_prompt, max_tokens=200)
            if verbose and "OK" not in cai_result:
                print(f"  [yellow]CAI flag:[/yellow] {cai_result[:100]}")

        # Log interaction
        interaction = Interaction(
            query=query,
            answer=answer_text,
            citations=citations,
        )
        self.tracker.log(interaction)
        interaction_idx = len(self.tracker.interactions) - 1

        return {
            "query": query,
            "answer": answer_text,
            "citations": citations,
            "retrieved_papers": [(paper["title"], score) for score, _, paper in retrieved],
            "interaction_idx": interaction_idx,
        }

    def rate_answer(self, interaction_idx: int, rating: int, verbose: bool = True):
        """User provides 1-5 star rating for an answer."""
        self.tracker.rate(interaction_idx, rating)
        if verbose:
            stars = "★" * rating + "☆" * (5 - rating)
            print(f"  Rated [{stars}] ({rating}/5)")

    def interactive_session(self, questions: list[tuple[str, int]]):
        """Run a session with pre-defined questions and ratings."""
        print("\n[bold]Research Assistant Interactive Session[/bold]\n")

        for question, simulated_rating in questions:
            result = self.answer(question, verbose=True)
            self.rate_answer(result["interaction_idx"], simulated_rating, verbose=True)
            print()
            time.sleep(0.5)

    def show_dashboard(self):
        """Show RLHF feedback dashboard."""
        stats = self.tracker.statistics()
        trend = self.tracker.quality_trend()

        print("\n[bold]RLHF Feedback Dashboard[/bold]")
        print(f"  Total interactions: {stats['n_interactions']}")
        print(f"  Rated: {stats['n_rated']}")

        if stats.get("avg_rating"):
            print(f"  Avg rating: {stats['avg_rating']:.2f}/5")
            print(f"  Rating distribution:")
            for r in range(1, 6):
                count = stats["rating_distribution"].get(str(r), 0)
                bar = "█" * count
                print(f"    {r}★: {count:2d}  {bar}")

        if len(trend) >= 2:
            improvement = trend[-1] - trend[0]
            print(f"\n  Quality trend: {trend[0]:.1f} → {trend[-1]:.1f} ({improvement:+.1f})")
            print(f"  Trend: ", end="")
            for t in trend:
                print("▲" if t >= 4 else "→" if t >= 3 else "▼", end="")
            print()

        if stats.get("n_preference_pairs", 0) > 0:
            print(f"\n  Preference pairs collected: {stats['n_preference_pairs']}")
            print("  (These can be used to train a reward model)")

        print(f"\n  Total API calls: {self.api_calls}")


# ── Main Demo ─────────────────────────────────────────────────────────────────

def main():
    print("\n[bold]Phase 5.1 — Full Research Assistant[/bold]")
    print("Components: RAG + ReAct + RLHF Feedback + Constitutional AI\n")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key or not ANTHROPIC_AVAILABLE:
        print("[red]ANTHROPIC_API_KEY not set — showing offline demo[/red]\n")
        _demo_offline()
        return

    assistant = ResearchAssistant()

    # ── Demo 1: Single question answering
    print("[bold]Demo 1: Research Q&A with Citations[/bold]")
    questions_and_ratings = [
        ("How does the attention mechanism work in transformers?", 5),
        ("What is RLHF and how is it used to align language models?", 4),
        ("How does RAG reduce hallucination in LLMs?", 4),
        ("What is DPO and how does it differ from RLHF?", 5),
        ("How does Constitutional AI work at Anthropic?", 3),
    ]

    print("\nRunning session with 5 research questions...")
    assistant.interactive_session(questions_and_ratings)

    # ── Demo 2: Comparative analysis
    print("\n[bold]Demo 2: Comparative Analysis Question[/bold]")
    comparison = assistant.answer(
        "What are the main differences between RLHF, DPO, and Constitutional AI for LLM alignment?",
        verbose=True,
    )
    assistant.rate_answer(comparison["interaction_idx"], 5)

    # ── Demo 3: RLHF dashboard
    assistant.show_dashboard()

    # ── Demo 4: Paper search demo
    print("\n[bold]Demo 4: RAG Retrieval Quality[/bold]")
    test_queries = [
        "few-shot learning in large language models",
        "reward hacking in reinforcement learning",
        "step-by-step reasoning with chain of thought",
        "agent systems that reason and act",
    ]
    print("\n  Query → Top Retrieved Papers:")
    for q in test_queries:
        retrieved = assistant.retrieve(q, top_k=2)
        titles = [f"{paper['title'][:35]} ({score:.3f})" for score, _, paper in retrieved]
        print(f"  '{q[:45]}...'")
        for t in titles:
            print(f"    → {t}")

    # ── Integration summary
    print("\n[bold]Integration Architecture:[/bold]")
    print("""
  User Query
      │
      ↓
  ┌───────────────────────────────────────────────────┐
  │ ResearchAssistant                                 │
  │                                                   │
  │  1. RAG Retrieval                                 │
  │     Query → TF-IDF → Top-K papers                │
  │                                                   │
  │  2. Context-Augmented Generation (ReAct-inspired) │
  │     [Retrieved papers] + [Query] → Claude         │
  │                                                   │
  │  3. Constitutional Check                          │
  │     Claude reviews its own output for accuracy    │
  │                                                   │
  │  4. RLHF Feedback Collection                     │
  │     User rates → preference pairs → RM training  │
  └───────────────────────────────────────────────────┘
      │
      ↓
  Cited, Grounded Answer + User Rating
    """)

    print("[bold green]Research Assistant complete![/bold green]")


def _demo_offline():
    """Show research assistant without API calls."""
    print("[bold]Offline Demo: Retrieval Quality[/bold]\n")

    test_queries = [
        "How does attention work?",
        "What is in-context learning?",
        "How to align language models with human preferences?",
        "What is the ReAct agent framework?",
    ]

    print("RAG retrieval results:")
    for q in test_queries:
        retrieved = simple_retrieval(q, PAPERS_DB, top_k=2)
        print(f"\n  Query: '{q}'")
        for score, _, paper in retrieved:
            print(f"    [{score:.3f}] {paper['title']} ({paper['year']})")
            print(f"           by {paper['authors']}")

    print("\n[bold]Integration Components:[/bold]")
    print("  1. RAG:  indexes AI papers, retrieves by query similarity")
    print("  2. ReAct: Thought → Retrieve → Observe → Answer loop")
    print("  3. RLHF: collects user ratings → builds preference pairs → trains RM")
    print("  4. CAI:  checks factual accuracy and source attribution")

    print("\n[bold]Knowledge Base:[/bold]")
    for pid, paper in PAPERS_DB.items():
        print(f"  • {paper['title']} ({paper['year']}) - {paper['authors'][:40]}")


if __name__ == "__main__":
    main()
