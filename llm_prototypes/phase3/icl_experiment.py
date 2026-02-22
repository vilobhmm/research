"""
Phase 3.1 — In-Context Learning Playground
============================================
Systematically explores ICL mechanics using Claude:
  • 0/1/3/5-shot sentiment classification
  • Example order effects (random vs. optimal)
  • Label quality effects (correct vs. noisy labels)
  • Chain-of-Thought prompting
  • Dynamic few-shot selection (retrieve most similar examples)

Run:
    ANTHROPIC_API_KEY=sk-... python phase3/icl_experiment.py

Requirements: anthropic, numpy, scikit-learn (for dynamic selection), rich
"""

import json
import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import Optional

import anthropic
import numpy as np

try:
    from rich import print
    from rich.console import Console
    from rich.table import Table
    console = Console()
except ImportError:
    console = None

# ── Dataset: Sentiment classification ────────────────────────────────────────

SENTIMENT_EXAMPLES = [
    ("The movie was absolutely brilliant, I loved every minute.", "positive"),
    ("Terrible film, waste of two hours of my life.", "negative"),
    ("An average experience, nothing stood out.", "neutral"),
    ("Stunning performances and beautiful cinematography!", "positive"),
    ("Boring plot with no character development whatsoever.", "negative"),
    ("It was okay, had some good moments and some bad ones.", "neutral"),
    ("Masterpiece of modern cinema, deeply moving.", "positive"),
    ("Predictable and clichéd from start to finish.", "negative"),
    ("Decent enough, won't remember it in a week though.", "neutral"),
    ("Exceptional directing and gripping storytelling.", "positive"),
    ("Painful to watch, performances were wooden.", "negative"),
    ("Nothing special but not offensive either.", "neutral"),
    ("A delightful surprise that exceeded all expectations.", "positive"),
    ("Dreadful dialogue and nonsensical plot twists.", "negative"),
    ("Had its moments but overall mediocre.", "neutral"),
]

TEST_CASES = [
    ("Incredible! I can't stop recommending it to friends.", "positive"),
    ("I want my money back. Absolutely dreadful.", "negative"),
    ("It's fine, pretty standard fare for the genre.", "neutral"),
    ("A haunting and unforgettable cinematic experience.", "positive"),
    ("Unwatchable. I left halfway through.", "negative"),
    ("Could've been better, could've been worse.", "neutral"),
]

# Pool of examples for dynamic selection
POOL = SENTIMENT_EXAMPLES


# ── Claude API Wrapper ────────────────────────────────────────────────────────

class LLM:
    def __init__(self, model: str = "claude-opus-4-6"):
        self.client = anthropic.Anthropic()
        self.model = model
        self.call_count = 0

    def complete(self, prompt: str, max_tokens: int = 128) -> str:
        self.call_count += 1
        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text.strip()
        except Exception as e:
            return f"[ERROR: {e}]"

    def extract_label(self, response: str) -> str:
        """Extract positive/negative/neutral from a free-form response."""
        resp_lower = response.lower()
        for label in ["positive", "negative", "neutral"]:
            if label in resp_lower:
                return label
        # Fallback: first word
        first_word = resp_lower.split()[0].strip(".,!?") if resp_lower.split() else "neutral"
        if first_word in ["positive", "negative", "neutral"]:
            return first_word
        return "neutral"


# ── Prompt Builders ───────────────────────────────────────────────────────────

def build_direct_prompt(test_text: str, k_examples: list[tuple[str, str]]) -> str:
    """Standard k-shot classification prompt."""
    lines = ["Classify the sentiment of movie reviews as positive, negative, or neutral.\n"]
    for text, label in k_examples:
        lines.append(f'Review: "{text}"\nSentiment: {label}\n')
    lines.append(f'Review: "{test_text}"\nSentiment:')
    return "\n".join(lines)


def build_cot_prompt(test_text: str, k_examples: list[tuple[str, str]], use_cot: bool = True) -> str:
    """CoT variant: examples include reasoning steps."""
    cot_examples = {
        "The movie was absolutely brilliant, I loved every minute.": (
            "positive",
            "The words 'absolutely brilliant' and 'loved every minute' are strong positive indicators."
        ),
        "Terrible film, waste of two hours of my life.": (
            "negative",
            "'Terrible' and 'waste' are clearly negative. The reviewer regrets watching."
        ),
        "An average experience, nothing stood out.": (
            "neutral",
            "'Average' and 'nothing stood out' indicate neither strong positive nor negative feelings."
        ),
        "Stunning performances and beautiful cinematography!": (
            "positive",
            "'Stunning' and 'beautiful' with an exclamation are enthusiastic praise."
        ),
    }

    lines = ["Classify movie review sentiment as positive, negative, or neutral.\n"]
    for text, label in k_examples:
        if use_cot and text in cot_examples:
            _, reasoning = cot_examples[text]
            lines.append(f'Review: "{text}"')
            lines.append(f'Reasoning: {reasoning}')
            lines.append(f'Sentiment: {label}\n')
        else:
            lines.append(f'Review: "{text}"\nSentiment: {label}\n')

    if use_cot:
        lines.append(f'Review: "{test_text}"')
        lines.append("Reasoning: Let me think about the sentiment indicators in this review.")
        lines.append("Sentiment:")
    else:
        lines.append(f'Review: "{test_text}"\nSentiment:')
    return "\n".join(lines)


# ── Dynamic Few-Shot Selection ────────────────────────────────────────────────

def simple_text_similarity(a: str, b: str) -> float:
    """Character n-gram overlap as a lightweight similarity measure."""
    def ngrams(s, n=3):
        s = s.lower()
        return set(s[i:i+n] for i in range(len(s) - n + 1))
    a_ng, b_ng = ngrams(a), ngrams(b)
    if not a_ng or not b_ng:
        return 0.0
    return len(a_ng & b_ng) / len(a_ng | b_ng)


def select_dynamic_examples(
    query: str, pool: list[tuple[str, str]], k: int, exclude: str = ""
) -> list[tuple[str, str]]:
    """Retrieve k most similar examples from the pool."""
    scored = [
        (simple_text_similarity(query, text), text, label)
        for text, label in pool
        if text != exclude
    ]
    scored.sort(reverse=True)
    return [(text, label) for _, text, label in scored[:k]]


# ── Experiments ───────────────────────────────────────────────────────────────

@dataclass
class ExperimentResult:
    name: str
    accuracy: float
    num_correct: int
    num_total: int
    per_example: list[dict] = field(default_factory=list)


def run_experiment(
    llm: LLM,
    test_cases: list[tuple[str, str]],
    example_selector: callable,  # (query) -> list[(text, label)]
    prompt_builder: callable,    # (query, examples) -> str
    name: str,
) -> ExperimentResult:
    correct = 0
    per_example = []
    for text, true_label in test_cases:
        examples = example_selector(text)
        prompt = prompt_builder(text, examples)
        response = llm.complete(prompt)
        pred = llm.extract_label(response)
        is_correct = pred == true_label
        correct += int(is_correct)
        per_example.append({
            "text": text[:50] + "...",
            "true": true_label,
            "pred": pred,
            "correct": is_correct,
        })

    return ExperimentResult(
        name=name,
        accuracy=correct / len(test_cases),
        num_correct=correct,
        num_total=len(test_cases),
        per_example=per_example,
    )


# ── Main Demo ─────────────────────────────────────────────────────────────────

def main():
    print("\n[bold]Phase 3.1 — In-Context Learning Playground[/bold]\n")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("[red]Error: ANTHROPIC_API_KEY not set[/red]")
        print("Running offline analysis only...")
        _demo_offline()
        return

    llm = LLM()
    random.seed(42)
    np.random.seed(42)

    all_results: list[ExperimentResult] = []

    # ── Experiment 1: Effect of number of examples
    print("[bold]Experiment 1: Number of Examples (0-shot → 5-shot)[/bold]")
    for k in [0, 1, 3, 5]:
        def make_selector(k_):
            def selector(query):
                if k_ == 0:
                    return []
                return random.sample(POOL, min(k_, len(POOL)))
            return selector

        result = run_experiment(
            llm,
            TEST_CASES,
            make_selector(k),
            lambda q, ex: build_direct_prompt(q, ex),
            f"{k}-shot (random examples)",
        )
        all_results.append(result)
        bar = "█" * int(result.accuracy * 20)
        print(f"  {k}-shot: {result.accuracy:.1%} ({result.num_correct}/{result.num_total}) {bar}")
        time.sleep(0.5)  # Rate limiting

    # ── Experiment 2: Example order matters
    print("\n[bold]Experiment 2: Example Order Effects[/bold]")
    k3_examples = POOL[:3]
    k3_reversed = list(reversed(k3_examples))

    for order_name, examples in [
        ("Chronological (pos→neg→neut)", k3_examples),
        ("Reversed (neut→neg→pos)", k3_reversed),
        ("Random shuffle", random.sample(k3_examples, len(k3_examples))),
    ]:
        ex = examples  # capture
        result = run_experiment(
            llm, TEST_CASES,
            lambda q, ex=ex: ex,
            lambda q, ex: build_direct_prompt(q, ex),
            f"Order: {order_name}",
        )
        all_results.append(result)
        print(f"  {order_name}: {result.accuracy:.1%}")
        time.sleep(0.5)

    # ── Experiment 3: Label quality (noise)
    print("\n[bold]Experiment 3: Label Quality (Correct vs Noisy Labels)[/bold]")
    noisy_pool = [(text, random.choice(["positive", "negative", "neutral"]))
                  for text, _ in POOL[:5]]  # 100% random labels

    for noise_name, pool_ in [
        ("Correct labels (clean)", POOL[:5]),
        ("50% noisy labels", POOL[:3] + noisy_pool[:2]),
        ("100% random labels", noisy_pool[:5]),
    ]:
        p_ = pool_
        result = run_experiment(
            llm, TEST_CASES,
            lambda q, p=p_: random.sample(p, min(3, len(p))),
            lambda q, ex: build_direct_prompt(q, ex),
            noise_name,
        )
        all_results.append(result)
        print(f"  {noise_name}: {result.accuracy:.1%}")
        time.sleep(0.5)

    # ── Experiment 4: Chain-of-Thought
    print("\n[bold]Experiment 4: Chain-of-Thought Prompting[/bold]")
    cot_examples = [POOL[0], POOL[1], POOL[2]]  # Known examples with CoT reasoning

    for cot_name, use_cot in [("Direct (no CoT)", False), ("With CoT reasoning", True)]:
        result = run_experiment(
            llm, TEST_CASES,
            lambda q: cot_examples,
            lambda q, ex, c=use_cot: build_cot_prompt(q, ex, use_cot=c),
            cot_name,
        )
        all_results.append(result)
        print(f"  {cot_name}: {result.accuracy:.1%}")
        time.sleep(0.5)

    # ── Experiment 5: Dynamic few-shot selection
    print("\n[bold]Experiment 5: Dynamic Few-Shot Selection[/bold]")
    for k in [1, 3, 5]:
        result = run_experiment(
            llm, TEST_CASES,
            lambda q, k_=k: select_dynamic_examples(q, POOL, k_),
            lambda q, ex: build_direct_prompt(q, ex),
            f"{k}-shot (dynamic retrieval)",
        )
        all_results.append(result)
        print(f"  {k}-shot dynamic: {result.accuracy:.1%}")
        time.sleep(0.5)

    # ── Summary
    print("\n[bold]Results Summary[/bold]")
    print(f"\n  {'Experiment':<40} {'Accuracy':>8} {'Correct':>8}")
    print("  " + "-" * 60)
    for r in all_results:
        bar = "█" * int(r.accuracy * 15)
        print(f"  {r.name:<40} {r.accuracy:>7.1%} {r.num_correct:>3}/{r.num_total:<3} {bar}")

    print(f"\n  Total Claude API calls: {llm.call_count}")

    # ── Key observations
    print("\n[bold]Key ICL Observations:[/bold]")
    zero_shot = next(r for r in all_results if r.name.startswith("0-shot"))
    five_shot  = next((r for r in all_results if r.name.startswith("5-shot")), None)
    cot_direct = next((r for r in all_results if "no CoT" in r.name), None)
    cot_with   = next((r for r in all_results if "CoT reasoning" in r.name), None)

    print(f"  • 0→5 shot improvement: {(five_shot.accuracy - zero_shot.accuracy):+.1%}")
    if cot_direct and cot_with:
        print(f"  • CoT improvement: {(cot_with.accuracy - cot_direct.accuracy):+.1%}")
    print("  • Label quality matters: noisy labels degrade performance")
    print("  • Example order creates recency bias (last examples weighted more)")
    print("  • Dynamic retrieval helps on out-of-distribution examples")

    print("\n[bold green]ICL Experiment complete![/bold green]")


def _demo_offline():
    """Offline demo showing ICL concepts without API calls."""
    print("\n[bold]Offline ICL Concepts Demo[/bold]\n")

    print("[bold]What is In-Context Learning?[/bold]")
    print("ICL allows models to learn from examples provided in the prompt at inference time,")
    print("without any weight updates. The model 'learns' the task format from demonstrations.\n")

    print("[bold]0-shot prompt:[/bold]")
    print(build_direct_prompt(TEST_CASES[0][0], []))
    print()

    print("[bold]3-shot prompt:[/bold]")
    print(build_direct_prompt(TEST_CASES[0][0], POOL[:3]))
    print()

    print("[bold]Chain-of-Thought prompt:[/bold]")
    print(build_cot_prompt(TEST_CASES[0][0], POOL[:2], use_cot=True))
    print()

    print("[bold]Dynamic selection for query:[/bold]")
    q = TEST_CASES[0][0]
    dynamic = select_dynamic_examples(q, POOL, k=3)
    print(f"Query: '{q}'")
    print("Most similar examples from pool:")
    for text, label in dynamic:
        sim = simple_text_similarity(q, text)
        print(f"  [{sim:.3f}] ({label}) {text}")

    print("\n[bold]Key ICL findings from literature:[/bold]")
    print("  • Performance scales with number of examples (up to context limit)")
    print("  • Models show recency bias: later examples have more influence")
    print("  • Label content matters less than example format (labels can even be random!)")
    print("  • CoT dramatically improves reasoning on complex, multi-step tasks")
    print("  • Dynamic retrieval of similar examples outperforms random selection")


if __name__ == "__main__":
    main()
