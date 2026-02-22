"""
Phase 3.3 — Multi-Agent Coordination System
=============================================
Scenario: AI Research Report Generation

Agents:
  • ManagerAgent    — orchestrates the workflow
  • ResearchAgent   — searches and retrieves relevant information
  • WriterAgent     — drafts sections based on research
  • EditorAgent     — reviews for clarity, accuracy, and coherence

Communication: structured JSON messages between agents
Metrics: report quality, agent utilisation, message count, latency

Run:
    ANTHROPIC_API_KEY=sk-... python phase3/multi_agent.py

Requirements: anthropic, rich
"""

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import anthropic

try:
    from rich import print
    from rich.console import Console
    from rich.panel import Panel
    console = Console()
except ImportError:
    console = None

# ── Local Research Database ───────────────────────────────────────────────────

RESEARCH_DB = {
    "attention mechanisms": """
Self-attention allows each token to attend to all other tokens in a sequence.
The Transformer uses scaled dot-product attention: Attention(Q,K,V) = softmax(QK^T/√d_k)V.
Multi-head attention enables attending to different representation subspaces simultaneously.
Cross-attention in the decoder allows attending to encoder outputs.
Flash Attention (Dao et al., 2022) achieves 2-4x speedup via memory-efficient tiling.
""",
    "large language models": """
LLMs are neural networks trained on massive text corpora to predict next tokens.
GPT-3 (175B parameters) demonstrated emergent capabilities with scale (Brown et al., 2020).
Scaling laws (Kaplan et al., 2020) predict performance from compute, data, and parameters.
Instruction tuning via RLHF significantly improves following user intent.
Modern LLMs include Claude (Anthropic), GPT-4 (OpenAI), Gemini (Google), and Llama (Meta).
""",
    "retrieval augmented generation": """
RAG combines parametric LLM knowledge with non-parametric retrieval memory.
The retriever finds relevant passages; the generator conditions on them.
Dense Passage Retrieval (DPR) uses bi-encoder architecture for efficient retrieval.
RAG reduces hallucination by grounding responses in retrieved evidence.
Challenges: retrieval quality, context length limits, and combining multiple passages.
""",
    "alignment and safety": """
AI alignment ensures AI systems pursue intended goals and values.
RLHF trains models using human preference comparisons (Ouyang et al., 2022).
Constitutional AI uses principles to guide self-critique and revision (Anthropic, 2022).
DPO provides a simpler alternative to RLHF without separate reward model training.
Key challenges: reward hacking, distributional shift, specification gaming.
""",
    "agent architectures": """
LLM agents augment language models with tools, memory, and planning capabilities.
ReAct (Yao et al., 2022) interleaves reasoning and acting for grounded decisions.
Tool use enables agents to search the web, execute code, and interact with APIs.
Multi-agent systems assign specialised roles to different agent instances.
Memory types: working memory (context), episodic memory, and semantic memory.
""",
}


def search_knowledge(topic: str, max_results: int = 2) -> str:
    """Search research database for a topic."""
    topic_lower = topic.lower()
    results = []
    for key, content in RESEARCH_DB.items():
        if any(word in key for word in topic_lower.split()) or topic_lower in key:
            results.append(f"[Topic: {key}]\n{content.strip()}")
    if not results:
        # Broader search
        for key, content in RESEARCH_DB.items():
            if any(word in content.lower() for word in topic_lower.split()[:2]):
                results.append(f"[Topic: {key}]\n{content.strip()}")
    return "\n\n".join(results[:max_results]) if results else f"No results for '{topic}'"


# ── Message Protocol ──────────────────────────────────────────────────────────

@dataclass
class Message:
    from_agent: str
    to_agent: str
    topic: str
    content: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "from": self.from_agent,
            "to": self.to_agent,
            "topic": self.topic,
            "content": self.content,
            "metadata": self.metadata,
        }

    def __str__(self):
        return f"[{self.from_agent}→{self.to_agent}] {self.topic}: {self.content[:100]}..."


# ── Agent Base Class ──────────────────────────────────────────────────────────

class Agent:
    def __init__(self, name: str, model: str = "claude-opus-4-6"):
        self.name = name
        self.model = model
        self.client = anthropic.Anthropic()
        self.message_log: list[Message] = []
        self.call_count = 0

    def _call_llm(self, system: str, user: str, max_tokens: int = 1024) -> str:
        self.call_count += 1
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
            return f"[LLM Error: {e}]"

    def process(self, message: Message) -> Message:
        raise NotImplementedError

    def log_message(self, msg: Message):
        self.message_log.append(msg)


# ── Specialised Agents ────────────────────────────────────────────────────────

class ResearchAgent(Agent):
    """Searches and synthesises information from the knowledge base."""

    SYSTEM = """You are a research specialist. Your job is to:
1. Search for information on the given topic
2. Synthesise findings into clear, factual summaries
3. Identify key concepts, methods, and findings
4. Return structured research notes with citations

Be factual and comprehensive. Format your output with clear sections."""

    def process(self, message: Message) -> Message:
        topic = message.content
        search_results = search_knowledge(topic)

        user_prompt = f"""Research the following topic for a technical report:

Topic: {topic}

Search results from knowledge base:
{search_results}

Synthesise this information into structured research notes covering:
1. Core concepts and definitions
2. Key methods and approaches
3. Important findings and contributions
4. Open challenges and limitations

Format as clear, numbered sections."""

        research_notes = self._call_llm(self.SYSTEM, user_prompt)

        return Message(
            from_agent=self.name,
            to_agent=message.from_agent,
            topic="research_complete",
            content=research_notes,
            metadata={"original_topic": topic, "sources": list(RESEARCH_DB.keys())},
        )


class WriterAgent(Agent):
    """Drafts report sections based on research notes."""

    SYSTEM = """You are a technical writer specialising in AI/ML research.
Your job is to transform research notes into clear, well-structured report sections.

Writing principles:
- Clear and accessible to a technical audience
- Logical flow from introduction to details to implications
- Use specific examples and concrete details
- Appropriate use of technical terminology
- Engaging but precise prose"""

    def process(self, message: Message) -> Message:
        topic = message.metadata.get("topic", "AI topic")
        research = message.content

        user_prompt = f"""Write a comprehensive report section on: {topic}

Research notes to draw from:
{research}

Structure your report section with:
1. Introduction and importance (2-3 sentences)
2. Technical overview (core concepts explained)
3. Key approaches and methods (specific techniques)
4. Practical implications and applications
5. Conclusion (1-2 sentences)

Length: ~400-500 words. Be technical but accessible."""

        draft = self._call_llm(self.SYSTEM, user_prompt, max_tokens=1024)

        return Message(
            from_agent=self.name,
            to_agent=message.from_agent,
            topic="draft_ready",
            content=draft,
            metadata={"topic": topic, "word_count": len(draft.split())},
        )


class EditorAgent(Agent):
    """Reviews and improves the draft for clarity and coherence."""

    SYSTEM = """You are a senior technical editor. Your job is to:
1. Check clarity and coherence of the draft
2. Ensure technical accuracy
3. Improve flow and transitions
4. Fix any awkward phrasing
5. Ensure consistent tone and style

Provide the improved version with brief editorial notes explaining your changes."""

    def process(self, message: Message) -> Message:
        topic = message.metadata.get("topic", "AI topic")
        draft = message.content

        user_prompt = f"""Edit and improve this technical report section on {topic}:

DRAFT:
{draft}

Please:
1. Improve clarity and flow
2. Fix any technical inaccuracies
3. Strengthen transitions
4. Make the writing more precise and engaging
5. Add any missing key information

Return the IMPROVED VERSION followed by brief EDITORIAL NOTES (bullet points)."""

        edited = self._call_llm(self.SYSTEM, user_prompt, max_tokens=1200)

        return Message(
            from_agent=self.name,
            to_agent=message.from_agent,
            topic="editing_complete",
            content=edited,
            metadata={"topic": topic, "edited_word_count": len(edited.split())},
        )


# ── Manager Agent ─────────────────────────────────────────────────────────────

class ManagerAgent(Agent):
    """Orchestrates the workflow across specialised agents."""

    def __init__(self, model: str = "claude-opus-4-6"):
        super().__init__("Manager", model)
        self.researcher = ResearchAgent("ResearchAgent", model)
        self.writer = WriterAgent("WriterAgent", model)
        self.editor = EditorAgent("EditorAgent", model)
        self.all_messages: list[Message] = []
        self.metrics = {
            "research_calls": 0,
            "writer_calls": 0,
            "editor_calls": 0,
            "total_messages": 0,
            "start_time": 0.0,
            "end_time": 0.0,
        }

    def _log(self, msg: Message, verbose: bool = True):
        self.all_messages.append(msg)
        self.metrics["total_messages"] += 1
        if verbose:
            icons = {
                "Manager": "📋",
                "ResearchAgent": "🔍",
                "WriterAgent": "✍️",
                "EditorAgent": "📝",
            }
            from_icon = icons.get(msg.from_agent, "•")
            to_icon   = icons.get(msg.to_agent, "•")
            print(f"  {from_icon} {msg.from_agent} → {to_icon} {msg.to_agent}: [{msg.topic}]")

    def generate_report(self, topic: str, verbose: bool = True) -> dict:
        """Run the full research → write → edit pipeline."""
        self.metrics["start_time"] = time.time()
        if verbose:
            print(f"\n[bold cyan]Generating report: '{topic}'[/bold cyan]")
            print("─" * 60)

        # ── Step 1: Research
        if verbose:
            print("\n[yellow]Stage 1: Research[/yellow]")
        research_request = Message(
            from_agent="Manager",
            to_agent="ResearchAgent",
            topic="research_request",
            content=topic,
        )
        self._log(research_request, verbose)
        research_result = self.researcher.process(research_request)
        self._log(research_result, verbose)
        self.metrics["research_calls"] += 1

        if verbose:
            preview = research_result.content[:200].replace("\n", " ")
            print(f"  Research preview: {preview}...")

        # ── Step 2: Write
        if verbose:
            print("\n[yellow]Stage 2: Writing[/yellow]")
        write_request = Message(
            from_agent="Manager",
            to_agent="WriterAgent",
            topic="write_request",
            content=research_result.content,
            metadata={"topic": topic},
        )
        self._log(write_request, verbose)
        write_result = self.writer.process(write_request)
        self._log(write_result, verbose)
        self.metrics["writer_calls"] += 1

        if verbose:
            wc = write_result.metadata.get("word_count", "?")
            print(f"  Draft: {wc} words")

        # ── Step 3: Edit
        if verbose:
            print("\n[yellow]Stage 3: Editing[/yellow]")
        edit_request = Message(
            from_agent="Manager",
            to_agent="EditorAgent",
            topic="edit_request",
            content=write_result.content,
            metadata={"topic": topic},
        )
        self._log(edit_request, verbose)
        edit_result = self.editor.process(edit_request)
        self._log(edit_result, verbose)
        self.metrics["editor_calls"] += 1

        self.metrics["end_time"] = time.time()
        latency = self.metrics["end_time"] - self.metrics["start_time"]

        return {
            "topic": topic,
            "final_report": edit_result.content,
            "research_notes": research_result.content,
            "draft": write_result.content,
            "metrics": {
                "latency_seconds": round(latency, 1),
                "total_messages": self.metrics["total_messages"],
                "agent_calls": {
                    "researcher": self.metrics["research_calls"],
                    "writer":     self.metrics["writer_calls"],
                    "editor":     self.metrics["editor_calls"],
                },
                "total_llm_calls": (
                    self.researcher.call_count
                    + self.writer.call_count
                    + self.editor.call_count
                ),
            },
        }

    def generate_multi_section_report(self, sections: list[str], verbose: bool = True) -> dict:
        """Generate a multi-section report in parallel (sequential for API limits)."""
        print(f"\n[bold]Generating multi-section report ({len(sections)} sections)[/bold]")
        reports = []
        for section in sections:
            r = self.generate_report(section, verbose=verbose)
            reports.append(r)

        # Combine into final report
        combined = f"# AI/ML Technical Report\n\n"
        for r in reports:
            combined += f"## {r['topic'].title()}\n\n"
            # Extract main content (before editorial notes if present)
            content = r["final_report"]
            if "EDITORIAL NOTES" in content.upper():
                content = content[:content.upper().find("EDITORIAL NOTES")].strip()
            combined += content + "\n\n"

        total_metrics = {
            "sections": len(sections),
            "total_llm_calls": sum(r["metrics"]["total_llm_calls"] for r in reports),
            "total_messages": sum(r["metrics"]["total_messages"] for r in reports),
            "total_latency": sum(r["metrics"]["latency_seconds"] for r in reports),
        }
        return {"combined_report": combined, "section_reports": reports, "metrics": total_metrics}


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_report(report_text: str, topic: str) -> dict:
    """Simple heuristic evaluation of report quality."""
    words = report_text.split()
    sentences = re.split(r"[.!?]+", report_text)
    paragraphs = [p for p in report_text.split("\n\n") if p.strip()]

    # Coherence: does it have intro, body, conclusion structure?
    has_intro = any(w in report_text[:200].lower() for w in ["introduction", "overview", "this section"])
    has_technical = any(w in report_text.lower() for w in ["method", "approach", "technique", "algorithm"])
    has_conclusion = any(w in report_text[-200:].lower() for w in ["conclusion", "summary", "overall", "thus"])

    # Coverage: topic keywords present
    topic_words = topic.lower().split()
    coverage = sum(1 for tw in topic_words if tw in report_text.lower()) / max(len(topic_words), 1)

    return {
        "word_count": len(words),
        "sentence_count": len([s for s in sentences if s.strip()]),
        "paragraph_count": len(paragraphs),
        "has_structure": int(has_intro) + int(has_technical) + int(has_conclusion),
        "topic_coverage": coverage,
        "quality_score": (
            min(len(words) / 400, 1.0) * 0.3    # length
            + coverage * 0.3                       # topic coverage
            + (int(has_technical) * 0.4)           # technical depth
        ),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n[bold]Phase 3.3 — Multi-Agent Coordination System[/bold]")
    print("Scenario: AI Research Report Generation\n")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("[red]ANTHROPIC_API_KEY not set — showing architecture demo[/red]\n")
        _demo_architecture()
        return

    manager = ManagerAgent()

    # ── Single topic report
    print("[bold]Demo 1: Single Section Report[/bold]")
    result = manager.generate_report("attention mechanisms in transformers", verbose=True)

    print("\n\n[bold]Generated Report:[/bold]")
    print("─" * 60)
    # Show first 800 chars
    report_preview = result["final_report"][:800]
    print(report_preview + ("..." if len(result["final_report"]) > 800 else ""))

    # Evaluate
    eval_metrics = evaluate_report(result["final_report"], "attention mechanisms")
    print("\n[bold]Report Quality Metrics:[/bold]")
    print(f"  Words: {eval_metrics['word_count']}")
    print(f"  Sentences: {eval_metrics['sentence_count']}")
    print(f"  Paragraphs: {eval_metrics['paragraph_count']}")
    print(f"  Structure score: {eval_metrics['has_structure']}/3")
    print(f"  Topic coverage: {eval_metrics['topic_coverage']:.0%}")
    print(f"  Quality score: {eval_metrics['quality_score']:.2f}")

    # System metrics
    m = result["metrics"]
    print("\n[bold]System Metrics:[/bold]")
    print(f"  Total messages:   {m['total_messages']}")
    print(f"  Total LLM calls:  {m['total_llm_calls']}")
    print(f"  Latency:          {m['latency_seconds']}s")
    print(f"  Agent utilisation:")
    for agent, calls in m["agent_calls"].items():
        print(f"    {agent}: {calls} call(s)")

    # ── Multi-section report
    print("\n\n[bold]Demo 2: Multi-Section Report[/bold]")
    sections = ["large language models", "alignment and safety"]
    multi_result = manager.generate_multi_section_report(sections, verbose=False)

    print(f"\n  Generated {multi_result['metrics']['sections']} sections")
    print(f"  Total LLM calls: {multi_result['metrics']['total_llm_calls']}")
    print(f"  Total latency: {multi_result['metrics']['total_latency']:.1f}s")
    print(f"\n  Combined report length: {len(multi_result['combined_report'].split())} words")

    # Compare single vs multi-agent
    print("\n[bold]Single Agent vs Multi-Agent Comparison:[/bold]")
    print("  Single agent (direct):")
    print("    - No specialisation, one-size-fits-all")
    print("    - No iterative refinement")
    print("    - Faster (fewer LLM calls)")
    print("  Multi-agent (this system):")
    print("    - Research → Write → Edit specialisation")
    print("    - Each agent focuses on what it does best")
    print("    - Higher quality through iteration")
    print("    - More expensive (3x LLM calls)")
    print("    - Communication overhead but worth it for quality")

    print("\n[bold green]Multi-Agent System complete![/bold green]")


def _demo_architecture():
    """Show architecture without API calls."""
    print("[bold]Multi-Agent Architecture:[/bold]")
    print("""
  ┌─────────────────────────────────────────────────────────┐
  │                    ManagerAgent                         │
  │                        │                               │
  │          ┌─────────────┼──────────────┐                │
  │          ↓             ↓              ↓                │
  │   ResearchAgent   WriterAgent   EditorAgent            │
  │       🔍               ✍️              📝               │
  │   Searches KB    Drafts report   Improves draft         │
  │                                                         │
  │   Message flow:                                         │
  │   Manager ──research_request──→ ResearchAgent          │
  │   ResearchAgent ──research_complete──→ Manager         │
  │   Manager ──write_request──→ WriterAgent               │
  │   WriterAgent ──draft_ready──→ Manager                 │
  │   Manager ──edit_request──→ EditorAgent                │
  │   EditorAgent ──editing_complete──→ Manager            │
  └─────────────────────────────────────────────────────────┘

  Message format (JSON):
  {
    "from": "Manager",
    "to": "ResearchAgent",
    "topic": "research_request",
    "content": "attention mechanisms in transformers",
    "metadata": {}
  }
""")

    # Show local research database
    print("[bold]Research database topics:[/bold]")
    for key in RESEARCH_DB:
        print(f"  • {key}")

    print("\n[bold]Sample search:[/bold]")
    print(search_knowledge("attention mechanisms")[:300] + "...")


if __name__ == "__main__":
    main()
