"""
Phase 3.2 — ReAct Agent
========================
Implements the Thought → Action → Observation loop from
"ReAct: Synergizing Reasoning and Acting in Language Models" (Yao et al., 2022)

Tools:
  • calculator  — evaluate arithmetic expressions
  • wiki_search  — search a small local knowledge base
  • lookup       — look up a specific entity

Run:
    ANTHROPIC_API_KEY=sk-... python phase3/react_agent.py

Requirements: anthropic, rich
"""

import json
import math
import operator
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import anthropic

try:
    from rich import print
    from rich.panel import Panel
    from rich.console import Console
    console = Console()
except ImportError:
    console = None

# ── Local Knowledge Base ─────────────────────────────────────────────────────

KNOWLEDGE_BASE = {
    "albert einstein": {
        "born": "March 14, 1879",
        "birthplace": "Ulm, Kingdom of Württemberg, German Empire",
        "died": "April 18, 1955",
        "nationality": "German-Swiss-American",
        "field": "Theoretical physics",
        "known_for": ["Theory of Relativity", "E=mc²", "Photoelectric effect"],
        "teacher": "Hermann Minkowski was his doctoral advisor at ETH Zurich",
        "awards": ["Nobel Prize in Physics 1921"],
    },
    "hermann minkowski": {
        "born": "June 22, 1864",
        "birthplace": "Aleksotas, Russian Empire (now Kaunas, Lithuania)",
        "died": "January 12, 1909",
        "field": "Mathematics and mathematical physics",
        "known_for": ["Minkowski space", "Minkowski spacetime"],
    },
    "ulm": {
        "country": "Germany",
        "state": "Baden-Württemberg",
        "description": "City in southern Germany, birthplace of Albert Einstein",
    },
    "theory of relativity": {
        "author": "Albert Einstein",
        "special_relativity": "1905",
        "general_relativity": "1915",
        "famous_equation": "E=mc²",
    },
    "marie curie": {
        "born": "November 7, 1867",
        "birthplace": "Warsaw, Kingdom of Poland (now Poland)",
        "field": "Physics and Chemistry",
        "known_for": ["Radioactivity", "Polonium", "Radium"],
        "awards": ["Nobel Prize in Physics 1903", "Nobel Prize in Chemistry 1911"],
        "teacher": "Gabriel Lippmann supervised her doctoral research",
    },
    "gabriel lippmann": {
        "born": "August 16, 1845",
        "birthplace": "Bonnevoie, Luxembourg",
        "field": "Physics",
        "awards": ["Nobel Prize in Physics 1908"],
    },
    "python language": {
        "creator": "Guido van Rossum",
        "first_released": "1991",
        "paradigm": ["Object-oriented", "Functional", "Procedural"],
        "typing": "Dynamic, strong",
    },
    "transformer architecture": {
        "paper": "Attention Is All You Need",
        "authors": "Vaswani et al.",
        "year": "2017",
        "key_innovation": "Self-attention mechanism",
        "applications": ["NLP", "Computer Vision", "Audio"],
    },
}


def wiki_search(query: str) -> str:
    """Search local knowledge base."""
    q = query.lower().strip()
    # Direct lookup
    for key, data in KNOWLEDGE_BASE.items():
        if q in key or key in q:
            return json.dumps(data, indent=2)
    # Partial match
    matches = [key for key in KNOWLEDGE_BASE if any(w in key for w in q.split())]
    if matches:
        results = {}
        for m in matches[:2]:
            results[m] = KNOWLEDGE_BASE[m]
        return json.dumps(results, indent=2)
    return f"No results found for '{query}'. Try a different search term."


def lookup(entity: str) -> str:
    """Direct entity lookup."""
    e = entity.lower().strip()
    for key, data in KNOWLEDGE_BASE.items():
        if e == key or e in key:
            return json.dumps(data, indent=2)
    return f"Entity '{entity}' not found."


def calculator(expression: str) -> str:
    """Safely evaluate a mathematical expression."""
    # Only allow safe operations
    allowed = set("0123456789+-*/()., ")
    expr = expression.strip()
    if not all(c in allowed for c in expr):
        # Try to extract numbers and operators
        expr = re.sub(r"[^0-9+\-*/().,\s]", "", expr)
    try:
        # Replace commas with dots for European notation
        expr = expr.replace(",", ".")
        result = eval(expr, {"__builtins__": {}}, {
            "sqrt": math.sqrt, "pow": math.pow,
            "abs": abs, "round": round, "pi": math.pi,
        })
        return str(result)
    except Exception as e:
        return f"Calculation error: {e}. Expression: {expr}"


# ── Tool Registry ─────────────────────────────────────────────────────────────

TOOLS = {
    "wiki_search": {
        "fn": wiki_search,
        "description": "Search the knowledge base for information about a topic.",
        "param": "query",
    },
    "lookup": {
        "fn": lookup,
        "description": "Look up a specific entity by name.",
        "param": "entity",
    },
    "calculator": {
        "fn": calculator,
        "description": "Evaluate a mathematical expression.",
        "param": "expression",
    },
}

SYSTEM_PROMPT = """You are a helpful research assistant that answers questions by reasoning step by step
and using tools to look up information.

You have access to these tools:
- wiki_search[query]: Search for information about a topic
- lookup[entity]: Look up specific facts about an entity
- calculator[expression]: Evaluate mathematical expressions

Use the following format EXACTLY:
Thought: <your reasoning about what to do next>
Action: tool_name[input]
Observation: <result of the action - provided by the system>
... (repeat Thought/Action/Observation as needed)
Thought: I now have enough information to answer.
Final Answer: <your complete answer>

Important rules:
- Always start with a Thought
- After each Action, wait for the Observation before continuing
- Only use tools listed above
- Give a complete Final Answer that addresses the full question
"""


# ── ReAct Agent ───────────────────────────────────────────────────────────────

@dataclass
class Step:
    thought: str
    action: Optional[str] = None
    action_input: Optional[str] = None
    observation: Optional[str] = None
    is_final: bool = False
    final_answer: Optional[str] = None


@dataclass
class AgentResult:
    question: str
    steps: list[Step] = field(default_factory=list)
    final_answer: str = ""
    success: bool = False
    num_steps: int = 0


class ReActAgent:
    def __init__(
        self,
        tools: dict = TOOLS,
        model: str = "claude-opus-4-6",
        max_steps: int = 8,
        verbose: bool = True,
    ):
        self.tools = tools
        self.model = model
        self.max_steps = max_steps
        self.verbose = verbose
        self.client = anthropic.Anthropic()

    def _parse_response(self, text: str) -> tuple[str, Optional[str], Optional[str], bool, Optional[str]]:
        """Parse LLM output into (thought, tool_name, tool_input, is_final, final_answer)."""
        thought, tool_name, tool_input, is_final, final_answer = "", None, None, False, None

        # Extract Thought
        thought_match = re.search(r"Thought:\s*(.+?)(?=\nAction:|\nFinal Answer:|$)", text, re.DOTALL)
        if thought_match:
            thought = thought_match.group(1).strip()

        # Check for Final Answer
        final_match = re.search(r"Final Answer:\s*(.+)", text, re.DOTALL)
        if final_match:
            is_final = True
            final_answer = final_match.group(1).strip()
            return thought, None, None, True, final_answer

        # Extract Action
        action_match = re.search(r"Action:\s*(\w+)\[(.+?)\]", text, re.DOTALL)
        if action_match:
            tool_name = action_match.group(1).strip()
            tool_input = action_match.group(2).strip()

        return thought, tool_name, tool_input, is_final, final_answer

    def _execute_tool(self, tool_name: str, tool_input: str) -> str:
        """Execute a tool and return its output."""
        if tool_name not in self.tools:
            return f"Error: Tool '{tool_name}' not found. Available: {list(self.tools.keys())}"
        try:
            return self.tools[tool_name]["fn"](tool_input)
        except Exception as e:
            return f"Tool execution error: {e}"

    def run(self, question: str) -> AgentResult:
        """Run the ReAct loop."""
        result = AgentResult(question=question)

        # Build conversation
        messages = [{"role": "user", "content": f"Question: {question}"}]

        if self.verbose:
            print(f"\n[cyan]Question:[/cyan] {question}")
            print("─" * 60)

        for step_num in range(self.max_steps):
            # Get LLM response
            try:
                resp = self.client.messages.create(
                    model=self.model,
                    max_tokens=512,
                    system=SYSTEM_PROMPT,
                    messages=messages,
                )
                response_text = resp.content[0].text
            except Exception as e:
                result.final_answer = f"API error: {e}"
                result.success = False
                break

            # Parse the response
            thought, tool_name, tool_input, is_final, final_answer = self._parse_response(response_text)

            step = Step(thought=thought)

            if self.verbose and thought:
                print(f"\n[yellow]Thought:[/yellow] {thought}")

            if is_final:
                step.is_final = True
                step.final_answer = final_answer
                result.steps.append(step)
                result.final_answer = final_answer or ""
                result.success = True
                result.num_steps = step_num + 1
                if self.verbose:
                    print(f"\n[green]Final Answer:[/green] {final_answer}")
                break

            if tool_name:
                step.action = tool_name
                step.action_input = tool_input

                # Execute tool
                observation = self._execute_tool(tool_name, tool_input)
                step.observation = observation

                if self.verbose:
                    print(f"[blue]Action:[/blue] {tool_name}[{tool_input}]")
                    obs_preview = observation[:200] + "..." if len(observation) > 200 else observation
                    print(f"[magenta]Observation:[/magenta] {obs_preview}")

                # Append to messages
                messages.append({"role": "assistant", "content": response_text})
                messages.append({
                    "role": "user",
                    "content": f"Observation: {observation}"
                })

            result.steps.append(step)

            if step_num == self.max_steps - 1:
                result.final_answer = "Max steps reached without finding answer."
                result.success = False

        return result

    def evaluate(self, test_set: list[dict]) -> dict:
        """Evaluate on a test set with expected answers/keywords."""
        successes = 0
        step_counts = []
        grounding_scores = []

        for item in test_set:
            res = self.run(item["question"])
            ans = res.final_answer.lower()

            # Check correctness via keywords
            keywords = [k.lower() for k in item.get("keywords", [])]
            correct = all(kw in ans for kw in keywords) if keywords else res.success
            successes += int(correct)
            step_counts.append(res.num_steps)

            # Grounding: answer uses tool observations
            tools_used = [s.action for s in res.steps if s.action]
            grounding_scores.append(1.0 if tools_used else 0.0)

        n = len(test_set)
        return {
            "success_rate": successes / max(n, 1),
            "avg_steps": np.mean(step_counts) if step_counts else 0,
            "grounding_rate": np.mean(grounding_scores) if grounding_scores else 0,
        }


# ── Test Questions ────────────────────────────────────────────────────────────

TEST_QUESTIONS = [
    {
        "question": "What city was Albert Einstein born in, and what country is that city in today?",
        "keywords": ["ulm", "germany"],
    },
    {
        "question": "Who was Marie Curie's doctoral advisor, and what Nobel Prize did they win?",
        "keywords": ["lippmann", "nobel"],
    },
    {
        "question": "What is 15% of 847?",
        "keywords": ["127.05"],
    },
    {
        "question": "What year was the Transformer architecture paper published, and who wrote it?",
        "keywords": ["2017", "vaswani"],
    },
    {
        "question": "If Einstein was born in 1879 and died in 1955, how old was he when he died?",
        "keywords": ["76"],
    },
]


def main():
    import numpy as np

    print("\n[bold]Phase 3.2 — ReAct Agent[/bold]\n")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("[red]ANTHROPIC_API_KEY not set — showing architecture demo[/red]\n")
        _demo_architecture()
        return

    agent = ReActAgent(max_steps=8, verbose=True)

    # ── Run a few demo questions
    print("[bold]Demo Questions[/bold]\n")
    demo_qs = TEST_QUESTIONS[:3]
    results = []
    for item in demo_qs:
        r = agent.run(item["question"])
        results.append(r)
        print("\n" + "═" * 60)
        time.sleep(1)  # Rate limiting

    # ── Evaluation
    print("\n\n[bold]Evaluation on Test Set[/bold]")
    eval_results = []
    for item in TEST_QUESTIONS:
        print(f"\n  Q: {item['question'][:60]}...")
        r = agent.run(item["question"])
        ans = r.final_answer.lower()
        keywords = [k.lower() for k in item.get("keywords", [])]
        correct = all(kw in ans for kw in keywords)
        tools_used = [s.action for s in r.steps if s.action]
        eval_results.append({
            "correct": correct,
            "steps": r.num_steps,
            "tools": tools_used,
        })
        icon = "✓" if correct else "✗"
        print(f"  {icon} Steps: {r.num_steps} | Tools: {tools_used}")
        time.sleep(0.5)

    # Summary
    n = len(eval_results)
    accuracy = sum(r["correct"] for r in eval_results) / max(n, 1)
    avg_steps = np.mean([r["steps"] for r in eval_results])
    grounding = sum(1 for r in eval_results if r["tools"]) / max(n, 1)

    print(f"\n[bold]Results:[/bold]")
    print(f"  Accuracy:      {accuracy:.1%} ({sum(r['correct'] for r in eval_results)}/{n})")
    print(f"  Avg steps:     {avg_steps:.1f}")
    print(f"  Grounding rate:{grounding:.1%}")

    # Tool usage breakdown
    tool_usage: dict[str, int] = {}
    for r in eval_results:
        for t in r["tools"]:
            tool_usage[t] = tool_usage.get(t, 0) + 1
    print(f"  Tool usage:    {tool_usage}")

    print("\n[bold green]ReAct Agent complete![/bold green]")


def _demo_architecture():
    """Show ReAct architecture without API calls."""
    print("[bold]ReAct Loop Architecture:[/bold]")
    print("""
  ┌─────────────────────────────────────────────────┐
  │                  ReAct Loop                     │
  │                                                 │
  │  Question ──→ LLM                               │
  │               ↓                                 │
  │          Thought: (reasoning)                   │
  │               ↓                                 │
  │          Action: tool[input]                    │
  │               ↓                                 │
  │          [Tool Execution]                       │
  │               ↓                                 │
  │         Observation: result                     │
  │               ↓                                 │
  │          (repeat until done)                    │
  │               ↓                                 │
  │         Final Answer: ...                       │
  └─────────────────────────────────────────────────┘
""")

    print("[bold]Example trace:[/bold]")
    print("""
  Question: What city was Einstein born in?

  Thought: I need to find where Einstein was born.
  Action: wiki_search[Albert Einstein birthplace]
  Observation: {"born": "March 14, 1879", "birthplace": "Ulm, Germany", ...}

  Thought: The search shows Einstein was born in Ulm, Germany.
  Final Answer: Albert Einstein was born in Ulm, which is in Germany.
""")

    print("[bold]Tool outputs:[/bold]")
    print(f"  wiki_search['albert einstein']: {wiki_search('albert einstein')[:200]}...")
    print(f"  calculator['15 * 847 / 100']:   {calculator('15 * 847 / 100')}")
    print(f"  lookup['marie curie']:           {lookup('marie curie')[:200]}...")


if __name__ == "__main__":
    main()
