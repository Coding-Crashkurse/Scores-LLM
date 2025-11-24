r"""
Typer CLI demo: Agent Routing with DEEP DIVE Logprobs (Raw Dump).
Shows formatted tables AND raw python data structures for maximum transparency.

Run:
  uv run python main.py
  uv run python main.py ask --query "I need a pitch"
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Literal

import typer
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from tabulate import tabulate

load_dotenv()

app = typer.Typer(help="Agent routing with Deep Dive Logprobs (Raw Dump).")

# --- 1. CONFIG: AGENTS ---------------------------------------------------------

AGENT_CARDS: List[Dict[str, str]] = [
    {
        "name": "TravelBuddy",
        "description": "Plan quick city trips with budgets, flights, and local tips.",
    },
    {
        "name": "CodeFixer",
        "description": "Debug and refactor small Python or JavaScript snippets.",
    },
    {
        "name": "HealthNote",
        "description": "Summarize lifestyle and nutrition questions into simple advice.",
    },
    {
        "name": "BizPitch",
        "description": "Draft short startup pitches and positioning statements.",
    },
    {
        "name": "DataScout",
        "description": "Explain CSV/Excel columns and basic data cleaning steps.",
    },
]

Answer = Literal["YES", "NO"]

# --- 2. RESULT-MODELS ----------------------------------------------------------


class RouterLLMOutput(BaseModel):
    """Binary router decision for one agent."""

    answer: Answer = Field(description="Either 'YES' or 'NO'")
    reasoning: str = Field(description="One short English sentence")


class YesNoStat(BaseModel):
    token: str
    logprob: float
    prob: float


class AgentDecision(BaseModel):
    """Final decision plus logprob analysis per agent."""

    agent_name: str
    answer: Answer = Field(description="'YES' or 'NO'")
    reasoning: str
    prob_yes: Optional[float] = None
    prob_no: Optional[float] = None
    yes_no_stats: List[YesNoStat] = Field(default_factory=list)
    # Raw candidates before filtering
    raw_candidates: List[Tuple[str, float]] = Field(default_factory=list)


# --- 3. LLM SETUP --------------------------------------------------------------

llm_base = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3,  # Allow a bit of variance
    logprobs=True,
    top_logprobs=10,  # Fetch the top 10 tokens
    max_tokens=64,
)

# Native structured output (json_schema is the default for ChatOpenAI)
router_llm = llm_base.with_structured_output(
    RouterLLMOutput,
    include_raw=True,  # keep response_metadata.logprobs
)

SYSTEM_PROMPT = """
You are a neutral binary classifier for agent routing.

You receive:
- a user query
- ONE agent card (name + description)

Your job:
- Decide if this agent is suitable to help with the query.

Rules:
- Set "answer" to "YES" or "NO" (uppercase).
- "reasoning" must be one short sentence in English.
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        (
            "user",
            "User Query:\n{query}\n\nAgent Card:\n{name}: {description}",
        ),
    ]
)


# --- 4. LOGPROB-UTILS (Enum-Style YES/NO) --------------------------------------


def _normalize_label_token(tok: str) -> str:
    """Normalize raw token to 'YES'/'NO' when applicable."""
    # Strip whitespace and simple quotes
    return tok.strip().strip("\"'").upper()


def extract_yes_no_enum_stats(
    logprobs: Dict[str, Any],
) -> Tuple[float, float, List[YesNoStat], List[Tuple[str, float]]]:
    """
    Analyze the token stream, find the YES/NO decision token,
    and return both normalized stats and the raw candidates.

    Assumptions (not defensive coding):
    - logprobs["content"] exists and is a list.
    - There is exactly one step where the label token ("YES" or "NO") is generated.
    - In that step, YES/NO appear in either token or top_logprobs.
    """

    content = logprobs["content"]

    decision_candidates: List[Tuple[str, float]] | None = None

    for step in content:
        # 1. Collect primary token and top-k candidates
        candidates: List[Tuple[str, float]] = [(step["token"], step["logprob"])]
        for alt in step["top_logprobs"]:
            candidates.append((alt["token"], alt["logprob"]))

        # 2. Check if this step carries the YES/NO label
        if any(_normalize_label_token(t) in ("YES", "NO") for t, _ in candidates):
            decision_candidates = candidates
            break

    if decision_candidates is None:
        # If this happens, the schema/prompt is broken
        raise RuntimeError(
            "No YES/NO decision token found in logprobs. Check schema/prompt."
        )

    # 3. Find best YES and NO by logprob
    best_lp_yes: float | None = None
    best_lp_no: float | None = None

    for tok_raw, lp_raw in decision_candidates:
        norm = _normalize_label_token(tok_raw)
        if norm == "YES":
            if best_lp_yes is None or lp_raw > best_lp_yes:
                best_lp_yes = lp_raw
        elif norm == "NO":
            if best_lp_no is None or lp_raw > best_lp_no:
                best_lp_no = lp_raw

    if best_lp_yes is None and best_lp_no is None:
        raise RuntimeError("Decision step does not contain YES or NO candidates.")

    logits: List[Tuple[str, float]] = []
    if best_lp_yes is not None:
        logits.append(("YES", best_lp_yes))
    if best_lp_no is not None:
        logits.append(("NO", best_lp_no))

    # Softmax only across YES/NO
    max_lp = max(lp for _, lp in logits)
    exps = [(label, math.exp(lp - max_lp)) for label, lp in logits]
    denom = sum(v for _, v in exps)

    stats: List[YesNoStat] = []
    p_yes = 0.0
    p_no = 0.0

    for label, val in exps:
        prob = val / denom
        lp_label = best_lp_yes if label == "YES" else best_lp_no  # type: ignore[arg-type]
        stats.append(
            YesNoStat(
                token=label,
                logprob=lp_label,
                prob=prob,
            )
        )
        if label == "YES":
            p_yes = prob
        else:
            p_no = prob

    return p_yes, p_no, stats, decision_candidates


# --- 5. SINGLE AGENT -> DECISION + LOGPROBS -----------------------------------


def classify_agent(
    query: str, card: Dict[str, str]
) -> Tuple[AgentDecision, List[YesNoStat]]:
    messages = prompt.format_messages(
        query=query,
        name=card["name"],
        description=card["description"],
    )

    # Structured output plus raw message
    result = router_llm.invoke(messages)
    router_out: RouterLLMOutput = result["parsed"]
    raw_msg = result["raw"]

    # The OpenAI-style logprobs + top_logprobs live here
    logprobs = raw_msg.response_metadata["logprobs"]

    p_yes, p_no, stats, raw_candidates = extract_yes_no_enum_stats(logprobs)

    decision = AgentDecision(
        agent_name=card["name"],
        answer=router_out.answer,
        reasoning=router_out.reasoning,
        prob_yes=p_yes,
        prob_no=p_no,
        yes_no_stats=stats,
        raw_candidates=raw_candidates,
    )

    return decision, stats


# --- 6. ROUTING FOR ONE QUERY --------------------------------------------------


def route_query(query: str) -> None:
    typer.echo("=" * 100)
    typer.echo(f"QUERY: {query}")
    typer.echo("=" * 100)

    decisions: List[AgentDecision] = []

    for card in AGENT_CARDS:
        decision, _ = classify_agent(query, card)
        decisions.append(decision)

    # Sort by p(YES)
    decisions_sorted = sorted(
        decisions, key=lambda d: (d.prob_yes or 0.0), reverse=True
    )

    # 6.1 Summary Table
    table_rows = []
    for d in decisions_sorted:
        p_yes_str = f"{d.prob_yes:.3f}" if d.prob_yes is not None else "-"
        p_no_str = f"{d.prob_no:.3f}" if d.prob_no is not None else "-"
        row = [
            d.agent_name,
            d.answer,
            p_yes_str,
            p_no_str,
            d.reasoning,
        ]
        table_rows.append(row)

    summary = tabulate(
        table_rows,
        headers=["Agent", "Ans", "p(YES)", "p(NO)", "Reason"],
        tablefmt="github",
        stralign="left",
    )
    typer.echo("ROUTING SUMMARY")
    typer.echo(summary)

    # 6.2 DEEP DIVE LOGPROBS
    typer.echo("\n" + "#" * 60)
    typer.echo("   DEEP DIVE: EVERY SINGLE TOKEN CONSIDERED")
    typer.echo("#" * 60)

    for d in decisions_sorted:
        typer.echo(
            f"\n>>> Agent: {typer.style(d.agent_name, bold=True, fg=typer.colors.CYAN)}"
        )

        if not d.raw_candidates:
            typer.echo("   (No logprobs captured for the decision step)")
            continue

        # Raw dump: unchanged for debugging
        typer.echo("   RAW PYTHON DATA (List[Tuple[str, float]]):")
        typer.echo(f"   {d.raw_candidates}")
        typer.echo("")

        # Table for the candidates
        raw_rows = []

        # Sort by logprob (best first)
        sorted_candidates = sorted(d.raw_candidates, key=lambda x: x[1], reverse=True)

        total_visible_prob = 0.0

        for tok, lp in sorted_candidates:
            prob = math.exp(lp)
            prob_percent = prob * 100
            total_visible_prob += prob_percent

            clean_tok = _normalize_label_token(tok)
            is_yes = clean_tok == "YES"
            is_no = clean_tok == "NO"

            if is_yes:
                display_tok = typer.style(repr(tok), fg=typer.colors.GREEN, bold=True)
                category = "YES bucket"
            elif is_no:
                display_tok = typer.style(repr(tok), fg=typer.colors.RED, bold=True)
                category = "NO bucket"
            else:
                display_tok = typer.style(repr(tok), fg=typer.colors.BRIGHT_BLACK)
                category = "Ignored"

            raw_rows.append(
                [
                    display_tok,
                    f"{lp:.4f}",
                    f"{prob_percent:.4f}%",
                    category,
                ]
            )

        print(
            tabulate(
                raw_rows,
                headers=["Token (Raw)", "Logprob", "Prob (%)", "Category"],
                tablefmt="simple",
            )
        )

        # Tail mass of the distribution
        missing_prob = 100.0 - total_visible_prob
        if missing_prob > 0.01:
            typer.echo(
                f"   ... plus {missing_prob:.4f}% probability spread across thousands of other tokens (Tail)."
            )

        if d.prob_yes is not None:
            p_yes = d.prob_yes * 100
            p_no = (d.prob_no or 0.0) * 100
            typer.echo(f"   ---> FINAL NORMALIZED: YES={p_yes:.1f}% | NO={p_no:.1f}%")


# --- 7. CLI COMMANDS -----------------------------------------------------------


@app.command()
def demo() -> None:
    queries = [
        "Plan a weekend trip to Porto with a $600 budget.",
        "Write a Python script to clean my marketing data CSV.",
        "Is a tomato a vegetable? Just give me the fact.",
        "Write the text for my startup's landing page.",
        "Generate an image of a flying toaster.",
    ]
    for q in queries:
        route_query(q)


@app.command()
def ask(query: str) -> None:
    route_query(query)


@app.callback(invoke_without_command=True)
def _default(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        demo()


if __name__ == "__main__":
    app()
