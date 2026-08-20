"""Benchmark for single-agent vs multi-agent runs."""

import re
from collections.abc import Callable
from time import perf_counter

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState

Runner = Callable[[str], ResearchState]

_CITATION_RE = re.compile(r"\[\d+\]")
_SENTENCE_RE = re.compile(r"[.!?]+(?:\s|$)")


def _estimated_cost_usd(state: ResearchState) -> float | None:
    costs: list[float] = [
        cost
        for result in state.agent_results
        if (cost := result.metadata.get("cost_usd")) is not None
    ]
    return round(sum(costs), 6) if costs else None


def _quality_score(state: ResearchState) -> float:
    """Cheap heuristic: rewards a substantive, cited, error-free answer (0-10)."""

    answer = state.final_answer or ""
    if not answer.strip():
        return 0.0

    word_count = len(answer.split())
    length_score = min(5.0, word_count / 60)  # up to 5 pts for ~300+ words
    citation_score = 3.0 if _CITATION_RE.search(answer) else 0.0
    error_penalty = min(3.0, len(state.errors))
    score = length_score + citation_score + 2.0 - error_penalty
    return round(max(0.0, min(10.0, score)), 2)


def _citation_coverage(state: ResearchState) -> float | None:
    answer = state.final_answer or ""
    sentences = [s for s in _SENTENCE_RE.split(answer) if s.strip()]
    if not sentences:
        return None
    cited = sum(1 for s in sentences if _CITATION_RE.search(s))
    return round(cited / len(sentences), 4)


def _failure_rate(state: ResearchState) -> float:
    total_steps = max(1, len(state.agent_results))
    return round(min(1.0, len(state.errors) / total_steps), 4)


def run_benchmark(
    run_name: str, query: str, runner: Runner
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Run `runner(query)` once and score latency, cost, quality, citations, and failures."""

    started = perf_counter()
    state = runner(query)
    latency = perf_counter() - started

    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=round(latency, 4),
        estimated_cost_usd=_estimated_cost_usd(state),
        quality_score=_quality_score(state),
        citation_coverage=_citation_coverage(state),
        failure_rate=_failure_rate(state),
        notes=f"{len(state.sources)} sources, {len(state.errors)} errors",
    )
    return state, metrics
