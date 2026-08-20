"""Benchmark report rendering."""

from multi_agent_research_lab.core.schemas import BenchmarkMetrics


def render_markdown_report(metrics: list[BenchmarkMetrics]) -> str:
    """Render benchmark metrics to markdown, with a short comparison when 2+ runs are given."""

    lines = [
        "# Benchmark Report",
        "",
        "| Run | Latency (s) | Cost (USD) | Quality | Citation cov. | Failure rate | Notes |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for item in metrics:
        cost = "" if item.estimated_cost_usd is None else f"{item.estimated_cost_usd:.4f}"
        quality = "" if item.quality_score is None else f"{item.quality_score:.1f}"
        citation = "" if item.citation_coverage is None else f"{item.citation_coverage:.0%}"
        failure = "" if item.failure_rate is None else f"{item.failure_rate:.0%}"
        lines.append(
            f"| {item.run_name} | {item.latency_seconds:.2f} | {cost} | {quality} "
            f"| {citation} | {failure} | {item.notes} |"
        )

    if len(metrics) >= 2:
        lines += ["", "## Comparison", ""]
        baseline, *rest = metrics
        for item in rest:
            latency_delta = item.latency_seconds - baseline.latency_seconds
            quality_delta = (
                (item.quality_score or 0) - (baseline.quality_score or 0)
                if item.quality_score is not None and baseline.quality_score is not None
                else None
            )
            latency_note = f"{latency_delta:+.2f}s latency vs {baseline.run_name}"
            quality_note = (
                f"{quality_delta:+.1f} quality vs {baseline.run_name}"
                if quality_delta is not None
                else "quality not comparable"
            )
            lines.append(f"- **{item.run_name}**: {latency_note}, {quality_note}.")

    return "\n".join(lines) + "\n"
