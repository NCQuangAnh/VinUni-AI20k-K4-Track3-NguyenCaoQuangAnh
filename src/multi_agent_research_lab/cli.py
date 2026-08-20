"""Command-line entrypoint for the lab starter."""

from typing import Annotated

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel

from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError, StudentTodoError
from multi_agent_research_lab.core.schemas import BenchmarkMetrics, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient
from multi_agent_research_lab.services.storage import LocalArtifactStore

app = typer.Typer(help="Multi-Agent Research Lab starter CLI")
console = Console()


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)


def _parse_query(query: str) -> ResearchQuery:
    try:
        return ResearchQuery(query=query)
    except ValidationError as exc:
        console.print(
            Panel.fit(
                f"Invalid query: {exc.errors()[0]['msg']}",
                title="Input Error",
                style="red",
            )
        )
        raise typer.Exit(code=1) from exc


def _run_baseline(query: str) -> ResearchState:
    """Single-agent baseline: one search call + one LLM call, no routing/critique."""

    request = _parse_query(query)
    state = ResearchState(request=request)
    state.sources = SearchClient().search(request.query, max_results=request.max_sources)
    state = WriterAgent(llm_client=LLMClient()).run(state)
    return state


def _run_multi_agent(query: str) -> ResearchState:
    state = ResearchState(request=_parse_query(query))
    return MultiAgentWorkflow().run(state)


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run a minimal single-agent baseline: search once, write once, no supervisor loop."""

    _init()
    state = _run_baseline(query)
    console.print(Panel.fit(state.final_answer or "(no answer)", title="Single-Agent Baseline"))


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the full supervisor-routed multi-agent workflow."""

    _init()
    request = _parse_query(query)
    state = ResearchState(request=request)
    workflow = MultiAgentWorkflow()
    try:
        result = workflow.run(state)
    except StudentTodoError as exc:
        console.print(Panel.fit(str(exc), title="Expected TODO", style="yellow"))
        raise typer.Exit(code=2) from exc
    except AgentExecutionError as exc:
        console.print(Panel.fit(str(exc), title="Agent Failure", style="red"))
        raise typer.Exit(code=1) from exc
    console.print(result.model_dump_json(indent=2))


@app.command()
def benchmark(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
    output: Annotated[
        str, typer.Option("--output", "-o", help="Report path under reports/")
    ] = "benchmark_report.md",
) -> None:
    """Run baseline and multi-agent on the same query and render a comparison report."""

    _init()
    _, baseline_metrics = run_benchmark("baseline", query, _run_baseline)
    _, multi_metrics = run_benchmark("multi-agent", query, _run_multi_agent)

    metrics: list[BenchmarkMetrics] = [baseline_metrics, multi_metrics]
    report = render_markdown_report(metrics)
    path = LocalArtifactStore().write_text(output, report)

    console.print(report)
    console.print(Panel.fit(str(path), title="Report written"))


if __name__ == "__main__":
    app()
