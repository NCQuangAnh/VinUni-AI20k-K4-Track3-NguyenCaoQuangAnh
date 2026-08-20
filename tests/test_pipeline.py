"""End-to-end tests for the offline pipeline (search -> agents -> workflow -> benchmark)."""

from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.services.search_client import SearchClient


def test_search_client_finds_corpus_matches() -> None:
    results = SearchClient().search("multi-agent architecture research", max_results=3)
    assert results
    assert all(r.snippet for r in results)


def test_workflow_runs_end_to_end_offline() -> None:
    state = ResearchState(request=ResearchQuery(query="multi-agent research architectures"))
    result = MultiAgentWorkflow(max_iterations=6).run(state)

    assert result.final_answer
    assert result.route_history[-1] == "done"
    assert result.iteration <= 6
    assert any(r.agent == "writer" for r in result.agent_results)


def test_workflow_respects_iteration_budget() -> None:
    state = ResearchState(request=ResearchQuery(query="multi-agent research architectures"))
    result = MultiAgentWorkflow(max_iterations=1).run(state)
    assert result.final_answer is not None


def test_benchmark_and_report_offline() -> None:
    def runner(query: str) -> ResearchState:
        state = ResearchState(request=ResearchQuery(query=query))
        return MultiAgentWorkflow(max_iterations=6).run(state)

    _, metrics = run_benchmark("multi-agent", "multi-agent research architectures", runner)
    assert metrics.latency_seconds >= 0
    assert metrics.quality_score is not None

    report = render_markdown_report([metrics])
    assert "Benchmark Report" in report
    assert "multi-agent" in report
