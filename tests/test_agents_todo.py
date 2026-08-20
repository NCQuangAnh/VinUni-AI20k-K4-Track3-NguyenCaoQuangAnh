"""Unit tests for the routing policy implemented in SupervisorAgent."""

from multi_agent_research_lab.agents import SupervisorAgent
from multi_agent_research_lab.agents.supervisor import DONE
from multi_agent_research_lab.core.schemas import ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState


def _state() -> ResearchState:
    return ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))


def test_supervisor_routes_to_researcher_first() -> None:
    state = _state()
    SupervisorAgent().run(state)
    assert state.route_history == ["researcher"]
    assert state.iteration == 1


def test_supervisor_routes_to_analyst_after_research() -> None:
    state = _state()
    state.sources = [SourceDocument(title="t", snippet="s")]
    state.research_notes = "notes"
    SupervisorAgent().run(state)
    assert state.route_history == ["analyst"]


def test_supervisor_routes_to_writer_after_analysis() -> None:
    state = _state()
    state.sources = [SourceDocument(title="t", snippet="s")]
    state.research_notes = "notes"
    state.analysis_notes = "analysis"
    SupervisorAgent().run(state)
    assert state.route_history == ["writer"]


def test_supervisor_done_when_final_answer_present() -> None:
    state = _state()
    state.sources = [SourceDocument(title="t", snippet="s")]
    state.research_notes = "notes"
    state.analysis_notes = "analysis"
    state.final_answer = "answer"
    SupervisorAgent().run(state)
    assert state.route_history == [DONE]


def test_supervisor_forces_writer_at_iteration_budget() -> None:
    state = _state()
    supervisor = SupervisorAgent(max_iterations=2)
    state.iteration = 2
    supervisor.run(state)
    assert state.route_history == ["writer"]


def test_supervisor_stops_at_budget_once_answer_exists() -> None:
    state = _state()
    state.final_answer = "answer"
    supervisor = SupervisorAgent(max_iterations=2)
    state.iteration = 2
    supervisor.run(state)
    assert state.route_history == [DONE]
