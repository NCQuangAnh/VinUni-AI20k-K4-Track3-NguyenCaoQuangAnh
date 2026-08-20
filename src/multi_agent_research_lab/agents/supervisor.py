"""Supervisor / router."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.state import ResearchState

DONE = "done"


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    def __init__(self, max_iterations: int = 6) -> None:
        self.max_iterations = max_iterations

    def run(self, state: ResearchState) -> ResearchState:
        """Append the next route to `state.route_history` and bump `state.iteration`."""

        route = self._decide(state)
        state.record_route(route)
        state.add_trace_event("supervisor.route", {"route": route, "iteration": state.iteration})
        return state

    def _decide(self, state: ResearchState) -> str:
        # Hard stop once the iteration budget is exhausted, unless a final answer is still
        # missing - then force one last writer pass so we always return something usable.
        if state.iteration >= self.max_iterations:
            return "writer" if state.final_answer is None else DONE

        if not state.sources or not state.research_notes:
            return "researcher"
        if not state.analysis_notes:
            return "analyst"
        if not state.final_answer:
            return "writer"
        return DONE
