"""Multi-agent workflow orchestration.

Implemented as a plain Python loop (no hard dependency on the optional `langgraph` extra):
`supervisor` decides the next route by inspecting `ResearchState`, and the loop dispatches to
the matching worker until the supervisor returns "done" or `max_iterations` is hit. Nodes,
edges, and the stop condition mirror what a LangGraph `StateGraph` would express - swap this
loop for `langgraph.graph.StateGraph` if the `[llm]` extra is installed and you want graph
visualization/checkpointing.
"""

import logging

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.agents.critic import CriticAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.supervisor import DONE, SupervisorAgent
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.config import get_settings, load_lab_config
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span

logger = logging.getLogger(__name__)


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph.

    Keep orchestration here; keep agent internals in `agents/`.
    """

    def __init__(self, max_iterations: int | None = None, run_critic: bool = True) -> None:
        lab_config = load_lab_config()
        self.max_iterations = (
            max_iterations
            or lab_config.get("lab", {}).get("max_iterations")
            or get_settings().max_iterations
        )
        self.run_critic = run_critic
        self._nodes: dict[str, BaseAgent] | None = None

    def build(self) -> dict[str, BaseAgent]:
        """Create the node map: route name -> agent instance."""

        nodes: dict[str, BaseAgent] = {
            "supervisor": SupervisorAgent(max_iterations=self.max_iterations),
            "researcher": ResearcherAgent(),
            "analyst": AnalystAgent(),
            "writer": WriterAgent(),
        }
        if self.run_critic:
            nodes["critic"] = CriticAgent()
        self._nodes = nodes
        return nodes

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the graph and return the final state."""

        nodes = self._nodes or self.build()
        supervisor = nodes["supervisor"]

        while True:
            with trace_span("supervisor", {"iteration": state.iteration}):
                supervisor.run(state)
            route = state.route_history[-1]

            if route == DONE:
                break

            agent = nodes.get(route)
            if agent is None:
                state.errors.append(f"workflow: unknown route '{route}'")
                break

            try:
                with trace_span(f"agent.{route}", {"iteration": state.iteration}):
                    agent.run(state)
            except Exception as exc:  # noqa: BLE001 - keep the loop alive, record the failure
                logger.exception("agent '%s' raised", route)
                state.errors.append(f"{route}: {exc}")
                raise AgentExecutionError(f"agent '{route}' failed: {exc}") from exc

        if self.run_critic and state.final_answer:
            with trace_span("agent.critic", {"iteration": state.iteration}):
                nodes["critic"].run(state)

        return state
