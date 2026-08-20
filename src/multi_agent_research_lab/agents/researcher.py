"""Researcher agent."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a research assistant. Read the numbered sources and write concise, "
    "citation-backed notes (use [n] markers referring to source order). Do not invent facts "
    "that are not supported by a source."
)


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def __init__(
        self, search_client: SearchClient | None = None, llm_client: LLMClient | None = None
    ) -> None:
        self.search_client = search_client or SearchClient()
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.sources` and `state.research_notes`."""

        try:
            sources = self.search_client.search(
                state.request.query, max_results=state.request.max_sources
            )
        except Exception:  # noqa: BLE001 - degrade gracefully, do not crash the pipeline
            logger.exception("search failed")
            state.errors.append("researcher: search_client.search failed")
            sources = []

        state.sources = sources or state.sources

        if not state.sources:
            state.research_notes = "No sources were found for this query."
            state.agent_results.append(
                AgentResult(agent=AgentName.RESEARCHER, content=state.research_notes)
            )
            return state

        catalog = "\n".join(
            f"[{i}] {s.title} - {s.snippet}" for i, s in enumerate(state.sources, start=1)
        )
        user_prompt = (
            f"Query: {state.request.query}\nAudience: {state.request.audience}\n\n"
            f"Sources:\n{catalog}\n\n"
            "Write research notes (bullet points) summarizing what each relevant source says, "
            "with a [n] citation per bullet."
        )
        response = self.llm_client.complete(SYSTEM_PROMPT, user_prompt)
        state.research_notes = response.content
        state.agent_results.append(
            AgentResult(
                agent=AgentName.RESEARCHER,
                content=response.content,
                metadata={
                    "source_count": len(state.sources),
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                },
            )
        )
        return state
