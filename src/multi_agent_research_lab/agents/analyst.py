"""Analyst agent."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a research analyst. Turn the research notes into structured insights: key "
    "claims, points of agreement/disagreement between sources, and any weak or unsupported "
    "evidence. Keep citation markers ([n]) intact."
)


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.analysis_notes`."""

        if not state.research_notes:
            state.analysis_notes = "No research notes available to analyze."
            state.errors.append("analyst: skipped, research_notes missing")
            state.agent_results.append(
                AgentResult(agent=AgentName.ANALYST, content=state.analysis_notes)
            )
            return state

        user_prompt = (
            f"Query: {state.request.query}\n\nResearch notes:\n{state.research_notes}\n\n"
            "Extract 3-6 key claims, note any disagreement between sources, and flag weak "
            "evidence (fewer than 1 supporting citation)."
        )
        try:
            response = self.llm_client.complete(SYSTEM_PROMPT, user_prompt)
        except Exception:  # noqa: BLE001 - degrade gracefully, do not crash the pipeline
            logger.exception("analyst LLM call failed")
            state.errors.append("analyst: llm_client.complete failed")
            state.analysis_notes = state.research_notes
            state.agent_results.append(
                AgentResult(agent=AgentName.ANALYST, content=state.analysis_notes)
            )
            return state

        state.analysis_notes = response.content
        state.agent_results.append(
            AgentResult(
                agent=AgentName.ANALYST,
                content=response.content,
                metadata={
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                },
            )
        )
        return state
