"""Writer agent."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a technical writer. Synthesize the research and analysis notes into a clear, "
    "well-organized answer for the target audience. Preserve [n] citation markers and end "
    "with a 'Sources' list mapping each [n] to its title."
)


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.final_answer`."""

        sources_list = "\n".join(
            f"[{i}] {s.title}" + (f" ({s.url})" if s.url else "")
            for i, s in enumerate(state.sources, start=1)
        )
        user_prompt = (
            f"Query: {state.request.query}\nAudience: {state.request.audience}\n\n"
            f"Research notes:\n{state.research_notes or '(none)'}\n\n"
            f"Analysis notes:\n{state.analysis_notes or '(none)'}\n\n"
            f"Available sources:\n{sources_list or '(none)'}\n\n"
            "Write the final answer now."
        )

        try:
            response = self.llm_client.complete(SYSTEM_PROMPT, user_prompt)
            content = response.content
            metadata = {
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "cost_usd": response.cost_usd,
            }
        except Exception:  # noqa: BLE001 - always leave the pipeline with *some* answer
            logger.exception("writer LLM call failed")
            state.errors.append("writer: llm_client.complete failed")
            content = (
                state.analysis_notes
                or state.research_notes
                or "Unable to produce an answer: no research or analysis notes were available."
            )
            metadata = {}

        state.final_answer = content
        state.agent_results.append(
            AgentResult(agent=AgentName.WRITER, content=content, metadata=metadata)
        )
        return state
