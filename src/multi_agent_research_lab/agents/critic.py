"""Optional critic agent for bonus work."""

import re

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState

_CITATION_RE = re.compile(r"\[(\d+)\]")


class CriticAgent(BaseAgent):
    """Fact-checking and citation-coverage review of the final answer."""

    name = "critic"

    def run(self, state: ResearchState) -> ResearchState:
        """Validate `state.final_answer` and append findings to `state.trace`/`state.errors`."""

        findings: list[str] = []
        answer = state.final_answer or ""

        if not answer.strip():
            findings.append("final_answer is empty")

        cited_indices = {int(m) for m in _CITATION_RE.findall(answer)}
        max_index = len(state.sources)
        out_of_range = {i for i in cited_indices if i < 1 or i > max_index}
        if out_of_range:
            findings.append(f"citations reference unknown sources: {sorted(out_of_range)}")

        if state.sources and not cited_indices:
            findings.append("final_answer cites no sources despite sources being available")

        for finding in findings:
            state.errors.append(f"critic: {finding}")

        state.add_trace_event(
            "critic.review",
            {
                "findings": findings,
                "citation_count": len(cited_indices),
                "source_count": max_index,
            },
        )
        state.agent_results.append(
            AgentResult(
                agent=AgentName.CRITIC,
                content="; ".join(findings) if findings else "no issues found",
                metadata={"citation_count": len(cited_indices), "source_count": max_index},
            )
        )
        return state
