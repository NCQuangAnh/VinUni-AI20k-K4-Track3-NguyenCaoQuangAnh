"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.

If `OPENAI_API_KEY` is configured and the `openai` package is installed (the `[llm]` extra),
`complete()` calls the real Chat Completions API with retry/backoff. Otherwise it falls back
to a deterministic offline synthesis so the whole lab (agents, workflow, benchmark) is runnable
without network access or secrets - important for grading and CI.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from tenacity import retry, stop_after_attempt, wait_exponential

from multi_agent_research_lab.core.config import get_settings

logger = logging.getLogger(__name__)

# Rough per-1K-token price for the default model family; good enough for benchmark estimates.
_USD_PER_1K_INPUT_TOKENS = 0.00015
_USD_PER_1K_OUTPUT_TOKENS = 0.0006


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


def _estimate_tokens(text: str) -> int:
    # ~4 chars per token is a standard rough heuristic when no tokenizer is available.
    return max(1, len(text) // 4)


class LLMClient:
    """Provider-agnostic LLM client with an offline fallback."""

    def __init__(self, model: str | None = None) -> None:
        settings = get_settings()
        self._api_key = settings.openai_api_key
        self._model = model or settings.openai_model

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a model completion, or a deterministic offline synthesis."""

        if self._api_key:
            try:
                return self._complete_openai(system_prompt, user_prompt)
            except ImportError:
                logger.warning("openai package not installed; falling back to offline mode")
            except Exception:  # noqa: BLE001 - any provider failure should degrade gracefully
                logger.exception("LLM call failed; falling back to offline mode")

        return self._complete_offline(system_prompt, user_prompt)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    def _complete_openai(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        from openai import OpenAI  # imported lazily so the base install has no hard dependency

        client = OpenAI(api_key=self._api_key)
        response = client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        choice = response.choices[0].message.content or ""
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else None
        output_tokens = usage.completion_tokens if usage else None
        cost = None
        if input_tokens is not None and output_tokens is not None:
            cost = (
                input_tokens / 1000 * _USD_PER_1K_INPUT_TOKENS
                + output_tokens / 1000 * _USD_PER_1K_OUTPUT_TOKENS
            )
        return LLMResponse(
            content=choice, input_tokens=input_tokens, output_tokens=output_tokens, cost_usd=cost
        )

    def _complete_offline(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Deterministic extractive synthesis used when no LLM provider is configured.

        Agents are expected to hand this a well-structured `user_prompt` (query, sources,
        prior notes); we simply trim/label it instead of generating new text, so the
        pipeline result is reproducible without an API key.
        """

        content = (
            f"[offline synthesis - no LLM provider configured]\n"
            f"Task: {system_prompt.strip()}\n\n{user_prompt.strip()}"
        )
        input_tokens = _estimate_tokens(system_prompt + user_prompt)
        output_tokens = _estimate_tokens(content)
        return LLMResponse(
            content=content, input_tokens=input_tokens, output_tokens=output_tokens, cost_usd=0.0
        )
