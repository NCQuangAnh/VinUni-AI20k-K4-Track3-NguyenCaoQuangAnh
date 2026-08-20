# Design Template

## Problem

Given a natural-language research query, produce a well-organized, citation-backed answer
grounded only in the offline research corpus (`ai_agent_offline_research_corpus_v2/`) — no
network access or API key required — while measuring latency, quality, citation coverage,
and failure rate against a single-agent baseline.

## Why multi-agent?

A single LLM call has no separation between "find evidence" and "write the answer": it tends
to either skip citations or hallucinate them. Splitting the work lets each stage be checked
independently — Researcher only cites what `SearchClient` actually returned, Analyst flags
weak/uncited claims before they reach the final answer, and Critic verifies citation coverage
after the fact. The cost is coordination overhead (more calls, more latency), which is exactly
what the benchmark (`malab benchmark`) is built to quantify against the baseline.

## Agent roles

| Agent | Responsibility | Input | Output | Failure mode |
|---|---|---|---|---|
| Supervisor | Route to the next worker; enforce the iteration budget | `ResearchState` | appended `route_history` entry | infinite loop if routing logic is wrong — bounded by `max_iterations` |
| Researcher | Retrieve sources, write cited research notes | `request.query`, `SearchClient` | `sources`, `research_notes` | search returns nothing → notes explicitly say so, loop still bounded |
| Analyst | Extract key claims, flag weak/uncited evidence | `research_notes` | `analysis_notes` | LLM call fails → falls back to raw `research_notes`, error recorded |
| Writer | Synthesize final answer with a Sources list | `research_notes`, `analysis_notes`, `sources` | `final_answer` | LLM call fails → falls back to `analysis_notes`/`research_notes` so the run never ends emptyhanded |
| Critic (bonus) | Post-hoc check: citation coverage, out-of-range citations, empty answer | `final_answer`, `sources` | findings appended to `trace`/`errors` | never blocks the pipeline; only records findings |

## Shared state

`ResearchState` (`core/state.py`) is the single object threaded through every node:
- `request` — the validated `ResearchQuery` (query, max_sources, audience)
- `iteration` / `route_history` — drives the stop condition and is inspectable for debugging
- `sources` / `research_notes` / `analysis_notes` / `final_answer` — the pipeline's working
  memory; each stage only writes its own field so responsibilities stay separated
- `agent_results` — per-agent output + token/cost metadata, feeds the benchmark
- `trace` — span-level timing events (`observability/tracing.py`) for post-run debugging
- `errors` — non-fatal failures (search miss, LLM fallback, citation problems); feeds
  `failure_rate` in the benchmark instead of crashing the run

## Routing policy

```
supervisor:
  if iteration >= max_iterations: writer (if no final_answer) else done
  elif no sources or no research_notes: researcher
  elif no analysis_notes: analyst
  elif no final_answer: writer
  else: done
```

This is a straight-line pipeline (researcher → analyst → writer) with the iteration budget as
the only loop-breaking guardrail; a step that fails to produce its output re-routes to itself
next iteration until the budget forces a writer pass. Critic always runs once after `done`, if
there is a `final_answer`.

## Guardrails

- Max iterations: `configs/lab_default.yaml: lab.max_iterations` (default 6), enforced in
  `SupervisorAgent`.
- Timeout: `Settings.timeout_seconds` (default 60s) — read by callers; not yet enforced inside
  `LLMClient`/`SearchClient` (both are synchronous and local/offline, so latency is bounded by
  corpus size, not network).
- Retry: `LLMClient._complete_openai` retries 3x with exponential backoff (`tenacity`) when a
  real provider is configured.
- Fallback: every LLM-calling agent catches exceptions and degrades to the best available
  notes instead of raising; `LLMClient` itself falls back to a deterministic offline synthesis
  when no `OPENAI_API_KEY` is set, so the whole pipeline is runnable without secrets.
- Validation: `ResearchQuery` (pydantic) rejects empty/short queries before any agent runs;
  `CriticAgent` validates citation ranges after the writer runs.

## Benchmark plan

Run `malab benchmark --query "<one of configs/lab_default.yaml: benchmark.queries>"` to compare
`baseline` (one search + one write call) against `multi-agent` (full supervisor loop + critic).

- **Queries**: the 3 sample queries in `configs/lab_default.yaml` — architecture comparison,
  workflow trade-offs, production guardrails — chosen because the offline corpus has dedicated
  topics for each.
- **Metrics** (`evaluation/benchmark.py`): `latency_seconds`, `estimated_cost_usd` (0 offline),
  `quality_score` (0-10 heuristic: length + citation presence - error penalty),
  `citation_coverage` (fraction of sentences with a `[n]` marker), `failure_rate` (errors /
  agent steps).
- **Expected outcome**: multi-agent should show higher `citation_coverage` and `quality_score`
  than baseline at the cost of higher `latency_seconds`, since baseline skips analysis and
  critique.
