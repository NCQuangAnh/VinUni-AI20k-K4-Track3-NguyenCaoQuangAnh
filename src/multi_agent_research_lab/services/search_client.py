"""Search client abstraction for ResearcherAgent.

Backed by the offline research corpus (`ai_agent_offline_research_corpus_v2/`) so the
whole pipeline runs without network access or API keys. Each corpus topic file supplies
knowledge articles and source documents; we rank them against the query with a simple
term-overlap score and return the top matches as `SourceDocument`.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from multi_agent_research_lab.core.schemas import SourceDocument

CORPUS_DIR = Path(__file__).resolve().parents[3] / "ai_agent_offline_research_corpus_v2" / "topics"

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


@lru_cache(maxsize=1)
def _load_corpus(corpus_dir: Path = CORPUS_DIR) -> list[dict[str, Any]]:
    """Load every topic JSON file once and cache the result."""

    documents: list[dict[str, Any]] = []
    if not corpus_dir.exists():
        return documents

    for path in sorted(corpus_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        topic_name = data.get("topic", {}).get("name", path.stem)
        knowledge_base = data.get("knowledge_base", {})

        for article in knowledge_base.get("knowledge_articles", []):
            documents.append(
                {
                    "kind": "article",
                    "id": article.get("article_id", path.stem),
                    "title": f"{topic_name}: {article.get('title', '')}".strip(": "),
                    "text": article.get("content", ""),
                    "url": None,
                    "is_synthetic": False,
                    "topic_file": path.name,
                }
            )

        for source in knowledge_base.get("source_documents", []):
            documents.append(
                {
                    "kind": "source",
                    "id": source.get("document_id", path.stem),
                    "title": source.get("title", topic_name),
                    "text": source.get("full_text", "") or "",
                    "url": source.get("provenance_url"),
                    "is_synthetic": bool(source.get("is_synthetic", False)),
                    "topic_file": path.name,
                }
            )

    return documents


class SearchClient:
    """Offline, corpus-backed search client."""

    def __init__(self, corpus_dir: Path = CORPUS_DIR) -> None:
        self._corpus_dir = corpus_dir

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Rank corpus documents against `query` and return the top matches."""

        query_terms = _tokenize(query)
        if not query_terms:
            return []

        scored: list[tuple[float, dict[str, Any]]] = []
        for doc in _load_corpus(self._corpus_dir):
            doc_terms = _tokenize(f"{doc['title']} {doc['text']}")
            overlap = len(query_terms & doc_terms)
            if overlap == 0:
                continue
            score = overlap / len(query_terms)
            scored.append((score, doc))

        scored.sort(key=lambda item: item[0], reverse=True)

        results: list[SourceDocument] = []
        for score, doc in scored[:max_results]:
            snippet = doc["text"].strip().replace("\n", " ")
            if len(snippet) > 400:
                snippet = snippet[:400].rsplit(" ", 1)[0] + "..."
            results.append(
                SourceDocument(
                    title=doc["title"],
                    url=doc["url"],
                    snippet=snippet or doc["title"],
                    metadata={
                        "document_id": doc["id"],
                        "kind": doc["kind"],
                        "is_synthetic": doc["is_synthetic"],
                        "topic_file": doc["topic_file"],
                        "relevance_score": round(score, 4),
                    },
                )
            )
        return results
