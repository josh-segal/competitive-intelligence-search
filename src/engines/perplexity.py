from __future__ import annotations

import os
import time
from dataclasses import dataclass

from perplexity import AsyncPerplexity

from models import EngineRunResult, SearchResultItem

from .base import SearchEngine


@dataclass(frozen=True)
class PerplexityConfig:
    api_key: str


class PerplexitySearchEngine(SearchEngine):
    def __init__(self, config: PerplexityConfig) -> None:
        # The Perplexity SDK supports reading from PERPLEXITY_API_KEY; set it explicitly
        # to avoid relying on the caller's environment.
        if config.api_key:
            os.environ["PERPLEXITY_API_KEY"] = config.api_key
        self._client = AsyncPerplexity()

    @property
    def name(self) -> str:
        return "perplexity"

    async def search(self, query: str, *, k: int) -> EngineRunResult:
        started = time.monotonic()

        # Per Perplexity Search API docs, max_results is capped at 20.
        k_eff = max(1, min(int(k), 20))

        try:
            search_resp = await self._client.search.create(query=query, max_results=k_eff)

            raw_results = None
            if hasattr(search_resp, "results"):
                raw_results = getattr(search_resp, "results")
            elif isinstance(search_resp, dict):
                raw_results = search_resp.get("results")

            items: list[SearchResultItem] = []
            for i, r in enumerate(raw_results or [], start=1):
                url = None
                if hasattr(r, "url"):
                    url = getattr(r, "url")
                elif isinstance(r, dict):
                    url = r.get("url")

                if not isinstance(url, str) or not url:
                    continue

                items.append(SearchResultItem(rank=i, url=url))

            latency_ms = (time.monotonic() - started) * 1000
            return EngineRunResult(
                engine_name=self.name,
                query_text=query,
                results=items,
                error=None,
                latency_ms=latency_ms,
            )
        except Exception as e:
            latency_ms = (time.monotonic() - started) * 1000
            return EngineRunResult(
                engine_name=self.name,
                query_text=query,
                results=[],
                error=str(e),
                latency_ms=latency_ms,
            )
