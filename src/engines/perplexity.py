from __future__ import annotations

import os
from dataclasses import dataclass

from perplexity import AsyncPerplexity

from models import EngineRunResult, SearchResultItem

from .base import SearchEngine


@dataclass(frozen=True)
class PerplexityConfig:
    api_key: str


class PerplexitySearchEngine(SearchEngine):
    def __init__(self, config: PerplexityConfig) -> None:
        if config.api_key:
            os.environ["PERPLEXITY_API_KEY"] = config.api_key
        self._client = AsyncPerplexity()

    @property
    def name(self) -> str:
        return "perplexity"

    async def search(self, query: str, *, k: int) -> EngineRunResult:
        k_eff = max(1, min(int(k), 20))  # per docs 20 cap

        try:
            search_resp = await self._client.search.create(query=query, max_results=k_eff)

            items: list[SearchResultItem] = []
            for i, r in enumerate(search_resp.results, start=1):
                items.append(SearchResultItem(rank=i, url=r.url))

            return EngineRunResult(
                engine_name=self.name,
                query_text=query,
                results=items,
                error=None,
            )
        except Exception as e:
            return EngineRunResult(
                engine_name=self.name,
                query_text=query,
                results=[],
                error=str(e),
            )
