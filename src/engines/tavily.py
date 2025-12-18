from __future__ import annotations

from dataclasses import dataclass

from tavily import AsyncTavilyClient

from models import EngineRunResult, SearchResultItem

from .base import SearchEngine


@dataclass(frozen=True)
class TavilyConfig:
    api_key: str


class TavilySearchEngine(SearchEngine):
    def __init__(self, config: TavilyConfig) -> None:
        self._config = config
        self._client = AsyncTavilyClient(config.api_key)

    @property
    def name(self) -> str:
        return "tavily"

    async def search(self, query: str, *, k: int) -> EngineRunResult:
        k_eff = max(1, min(int(k), 20))  # per docs 20 cap

        try:
            resp = await self._client.search(
                query=query, max_results=k_eff, include_raw_content=True
            )

            items: list[SearchResultItem] = []
            raw_results = [r for r in resp.get("results", []) if (r.get("url") or "").strip()]
            for i, r in enumerate(raw_results, start=1):
                items.append(
                    SearchResultItem(
                        rank=i,
                        url=r.get("url"),
                        title=r.get("title"),
                        content=r.get("raw_content"),
                    )
                )

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
