from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

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
        started = time.monotonic()

        # Tavily caps max_results at 20 per docs.
        k_eff = max(1, min(int(k), 20))

        try:
            resp = await self._client.search(
                query=query,
                max_results=k_eff,
            )

            raw_results = None
            if isinstance(resp, dict):
                raw_results = resp.get("results")

            items: list[SearchResultItem] = []
            for i, r in enumerate(raw_results or [], start=1):
                url = r.get("url") if isinstance(r, dict) else None
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
