from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

from exa_py import Exa

from models import EngineRunResult, SearchResultItem

from .base import SearchEngine


@dataclass(frozen=True)
class ExaConfig:
    api_key: str
    search_type: Literal["auto", "neural", "fast", "deep"] = "auto"


class ExaSearchEngine(SearchEngine):
    def __init__(self, config: ExaConfig) -> None:
        self._exa = Exa(api_key=config.api_key)
        self._search_type = config.search_type

    @property
    def name(self) -> str:
        return "exa"

    async def search(self, query: str, *, k: int) -> EngineRunResult:
        started = time.monotonic()

        try:
            response = self._exa.search(query, num_results=k, type=self._search_type)

            items: list[SearchResultItem] = []
            for i, r in enumerate(response.results, start=1):
                items.append(SearchResultItem(rank=i, url=r.url))

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
            msg = str(e)
            return EngineRunResult(
                engine_name=self.name,
                query_text=query,
                results=[],
                error=msg,
                latency_ms=latency_ms,
            )
