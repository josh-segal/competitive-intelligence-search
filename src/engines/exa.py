from __future__ import annotations

import asyncio
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
        return "exa-" + self._search_type

    async def search(self, query: str, *, k: int) -> EngineRunResult:
        try:
            response = await asyncio.to_thread(
                self._exa.search, query, num_results=k, type=self._search_type
            )

            items: list[SearchResultItem] = []
            for i, r in enumerate(response.results, start=1):
                items.append(SearchResultItem(rank=i, url=r.url))

            return EngineRunResult(
                engine_name=self.name,
                query_text=query,
                results=items,
                error=None,
            )
        except Exception as e:
            msg = str(e)
            return EngineRunResult(
                engine_name=self.name,
                query_text=query,
                results=[],
                error=msg,
            )
