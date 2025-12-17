from __future__ import annotations

from abc import ABC, abstractmethod

from models import EngineRunResult


class SearchAdapter(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    async def __aenter__(self) -> SearchAdapter:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        return None

    @abstractmethod
    async def search(self, query: str, *, k: int) -> EngineRunResult:
        raise NotImplementedError
