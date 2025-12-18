from __future__ import annotations

from abc import ABC, abstractmethod

from models import EngineRunResult


class SearchEngine(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    async def search(self, query: str, *, k: int) -> EngineRunResult:
        raise NotImplementedError
