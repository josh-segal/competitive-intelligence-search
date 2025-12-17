from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Literal

import httpx

from models import EngineRunResult, SearchResultItem

from .base import SearchAdapter


@dataclass(frozen=True)
class ExaConfig:
    api_key: str
    base_url: str = "https://api.exa.ai"
    search_type: Literal["auto", "neural", "keyword"] = "auto"
    use_autoprompt: bool = True
    timeout_s: float = 30.0


class ExaSearchAdapter(SearchAdapter):
    def __init__(self, config: ExaConfig) -> None:
        self._config = config
        self._client: httpx.AsyncClient | None = None

    @property
    def name(self) -> str:
        return "exa"

    async def __aenter__(self) -> ExaSearchAdapter:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._config.base_url,
                headers={
                    "Authorization": f"Bearer {self._config.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=httpx.Timeout(self._config.timeout_s),
            )
        return self

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def search(self, query: str, *, k: int) -> EngineRunResult:
        started = time.monotonic()

        if self._client is None:
            # Allow use without `async with ...` for convenience.
            await self.__aenter__()
        assert self._client is not None

        try:
            resp = await self._client.post(
                "/search",
                json={
                    "query": query,
                    "numResults": k,
                    "type": self._config.search_type,
                    "useAutoprompt": self._config.use_autoprompt,
                },
            )
            resp.raise_for_status()
            payload: dict[str, Any] = resp.json()

            items: list[SearchResultItem] = []
            for i, r in enumerate(payload.get("results", []) or [], start=1):
                url = r.get("url")
                if not isinstance(url, str) or not url:
                    continue
                items.append(
                    SearchResultItem(
                        rank=i,
                        url=url,
                        title=r.get("title") if isinstance(r.get("title"), str) else None,
                        snippet=r.get("text") if isinstance(r.get("text"), str) else None,
                        raw=r if isinstance(r, dict) else None,
                    )
                )

            latency_ms = (time.monotonic() - started) * 1000
            return EngineRunResult(
                engine_name=self.name,
                query_text=query,
                results=items,
                error=None,
                latency_ms=latency_ms,
            )
        except httpx.HTTPStatusError as e:
            latency_ms = (time.monotonic() - started) * 1000
            msg = f"HTTP {e.response.status_code}: {e.response.text}"
            return EngineRunResult(
                engine_name=self.name,
                query_text=query,
                results=[],
                error=msg,
                latency_ms=latency_ms,
            )
        except httpx.HTTPError as e:
            latency_ms = (time.monotonic() - started) * 1000
            return EngineRunResult(
                engine_name=self.name,
                query_text=query,
                results=[],
                error=str(e),
                latency_ms=latency_ms,
            )
