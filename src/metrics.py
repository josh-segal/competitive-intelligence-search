from __future__ import annotations

from collections.abc import Iterable

from models import SearchResultItem


def first_relevant_rank_exact(
    results: Iterable[SearchResultItem],
    relevant_urls: set[str],
) -> int | None:
    for item in results:
        if item.url in relevant_urls:
            return item.rank
    return None


def reciprocal_rank(first_rank: int | None) -> float:
    if first_rank is None:
        return 0.0
    return 1.0 / float(first_rank)
