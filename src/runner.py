from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from adapters.base import SearchAdapter
from metrics import first_relevant_rank_exact, reciprocal_rank
from models import (
    EvaluationAggregates,
    EvaluationReport,
    QueryCase,
    QueryEvaluation,
    RunMetadata,
)


@dataclass(frozen=True)
class RunnerConfig:
    k: int = 10
    concurrency: int = 10


class EvaluationRunner:
    def __init__(
        self,
        *,
        adapter: SearchAdapter,
        config: RunnerConfig,
        dataset_path: str,
        dataset_sha256: str,
        engine_config: dict[str, Any] | None = None,
    ) -> None:
        self._adapter = adapter
        self._config = config
        self._dataset_path = dataset_path
        self._dataset_sha256 = dataset_sha256
        self._engine_config = engine_config or {}

    async def run(
        self,
        cases: list[QueryCase],
        *,
        on_evaluation_complete: (
            Callable[[QueryEvaluation, int, int], None] | None  # (evaluation, done, total)
        ) = None,
    ) -> EvaluationReport:
        sem = asyncio.Semaphore(self._config.concurrency)
        total = len(cases)

        async def eval_one(case: QueryCase) -> QueryEvaluation:
            async with sem:
                engine_run = await self._adapter.search(case.query_text, k=self._config.k)

            relevant_set = set(case.relevant_urls)
            first_rank = (
                None
                if engine_run.error
                else first_relevant_rank_exact(engine_run.results, relevant_set)
            )
            rr = reciprocal_rank(first_rank)

            return QueryEvaluation(
                product=case.product,
                query_text=case.query_text,
                relevant_urls=list(case.relevant_urls),
                first_relevant_rank=first_rank,
                reciprocal_rank=rr,
                engine_run=engine_run,
            )

        tasks = [asyncio.create_task(eval_one(c)) for c in cases]
        evaluations: list[QueryEvaluation] = []
        done = 0
        for fut in asyncio.as_completed(tasks):
            e = await fut
            evaluations.append(e)
            done += 1
            if on_evaluation_complete is not None:
                on_evaluation_complete(e, done, total)

        evaluations.sort(key=lambda e: (e.product, e.query_text))

        overall_mrr = (
            sum(e.reciprocal_rank for e in evaluations) / float(len(evaluations))
            if evaluations
            else 0.0
        )

        by_product: dict[str, list[QueryEvaluation]] = defaultdict(list)
        for e in evaluations:
            by_product[e.product].append(e)

        per_product_mrr: dict[str, float] = {}
        for product, es in by_product.items():
            per_product_mrr[product] = (
                sum(x.reciprocal_rank for x in es) / float(len(es)) if es else 0.0
            )

        hit_count = sum(1 for e in evaluations if e.first_relevant_rank is not None)

        aggregates = EvaluationAggregates(
            query_count=len(evaluations),
            hit_count=hit_count,
            overall_mrr=overall_mrr,
            per_product_mrr=per_product_mrr,
        )

        metadata = RunMetadata(
            engine_name=self._adapter.name,
            k=self._config.k,
            concurrency=self._config.concurrency,
            dataset_path=self._dataset_path,
            dataset_sha256=self._dataset_sha256,
            engine_config=self._engine_config,
        )

        return EvaluationReport(metadata=metadata, evaluations=evaluations, aggregates=aggregates)
