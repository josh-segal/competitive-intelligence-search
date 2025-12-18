from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass

from engines.base import SearchEngine
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
    concurrency: int = 1


class EvaluationRunner:
    def __init__(
        self,
        *,
        engines: list[SearchEngine],
        config: RunnerConfig,
        dataset_path: str,
        dataset_sha256: str,
    ) -> None:
        self._engines = engines
        self._config = config
        self._dataset_path = dataset_path
        self._dataset_sha256 = dataset_sha256

    async def run(
        self,
        cases: list[QueryCase],
        *,
        on_evaluation_complete: (
            Callable[[QueryEvaluation, int, int], None] | None  # (evaluation, done, total)
        ) = None,
    ) -> EvaluationReport:
        engines = list(self._engines or [])
        if not engines:
            raise ValueError("EvaluationRunner requires at least one engine.")

        sem_by_engine: dict[str, asyncio.Semaphore] = {
            e.name: asyncio.Semaphore(self._config.concurrency) for e in engines
        }

        total = len(cases) * len(engines)

        async def eval_one(engine: SearchEngine, case: QueryCase) -> QueryEvaluation:
            sem = sem_by_engine[engine.name]
            async with sem:
                engine_run = await engine.search(case.query_text, k=self._config.k)

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

        tasks: list[asyncio.Task[QueryEvaluation]] = []
        for engine in engines:
            for case in cases:
                tasks.append(asyncio.create_task(eval_one(engine, case)))

        evaluations: list[QueryEvaluation] = []
        done = 0
        for fut in asyncio.as_completed(tasks):
            e = await fut
            evaluations.append(e)
            done += 1
            if on_evaluation_complete is not None:
                on_evaluation_complete(e, done, total)

        evaluations.sort(key=lambda e: (e.engine_run.engine_name, e.product, e.query_text))

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

        by_engine: dict[str, list[QueryEvaluation]] = defaultdict(list)
        for e in evaluations:
            by_engine[e.engine_run.engine_name].append(e)

        per_engine_overall_mrr: dict[str, float] = {}
        per_engine_hit_count: dict[str, int] = {}
        per_engine_per_product_mrr: dict[str, dict[str, float]] = {}

        for engine_name, es in by_engine.items():
            per_engine_overall_mrr[engine_name] = (
                sum(x.reciprocal_rank for x in es) / float(len(es)) if es else 0.0
            )
            per_engine_hit_count[engine_name] = sum(
                1 for x in es if x.first_relevant_rank is not None
            )

            by_prod: dict[str, list[QueryEvaluation]] = defaultdict(list)
            for x in es:
                by_prod[x.product].append(x)
            per_engine_per_product_mrr[engine_name] = {
                product: (sum(y.reciprocal_rank for y in ys) / float(len(ys)) if ys else 0.0)
                for product, ys in by_prod.items()
            }

        aggregates = EvaluationAggregates(
            query_count=len(evaluations),
            hit_count=hit_count,
            overall_mrr=overall_mrr,
            per_product_mrr=per_product_mrr,
            per_engine_overall_mrr=per_engine_overall_mrr,
            per_engine_hit_count=per_engine_hit_count,
            per_engine_per_product_mrr=per_engine_per_product_mrr,
        )

        engine_names = sorted({e.name for e in engines})
        metadata = RunMetadata(
            engine_names=engine_names,
            k=self._config.k,
            concurrency=self._config.concurrency,
            dataset_path=self._dataset_path,
            dataset_sha256=self._dataset_sha256,
        )

        return EvaluationReport(metadata=metadata, evaluations=evaluations, aggregates=aggregates)
