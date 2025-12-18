from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from pydantic import BaseModel, Field


class EvalJson(BaseModel):
    """
    Mirrors the on-disk `eval.json` structure exactly:

    {
      "ProductA": {
        "query text 1": ["https://...", ...],
        "query text 2": ["https://...", ...]
      },
      "ProductB": { ... }
    }
    """

    root: dict[str, dict[str, list[str]]]

    model_config = {
        "extra": "forbid",
    }


@dataclass(frozen=True)
class QueryCase:
    product: str
    query_text: str
    relevant_urls: tuple[str, ...]


class SearchResultItem(BaseModel):
    rank: int = Field(ge=1)
    url: str = Field(min_length=1)

    model_config = {
        "extra": "forbid",
    }


class EngineRunResult(BaseModel):
    engine_name: str = Field(min_length=1)
    query_text: str = Field(min_length=1)
    results: list[SearchResultItem] = Field(default_factory=list)
    error: str | None = None

    model_config = {
        "extra": "forbid",
    }


class QueryEvaluation(BaseModel):
    product: str = Field(min_length=1)
    query_text: str = Field(min_length=1)
    relevant_urls: list[str] = Field(default_factory=list)
    first_relevant_rank: int | None = Field(default=None, ge=1)
    reciprocal_rank: float = Field(ge=0)
    engine_run: EngineRunResult

    model_config = {
        "extra": "forbid",
    }


class EvaluationAggregates(BaseModel):
    query_count: int = Field(ge=0)
    hit_count: int = Field(ge=0)
    overall_mrr: float = Field(ge=0, le=1)
    per_product_mrr: dict[str, float] = Field(default_factory=dict)
    # Optional multi-engine breakdowns (populated when running with multiple engines).
    per_engine_overall_mrr: dict[str, float] = Field(default_factory=dict)
    per_engine_hit_count: dict[str, int] = Field(default_factory=dict)
    per_engine_per_product_mrr: dict[str, dict[str, float]] = Field(default_factory=dict)

    model_config = {
        "extra": "forbid",
    }


class RunMetadata(BaseModel):
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    engine_names: list[str] = Field(min_length=1)
    k: int = Field(ge=1)
    concurrency: int = Field(ge=1)
    dataset_path: str = Field(min_length=1)
    dataset_sha256: str = Field(min_length=1)

    model_config = {
        "extra": "forbid",
    }


class EvaluationReport(BaseModel):
    metadata: RunMetadata
    evaluations: list[QueryEvaluation] = Field(default_factory=list)
    aggregates: EvaluationAggregates

    model_config = {
        "extra": "forbid",
    }
