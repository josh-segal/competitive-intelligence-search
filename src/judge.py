from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from models import EvaluationReport, LLMJudgementItem, LLMJudgeMetadata

SYSTEM_PROMPT = "\n".join(
    [
        "You are a helpful assistant that grades the relevance of search results "
        "for given queries.",
        "Your task is to assign a relevance score between 0.0 and 1.0 to each result, "
        "based on how good a result is for the query.",
        "",
        "For each search result, carefully read the query and the result. "
        "Assign a value for each criterion as follows:",
        "- Provide a brief explanation of your reasoning.",
        "- Assign a query_relevance score between 0.0 and 1.0.",
        "- Assign a result_quality score between 0.0 and 1.0.",
        "- Indicate if there are any content_issues (True/False).",
        "- Assign a confidence score between 0.0 and 1.0.",
        "- Assign an overall score between 0.0 and 1.0.",
    ]
)


class _JudgeResponse(BaseModel):
    judgements: list[LLMJudgementItem] = Field(default_factory=list)

    model_config = {
        "extra": "forbid",
    }


@dataclass(frozen=True)
class JudgeConfig:
    api_key: str
    model: str = "gpt-4o-mini"
    concurrency: int = 1
    max_k: int | None = None
    prompt_version: str = "v1"
    # prevent oversized judge requests
    max_title_chars: int = 200
    max_content_chars: int = 2000


def _build_payload(
    *,
    query_text: str,
    results: list[Any],
    max_title_chars: int,
    max_content_chars: int,
) -> dict[str, Any]:
    def _truncate(text: str | None, *, max_chars: int) -> str | None:
        if text is None:
            return None
        s = str(text)
        if len(s) <= max_chars:
            return s
        return s[: max(0, max_chars - 1)] + "…"

    return {
        "query": query_text,
        "results": [
            {
                "rank": r.rank,
                "url": r.url,
                "title": _truncate(getattr(r, "title", None), max_chars=max_title_chars),
                "content": _truncate(getattr(r, "content", None), max_chars=max_content_chars),
            }
            for r in results
        ],
        "output_format": {
            "judgements": [
                {
                    "rank": 1,
                    "explanation": "string",
                    "query_relevance": 0.0,
                    "result_quality": 0.0,
                    "content_issues": False,
                    "confidence": 0.0,
                    "overall": 0.0,
                }
            ]
        },
        "rules": [
            "Return ONLY valid JSON.",
            "Provide exactly one judgement per provided result, matching by rank.",
            "All scores must be in [0.0, 1.0].",
        ],
    }


def _align_judgements(
    *, ranks: list[int], returned: list[LLMJudgementItem]
) -> list[LLMJudgementItem]:
    by_rank: dict[int, LLMJudgementItem] = {j.rank: j for j in returned}
    aligned: list[LLMJudgementItem] = []
    for rank in ranks:
        j = by_rank.get(rank)
        if j is None:
            aligned.append(
                LLMJudgementItem(
                    rank=rank,
                    explanation="No judgement returned for this rank.",
                    query_relevance=0.0,
                    result_quality=0.0,
                    content_issues=False,
                    confidence=0.0,
                    overall=0.0,
                )
            )
        else:
            aligned.append(j)
    return aligned


async def judge_report(
    report: EvaluationReport,
    *,
    config: JudgeConfig,
    on_judgement_complete: Callable[[int, int], None] | None = None,  # (done, total)
) -> EvaluationReport:
    client = AsyncOpenAI(api_key=config.api_key)
    sem = asyncio.Semaphore(max(1, int(config.concurrency)))

    async def judge_one(qe):
        # Perplexity skip judging for now
        if qe.engine_run.engine_name == "perplexity":
            return qe
        if qe.engine_run.error:
            return qe

        results = list(qe.engine_run.results or [])
        if not results:
            return qe

        if config.max_k is not None:
            results = results[: max(1, int(config.max_k))]

        ranks = [r.rank for r in results]
        payload = _build_payload(
            query_text=qe.query_text,
            results=results,
            max_title_chars=config.max_title_chars,
            max_content_chars=config.max_content_chars,
        )

        async with sem:
            try:
                resp = await client.chat.completions.create(
                    model=config.model,
                    temperature=0,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": json.dumps(payload)},
                    ],
                )
            except Exception:
                return qe

        text = (resp.choices[0].message.content or "").strip()
        try:
            parsed = _JudgeResponse.model_validate(json.loads(text or "{}"))
        except Exception:
            return qe

        aligned = _align_judgements(ranks=ranks, returned=parsed.judgements)
        return qe.model_copy(update={"llm_judgements": aligned})

    tasks = [asyncio.create_task(judge_one(qe)) for qe in report.evaluations]

    new_evals = []
    done = 0
    total = len(tasks)
    for fut in asyncio.as_completed(tasks):
        new_evals.append(await fut)
        done += 1
        if on_judgement_complete is not None:
            on_judgement_complete(done, total)
    new_evals.sort(key=lambda e: (e.engine_run.engine_name, e.product, e.query_text))

    new_meta = report.metadata.model_copy(
        update={
            "llm_judge": LLMJudgeMetadata(
                model=config.model,
                prompt_version=config.prompt_version,
            )
        }
    )
    return report.model_copy(update={"metadata": new_meta, "evaluations": new_evals})
