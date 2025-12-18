from __future__ import annotations

import asyncio
import os
from collections.abc import Iterable
from contextlib import AsyncExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from engines.base import SearchEngine
from engines.exa import ExaConfig, ExaSearchEngine
from engines.perplexity import PerplexityConfig, PerplexitySearchEngine
from engines.tavily import TavilyConfig, TavilySearchEngine
from ground_truth import load_eval_json
from judge import JudgeConfig, judge_report
from models import EvaluationReport, QueryCase
from persistence import load_report, save_report
from runner import EvaluationRunner, RunnerConfig

console = Console()


def _default_output_path() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return str(Path("results") / f"results-{ts}.json")


def _iter_all_judgements(report: EvaluationReport):
    for e in report.evaluations:
        for j in getattr(e, "llm_judgements", []) or []:
            yield e, j


def _has_judgements(report: EvaluationReport) -> bool:
    meta = report.metadata
    if getattr(meta, "llm_judge", None) is not None:
        return True
    return any(bool(getattr(e, "llm_judgements", []) or []) for e in report.evaluations)


def _render_judge_summary_tables(*, report: EvaluationReport) -> None:
    pairs = list(_iter_all_judgements(report))
    if not pairs:
        return

    meta = report.metadata
    judge_meta = getattr(meta, "llm_judge", None)
    if judge_meta is not None:
        console.print(
            f"[bold]LLM judge[/bold]: {judge_meta.model} "
            f"[dim]prompt={judge_meta.prompt_version} "
            f"judged_at={judge_meta.judged_at.isoformat()}[/dim]"
        )
    else:
        console.print("[bold]LLM judge[/bold]")

    total = len(pairs)
    issues = sum(1 for _e, j in pairs if j.content_issues)

    avg_overall = sum(j.overall for _e, j in pairs) / float(total)
    avg_qr = sum(j.query_relevance for _e, j in pairs) / float(total)
    avg_rq = sum(j.result_quality for _e, j in pairs) / float(total)
    avg_conf = sum(j.confidence for _e, j in pairs) / float(total)
    issue_rate = issues / float(total) if total else 0.0

    t0 = Table(show_lines=False)
    t0.add_column("Metric", style="bold")
    t0.add_column("Value", justify="right")
    t0.add_row("Avg overall", f"{avg_overall:.4f}")
    t0.add_row("Avg query_relevance", f"{avg_qr:.4f}")
    t0.add_row("Avg result_quality", f"{avg_rq:.4f}")
    t0.add_row("Avg confidence", f"{avg_conf:.4f}")
    t0.add_row("Content issues rate", f"{issue_rate:.2%}")
    console.print(t0)

    # Per-engine summary
    by_engine: dict[str, list[float]] = {}
    by_engine_issues: dict[str, int] = {}
    by_engine_total: dict[str, int] = {}
    for e, j in pairs:
        eng = e.engine_run.engine_name
        by_engine.setdefault(eng, []).append(j.overall)
        by_engine_total[eng] = by_engine_total.get(eng, 0) + 1
        if j.content_issues:
            by_engine_issues[eng] = by_engine_issues.get(eng, 0) + 1

    engine_names = sorted(by_engine.keys())
    if engine_names:
        console.print("[bold]Judge per engine[/bold]")
        te = Table(show_lines=False)
        te.add_column("", style="bold")
        te.add_column("Avg overall", justify="right")
        te.add_column("Issues", justify="right")
        for eng in engine_names:
            avg = sum(by_engine[eng]) / float(len(by_engine[eng]))
            issues = by_engine_issues.get(eng, 0)
            tot = by_engine_total.get(eng, 0)
            te.add_row(eng, f"{avg:.4f}", f"{issues}/{tot}")
        console.print(te)

    # Per-product summary (and optional product x engine matrix if multi-engine)
    by_product_engine: dict[str, dict[str, list[float]]] = {}
    for e, j in pairs:
        by_product_engine.setdefault(e.product, {}).setdefault(e.engine_run.engine_name, []).append(
            j.overall
        )

    products = sorted(by_product_engine.keys())
    if not products:
        return

    if len(engine_names) > 1:
        console.print("[bold]Judge per product[/bold]")
        tp = Table(show_lines=False)
        tp.add_column("", style="bold")
        for eng in engine_names:
            tp.add_column(eng, justify="right")
        for product in products:
            row_vals: list[str] = []
            per_eng = by_product_engine.get(product, {})
            for eng in engine_names:
                scores = per_eng.get(eng)
                if scores:
                    row_vals.append(f"{sum(scores) / float(len(scores)):.4f}")
                else:
                    row_vals.append("")
            tp.add_row(
                product,
                *row_vals,
            )
        console.print(tp)
    else:
        console.print("[bold]Judge per product[/bold]")
        tp = Table(show_lines=False)
        tp.add_column("Product", style="bold")
        tp.add_column("Avg overall", justify="right")
        for product in products:
            scores = []
            for eng_scores in by_product_engine[product].values():
                scores.extend(eng_scores)
            tp.add_row(product, f"{sum(scores) / float(len(scores)):.4f}")
        console.print(tp)


def _render_summary_tables(*, report) -> None:
    meta = report.metadata
    aggs = report.aggregates
    engine_label = "all" if len(meta.engine_names) > 1 else meta.engine_names[0]

    console.print(
        f"[bold]MRR[/bold]: {aggs.overall_mrr:.4f} "
        f"(hits {aggs.hit_count}/{aggs.query_count})  "
        f"[dim]{engine_label} k={meta.k} concurrency={meta.concurrency}[/dim]"
    )

    if aggs.per_engine_overall_mrr:
        engine_names = sorted(aggs.per_engine_overall_mrr.keys())

        console.print("[bold]MRR per engine[/bold]")
        t1 = Table(show_lines=False)
        t1.add_column("", style="bold")
        for engine_name in engine_names:
            t1.add_column(engine_name, justify="right")
        t1.add_row(
            "MRR",
            *[
                f"{aggs.per_engine_overall_mrr.get(engine_name, 0.0):.4f}"
                for engine_name in engine_names
            ],
        )
        console.print(t1)

        # Build product x engine MRR matrix (union across engines).
        products: set[str] = set()
        for per_prod in aggs.per_engine_per_product_mrr.values():
            products.update(per_prod.keys())

        console.print("[bold]MRR per product[/bold]")
        t2 = Table(show_lines=False)
        t2.add_column("", style="bold")
        for engine_name in engine_names:
            t2.add_column(engine_name, justify="right")

        for product in sorted(products):
            t2.add_row(
                product,
                *[
                    f"{aggs.per_engine_per_product_mrr.get(engine_name, {}).get(product, 0.0):.4f}"
                    for engine_name in engine_names
                ],
            )
        console.print(t2)
    else:
        engine_label = (engine_label or "").strip().title() or "MRR"
        t = Table(show_lines=False)
        t.add_column("Product", style="bold")
        t.add_column(engine_label, justify="right")
        for product in sorted(aggs.per_product_mrr):
            t.add_row(product, f"{aggs.per_product_mrr[product]:.4f}")
        console.print(t)

    if _has_judgements(report):
        console.print()
        _render_judge_summary_tables(report=report)


def _resolve_product_and_query_from_cases(
    *, cases: Iterable[QueryCase], product_idx: int, query_idx: int
) -> tuple[str, str]:
    products = sorted({c.product for c in cases})
    if product_idx < 0 or product_idx >= len(products):
        raise click.ClickException(
            f"Invalid --query PRODUCT_IDX={product_idx}. "
            f"Valid range: [0, {max(0, len(products) - 1)}]. "
            f"Products: {products}"
        )

    product = products[product_idx]
    queries = sorted({c.query_text for c in cases if c.product == product})
    if query_idx < 0 or query_idx >= len(queries):
        raise click.ClickException(
            f"Invalid --query QUERY_IDX={query_idx} for product={product!r}. "
            f"Valid range: [0, {max(0, len(queries) - 1)}]. "
            f"Queries: {queries}"
        )

    return product, queries[query_idx]


def _resolve_product_and_query_from_report(
    *, report: EvaluationReport, product_idx: int, query_idx: int
) -> tuple[str, str]:
    products = sorted({e.product for e in report.evaluations})
    if product_idx < 0 or product_idx >= len(products):
        raise click.ClickException(
            f"Invalid --query PRODUCT_IDX={product_idx}. "
            f"Valid range: [0, {max(0, len(products) - 1)}]. "
            f"Products: {products}"
        )

    product = products[product_idx]
    queries = sorted({e.query_text for e in report.evaluations if e.product == product})
    if query_idx < 0 or query_idx >= len(queries):
        raise click.ClickException(
            f"Invalid --query QUERY_IDX={query_idx} for product={product!r}. "
            f"Valid range: [0, {max(0, len(queries) - 1)}]. "
            f"Queries: {queries}"
        )

    return product, queries[query_idx]


def _render_single_query_table(*, report) -> None:
    evals = list(report.evaluations or [])
    if not evals:
        raise click.ClickException("No evaluations found to render.")

    def _display_url(url: str) -> str:
        if url.startswith("https://www."):
            return url[len("https://www.") :]
        if url.startswith("https://"):
            return url[len("https://") :]
        return url

    products = {e.product for e in evals}
    queries = {e.query_text for e in evals}
    if len(products) != 1 or len(queries) != 1:
        raise click.ClickException(
            "Single-query table requires a report filtered to exactly one (product, query). "
            f"Got products={sorted(products)} queries={sorted(queries)}."
        )

    product = next(iter(products))
    query_text = next(iter(queries))

    console.print(f"[bold]Product[/bold]: {product}")
    console.print(f"[bold]Query[/bold]: {query_text}")

    by_engine_eval = {e.engine_run.engine_name: e for e in evals}
    engine_names = sorted(by_engine_eval.keys())

    for engine_name in engine_names:
        qe = by_engine_eval[engine_name]
        run = qe.engine_run

        console.print(f"\n[bold]{engine_name}[/bold]")
        if run.error:
            console.print(f"[red]ERROR[/red]: {run.error}")

        t = Table(show_lines=False, expand=True)
        t.add_column("", style="bold", justify="right")  # Rank
        t.add_column("URL", overflow="fold")
        t.add_column("Title", overflow="fold")

        results = list(run.results or [])
        max_k_seen = max(1, len(results))
        for rank in range(1, max_k_seen + 1):
            idx = rank - 1
            if 0 <= idx < len(results):
                r = results[idx]
                t.add_row(str(rank), _display_url(r.url), (r.title or ""))
            else:
                t.add_row(str(rank), "", "")

        console.print(t)

        judgements = list(getattr(qe, "llm_judgements", []) or [])
        if judgements:
            by_rank = {j.rank: j for j in judgements}
            tj = Table(show_lines=False, expand=True)
            tj.add_column("", style="bold", justify="right")  # Rank
            tj.add_column("Overall", justify="right")
            tj.add_column("Issues", justify="right")
            tj.add_column("Conf", justify="right")
            tj.add_column("Explanation", overflow="fold")
            for r in results:
                j = by_rank.get(r.rank)
                if j is None:
                    continue
                tj.add_row(
                    str(r.rank),
                    f"{j.overall:.3f}",
                    "Y" if j.content_issues else "",
                    f"{j.confidence:.3f}",
                    j.explanation,
                )
            console.print(tj)


@click.group()
def cli() -> None:
    """Evaluate search engines against a ground-truth URL set and report MRR."""


@cli.command()
@click.option(
    "--dataset",
    "dataset_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=Path("eval.json"),
    show_default=True,
)
@click.option("--out", "out_path", type=click.Path(dir_okay=False), default=_default_output_path)
@click.option("-k", type=int, default=10, show_default=True)
@click.option("--concurrency", type=int, default=1, show_default=True)
@click.option(
    "--engine",
    type=click.Choice(["exa", "perplexity", "tavily", "all", "max"], case_sensitive=False),
    default="all",
    show_default=True,
)
@click.option(
    "--query",
    "query_selector",
    type=int,
    nargs=2,
    default=None,
    help="Select a single dataset query to run: --query PRODUCT_IDX QUERY_IDX (0-based).",
)
@click.option("--exa-api-key", type=str, default=None)
@click.option(
    "--exa-search-type",
    type=click.Choice(["auto", "neural", "fast", "deep"], case_sensitive=False),
    default="auto",
    show_default=True,
)
@click.option("--perplexity-api-key", type=str, default=None)
@click.option("--tavily-api-key", type=str, default=None)
@click.option("--judge/--no-judge", default=False, show_default=True)
@click.option("--openai-api-key", type=str, default=None)
@click.option("--judge-model", type=str, default="gpt-4o-mini", show_default=True)
@click.option("--judge-concurrency", type=int, default=5, show_default=True)
@click.option("--judge-max-k", type=int, default=None)
def run(
    *,
    dataset_path: Path,
    out_path: str,
    k: int,
    concurrency: int,
    engine: str,
    query_selector: tuple[int, int] | None,
    exa_api_key: str | None,
    exa_search_type: Literal["auto", "neural", "fast", "deep"],
    perplexity_api_key: str | None,
    tavily_api_key: str | None,
    judge: bool,
    openai_api_key: str | None,
    judge_model: str,
    judge_concurrency: int,
    judge_max_k: int | None,
) -> None:
    """Run evaluation (calls selected engine) and write a JSON artifact to disk."""
    load_dotenv()

    cases, dataset_sha256 = load_eval_json(dataset_path)
    if query_selector is not None:
        product_idx, query_idx = query_selector
        product, query_text = _resolve_product_and_query_from_cases(
            cases=cases, product_idx=product_idx, query_idx=query_idx
        )
        cases = [c for c in cases if c.product == product and c.query_text == query_text]

    engine = (engine or "").strip().lower()

    engines: list[SearchEngine] = []

    if engine == "max":
        engines = [
            ExaSearchEngine(
                ExaConfig(api_key=exa_api_key or os.getenv("EXA_API_KEY"), search_type="auto")
            ),
            ExaSearchEngine(
                ExaConfig(api_key=exa_api_key or os.getenv("EXA_API_KEY"), search_type="neural")
            ),
            ExaSearchEngine(
                ExaConfig(api_key=exa_api_key or os.getenv("EXA_API_KEY"), search_type="fast")
            ),
            ExaSearchEngine(
                ExaConfig(api_key=exa_api_key or os.getenv("EXA_API_KEY"), search_type="deep")
            ),
            PerplexitySearchEngine(
                PerplexityConfig(api_key=perplexity_api_key or os.getenv("PERPLEXITY_API_KEY"))
            ),
            TavilySearchEngine(TavilyConfig(api_key=tavily_api_key or os.getenv("TAVILY_API_KEY"))),
        ]
    elif engine == "all":
        engines = [
            ExaSearchEngine(
                ExaConfig(api_key=exa_api_key or os.getenv("EXA_API_KEY"), search_type="fast")
            ),
            PerplexitySearchEngine(
                PerplexityConfig(api_key=perplexity_api_key or os.getenv("PERPLEXITY_API_KEY"))
            ),
            TavilySearchEngine(TavilyConfig(api_key=tavily_api_key or os.getenv("TAVILY_API_KEY"))),
        ]
    elif engine == "exa":
        api_key = exa_api_key or os.getenv("EXA_API_KEY")
        if not api_key:
            raise click.ClickException("Missing EXA_API_KEY (set env var or pass --exa-api-key).")

        engines = [
            ExaSearchEngine(
                ExaConfig(
                    api_key=api_key,
                    search_type=exa_search_type,
                )
            )
        ]
    elif engine == "perplexity":
        api_key = perplexity_api_key or os.getenv("PERPLEXITY_API_KEY")
        if not api_key:
            raise click.ClickException(
                "Missing PERPLEXITY_API_KEY (set env var or pass --perplexity-api-key)."
            )
        engines = [PerplexitySearchEngine(PerplexityConfig(api_key=api_key))]
    elif engine == "tavily":
        api_key = tavily_api_key or os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise click.ClickException(
                "Missing TAVILY_API_KEY (set env var or pass --tavily-api-key)."
            )
        engines = [TavilySearchEngine(TavilyConfig(api_key=api_key))]
    else:
        raise click.ClickException(f"Unknown engine: {engine}")

    runner = EvaluationRunner(
        engines=engines,
        config=RunnerConfig(k=k, concurrency=concurrency),
        dataset_path=str(dataset_path),
        dataset_sha256=dataset_sha256,
    )

    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("[red]E:{task.fields[errors]}[/red]"),
        TimeElapsedColumn(),
        console=console,
    )

    async def _go():
        async with AsyncExitStack() as stack:
            for e in engines:
                await stack.enter_async_context(e)

            per_engine_total = len(cases)
            total = per_engine_total * len(engines)

            overall_task_id: int | None = None
            if len(engines) > 1:
                overall_task_id = progress.add_task("Overall", total=total, errors=0)

            task_id_by_engine: dict[str, int] = {
                e.name: progress.add_task(e.name, total=per_engine_total, errors=0) for e in engines
            }

            errors_by_engine: dict[str, int] = {e.name: 0 for e in engines}
            overall_errors = 0

            def on_done(evaluation, _done: int, _total: int) -> None:
                nonlocal overall_errors
                engine_name = evaluation.engine_run.engine_name
                had_error = evaluation.engine_run.error is not None
                task_id = task_id_by_engine.get(engine_name)
                if task_id is not None:
                    if had_error:
                        errors_by_engine[engine_name] = errors_by_engine.get(engine_name, 0) + 1
                    progress.update(task_id, advance=1, errors=errors_by_engine.get(engine_name, 0))
                if overall_task_id is not None:
                    if had_error:
                        overall_errors += 1
                    progress.update(overall_task_id, advance=1, errors=overall_errors)

            with progress:
                return await runner.run(cases, on_evaluation_complete=on_done)

    report = asyncio.run(_go())

    if judge:
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise click.ClickException(
                "Missing OPENAI_API_KEY (set env var or pass --openai-api-key)."
            )

        judge_progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        )

        total = len(report.evaluations)
        task_id = judge_progress.add_task("Judging", total=total)

        def _on_judge_done(done: int, _total: int) -> None:
            judge_progress.update(task_id, completed=done)

        async def _judge():
            with judge_progress:
                max_k_eff = judge_max_k if judge_max_k is not None else k
                return await judge_report(
                    report,
                    config=JudgeConfig(
                        api_key=api_key,
                        model=judge_model,
                        concurrency=judge_concurrency,
                        max_k=max_k_eff,
                    ),
                    on_judgement_complete=_on_judge_done,
                )

        report = asyncio.run(_judge())

    save_report(report, out_path)

    console.print(f"[bold]Wrote[/bold] {out_path}")
    if query_selector is not None:
        _render_single_query_table(report=report)
    else:
        _render_summary_tables(report=report)


@cli.command()
@click.argument("artifact", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--query",
    "query_selector",
    type=int,
    nargs=2,
    default=None,
    help="Select a single query to render: --query PRODUCT_IDX QUERY_IDX (0-based).",
)
def report(*, artifact: str, query_selector: tuple[int, int] | None) -> None:
    """Render a saved JSON artifact."""
    r = load_report(artifact)
    meta = r.metadata
    engine_label = "all" if len(meta.engine_names) > 1 else meta.engine_names[0]
    judge_meta = getattr(meta, "llm_judge", None)
    judge_part = f" judge={judge_meta.model}" if judge_meta is not None else ""
    console.print(
        f"[bold]Artifact[/bold]: {artifact}\n"
        f"[dim]created_at={meta.created_at.isoformat()} engine={engine_label} "
        f"k={meta.k} concurrency={meta.concurrency}{judge_part} "
        f"dataset_sha256={meta.dataset_sha256}[/dim]"
    )
    if query_selector is not None:
        product_idx, query_idx = query_selector
        product, query_text = _resolve_product_and_query_from_report(
            report=r, product_idx=product_idx, query_idx=query_idx
        )
        filtered = [e for e in r.evaluations if e.product == product and e.query_text == query_text]
        r2 = r.model_copy(update={"evaluations": filtered})
        _render_single_query_table(report=r2)
    else:
        _render_summary_tables(report=r)


if __name__ == "__main__":
    cli()
