from __future__ import annotations

import asyncio
import os
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
from persistence import load_report, save_report
from runner import EvaluationRunner, RunnerConfig

console = Console()


def _default_output_path() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return str(Path("results") / f"results-{ts}.json")


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
    type=click.Choice(["exa", "perplexity", "tavily", "all"], case_sensitive=False),
    default="all",
    show_default=True,
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
def run(
    *,
    dataset_path: Path,
    out_path: str,
    k: int,
    concurrency: int,
    engine: str,
    exa_api_key: str | None,
    exa_search_type: Literal["auto", "neural", "fast", "deep"],
    perplexity_api_key: str | None,
    tavily_api_key: str | None,
) -> None:
    """Run evaluation (calls selected engine) and write a JSON artifact to disk."""
    load_dotenv()

    cases, dataset_sha256 = load_eval_json(dataset_path)

    engine = (engine or "").strip().lower()

    engines: list[SearchEngine] = []

    if engine == "all":
        engines = [
            ExaSearchEngine(
                ExaConfig(
                    api_key=exa_api_key or os.getenv("EXA_API_KEY"), search_type=exa_search_type
                )
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
        TimeElapsedColumn(),
        console=console,
    )

    async def _go():
        async with AsyncExitStack() as stack:
            for e in engines:
                await stack.enter_async_context(e)

            total = len(cases) * len(engines)
            task_id = progress.add_task("Searching", total=total)

            def on_done(_evaluation, done: int, _total: int) -> None:
                progress.update(task_id, completed=done)

            with progress:
                return await runner.run(cases, on_evaluation_complete=on_done)

    report = asyncio.run(_go())
    save_report(report, out_path)

    console.print(f"[bold]Wrote[/bold] {out_path}")
    _render_summary_tables(report=report)


@cli.command()
@click.argument("artifact", type=click.Path(exists=True, dir_okay=False))
def report(*, artifact: str) -> None:
    """Render a saved JSON artifact (no network calls)."""
    r = load_report(artifact)
    meta = r.metadata
    engine_label = "all" if len(meta.engine_names) > 1 else meta.engine_names[0]
    console.print(
        f"[bold]Artifact[/bold]: {artifact}\n"
        f"[dim]created_at={meta.created_at.isoformat()} engine={engine_label} "
        f"k={meta.k} concurrency={meta.concurrency} dataset_sha256={meta.dataset_sha256}[/dim]"
    )
    _render_summary_tables(report=r)


if __name__ == "__main__":
    cli()
