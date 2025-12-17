from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from adapters.exa import ExaConfig, ExaSearchAdapter
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

    console.print(
        f"[bold]MRR[/bold]: {aggs.overall_mrr:.4f} "
        f"(hits {aggs.hit_count}/{aggs.query_count})  "
        f"[dim]{meta.engine_name} k={meta.k} concurrency={meta.concurrency}[/dim]"
    )

    engine_label = (meta.engine_name or "").strip().title() or "MRR"
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
@click.option("--exa-api-key", type=str, default=None)
@click.option("--exa-base-url", type=str, default="https://api.exa.ai", show_default=True)
@click.option(
    "--exa-search-type",
    type=click.Choice(["auto", "neural", "keyword"], case_sensitive=False),
    default="auto",
    show_default=True,
)
@click.option("--exa-use-autoprompt/--no-exa-use-autoprompt", default=True, show_default=True)
@click.option("--exa-timeout-s", type=float, default=30.0, show_default=True)
def run(
    *,
    dataset_path: Path,
    out_path: str,
    k: int,
    concurrency: int,
    exa_api_key: str | None,
    exa_base_url: str,
    exa_search_type: str,
    exa_use_autoprompt: bool,
    exa_timeout_s: float,
) -> None:
    """Run evaluation (calls Exa) and write a JSON artifact to disk."""
    load_dotenv()
    api_key = exa_api_key or os.getenv("EXA_API_KEY")
    if not api_key:
        raise click.ClickException("Missing EXA_API_KEY (set env var or pass --exa-api-key).")

    cases, dataset_sha256 = load_eval_json(dataset_path)

    engine_cfg: dict[str, Any] = {
        "base_url": exa_base_url,
        "search_type": exa_search_type,
        "use_autoprompt": exa_use_autoprompt,
        "timeout_s": exa_timeout_s,
    }

    adapter = ExaSearchAdapter(
        ExaConfig(
            api_key=api_key,
            base_url=exa_base_url,
            search_type=exa_search_type.lower(),  # click normalizes but keep explicit
            use_autoprompt=exa_use_autoprompt,
            timeout_s=exa_timeout_s,
        )
    )

    runner = EvaluationRunner(
        adapter=adapter,
        config=RunnerConfig(k=k, concurrency=concurrency),
        dataset_path=str(dataset_path),
        dataset_sha256=dataset_sha256,
        engine_config=engine_cfg,
    )

    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    )

    async def _go():
        async with adapter:
            task_id = progress.add_task("Searching", total=len(cases))

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
    console.print(
        f"[bold]Artifact[/bold]: {artifact}\n"
        f"[dim]created_at={meta.created_at.isoformat()} engine={meta.engine_name} "
        f"k={meta.k} concurrency={meta.concurrency} dataset_sha256={meta.dataset_sha256}[/dim]"
    )
    _render_summary_tables(report=r)


if __name__ == "__main__":
    cli()
