from __future__ import annotations

import json
from pathlib import Path

from models import EvaluationReport


def save_report(report: EvaluationReport, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    data = report.model_dump(mode="json")

    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(p)


def load_report(path: str | Path) -> EvaluationReport:
    p = Path(path)
    raw = json.loads(p.read_text(encoding="utf-8"))
    return EvaluationReport.model_validate(raw)
