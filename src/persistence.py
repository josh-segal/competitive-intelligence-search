from __future__ import annotations

import json
from pathlib import Path

from models import SCHEMA_VERSION, EvaluationReport


class PersistenceError(RuntimeError):
    pass


def save_report(report: EvaluationReport, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    data = report.model_dump(mode="json")
    if data.get("metadata", {}).get("schema_version") != SCHEMA_VERSION:
        raise PersistenceError("Refusing to write report with unexpected schema_version")

    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(p)


def load_report(path: str | Path) -> EvaluationReport:
    p = Path(path)
    raw = json.loads(p.read_text(encoding="utf-8"))

    schema_version = raw.get("metadata", {}).get("schema_version")
    if schema_version != SCHEMA_VERSION:
        raise PersistenceError(
            f"Unsupported schema_version={schema_version}; expected {SCHEMA_VERSION}"
        )

    return EvaluationReport.model_validate(raw)
