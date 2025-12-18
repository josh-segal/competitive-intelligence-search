from __future__ import annotations

import hashlib
import json
from pathlib import Path

from models import EvalJson, QueryCase


def fingerprint_bytes_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_eval_json(path: str | Path) -> tuple[list[QueryCase], str]:
    """
    Load and validate the dataset at `path`.

    Returns:
      (query_cases, dataset_sha256)
    """
    p = Path(path)
    raw_bytes = p.read_bytes()
    dataset_sha256 = fingerprint_bytes_sha256(raw_bytes)
    raw = json.loads(raw_bytes.decode("utf-8"))
    eval_json = EvalJson.model_validate({"root": raw})

    cases: list[QueryCase] = []
    for product, queries in eval_json.root.items():
        for query_text, relevant_urls in queries.items():
            cases.append(
                QueryCase(
                    product=product,
                    query_text=query_text,
                    relevant_urls=tuple(relevant_urls),
                )
            )

    cases.sort(key=lambda c: (c.product, c.query_text))
    return cases, dataset_sha256
