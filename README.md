## Competitive Intelligence Search (Retrieval Evaluation)

This repo evaluates search retrieval quality against a ground-truth URL set (MRR), using a thin Click + Rich CLI on top of a pure-Python core.

### Dataset

The ground truth lives in `eval.json` with this structure:

- `product -> query -> [relevant_url, ...]`

Relevance is scored using **exact URL string matching**.

### Setup

- **Environment**: set `EXA_API_KEY` (or pass `--exa-api-key`).
- **Install** (editable):

```bash
python -m pip install -e .
```

### Run an evaluation (calls Exa)

```bash
ci-eval run --dataset eval.json -k 10 --concurrency 10 --out results/results.json
```

Outputs:
- A JSON artifact at `--out` (reloadable without any API calls)
- Rich tables summarizing overall MRR and per-product MRR

### Render a saved artifact (no network)

```bash
ci-eval report results/results.json
```

### Notes

- API failures are recorded per query and do not crash the run.
- The JSON artifact includes a `schema_version` for forwards-compatible persistence.


