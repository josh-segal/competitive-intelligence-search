## Competitive Intelligence Search (Retrieval Evaluation)

This repo evaluates search retrieval quality against a ground-truth URL set (MRR), using a thin Click + Rich CLI on top of a pure-Python core.

### Dataset

The ground truth lives in `eval.json` with this structure:

- `product -> query -> [relevant_url, ...]`

Relevance is scored using **exact URL string matching**.

### Setup

- **Environment**: set the relevant API key for the engine you’re running:
  - `EXA_API_KEY` (or pass `--exa-api-key`)
  - `PERPLEXITY_API_KEY` (or pass `--perplexity-api-key`)
  - `TAVILY_API_KEY` (or pass `--tavily-api-key`)
- **Install** (editable):

```bash
python -m pip install -e .
```

### Run an evaluation

```bash
ci-eval run --engine exa --dataset eval.json -k 10 --concurrency 10 --out results/results.json
```

Perplexity:

```bash
ci-eval run --engine perplexity --dataset eval.json -k 10 --concurrency 10 --out results/results.json
```

Tavily:

```bash
ci-eval run --engine tavily --dataset eval.json -k 10 --concurrency 10 --out results/results.json
```

Outputs:
- A JSON artifact at `--out` (reloadable without any API calls)
- Rich tables summarizing overall MRR and per-product MRR

### Render a saved artifact

```bash
ci-eval report results/results.json
```

### Notes

- API failures are recorded per query and do not crash the run.


