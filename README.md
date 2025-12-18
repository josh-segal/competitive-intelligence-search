## Competitive Intelligence Search (Retrieval Evaluation)

This project evaluates search retrieval quality for competitive intelligence search across software products. Two evaluation methods are used: MRR (Mean Reciprocal Rank) against a ground truth URL set, and LLM-as-judge scoring across multiple dimensions including query relevance, result quality, content issues, confidence score, and overall score.


### Results

#### Judge per engine

| Engine | Avg overall | Issues |
|---|---:|---:|
| exa-auto | 0.5199 | 5/500 |
| exa-deep | 0.5269 | 5/398 |
| exa-fast | 0.4450 | 1/500 |
| exa-neural | 0.4081 | 5/497 |
| tavily | 0.4704 | 59/488 |

#### Judge per Product
| Product | exa-auto | exa-deep | exa-fast | exa-neural | tavily |
|---|---:|---:|---:|---:|---:|
| Asana | 0.5190 | 0.4643 | 0.3880 | 0.3640 | 0.4965 |
| Confluence | 0.5756 | 0.5563 | 0.4720 | 0.4750 | 0.4980 |
| HubSpot | 0.5690 | 0.5513 | 0.4270 | 0.4250 | 0.4827 |
| Intercom | 0.4106 | 0.5156 | 0.4080 | 0.3265 | 0.3800 |
| Monday.com | 0.5230 | 0.5342 | 0.4700 | 0.3449 | 0.4708 |
| Shopify | 0.5940 | 0.6315 | 0.4820 | 0.5190 | 0.5504 |
| Slack | 0.5378 | 0.5350 | 0.4710 | 0.4630 | 0.4851 |
| Trello | 0.4430 | 0.4871 | 0.3440 | 0.3120 | 0.4250 |
| Zendesk | 0.4620 | 0.5369 | 0.4800 | 0.3469 | 0.4308 |
| Zoom | 0.5650 | 0.4723 | 0.5075 | 0.5005 | 0.4875 |

### Dataset

Datasets are plain JSON files with this shape:

```json
{
  "<product>": {
    "<query>": ["<relevant_url>", "..."]
  }
}
```


- **`eval.json` (ground truth)**: each query maps to **one or more** relevant URLs.
- **`eval2.json` (non-ground-truth / judge-only)**: uses the same schema, but each query’s relevant-URL list is **empty** (`[]`).


### Setup

- **Environment**: set the relevant API key for the engine you’re running:
  - `EXA_API_KEY` (or pass `--exa-api-key`)
  - `PERPLEXITY_API_KEY` (or pass `--perplexity-api-key`)
  - `TAVILY_API_KEY` (or pass `--tavily-api-key`)
  - `OPENAI_API_KEY` (for LLM-as-judge feature)
- **Install** (editable):

```bash
python -m pip install -e .
```

### Run an evaluation

#### `ci-eval run`

**Usage**:

```bash
ci-eval run [OPTIONS]
```

**Options**:

| Option | Type | Required | Default | Notes |
|---|---:|:---:|---:|---|
| `--dataset PATH` | file path | No | `eval.json` | Must exist. JSON dataset to evaluate. |
| `--out PATH` | file path | No | `results/results-<UTC_TIMESTAMP>.json` | Output JSON artifact path. |
| `-k INT` | int | No | `10` | Number of results to request per query. |
| `--concurrency INT` | int | No | `1` | Per-engine concurrency for retrieval. |
| `--engine [exa\|perplexity\|tavily\|all\|max]` | enum | No | `all` | `all` runs Exa (fast) + Perplexity + Tavily. `max` runs Exa in all search types + Perplexity + Tavily. |
| `--query PRODUCT_IDX QUERY_IDX` | 2 ints | No | — | Run a single dataset query by index (0-based) for quick inspection. |
| `--exa-api-key TEXT` | string | Conditionally | — | Required when running an Exa engine (`--engine exa`, `all`, or `max`) unless `EXA_API_KEY` is set. |
| `--exa-search-type [auto\|neural\|fast\|deep]` | enum | No | `auto` | Only used when `--engine exa`. (For `all`/`max`, Exa search types are fixed as described above.) |
| `--perplexity-api-key TEXT` | string | Conditionally | — | Required when running Perplexity (`--engine perplexity`, `all`, or `max`) unless `PERPLEXITY_API_KEY` is set. |
| `--tavily-api-key TEXT` | string | Conditionally | — | Required when running Tavily (`--engine tavily`, `all`, or `max`) unless `TAVILY_API_KEY` is set. |
| `--judge / --no-judge` | bool | No | `--no-judge` | If enabled, runs LLM-as-judge after retrieval and stores judgement details in the artifact. |
| `--openai-api-key TEXT` | string | Conditionally | — | Required when `--judge` unless `OPENAI_API_KEY` is set. |
| `--judge-model TEXT` | string | No | `gpt-4o-mini` | OpenAI model used for judging. |
| `--judge-concurrency INT` | int | No | `5` | Concurrency for LLM judging. |
| `--judge-max-k INT` | int | No | — | Judge only the top-k results. Defaults to `k` when omitted. |

**Outputs**:

- **JSON artifact**: written to `--out` (contains per-query runs, MRR aggregates, and optional LLM judgements).
- **Console tables**:
  - MRR overall + per-product (and per-engine when running multiple engines)
  - If `--judge` is enabled: averages for `overall`, `query_relevance`, `result_quality`, `confidence`, and a content-issues rate (plus per-engine/per-product breakdowns).

### Render a saved artifact

#### `ci-eval report`

**Usage**:

```bash
ci-eval report ARTIFACT [OPTIONS]
```

**Arguments**:

| Argument | Type | Required | Notes |
|---|---:|:---:|---|
| `ARTIFACT` | file path | Yes | Path to a saved JSON artifact produced by `ci-eval run`. |

**Options**:

| Option | Type | Required | Default | Notes |
|---|---:|:---:|---:|---|
| `--query PRODUCT_IDX QUERY_IDX` | 2 ints | No | — | Render a single query’s results (0-based) as a detailed table. |


