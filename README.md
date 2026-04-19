# LakePrompt

LakePrompt is a Python research prototype for answering natural-language questions over a data lake of CSV files. It profiles columns, retrieves relevant schema signals, plans joins, executes SQL over Polars, packages evidence for an LLM, and returns grounded answers with citations.

The repository currently includes:

- the core `lakeprompt` package
- a Spider-style evaluation runner
- a generated join-question benchmark file
- optional tooling to derive SQL-result ground truth offline
- project documents under `Documents/`

## Current State

The current pipeline supports:

- recursive CSV discovery from nested directories
- stable table naming for nested paths such as `region/sales/customers.csv -> region__sales__customers`
- local and remote lake initialization via:
  - local CSV directories
  - direct `.csv` URLs
  - direct `.zip` URLs containing CSV files
  - GitHub repository URLs
- cached table summaries and prepared remote sources under `.lakeprompt_cache/`
- semantic column retrieval with `sentence-transformers` and `hnswlib`
- planner-context expansion from retrieved tables into the local join neighborhood before LLM query planning
- LLM extraction of a structured `QueryPlan` with:
  - filters
  - projections
  - group-by
  - having
  - order-by
  - limit
- join-path discovery with:
  - Jaccard overlap
  - schema heuristics
  - namespace-aware pruning for Spider-style multi-database lakes
  - plan-aware path validation
  - single-table short-circuiting when no join is required
- deterministic SQL generation for Polars SQL from `QueryPlan`
- typed filter coercion before execution
- aggregate-aware output metadata so selected aggregate fields survive into evidence packaging
- row-level evidence ranking and diversity filtering
- prompt-time packaging with:
  - TOON-encoded final prompts
  - prompt-friendly display names for row fields
  - stable `field_map` metadata back to internal execution keys
- optional stdout tracing for:
  - semantic retrieval
  - expanded planner schema context
  - LLM prompts and responses
  - parsed and transformed query plans
  - ranked join paths
  - refined SQL
  - executed SQL
  - ranked evidence rows
  - final prompt context metadata
- local fallback behavior when no evidence is found, without calling the final answer LLM

## Repository Layout

Important files and directories:

- `lakeprompt/`: core package
- `eval/evaluation.py`: evaluation runner
- `eval/spider_join_questions.json`: Spider join questions with expected SQL
- `eval/spider_join_questions_with_ground_truth.json`: same questions plus derived SQL-result ground truth
- `eval/generate_ground_truth.py`: optional offline ground-truth generator
- `debug_one_question.py`: single-question debug runner with `logger=True`
- `data/`: local CSV schemas used by evaluation/debugging
- `requirements.txt`: pinned dependencies
- `Documents/`: proposal and progress notes

Important package modules:

- `lakeprompt/_lakeprompt.py`: public `LakePrompt` orchestration
- `lakeprompt/_datalake.py`: CSV loading and Polars SQL execution
- `lakeprompt/_ingest.py`: remote-source preparation and caching
- `lakeprompt/_profiler.py`: profiling, join discovery, and join-path ranking
- `lakeprompt/_retrieval.py`: semantic retrieval over `ColumnCard`s
- `lakeprompt/_executor.py`: execution coordination and tuple packaging
- `lakeprompt/_executor_sql.py`: SQL compilation, query-plan application, and output-column metadata
- `lakeprompt/_executor_ranking.py`: row scoring and duplicate suppression
- `lakeprompt/_packager.py`: evidence packaging into TOON prompts
- `lakeprompt/_llm_utilities.py`: Anthropic-backed table summaries and query planning
- `lakeprompt/_models.py`: shared dataclasses
- `lakeprompt/_tracing.py`: stdout pipeline logger

## Public API

```python
from lakeprompt import LakePrompt
```

Main entry points:

- `LakePrompt(...)`: initialize from a local CSV directory
- `LakePrompt.from_url(...)`: initialize from a supported remote source
- `LakePrompt.query(question)`: run the full retrieval/planning/execution pipeline

`LakePrompt.query(...)` currently returns a `LakeAnswer` with:

- `text`
- `evidence`
- `cited_ids`
- `prompt`

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Set your Anthropic key for table summarization, query planning, evaluation helpers, and final answering:

```bash
export ANTHROPIC_API_KEY="your_api_key_here"
```

Optional but useful for model downloads:

```bash
export HF_TOKEN="your_hf_token_here"
```

## Usage

### Local lake

```python
from lakeprompt import LakePrompt

lp = LakePrompt(
    lake_dir="./data/department_management",
    model="claude-sonnet-4-20250514",
    logger=True,
)

answer = lp.query("What are the distinct creation years of the departments managed by a secretary born in state 'Alabama'?")
print(answer.text)
print(answer.cited_ids)
print(answer.evidence)
```

### Remote source

```python
from lakeprompt import LakePrompt

lp = LakePrompt.from_url(
    "https://github.com/your-org/your-data-repo",
    model="claude-sonnet-4-20250514",
    logger=True,
)
```

Supported remote sources:

- direct `.csv` URLs
- direct `.zip` URLs containing CSV files
- GitHub repository URLs, including `.../tree/<branch>` links

## Debugging One Question

Use the included debug runner to trace one question with full logging:

```bash
source .venv/bin/activate
export ANTHROPIC_API_KEY="your_api_key_here"
./.venv/bin/python debug_one_question.py
```

This is useful when you want to inspect:

- semantic retrieval results
- planner-visible schema context
- raw / coerced / sanitized query plans
- refined SQL
- final evidence rows

## Running Evaluation

The evaluation runner compares several baselines:

- `no_context`
- `single_table`
- `naive_multitable`
- `schema_baseline`
- `join_no_ranking`
- `lakeprompt_ranked`

### Recommended command

Use the ground-truth-enriched question file:

```bash
source .venv/bin/activate
export ANTHROPIC_API_KEY="your_api_key_here"

python eval/evaluation.py \
  --dataset-root ./data \
  --questions-file ./eval/spider_join_questions_with_ground_truth.json \
  --output-json ./eval/results.json \
  --output-txt ./eval/results.md
```

Useful flags:

- `--max-questions 100`: run only a subset
- `--sample-rows 10`: increase rows shown for the simpler baselines
- `--claude-model claude-sonnet-4-20250514`
- `--lakeprompt-model claude-sonnet-4-20250514`
- `--cache-path ./eval/summary_cache.json`

The human-readable report is Markdown, even though the CLI flag remains `--output-txt`.

### If you only have Spider SQL and no ground truth

You can still run evaluation with:

```bash
python eval/evaluation.py \
  --dataset-root ./data \
  --questions-file ./eval/spider_join_questions.json \
  --output-json ./eval/results.json \
  --output-txt ./eval/results.md
```

The evaluator will derive ground truth live from the expected Spider SQL during the run when `query` is present.

### Optional offline ground-truth generation

If you want a fully enriched question file ahead of time:

```bash
./.venv/bin/python eval/generate_ground_truth.py \
  --input-json ./eval/spider_join_questions.json \
  --dataset-root ./data \
  --output-json ./eval/spider_join_questions_with_ground_truth.json
```

This loads each schema into in-memory SQLite, executes the Spider SQL, and stores the resulting answer string as `ground_truth`.

## Evaluation Outputs

The evaluation report includes:

- run metadata
- a metric glossary
- aggregate metrics per baseline
- per-example sections with:
  - question
  - expected SQL
  - shared LakePrompt generated SQL
  - each baseline prompt
  - each baseline response
  - per-example scoring metrics

Generated files:

- `results.json`: structured output
- `results.md`: human-readable Markdown report

## Caching Behavior

- `save_artifacts=True` by default
- if `cache_path` is not supplied, table summaries are written under `.lakeprompt_cache/table_summaries/`
- prepared remote sources are cached under `.lakeprompt_cache/` unless `source_cache_dir` is provided
- summary cache files are written incrementally

## Logger Behavior

Passing `logger=True` enables stdout tracing. Current trace output includes:

- retrieved semantic columns
- expanded planner schema context
- LLM prompts and responses
- parsed and transformed query plans
- ranked join paths
- refined SQL
- executed SQL
- ranked evidence rows
- final prompt context metadata

## Behavior Notes

- if no evidence is found, `query(...)` returns `"Could not find evidence in the data lake"` and does not call the final answer LLM
- join execution is limited to the top-ranked candidate paths instead of exhaustively running every path
- Spider-style multi-database lakes are constrained by namespace-aware join planning to reduce false cross-database joins
- the final prompt uses user-friendly field names, but retains exact internal key mappings in `field_map`
- evaluation scoring can compare against live SQL-derived ground truth when the expected Spider SQL is present

## Constraints

- Anthropic is required for multiple pipeline stages
- embeddings use `sentence-transformers`
- nearest-neighbor retrieval uses `hnswlib`
- the pinned dependency set is relatively heavy
- some schemas in the local CSV export may not perfectly match the original Spider relational schemas

## Acknowledgments

- Spider Join Data: https://github.com/superctj/spider-join-data
- Spider paper: https://aclanthology.org/D18-1425/
- Spider dataset repository: https://github.com/taoyds/spider
- TOON specification: https://github.com/toon-format/spec
- TOON reference implementation: https://github.com/toon-format/toon
