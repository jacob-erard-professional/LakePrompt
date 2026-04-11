# LakePrompt

LakePrompt is a Python research prototype for answering natural-language questions over a data lake of CSV files. It profiles columns, retrieves semantically relevant columns, plans joins, executes SQL over Polars, packages evidence for an LLM, and returns a grounded answer with supporting evidence.

The repository currently contains the core LakePrompt package plus documentation for the class project proposal and progress.

## Current State

The pipeline currently supports:

- recursive CSV discovery from nested directories
- stable table naming for nested paths such as `region/sales/customers.csv -> region__sales__customers`
- remote data-lake preparation from:
  - direct `.csv` URLs
  - `.zip` archives
  - GitHub repository URLs
- cached table summaries and prepared remote sources under `.lakeprompt_cache/` by default
- semantic column retrieval with `sentence-transformers` and `hnswlib`
- score-window filtering of semantic retrieval results so weak tail matches are dropped
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
- profile-based key heuristics on columns:
  - uniqueness
  - null ratio
  - sequentiality
  - surrogate-key score
  - foreign-key score
- deterministic SQL generation from `QueryPlan`
- typed filter coercion before SQL execution
- row-level evidence ranking and diversity filtering
- prompt-time projection of rows down to plan-relevant columns
- optional stdout tracing for:
  - semantic retrieval
  - Jaccard matches
  - LLM prompts and responses
  - join paths
  - refined SQL
  - executed SQL
  - ranked evidence rows
- local fallback behavior when no evidence is found, without calling the final LLM

## Repository Layout

Top-level visible files and folders:

- `README.md`
- `requirements.txt`
- `Documents/`
- `lakeprompt/`

Key documents:

- `Documents/Final Project Proposal.md`: original proposal
- `Documents/Progress.MD`: running implementation/progress notes

Key package modules:

- `lakeprompt/__init__.py`: package exports
- `lakeprompt/_datalake.py`: internal CSV loading and Polars SQL execution
- `lakeprompt/_ingest.py`: internal remote-source preparation and caching used by `LakePrompt.from_url(...)`
- `lakeprompt/_profiler.py`: internal column profiling, join discovery, and join-path ranking
- `lakeprompt/_retrieval.py`: internal semantic retrieval over `ColumnCard`s
- `lakeprompt/_executor.py`: internal SQL generation, execution, and tuple scoring
- `lakeprompt/_packager.py`: internal evidence packaging into TOON prompts
- `lakeprompt/_llm_utilities.py`: internal Anthropic-backed table summaries and query planning
- `lakeprompt/_lakeprompt.py`: internal implementation module for the public `LakePrompt` class
- `lakeprompt/_models.py`: internal shared dataclasses such as `LakeAnswer`, `JoinPath`, `QueryPlan`, and `ColumnCard`
- `lakeprompt/_tracing.py`: internal stdout tracing logger
- `lakeprompt/evaluation.py`: evaluation runner

## Public API

```python
from lakeprompt import LakePrompt
```

Main entry points:

- `LakePrompt(...)`: initialize from a local CSV directory
- `LakePrompt.from_url(...)`: initialize from a supported remote source
- `LakePrompt.query(question)`: run the full retrieval/planning/execution pipeline

`LakePrompt.query(...)` returns an answer object with:

- `text`
- `evidence`
- `cited_ids`

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

Optional but useful for faster Hugging Face downloads:

```bash
export HF_TOKEN="your_hf_token_here"
```

## Usage

### Local lake

```python
from lakeprompt import LakePrompt

lp = LakePrompt(
    lake_dir="./data",
    model="claude-sonnet-4-20250514",
    logger=True,
)

answer = lp.query("Which customers spent the most in January?")
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

Remote-source preparation is intentionally kept behind `LakePrompt.from_url(...)`. Users do not need to instantiate ingestion helpers directly.
Internal package modules now use underscore-prefixed names to signal that they are not part of the supported public API.

### Caching behavior

- `save_artifacts=True` by default
- if `cache_path` is not supplied, table summaries are written under `.lakeprompt_cache/table_summaries/`
- prepared remote sources are cached under `.lakeprompt_cache/` unless `source_cache_dir` is provided
- summary cache files are written incrementally, so interrupted runs do not lose all progress

### Logger behavior

Passing `logger=True` enables stdout tracing. Current trace output includes:

- profiled column cards
- discovered Jaccard/schema join matches
- semantically retrieved columns
- LLM prompts and responses
- parsed `QueryPlan`
- ranked join paths
- refined SQL as plain text
- executed SQL as plain text
- ranked evidence rows
- final prompt context metadata

## Behavior Notes

- if no evidence is found, `query(...)` returns `"Could not find evidence in the data lake"` and does not call the final LLM
- join execution is limited to the top-ranked candidate paths rather than exhaustively running every path
- Spider-style multi-database lakes are constrained by namespace-aware join planning to reduce false cross-database joins
- join heuristics now use profile-based key signals, not only literal `id` naming

## Changes Made By Codex

The following major repository changes were implemented in this working branch:

- added remote ingestion support for CSV, ZIP, and GitHub sources
- added recursive CSV discovery so nested files are loaded automatically
- added default-on artifact caching and incremental summary-cache writes
- introduced structured `QueryPlan` planning instead of opaque LLM SQL rewrites
- reused `QueryPlan` across join planning, execution, and prompt packaging
- added prompt-time projection to only include plan-relevant evidence columns
- added stdout pipeline tracing with detailed SQL, LLM, retrieval, and join logs
- added Spider-based join-metric tests and expanded join workflow coverage
- hardened LLM parsing for fenced JSON and partial table-summary outputs
- quoted SQL identifiers safely for Polars execution
- improved join planning with:
  - namespace-aware pruning
  - path validation against query-plan-required tables
  - same-namespace schema heuristics
  - profile-based surrogate/foreign-key scoring
- improved executor behavior with:
  - typed filter coercion
  - aggregate aliasing for grouped SQL
  - capped path execution
  - local no-evidence fallback
- updated the docs and progress notes to match the current implementation

## Constraints

- Anthropic is required for several pipeline stages
- embeddings use `sentence-transformers`
- nearest-neighbor retrieval uses `hnswlib`
- the pinned dependency set is relatively heavy
- benchmark data is not committed in the visible repository

## Acknowledgments

- Spider Join Data: https://github.com/superctj/spider-join-data
- Spider paper: https://aclanthology.org/D18-1425/
- Spider dataset repository: https://github.com/taoyds/spider
- TOON specification: https://github.com/toon-format/spec
- TOON reference implementation: https://github.com/toon-format/toon
