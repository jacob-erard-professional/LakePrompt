# LakePrompt Progress

This document summarizes project status from the public repository surface only: files currently visible in the repo, excluding anything covered by `.gitignore`.

## Current Public Surface

A public user currently sees:

- `README.md`
- `requirements.txt`
- `Documents/Final Project Proposal.md`
- `Documents/Progress.MD`
- `lakeprompt/__init__.py`
- `lakeprompt/LLM_utilities.py`
- `lakeprompt/datalake.py`
- `lakeprompt/ingest.py`
- `eval/evaluation.py`
- `lakeprompt/executor.py`
- `lakeprompt/lakeprompt.py`
- `lakeprompt/models.py`
- `lakeprompt/packager.py`
- `lakeprompt/_profiler.py`
- `lakeprompt/retrieval.py`
- `lakeprompt/tracing.py`

Ignored items such as `tests/`, local caches, datasets, and demo files are intentionally not part of this summary.

## What Is Implemented

### Core local-lake pipeline

The public package contains a working end-to-end local CSV workflow:

- recursive CSV discovery through nested directories
- stable table naming for nested CSV paths
- lazy CSV loading and SQL execution through Polars
- column profiling with sample values and basic column statistics
- Jaccard-based joinability signals and join-graph construction
- semantic retrieval over column cards using SentenceTransformers and HNSW
- join-path generation
- join execution and row-level relevance scoring
- evidence diversity filtering and packaging
- TOON-based prompt serialization
- final answer generation through Anthropic

### Public API and data models

The visible exported API currently includes:

- `LakePrompt`

The internal models layer includes structured types for:

- join edges and join paths
- joined-tuple provenance
- packaged prompt context
- structured query plans with filters, projections, grouping, ordering, and limits

### Remote-source ingestion

`LakePrompt.from_url(...)` uses internal ingestion helpers in `lakeprompt/_ingest.py` to normalize:

- direct CSV URLs
- ZIP archives containing CSV files
- GitHub repositories, including branch-specific URLs

Prepared sources are cached locally under `.lakeprompt_cache/` when artifacts are enabled.

### Logging and artifacts

The public code supports:

- optional structured stdout tracing through `PipelineLogger`
- cached table summaries
- cached prepared remote-source lakes
- disabling summary-cache writes through `save_artifacts=False`

### Evaluation scaffold

The repo includes `eval/evaluation.py`, which can:

- read Spider-style join metadata
- generate join-requiring questions with Claude
- run a schema-context baseline
- run LakePrompt on the same schema
- write JSON and TXT evaluation outputs

## What Is Partially Done

### Proposal alignment

The public code now covers most of the pipeline described in `Documents/Final Project Proposal.md`, including join discovery, joined-tuple retrieval, provenance, and context packaging. The remaining mismatch is mostly around evaluation depth and rigor rather than the core retrieval pipeline.

### Evidence-grounded answers

`LakeAnswer` preserves both evidence rows and `cited_ids`, and the prompt format asks the model to cite evidence IDs. What is still missing is a validation layer that checks whether cited IDs actually support the answer.

### Query planning

The public code includes LLM-based extraction of a structured `QueryPlan`, and downstream stages use that plan to guide join selection and row projection. This is implemented, but it still depends on heuristic execution and graceful fallback rather than a fully validated semantic parser.

### Evaluation

The evaluation module is real and runnable, but it is still a scaffold:

- it generates questions dynamically instead of using a committed benchmark
- it records examples rather than computing a full metrics suite
- it depends on external datasets that are not included in the repo

## What Is Still Missing

### Benchmarks in the repo

The public repo does not currently include:

- a committed benchmark dataset
- stored ground-truth answers
- stored ground-truth SQL
- stored gold evidence rows

### Metric computation

The public evaluation code does not yet provide:

- exact-match scoring
- F1 or partial-credit scoring
- numeric-tolerance scoring
- execution-accuracy scoring
- aggregate benchmark summaries

### Baseline breadth

The proposal mentions several baselines that are not yet fully implemented in the public codebase:

- no-context LLM baseline
- single-table retrieval baseline
- naive multi-table concatenation baseline
- join-without-ranking baseline

### Faithfulness checks

Still missing from the visible repo:

- answer-to-evidence validation
- unsupported-claim detection
- citation validation against returned evidence rows

### Public automated tests

No public automated test suite is currently visible in the repo because the `tests/` path is ignored and therefore outside the public repository surface described here.

## Current Constraints

From the public code and dependency list:

- Anthropic is required for multiple stages, including summarization, planning, evaluation question generation, and final answering
- semantic retrieval depends on `sentence-transformers` and `hnswlib`
- the pinned dependency set is relatively heavy and currently includes many ML and CUDA packages
- remote-source workflows and LLM-backed steps require network access at runtime

## Bottom Line

Based on the visible repository contents, LakePrompt already exposes a substantial prototype:

- the main local and remote ingestion paths are implemented
- the join-aware retrieval and evidence-packaging pipeline is implemented
- structured result objects, logging, and evaluation scaffolding are present

The largest remaining public gaps are benchmark assets, scoring infrastructure, citation validation, and broader public-facing validation of the system.
