# LakePrompt Progress

This document summarizes project status based only on code and files currently present in the git repository.

## What's Been Done

### Core data-lake pipeline

The git repo contains a working pipeline structure across these modules:

- `lakeprompt/datalake.py`
- `lakeprompt/profiler.py`
- `lakeprompt/retrieval.py`
- `lakeprompt/executor.py`
- `lakeprompt/packager.py`
- `lakeprompt/lakeprompt.py`
- `lakeprompt/models.py`

Implemented pieces in the repo:

- CSV loading through a `DataLake` abstraction using Polars SQL
- column profiling with sample values and Jaccard-based joinability detection
- a join graph stored on the lake
- semantic column retrieval using SentenceTransformers and HNSW
- join-path generation in the profiler
- join execution and row-level scoring in the executor
- diversity filtering for evidence rows
- TOON-based prompt packaging
- final answer generation through Anthropic
- returned evidence rows in `LakeAnswer`
- preserved `cited_ids` in `LakeAnswer`

### Models and public API

The repo already exposes:

- `LakePrompt` as the main entry point
- `LakeAnswer` as the return type
- structured evidence objects with provenance metadata

### Evaluation scaffold

The repo includes an evaluation runner in `lakeprompt/evaluation.py` that can:

- load Spider join metadata
- generate questions with Claude
- run a baseline prompt
- run LakePrompt on the same schemas
- write JSON and TXT outputs

### Tests currently present

The repo includes `tests/test_join_workflow.py`, which covers:

- three-table join-path discovery
- row-level tuple scoring
- provenance preservation
- provenance packaging into prompt context

## What's Partially Done

### Join-path workflow alignment

The repo contains join discovery, tuple scoring, and provenance support, but `Documents/Final Project Proposal.md` and the current code are not yet fully aligned on the full intended join-path workflow. This is the area currently being finished.

### Cited answers

The prompt format asks the LLM for `cited_ids`, and the public result now preserves them, but there is not yet a validation layer that checks whether the citations genuinely support the answer.

### Evaluation

The repo has an evaluation runner, but it is still a scaffold rather than a full benchmark-and-metrics system.

## What's Left To Do

### 1. Fixed benchmark data

Still missing:

- a stable benchmark dataset in the repo
- labeled question difficulty categories
- stored ground-truth answers
- stored ground-truth SQL or gold evidence

### 2. Real scoring metrics

Still missing:

- exact match scoring
- F1 scoring
- numeric tolerance scoring
- execution accuracy scoring
- aggregate metric summaries

### 3. Proposal baselines

Still missing:

- no-context baseline
- single-table retrieval baseline
- naive multi-table concatenation baseline
- join-without-ranking baseline

### 4. Faithfulness checks

Still missing:

- answer-to-evidence validation
- unsupported-claim detection
- citation validation against evidence rows

### 5. Ablation support

Still missing:

- experiments that disable joins, filtering, or ranking
- comparisons across different context formats

### 6. Broader automated tests

Still missing:

- tests for benchmark loading
- tests for evaluation metrics
- tests for baselines
- tests for faithfulness and citation validation
- stronger end-to-end tests

## Current Constraints

From the repo as it stands:

- Anthropic is required for table summarization and final answer generation.
- SentenceTransformers and HNSW are required for semantic retrieval.
- The full system depends on external Python packages listed in `requirements.txt`.

## Current Repository Structure

Tracked files relevant to the project:

- `Documents/Final Project Proposal.md`
- `lakeprompt/LLM_utilities.py`
- `lakeprompt/__init__.py`
- `lakeprompt/datalake.py`
- `lakeprompt/evaluation.py`
- `lakeprompt/executor.py`
- `lakeprompt/lakeprompt.py`
- `lakeprompt/models.py`
- `lakeprompt/packager.py`
- `lakeprompt/profiler.py`
- `lakeprompt/retrieval.py`
- `requirements.txt`
- `README.md`

## Bottom Line

Based only on the git repo contents:

- the core LakePrompt pipeline is largely implemented
- the join-path portion is the main core workflow area still being aligned with the proposal
- the largest remaining gap after that is evaluation and comparison infrastructure
