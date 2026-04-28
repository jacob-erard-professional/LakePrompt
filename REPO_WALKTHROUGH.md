# LakePrompt Repo Walkthrough

This document is a step-by-step tour of the repository. It starts with the Python files at the repo level and in `eval/`, then moves into the `lakeprompt/` package and explains how the main classes fit together at runtime.

## 1. Start From The Entrypoints

### `run.py`

Purpose:
- Main interactive CLI for the project.
- Accepts a local directory path or remote URL through `--database_link`.
- Builds a `LakePrompt` instance, installs a compact progress logger, prints loaded tables, and opens a REPL.

What matters in this file:
- `CliPipelineTracker(PipelineLogger)` is a small UI-focused logger. It translates internal pipeline section names into cleaner CLI stage messages.
- `_build_lakeprompt(...)` decides between `LakePrompt(...)` and `LakePrompt.from_url(...)`.
- `repl(...)` is the loop that repeatedly calls `lp.query(question)`.

Use this file when:
- You want the “real” interactive experience.
- You want progress updates for each query stage.

### `demo.py`

Purpose:
- A simpler demo CLI than `run.py`.
- Supports either `--lake-dir` or `--source-url`.
- Can optionally print the final prompt and executed SQL.

What matters in this file:
- It is still just a thin wrapper around `LakePrompt`.
- `_print_answer(...)` exposes more debugging detail than `run.py`, especially evidence rows and path metadata.

Use this file when:
- You want to inspect answers, evidence rows, and SQL more directly.

### `eval/generate_ground_truth.py`

Purpose:
- Offline utility for executing known Spider SQL queries against local CSV schemas and storing the resulting answer as ground truth.

What it does:
- Loads each schema’s CSV files into an in-memory SQLite database.
- Executes the provided SQL query.
- Serializes the result into a canonical string or JSON string.

Use this file when:
- You want reference answers for evaluation without calling LakePrompt.

### `eval/update_metric_names.py`

Purpose:
- Small maintenance script that renames old metric labels in a markdown report.

What it does:
- Reads a markdown file.
- Replaces `token_f1` wording with `ground_truth_coverage`.

Use this file when:
- You need to update older evaluation output docs.

### `eval/evaluation.py`

Purpose:
- Main benchmark runner for Spider-style join evaluation.
- Generates or loads examples, runs different answering conditions, and computes metrics.

Main classes in this file:
- `ConditionResult`: one condition’s answer plus metrics.
- `EvaluationExample`: one benchmark example and all condition outputs.
- `ClaudeCompletion`: response container for Anthropic usage data.
- `ClaudeClient`: small wrapper around the Anthropic API for evaluation-time calls.
- `SpiderJoinEvaluation`: the orchestration class for the entire benchmark pipeline.

Use this file when:
- You want to compare LakePrompt against baselines.
- You want JSON and text evaluation reports.

### `lakeprompt/__init__.py`

Purpose:
- Exposes the public package API.

What it exports:
- `LakePrompt`

That means external code is expected to start here:

```python
from lakeprompt import LakePrompt
```

## 2. The Core Runtime Flow

The main runtime path is:

1. `run.py` or `demo.py` builds a `LakePrompt`.
2. `LakePrompt.__init__()` loads data, profiles schema, generates summaries, and builds retrieval indexes.
3. `LakePrompt.query(question)` runs retrieval, planning, join search, SQL execution, evidence ranking, prompt packaging, and final answering.

In short:

`CLI -> LakePrompt -> DataLake -> LakeProfiler -> SemanticRetriever -> TupleExecutor -> ContextPackager -> Anthropic`

## 3. `lakeprompt/` Package By Class

### `LakePrompt` in `lakeprompt/_lakeprompt.py`

Purpose:
- This is the main orchestrator and public entrypoint for the package.

How it is used:
- `run.py` and `demo.py` construct it directly for local data.
- `LakePrompt.from_url(...)` is used for remote CSV/ZIP/GitHub sources.
- `LakePrompt.query(question)` is the central method the rest of the repo depends on.

What it owns:
- `self.lake`: a `DataLake`
- `self.profiler`: a `LakeProfiler`
- `self.retriever`: a `SemanticRetriever`
- `self.executor`: a `TupleExecutor`
- `self.packager`: a `ContextPackager`
- `self.logger`: a `PipelineLogger`

Query flow inside `LakePrompt.query(...)`:
1. `SemanticRetriever.find_columns(...)` gets relevant `ColumnCard`s.
2. `_expand_cards_for_planning(...)` adds nearby join-neighbor tables for the planner.
3. `_plan_query(...)` calls the LLM to build a `QueryPlan`.
4. `LakeProfiler.get_join_paths(...)` builds candidate `JoinPath`s.
5. `TupleExecutor.get_tuples(...)` executes top paths and returns ranked `JoinedTuple`s.
6. `ContextPackager.build_context(...)` turns evidence into a final prompt.
7. `_llm_complete(...)` asks Anthropic for the final answer and citations.
8. The result is returned as `LakeAnswer`.

### `DataLake` in `lakeprompt/_datalake.py`

Purpose:
- Owns the loaded CSV lake and the SQLite database used for execution.

How it is used:
- Built first by `LakePrompt`.
- Read by almost every later stage.

What it does:
- Recursively finds CSV files under a lake directory.
- Loads them into a temporary SQLite database.
- Creates stable table names, including nested-path names like `a/b/customers.csv -> a__b__customers`.
- Exposes helpers like `query(...)`, `get_sample(...)`, `get_table_columns(...)`, `get_column_dtype(...)`, and `get_column_values(...)`.

Why it matters:
- It is the shared data substrate for profiling, join discovery, SQL execution, and evaluation.

### `_DataLakePreparer` and `_PreparedLake` in `lakeprompt/_ingest.py`

Purpose:
- Prepare remote sources so LakePrompt can treat them like a local CSV lake.

How they are used:
- Only through `LakePrompt.from_url(...)`.

What they do:
- Detect whether a URL is a direct CSV, ZIP, or GitHub repo URL.
- Download it into `.lakeprompt_cache/`.
- Extract CSV files, including nested ZIPs.
- Return a `_PreparedLake` describing the prepared local directory.

Why it matters:
- It is the bridge between remote datasets and the rest of the local pipeline.

### `LakeProfiler` in `lakeprompt/_profiler.py`

Purpose:
- Turns raw tables into profile metadata and a join graph.

How it is used:
- Created during `LakePrompt.__init__()`.
- Called once up front through `profile()`.
- Called at query time through `get_join_paths(...)`.

What it produces:
- `ColumnCard` objects for every column.
- `jaccard_matches` between likely join columns.
- `lake.join_graph`, a table-level graph of joinable edges.

Key methods:
- `profile()`: builds all `ColumnCard`s and the join graph.
- `build_join_graph(...)`: converts column matches into table graph edges.
- `get_join_paths(...)`: searches the join graph and returns ranked `JoinPath`s.

Why it matters:
- It is where LakePrompt moves from “a pile of CSVs” to “a searchable multi-table graph.”

### `SemanticRetriever` in `lakeprompt/_retrieval.py`

Purpose:
- Finds the columns most relevant to a natural-language question.

How it is used:
- Built during `LakePrompt.__init__()`.
- `build_index()` is called once after table summaries are generated.
- `find_columns(question)` is the first step of `LakePrompt.query(...)`.

What it does:
- Embeds `ColumnCard` descriptions with SentenceTransformers.
- Builds an HNSW index with `hnswlib`.
- Merges semantic similarity with lexical overlap, so exact schema words still matter.

Why it matters:
- It narrows the search space before join planning and SQL execution.

### `TupleExecutor` in `lakeprompt/_executor.py`

Purpose:
- Turns candidate join paths into ranked evidence rows.

How it is used:
- Called from `LakePrompt.query(...)` after join paths are found.

What it delegates to:
- `SqlBuilder`
- `QueryPlanApplier`
- `RowRanker`

Runtime flow:
1. For each top join path, `_execute_path(...)` builds base SQL.
2. `QueryPlanApplier.apply(...)` refines the SQL using the `QueryPlan`.
3. `DataLake.query(...)` executes the SQL.
4. `RowRanker.score_rows(...)` scores returned rows.
5. `RowRanker.select_diverse_rows(...)` removes near-duplicates.
6. `_to_joined_tuples(...)` wraps rows as `JoinedTuple`.

Why it matters:
- This is the stage where abstract join candidates become concrete evidence.

### `SqlBuilder` in `lakeprompt/_executor_sql.py`

Purpose:
- Compiles a `JoinPath` into executable SQLite SQL.

How it is used:
- Usually called from `TupleExecutor._execute_path(...)`.
- Also used indirectly through `QueryPlanApplier`.

What it does:
- Builds the base `SELECT ... FROM ... JOIN ... ON ...` skeleton.
- Handles duplicate output-column names.
- Produces `OutputColumn` metadata so later stages know how result keys map back to logical fields.

Why it matters:
- It preserves the chosen join structure before question-specific refinement is applied.

### `QueryPlanApplier` in `lakeprompt/_executor_sql.py`

Purpose:
- Applies structured intent from the LLM to a base SQL query.

How it is used:
- Called by `TupleExecutor` after `SqlBuilder` creates the join skeleton.

What it does:
- Accepts a `QueryPlan`.
- Coerces filter values to the right types when possible.
- Sanitizes invalid plan fields against the actual schema.
- Recompiles the SQL with filters, projections, grouping, having, ordering, and limit.

Why it matters:
- It lets the LLM control query intent without letting it invent free-form SQL.

### `RowRanker` in `lakeprompt/_executor_ranking.py`

Purpose:
- Scores materialized rows and suppresses duplicates.

How it is used:
- Instantiated inside `TupleExecutor.get_tuples(...)`.

What it does:
- Embeds the user question.
- Scores each row using semantic similarity plus light coverage bonuses.
- Keeps a diverse top set instead of many near-identical rows.

Why it matters:
- It improves the quality of the evidence passed to the final answering prompt.

### `ContextPackager` in `lakeprompt/_packager.py`

Purpose:
- Converts ranked evidence into a prompt for the final LLM answer.

How it is used:
- Called near the end of `LakePrompt.query(...)`.

What it does:
- Trims evidence to fit a token budget.
- Creates shared source metadata and row schemas.
- Renames internal row keys into more readable display keys.
- Uses TOON encoding to render the final prompt.
- Returns a `LakeContext`.

Why it matters:
- This is the boundary between retrieval/execution and answer generation.

### `PipelineLogger` in `lakeprompt/_tracing.py`

Purpose:
- Lightweight structured logger for pipeline debugging.

How it is used:
- Created in `LakePrompt`.
- Reassigned in `run.py` to `CliPipelineTracker` for nicer CLI output.
- Passed into profiler, retriever, executor, and packager.

Why it matters:
- It makes the pipeline inspectable without changing the core logic.

## 4. Shared Model Classes In `lakeprompt/_models.py`

These classes mostly carry state between stages.

### Query and answer objects

- `QueryFilter`: one structured predicate.
- `QuerySelect`: one projected column, optionally aggregated.
- `QueryOrder`: one sort instruction.
- `QueryPlan`: the LLM-produced structured interpretation of the question.
- `LakeContext`: packaged prompt plus chosen evidence.
- `LakeAnswer`: final return object from `LakePrompt.query(...)`.

How they are used:
- `plan_llm_query(...)` builds a `QueryPlan`.
- `QueryPlanApplier` turns the `QueryPlan` into SQL.
- `ContextPackager` turns ranked evidence into a `LakeContext`.
- `LakePrompt._llm_complete(...)` turns that context into a `LakeAnswer`.

### Join and evidence objects

- `JoinEdge`: one join edge between two tables.
- `JoinPath`: a candidate multi-table execution path.
- `JoinProvenance`: path metadata attached to an evidence row.
- `OutputColumn`: metadata for one output field produced by SQL execution.
- `JoinedTuple`: one evidence row plus provenance and ranking score.

How they are used:
- `LakeProfiler.get_join_paths(...)` creates `JoinPath`s and `JoinEdge`s.
- `TupleExecutor` executes a `JoinPath`.
- `SqlBuilder` and `QueryPlanApplier` emit `OutputColumn`s.
- `TupleExecutor._to_joined_tuples(...)` creates `JoinedTuple`s with `JoinProvenance`.
- `ContextPackager` packages `JoinedTuple`s into the final prompt.

### Schema-profile object

- `ColumnCard`: profile of one column.

How it is used:
- `LakeProfiler.profile()` creates it.
- `SemanticRetriever` embeds and retrieves it.
- `plan_llm_query(...)` sends selected cards to the LLM as schema context.
- `LakeProfiler.get_join_paths(...)` uses it indirectly through the join graph and plan coverage.

## 5. LLM Utilities

### Functions in `lakeprompt/_llm_utilities.py`

Main responsibilities:
- `generate_table_summaries(...)`: summarize each table for better retrieval.
- `plan_llm_query(...)`: convert a question into a structured `QueryPlan`.
- `_package_with_toon(...)`: serialize structured prompt data in TOON format.

How they are used:
- `LakePrompt.__init__()` calls `generate_table_summaries(...)`.
- `LakePrompt._plan_query(...)` calls `plan_llm_query(...)`.
- `ContextPackager` calls `_package_with_toon(...)`.

Why this module matters:
- It contains the LLM-facing glue code, but keeps that code outside the orchestration and execution classes.

## 6. Putting It Together As One Query

If a user runs `python run.py --database_link ...` and asks one question, the path is:

1. `run.py` parses args and builds `LakePrompt`.
2. `LakePrompt` loads CSVs through `DataLake`.
3. `LakeProfiler.profile()` creates `ColumnCard`s and a join graph.
4. `generate_table_summaries(...)` adds table-level descriptions.
5. `SemanticRetriever.build_index()` creates the retrieval index.
6. The user asks a question.
7. `SemanticRetriever.find_columns(...)` picks likely columns.
8. `plan_llm_query(...)` produces a `QueryPlan`.
9. `LakeProfiler.get_join_paths(...)` finds candidate `JoinPath`s.
10. `TupleExecutor` runs SQL for the best paths.
11. `RowRanker` scores and deduplicates rows.
12. `ContextPackager.build_context(...)` builds the final prompt.
13. `LakePrompt._llm_complete(...)` asks Anthropic for the answer.
14. `LakePrompt` returns a `LakeAnswer`.
15. `run.py` prints the answer, cited evidence IDs, and evidence rows.

## 7. Fast Mental Model

If you want the shortest useful mental model of the repo:

- `run.py` and `demo.py` are wrappers.
- `LakePrompt` is the orchestrator.
- `DataLake` loads and queries CSV-backed SQLite tables.
- `LakeProfiler` figures out what columns and joins exist.
- `SemanticRetriever` decides what parts of the schema matter for a question.
- `QueryPlan` captures what the question is asking for.
- `TupleExecutor` turns join paths into ranked evidence rows.
- `ContextPackager` turns evidence into a final prompt.
- `LakeAnswer` is the final returned object.
