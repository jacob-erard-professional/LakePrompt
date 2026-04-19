from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class JoinEdge:
    """
    A single directed join edge between two tables.

    Stored in join paths and provenance so later pipeline stages can
    reconstruct the exact join chain used to produce a row.

    Attributes:
        left_table: Source table for the edge.
        left_column: Source column used in the join.
        right_table: Destination table for the edge.
        right_column: Destination column used in the join.
        score: Joinability score, currently derived from Jaccard overlap.
    """

    left_table: str
    left_column: str
    right_table: str
    right_column: str
    score: float


@dataclass
class JoinPath:
    """
    A candidate join path across two or more tables.

    Produced by `LakeProfiler.get_join_paths()` and consumed by
    `TupleExecutor.get_tuples()`. The object is needed because the
    executor requires an ordered, inspectable plan rather than a loose
    collection of related tables.

    Attributes:
        path_id: Stable identifier for the candidate path.
        tables: Ordered list of tables in execution order.
        join_keys: Ordered join key tuples for SQL compilation.
        score: Planner score for the path.
        estimated_output_rows: Heuristic output-size estimate.
        join_edges: Rich edge metadata for the path.
    """
    path_id: str
    tables: list[str]
    join_keys: list[tuple[str, str, str, str]]  # (t1, col1, t2, col2)
    score: float
    estimated_output_rows: int | None
    join_edges: list[JoinEdge] = field(default_factory=list)


@dataclass
class JoinProvenance:
    """
    Structured provenance for an evidence tuple.

    Carries the exact join path metadata promised by the proposal.

    Attributes:
        path_id: Identifier of the path that produced the row.
        tables: Ordered list of tables used in the path.
        join_keys: Ordered join keys used by the path.
        path_score: Planner score assigned to the source path.
        sql: Final executed SQL query, normalized to a single line for
            easier logging and downstream display.
    """

    path_id: str
    tables: list[str]
    join_keys: list[tuple[str, str, str, str]]
    path_score: float
    sql: str


@dataclass(frozen=True)
class OutputColumn:
    """
    Metadata describing one concrete output column produced by execution.

    Attributes:
        output_key: Actual key present in the executed row payload.
        table: Source table when applicable.
        column: Source column when applicable.
        aggregation: Aggregate wrapper such as `COUNT` or `DISTINCT`.
    """
    output_key: str
    table: str | None
    column: str
    aggregation: str | None = None


@dataclass
class JoinedTuple:
    """
    A single row of evidence produced by executing a join path.

    Produced by `TupleExecutor` and consumed by `ContextPackager`.
    This object is needed because the project returns evidence as
    retrieved rows with explicit provenance, not just as plain text.

    Attributes:
    evidence_id: Stable evidence identifier such as `E1`.
        data: Joined row payload.
        output_columns: Concrete output-column metadata for `data`.
        provenance: Structured description of how the row was produced.
        join_path: Full join path object that produced the row.
        relevance_score: Ranking score assigned by the executor.
    """
    evidence_id: str
    data: dict[str, Any]
    output_columns: list[OutputColumn]
    provenance: JoinProvenance
    join_path: JoinPath
    relevance_score: float


@dataclass
class LakeContext:
    """
    A fully packaged prompt ready to send to the LLM.

    Produced by `ContextPackager` and consumed by `LakePrompt._llm_complete()`.
    This object is needed so the system can carry prompt text, selected
    evidence, and prompt-size metadata together.

    Attributes:
        question: Original user question.
        evidence: Evidence rows included in the prompt.
        prompt: Final prompt string sent to the model.
        token_count: Approximate token count for the prompt.
    """
    question: str
    evidence: list[JoinedTuple]
    prompt: str
    token_count: int


@dataclass
class LakeAnswer:
    """
    The final output of a LakePrompt.query() call.

    Returned to the end user with the answer text and the evidence
    tuples that support it.

    Attributes:
        text: Final answer text.
        evidence: Evidence rows supporting the answer.
        cited_ids: Evidence IDs cited by the model when available.
        prompt: Final prompt string sent to the model when available.
    """
    text: str
    evidence: list[JoinedTuple]
    cited_ids: list[str] = field(default_factory=list)
    prompt: str = ""


@dataclass(frozen=True)
class QueryFilter:
    """
    A structured filter extracted from a natural-language question.

    Attributes:
        column: Target column for the predicate.
        operator: SQL-style comparison operator.
        value: Literal value or values used by the predicate.
        table: Optional table name for disambiguation.
    """

    column: str
    operator: str
    value: Any
    table: str | None = None


@dataclass(frozen=True)
class QuerySelect:
    """
    A projected output column, optionally with aggregation.

    Attributes:
        column: Selected column.
        table: Optional table name for disambiguation.
        aggregation: Optional aggregation such as `SUM` or `COUNT`.
    """

    column: str
    table: str | None = None
    aggregation: str | None = None


@dataclass(frozen=True)
class QueryOrder:
    """
    A structured ordering directive for the final SQL query.

    Attributes:
        column: Column to order by.
        direction: Sort direction, usually `asc` or `desc`.
        table: Optional table name for disambiguation.
        aggregation: Optional aggregate wrapper for the sort key.
    """

    column: str
    direction: str = "asc"
    table: str | None = None
    aggregation: str | None = None


@dataclass
class QueryPlan:
    """
    Structured question intent used to refine a candidate SQL query.

    The plan is needed because the LLM should contribute inspectable intent
    rather than opaque SQL rewrites.

    Attributes:
        filters: Predicates for `WHERE`.
        projections: Requested output columns or aggregations.
        group_by: Grouping columns.
        having: Predicates for `HAVING`.
        order_by: Sorting directives.
        limit: Optional row limit.
    """

    filters: list[QueryFilter] = field(default_factory=list)
    projections: list[QuerySelect] = field(default_factory=list)
    group_by: list[str] = field(default_factory=list)
    having: list[QueryFilter] = field(default_factory=list)
    order_by: list[QueryOrder] = field(default_factory=list)
    limit: int | None = None


@dataclass
class ColumnCard:
    """
    A profile of a single column within a table. This class is used
    to store information on columns as the tables go through the pipeline.
    It is needed because retrieval, join discovery, and LLM planning all
    need the same compact representation of a column.

    Produced by DataProfiler.profile_table() and used throughout the
    LakePrompt pipeline. The embedding field is empty until
    SemanticRetriever.embed_cards() is called.

    Args:
        table_name: Name of the table this column belongs to.
        column_name: Name of the column.
        dtype: Data type of the column (e.g. 'Int64', 'Utf8').
        sample_values: A small list of representative non-null values.
        summary: Auto-generated or LLM-generated description of the column.
        table_summary: LLM-generated description of the whole table.
            Shared across all ColumnCards belonging to the same table.
        uniqueness_ratio: Fraction of non-null sampled values that are unique.
        null_ratio: Fraction of sampled values that are null.
        sequentiality_ratio: Fraction of adjacent sorted numeric deltas equal
            to 1.0 when the column looks numeric.
        surrogate_key_score: Heuristic score that the column behaves like a
            generated row identifier.
        foreign_key_score: Heuristic score that the column behaves like a
            reference into another table.
        jaccard_matches: Mapping from (table_name, column_name) to
            Jaccard similarity for columns above the profiler threshold.
        embedding: Vector embedding of the summary, filled by
            SemanticRetriever. None until embedded.
    """
    table_name: str
    column_name: str
    dtype: str
    sample_values: list[Any] = field(default_factory=list)
    summary: str = ""
    table_summary: str = ""
    uniqueness_ratio: float = 0.0
    null_ratio: float = 0.0
    sequentiality_ratio: float = 0.0
    surrogate_key_score: float = 0.0
    foreign_key_score: float = 0.0
    jaccard_matches: dict[tuple[str, str], float] = field(default_factory=dict)
    embedding: list[float] | None = None
