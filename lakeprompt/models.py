from dataclasses import dataclass, field
from typing import Any

@dataclass
class JoinPath:
    """
    A candidate join path across two or more tables.

    Produced by DataProfiler.get_join_paths() and consumed by
    TupleExecutor.get_tuples().
    """
    tables: list[str]
    join_keys: list[tuple[str, str, str, str]]  # (t1, col1, t2, col2)
    score: float
    estimated_output_rows: int


@dataclass
class JoinedTuple:
    """
    A single row of evidence produced by executing a join path.

    Produced by TupleExecutor and consumed by ContextPackager.
    """
    evidence_id: str
    data: dict[str, Any]
    provenance: list[str]
    join_path: JoinPath
    relevance_score: float


@dataclass
class LakeContext:
    """
    A fully packaged prompt ready to send to the LLM.

    Produced by ContextPackager and consumed by LakePrompt._llm_complete().
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
    """
    text: str
    evidence: list[JoinedTuple]
    
@dataclass
class ColumnCard:
    """
    A profile of a single column within a table. This class is used
    to store information on columns as the tables go through the pipeline.

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
    jaccard_matches: dict[tuple[str, str], float] = field(default_factory=dict)
    embedding: list[float] | None = None
