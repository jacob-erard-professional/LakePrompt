"""
TupleExecutor — Stream S3 of the LakePrompt pipeline.

Receives JoinPath objects, executes the joins against the DataLake,
ranks the resulting rows, and returns the top-r rows as JoinedTuple
evidence objects for the packager.
"""

import logging
logger = logging.getLogger(__name__)

from dataclasses import dataclass
from .datalake import DataLake
from .models import JoinPath, JoinedTuple


DEFAULT_TOP_R = 20

# model cache so SentenceTransformer is only loaded once per process.
_ST_MODEL: object = None


@dataclass
class TupleExecutor:
    """
    Executes join paths against the DataLake and returns ranked evidence tuples.

    Rows are scored by the average cosine similarity between the question
    embedding and the ColumnCard embeddings for the path's tables.  All rows
    from the same join path receive the same score, so ranking differentiates
    across paths rather than within them.

    Args:
        lake: An initialised DataLake instance.
    """

    lake: DataLake

    # Public API
    def get_tuples(
        self,
        question: str,
        paths: list[JoinPath],
        top_r: int = DEFAULT_TOP_R,
    ) -> list[JoinedTuple]:
        """
        Execute join paths and return the top-r most relevant evidence tuples.

        Args:
            question: The natural language question driving retrieval.
            paths: Join paths produced by the profiler/retriever.
            top_r: Maximum number of JoinedTuple objects to return.

        Returns:
            A list of at most top_r JoinedTuple objects, sorted by descending
            relevance score.
        """
        # Embed the question once; reuse for every path.
        q_emb = self._embed_question(question)

        # Accumulate (score, row, path) across all join paths so we can rank
        # globally and pick the best top_r tuples regardless of which path
        # they came from.
        all_scored: list[tuple[float, dict, JoinPath]] = []

        for path in paths:
            # TODO: push down filters extracted from the question before executing
            # join to have smaller outputs.
            # See _extract_filters (not yet implemented).
            rows = self._execute_path(path)
            if not rows:
                continue

            for score, row in self._rank_by_card_similarity(rows, q_emb, path):
                all_scored.append((score, row, path))

        # Sort globally so the highest-scoring rows from any path rise to the top.
        all_scored.sort(key=lambda x: x[0], reverse=True)
        return self._to_joined_tuples(all_scored[:top_r])

    # SQL query building
    def _build_join_sql(self, path: JoinPath, filter_clause: str = "") -> str:
        """
        Convert a JoinPath into a SQL SELECT ... JOIN ... ON ... string.

        Duplicate column names across tables are aliased as <table>__<col>
        to prevent dict key collisions.

        Args:
            path: The join path to compile.
            filter_clause: Optional SQL WHERE fragment (no 'WHERE' keyword).

        Returns:
            A complete SQL query string ready for DataLake.query().

        Raises:
            ValueError: If path.tables contains fewer than two entries.
        """
        tables: list[str] = path.tables
        join_keys: list[tuple[str, str, str, str]] = path.join_keys  # (t1, col1, t2, col2)

        if len(tables) < 2:
            raise ValueError(
                f"JoinPath must reference at least two tables, got: {tables}"
            )

        # Get (table, col, alias) triples for every column across all tables.
        # Columns that appear in multiple tables get a disambiguating alias.
        col_aliases = self._build_column_aliases(tables)

        # Build explicit SELECT col [AS alias] items instead of SELECT *.
        # This avoids duplicate column names in the result when two tables
        # share a column name (e.g. both have customer_id).
        select_parts: list[str] = []
        for tbl, col, alias in col_aliases:
            if alias != col:
                select_parts.append(f"{tbl}.{col} AS {alias}")
            else:
                select_parts.append(f"{tbl}.{col}")

        # Start with the first (leftmost) table, then chain joins.
        sql = f"SELECT {', '.join(select_parts)}\nFROM {tables[0]}"

        for t1, col1, t2, col2 in join_keys:
            sql += f"\nJOIN {t2} ON {t1}.{col1} = {t2}.{col2}"

        if filter_clause:
            sql += f"\nWHERE {filter_clause}"

        return sql

    def _build_column_aliases(
        self, tables: list[str]
    ) -> list[tuple[str, str, str]]:
        """
        Return (table, column, alias) triples for all columns across tables.

        Columns that appear in more than one table are aliased as
        '<table>__<column>' to avoid key collisions in result dicts.

        Args:
            tables: Ordered list of table names in the join.

        Returns:
            List of (table_name, column_name, alias) tuples.
        """
        # First pass: count how many tables each column name appears in.
        # Any column with a count > 1 needs an alias to avoid collisions.
        col_count: dict[str, int] = {}
        table_cols: dict[str, list[str]] = {}

        for tbl in tables:
            if tbl not in self.lake.tables:
                logger.warning("Table '%s' not found in lake; skipping.", tbl)
                table_cols[tbl] = []
                continue
            # Fetch a single row just to get the column names — no data needed.
            cols = self.lake.get_sample(tbl, n=1).columns
            table_cols[tbl] = cols
            for col in cols:
                col_count[col] = col_count.get(col, 0) + 1

        # Second pass: build the alias list. Columns shared across tables get
        # prefixed with their table name (e.g. customers__customer_id).
        result: list[tuple[str, str, str]] = []
        for tbl in tables:
            for col in table_cols.get(tbl, []):
                alias = f"{tbl}__{col}" if col_count.get(col, 1) > 1 else col
                result.append((tbl, col, alias))

        return result

    # Execution of joins and retrieval of rows
    def _execute_path(self, path: JoinPath, filter_clause: str = "") -> list[dict]:
        """
        Build and execute SQL for a join path, returning rows as list[dict].

        Args:
            path: The join path to execute.
            filter_clause: Optional SQL WHERE fragment.

        Returns:
            A (possibly empty) list of row dicts.
        """
        try:
            sql = self._build_join_sql(path, filter_clause)
        except ValueError as exc:
            logger.warning("Skipping malformed JoinPath: %s", exc)
            return []

        logger.debug("Executing SQL:\n%s", sql)

        try:
            result_df = self.lake.query(sql)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Query failed for path %s: %s", path.tables, exc)
            return []

        # lake.query() returns different types depending on the backend.
        # Normalise to list[dict] immediately so all downstream code is the same regardless of backend.
        if self.lake.backend == "spark":
            rows = [row.asDict() for row in result_df.collect()]
        else:
            rows = result_df.to_dicts()

        if not rows:
            logger.warning("Join path %s produced zero rows.", path.tables)

        return rows

    # Ranking rows based on ColumnCard similarity
    def _embed_question(self, question: str) -> "np.ndarray | None":  # type: ignore[name-defined]
        """
        Encode the question with SentenceTransformer and return a unit vector.

        Returns None if sentence-transformers is not installed.
        """
        global _ST_MODEL
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]
        except ImportError:
            logger.warning(
                "sentence-transformers not installed; card-similarity ranking unavailable."
            )
            return None

        if _ST_MODEL is None:
            _ST_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

        return _ST_MODEL.encode(question, normalize_embeddings=True)  # type: ignore[union-attr]

    def _rank_by_card_similarity(
        self,
        rows: list[dict],
        q_emb: "np.ndarray | None",  # type: ignore[name-defined]
        path: JoinPath,
    ) -> list[tuple[float, dict]]:
        """
        Score rows by the average cosine similarity between the question and
        the ColumnCard embeddings for the path's tables.

        All rows in a path receive the same score because they share the same
        set of contributing tables/cards.  Ranking therefore differentiates
        across paths, not within them.

        Falls back to score 0.0 per row if embeddings are unavailable.

        Args:
            rows: Raw row dicts from _execute_path.
            q_emb: Unit-vector question embedding, or None if unavailable.
            path: The join path whose tables determine which cards to use.

        Returns:
            List of (score, row) pairs sorted by descending score.
        """
        score = 0.0

        if q_emb is not None:
            try:
                import numpy as np
            except ImportError:
                pass
            else:
                path_table_set = set(path.tables)
                embedded_cards = [
                    card
                    for card in (getattr(self.lake, "cards", None) or [])
                    if card.table_name in path_table_set and card.embedding is not None
                ]

                if not embedded_cards:
                    logger.warning(
                        "No embedded ColumnCards found for path %s; scoring rows as 0.0.",
                        path.tables,
                    )
                else:
                    card_matrix = np.array([card.embedding for card in embedded_cards])
                    # Normalise in case cards were stored without unit-length guarantee.
                    norms = np.linalg.norm(card_matrix, axis=1, keepdims=True)
                    norms = np.where(norms == 0, 1.0, norms)
                    card_matrix = card_matrix / norms
                    score = float((card_matrix @ q_emb).mean())

        return [(score, row) for row in rows]

    # Packaging the rows
    def _to_joined_tuples(
        self,
        ranked: list[tuple[float, dict, JoinPath]],
    ) -> list[JoinedTuple]:
        """
        Wrap scored rows into JoinedTuple objects with sequential evidence IDs.

        IDs are assigned E1, E2, … in score-descending order and are stable
        within a single get_tuples() call.

        Args:
            ranked: List of (score, row_dict, join_path) triples.

        Returns:
            List of JoinedTuple objects.
        """
        return [
            JoinedTuple(
                evidence_id=f"E{idx}",  # E1, E2, E3, … used by the packager to cite evidence
                data=row,
                provenance=list(path.tables),  # which tables contributed to this row (the path)
                join_path=path,
                relevance_score=score,
            )
            for idx, (score, row, path) in enumerate(ranked, start=1)
        ]
