import logging
logger = logging.getLogger(__name__)

from collections import defaultdict
from dataclasses import dataclass
from .datalake import DataLake
from .models import JoinPath, JoinedTuple, JoinProvenance
from .LLM_utilities import apply_llm_filters_to_sql

"""
TupleExecutor — Stream S3 of the LakePrompt pipeline.

Receives JoinPath objects, executes the joins against the DataLake,
ranks the resulting rows, and returns the top-r rows as JoinedTuple
evidence objects for the packager.
"""



DEFAULT_TOP_R = 20

# model cache so SentenceTransformer is only loaded once per process.
_ST_MODEL: object = None


@dataclass
class TupleExecutor:
    """
    Executes join paths against the DataLake and returns ranked evidence tuples.

    Rows are scored individually after execution, then diversity-filtered so
    the final evidence set is not flooded by near-duplicate rows.

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
        q_emb = self._embed_question(question)
        all_scored: list[tuple[float, dict, JoinPath]] = []

        for path in paths:
            rows = self._execute_path(path, question=question)
            if not rows:
                continue

            for score, row in self._score_rows(rows, q_emb, path):
                all_scored.append((score, row, path))

        all_scored.sort(key=lambda x: x[0], reverse=True)
        diverse = self._select_diverse_rows(all_scored, top_r=top_r)
        return self._to_joined_tuples(diverse)

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

        In multi-table joins, every column is aliased as '<table>__<column>'
        so downstream tuple ranking sees stable table-qualified row keys.

        Args:
            tables: Ordered list of table names in the join.

        Returns:
            List of (table_name, column_name, alias) tuples.
        """
        table_cols: dict[str, list[str]] = {}

        for tbl in tables:
            if tbl not in self.lake.tables:
                logger.warning("Table '%s' not found in lake; skipping.", tbl)
                table_cols[tbl] = []
                continue
            # Fetch a single row just to get the column names — no data needed.
            cols = self.lake.get_sample(tbl, n=1).columns
            table_cols[tbl] = cols
        result: list[tuple[str, str, str]] = []
        for tbl in tables:
            for col in table_cols.get(tbl, []):
                result.append((tbl, col, f"{tbl}__{col}"))

        return result

    # Execution of joins and retrieval of rows
    def _execute_path(
        self, path: JoinPath, question: str = "", filter_clause: str = ""
    ) -> list[dict]:
        """
        Build and execute SQL for a join path, returning rows as list[dict].

        If a question is provided and no explicit filter_clause is given,
        the raw join SQL is first refined by _extract_filters before execution.

        Args:
            path: The join path to execute.
            question: Optional natural language question used to infer filters.
            filter_clause: Optional SQL WHERE fragment. When provided, skips
                LLM-based filter extraction and uses this clause directly.

        Returns:
            A (possibly empty) list of row dicts.
        """
        if len(path.tables) == 1:
            where = f" WHERE {filter_clause}" if filter_clause else ""
            sql = f"SELECT * FROM {path.tables[0]}{where}"
        else:
            try:
                sql = self._build_join_sql(path, filter_clause)
            except ValueError as exc:
                logger.warning("Skipping malformed JoinPath: %s", exc)
                return []

        if question and not filter_clause:
            sql = self._extract_filters(question, sql, path)

        logger.debug("Executing SQL:\n%s", sql)

        try:
            result_df = self.lake.query(sql)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Query failed for path %s: %s", path.tables, exc)
            return []

        rows = result_df.to_dicts()

        if not rows:
            logger.warning("Join path %s produced zero rows.", path.tables)

        return rows

    def _extract_filters(self, question: str, sql: str, path: JoinPath) -> str:
        """
        Refine a raw join SQL query with filters inferred from the question.

        Passes the question, the raw SQL, and the ColumnCards for the path's
        tables to apply_llm_filters_to_sql. Falls back to the original SQL if
        the LLM call fails or returns an invalid result.

        Args:
            question: The natural language question driving retrieval.
            sql: Raw join SQL produced by _build_join_sql.
            path: The join path whose tables determine which cards to pass.

        Returns:
            The refined SQL string with filters added, or the original sql
            if refinement fails.
        """
        path_table_set = set(path.tables)
        involved_cards = [
            card
            for card in (getattr(self.lake, "cards", None) or [])
            if card.table_name in path_table_set
        ]

        try:
            return apply_llm_filters_to_sql(question, sql, involved_cards)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Filter extraction failed for path %s: %s", path.tables, exc
            )
            return sql

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

    def _score_rows(
        self,
        rows: list[dict],
        q_emb: "np.ndarray | None",  # type: ignore[name-defined]
        path: JoinPath,
    ) -> list[tuple[float, dict]]:
        """
        Score rows individually using row text similarity plus a light
        path-coverage bonus.

        Args:
            rows: Raw row dicts from _execute_path.
            q_emb: Unit-vector question embedding, or None if unavailable.
            path: The join path whose tables determine which cards to use.

        Returns:
            List of (score, row) pairs sorted by descending score.
        """
        scored_rows: list[tuple[float, dict]] = []
        path_table_set = set(path.tables)
        relevant_columns = [
            card.column_name
            for card in (getattr(self.lake, "cards", None) or [])
            if card.table_name in path_table_set
        ]
        coverage_bonus = self._path_coverage_bonus(path, relevant_columns)

        for row in rows:
            row_score = self._row_semantic_score(row, q_emb)
            row_score += coverage_bonus
            row_score += self._row_value_coverage_bonus(row, path)
            scored_rows.append((row_score, row))

        scored_rows.sort(key=lambda item: item[0], reverse=True)
        return scored_rows

    def _row_semantic_score(
        self,
        row: dict,
        q_emb: "np.ndarray | None",  # type: ignore[name-defined]
    ) -> float:
        """
        Score a row by embedding similarity against the question.
        """
        if q_emb is None:
            return 0.0

        try:
            import numpy as np
        except ImportError:
            return 0.0

        row_text = self._row_to_text(row)
        if not row_text.strip():
            return 0.0

        global _ST_MODEL
        if _ST_MODEL is None:
            return 0.0

        row_emb = _ST_MODEL.encode(row_text, normalize_embeddings=True)  # type: ignore[union-attr]
        return float(np.dot(row_emb, q_emb))

    def _path_coverage_bonus(self, path: JoinPath, relevant_columns: list[str]) -> float:
        """
        Reward paths that cover more relevant tables and columns.
        """
        table_bonus = 0.03 * len(path.tables)
        column_bonus = 0.005 * len(set(relevant_columns))
        return table_bonus + min(column_bonus, 0.05)

    def _row_value_coverage_bonus(self, row: dict, path: JoinPath) -> float:
        """
        Reward rows that carry values from more joined tables.
        """
        populated_tables = 0
        for table in path.tables:
            has_value = any(
                value is not None
                for key, value in row.items()
                if key.startswith(f"{table}__")
            )
            if has_value:
                populated_tables += 1

        if len(path.tables) == 1 and row:
            populated_tables = 1

        return 0.04 * (populated_tables / max(len(path.tables), 1))

    def _row_to_text(self, row: dict) -> str:
        """
        Convert a row into a stable table-qualified text representation.
        """
        parts: list[str] = []
        for key in sorted(row):
            value = row[key]
            if value is None:
                continue
            parts.append(f"{key}={value}")
        return " | ".join(parts)

    def _select_diverse_rows(
        self,
        ranked: list[tuple[float, dict, JoinPath]],
        top_r: int,
    ) -> list[tuple[float, dict, JoinPath]]:
        """
        Greedily select the best rows while penalizing near-duplicates from
        the same join path.
        """
        selected: list[tuple[float, dict, JoinPath]] = []
        selected_tokens_by_path: dict[str, list[set[str]]] = defaultdict(list)

        for base_score, row, path in ranked:
            row_tokens = self._row_token_set(row)
            similarity_penalty = 0.0

            for existing_tokens in selected_tokens_by_path[path.path_id]:
                similarity_penalty = max(
                    similarity_penalty,
                    self._token_jaccard_similarity(row_tokens, existing_tokens),
                )

            adjusted_score = base_score - (0.15 * similarity_penalty)
            if similarity_penalty >= 0.98:
                continue

            inserted = False
            for index, (current_score, _, _) in enumerate(selected):
                if adjusted_score > current_score:
                    selected.insert(index, (adjusted_score, row, path))
                    inserted = True
                    break
            if not inserted:
                selected.append((adjusted_score, row, path))

            selected_tokens_by_path[path.path_id].append(row_tokens)
            if len(selected) >= top_r:
                selected = selected[:top_r]

        return selected[:top_r]

    def _row_token_set(self, row: dict) -> set[str]:
        """
        Convert a row into a coarse token set for duplicate suppression.
        """
        tokens: set[str] = set()
        for key, value in row.items():
            if value is None:
                continue
            tokens.add(str(key).lower())
            tokens.update(str(value).lower().split())
        return tokens

    @staticmethod
    def _token_jaccard_similarity(left: set[str], right: set[str]) -> float:
        """
        Return Jaccard similarity between two token sets.
        """
        if not left and not right:
            return 1.0
        if not left or not right:
            return 0.0
        return len(left & right) / len(left | right)

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
                provenance=JoinProvenance(
                    path_id=path.path_id,
                    tables=list(path.tables),
                    join_keys=list(path.join_keys),
                    path_score=path.score,
                ),
                join_path=path,
                relevance_score=score,
            )
            for idx, (score, row, path) in enumerate(ranked, start=1)
        ]
