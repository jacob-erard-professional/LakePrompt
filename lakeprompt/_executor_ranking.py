import logging
from dataclasses import dataclass

from ._datalake import DataLake
from ._models import JoinPath

logger = logging.getLogger(__name__)

# model cache so SentenceTransformer is only loaded once per process.
_ST_MODEL: object = None


@dataclass
class RowRanker:
    """
    Score materialized rows and suppress near-duplicates.

    This class is needed because execution produces raw rows, while later
    stages need a smaller ranked evidence set.
    """

    lake: DataLake

    def embed_question(self, question: str) -> "np.ndarray | None":  # type: ignore[name-defined]
        """
        Encode the question with SentenceTransformer and return a unit vector.

        This method is needed because row ranking compares row semantics to
        the user's question.
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

    def score_rows(
        self,
        rows: list[dict],
        q_emb: "np.ndarray | None",  # type: ignore[name-defined]
        path: JoinPath,
    ) -> list[tuple[float, dict]]:
        """
        Score rows individually using semantic similarity plus light
        path- and value-coverage bonuses.

        This method is needed because the system ranks actual evidence rows,
        not just the join paths that produced them.
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
            semantic_score = self._row_semantic_score(row, q_emb)
            value_bonus = self._row_value_coverage_bonus(row, path)
            total_score = semantic_score + coverage_bonus + value_bonus
            scored_rows.append((total_score, row))

        scored_rows.sort(key=lambda item: item[0], reverse=True)
        return scored_rows

    def select_diverse_rows(
        self,
        ranked: list[tuple[float, dict, JoinPath]],
        top_r: int,
    ) -> list[tuple[float, dict, JoinPath]]:
        """
        Select the best rows while suppressing near-duplicates across all
        join paths.

        This method is needed because top-ranked results otherwise tend to
        collapse into redundant variants of the same evidence, including
        identical rows produced by two different paths covering the same tables.
        """
        selected: list[tuple[float, dict, JoinPath]] = []
        all_selected_tokens: list[set[str]] = []

        for base_score, row, path in ranked:
            row_tokens = self._row_token_set(row)
            similarity_penalty = 0.0

            for existing_tokens in all_selected_tokens:
                sim = self._token_jaccard_similarity(row_tokens, existing_tokens)
                if sim > similarity_penalty:
                    similarity_penalty = sim
                if similarity_penalty >= 0.98:
                    break

            if similarity_penalty >= 0.98:
                continue

            adjusted_score = base_score - (0.15 * similarity_penalty)
            selected.append((adjusted_score, row, path))
            all_selected_tokens.append(row_tokens)

            if len(selected) >= top_r:
                break

        return selected

    def _row_semantic_score(
        self,
        row: dict,
        q_emb: "np.ndarray | None",  # type: ignore[name-defined]
    ) -> float:
        """
        Score a row by embedding similarity against the question.

        This helper is needed because semantic relevance is one component
        of the final row score.
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

        This helper is needed because broader path coverage is treated as a
        small ranking advantage.
        """
        table_bonus = 0.03 * len(path.tables)
        column_bonus = 0.005 * len(set(relevant_columns))
        return table_bonus + min(column_bonus, 0.05)

    def _row_value_coverage_bonus(self, row: dict, path: JoinPath) -> float:
        """
        Reward rows that carry values from more joined tables.

        This helper is needed because rows with more populated joined
        context are usually more useful as evidence.
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

        This helper is needed because embedding models consume text rather
        than structured dictionaries.
        """
        parts: list[str] = []
        for key in sorted(row):
            value = row[key]
            if value is None:
                continue
            parts.append(f"{key}={value}")
        return " | ".join(parts)

    def _row_token_set(self, row: dict) -> set[str]:
        """
        Convert a row into a coarse token set for duplicate suppression.

        This helper is needed because diversity filtering operates on a
        simple approximate similarity signal.
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

        This helper is needed because duplicate suppression compares rows by
        overlap rather than exact equality.
        """
        if not left and not right:
            return 1.0
        if not left or not right:
            return 0.0
        return len(left & right) / len(left | right)
