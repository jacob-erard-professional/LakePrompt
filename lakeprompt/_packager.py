import re
from dataclasses import asdict

from ._datalake import DataLake
from ._models import JoinedTuple, LakeContext, QueryPlan
from ._llm_utilities import _package_with_toon
from ._tracing import NULL_LOGGER, PipelineLogger


_AGGREGATE_KEY_PATTERN = re.compile(
    r"^agg_\d+__(?P<aggregation>[a-z0-9]+)__(?:(?P<table>.+?)__)?(?P<column>.+)$"
)


class ContextPackager:
    """
    Assembles ranked evidence tuples and the user question into a
    token-budgeted, TOON-encoded prompt ready for the LLM. This class is
    needed because high-quality retrieval still has to be transformed into
    a compact, predictable prompt structure before the model can answer
    faithfully from evidence.

    Args:
        lake: The active DataLake instance (reserved for future use,
            e.g. fetching additional rows if the budget allows).
        max_tokens: Soft upper bound on prompt length in tokens.
            Evidence rows are dropped from the tail until the prompt
            fits within this budget.
    """

    def __init__(self, lake: DataLake, max_tokens: int = 3_000):
        self.lake = lake
        self.max_tokens = max_tokens
        self.logger: PipelineLogger = NULL_LOGGER

    def build_context(
        self,
        question: str,
        tuples: list[JoinedTuple],
        query_plan: QueryPlan | None = None,
    ) -> LakeContext:
        """
        Package evidence tuples into a `LakeContext` with a TOON prompt.

        Tuples are assumed to arrive pre-sorted by descending relevance
        score (as produced by TupleExecutor).  The method includes as
        many rows as fit within `max_tokens`, then builds the prompt. This
        keeps evidence selection and prompt rendering aligned in one place.

        Args:
            question: The original natural-language question.
            tuples: Ranked evidence tuples from TupleExecutor.
            query_plan: Optional structured intent used to project rows down
                to plan-relevant columns before packaging.

        Returns:
            A `LakeContext` containing the final prompt, the included
            evidence rows, and an estimated token count.
        """
        included = self._fit_to_budget(question, tuples, query_plan=query_plan)

        evidence_payload = [
            {
                "id": t.evidence_id,
                "provenance": asdict(t.provenance),
                **self._display_row_payload(t.data, query_plan),
            }
            for t in included
        ]

        prompt = _package_with_toon(
            task=(
                "Answer the question using only the provided evidence rows "
                "from the data lake. Cite the evidence IDs that support "
                "your answer."
            ),
            payload={"question": question, "evidence": evidence_payload},
            response_format='{"answer":"Plain-text answer here","cited_ids":["id1","id2"]}',
        )
        self.logger.log(
            "prompt_context",
            "Built prompt context.",
            {"question": question, "evidence": evidence_payload, "token_count": self._estimate_tokens(prompt)},
        )

        return LakeContext(
            question=question,
            evidence=included,
            prompt=prompt,
            token_count=self._estimate_tokens(prompt),
        )

    def _fit_to_budget(
        self,
        question: str,
        tuples: list[JoinedTuple],
        query_plan: QueryPlan | None = None,
    ) -> list[JoinedTuple]:
        """
        Return the longest prefix of `tuples` whose prompt fits within
        `max_tokens`. Always includes at least one tuple even if it
        exceeds the budget, so the LLM always has something to work with.

        This helper is needed because evidence quality is constrained by
        prompt size; without explicit budgeting, useful rows could be lost
        arbitrarily to context limits instead of being selected on purpose.

        Args:
            question: User question (included in the overhead estimate).
            tuples: Candidate evidence tuples, best-first.
            query_plan: Optional structured intent used for projected token
                estimation.

        Returns:
            A possibly truncated list of `JoinedTuple` objects.
        """
        if not tuples:
            return []

        included: list[JoinedTuple] = []
        for t in tuples:
            candidate = included + [t]
            overhead = self._estimate_tokens(question) + len(candidate) * 10
            data_tokens = sum(
                self._estimate_tokens(str(self._project_row_data(c.data, query_plan))) for c in candidate
            )
            if included and overhead + data_tokens > self.max_tokens:
                break
            included = candidate

        return included

    def _project_row_data(
        self,
        row: dict,
        query_plan: QueryPlan | None,
    ) -> dict:
        """
        Project a joined row down to plan-relevant columns for prompting.
        """
        if query_plan is None:
            return row

        relevant_keys = self._relevant_row_keys(query_plan, row)
        if not relevant_keys:
            return row

        projected = {key: row[key] for key in relevant_keys if key in row}
        return projected if projected else row

    def _display_row_payload(
        self,
        row: dict,
        query_plan: QueryPlan | None,
    ) -> dict:
        """
        Return prompt-facing row data with readable field names plus a stable
        mapping back to internal execution keys.
        """
        projected = self._project_row_data(row, query_plan)
        display_data: dict[str, object] = {}
        field_map: dict[str, str] = {}

        for key, value in projected.items():
            display_key = self._display_key_for_row_key(key, projected)
            if display_key in display_data:
                display_key = key.replace("__", "_")
            if display_key in display_data:
                display_key = key
            display_data[display_key] = value
            field_map[display_key] = key

        return {
            "data": display_data,
            "field_map": field_map,
        }

    def _display_key_for_row_key(self, key: str, row: dict) -> str:
        """
        Convert internal row keys into prompt-friendly labels.
        """
        aggregate_match = _AGGREGATE_KEY_PATTERN.match(key)
        if aggregate_match:
            aggregation = aggregate_match.group("aggregation")
            column = aggregate_match.group("column")
            return f"{aggregation}_{column}" if aggregation else column

        if "__" not in key:
            return key

        table, column = key.split("__", 1)
        sibling_columns = [
            other.split("__", 1)[1]
            for other in row
            if other != key and "__" in other
        ]
        if column not in sibling_columns:
            return column
        return f"{table}_{column}"

    def _relevant_row_keys(self, query_plan: QueryPlan, row: dict) -> list[str]:
        """
        Resolve plan fields to concrete row keys.
        """
        wanted: list[tuple[str | None, str]] = []

        for filter_ in query_plan.filters + query_plan.having:
            wanted.append((filter_.table, filter_.column))
        for projection in query_plan.projections:
            wanted.append((projection.table, projection.column))
        for order in query_plan.order_by:
            wanted.append((order.table, order.column))
        for item in query_plan.group_by:
            if "." in item:
                table, column = item.split(".", 1)
                wanted.append((table, column))
            else:
                wanted.append((None, item))

        seen: set[str] = set()
        resolved: list[str] = []
        for table, column in wanted:
            for key in self._match_row_keys(row, table, column):
                if key not in seen:
                    seen.add(key)
                    resolved.append(key)
        return resolved

    @staticmethod
    def _match_row_keys(row: dict, table: str | None, column: str) -> list[str]:
        """
        Match a logical table/column reference to concrete row keys.
        """
        matches: list[str] = []
        qualified = f"{table}__{column}" if table else None

        for key in row:
            if qualified and key == qualified:
                matches.append(key)
            elif key == column:
                matches.append(key)
            elif key.endswith(f"__{column}") and table is None:
                matches.append(key)

        return matches

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """
        Estimate token count from raw text length.

        This approximation is needed so prompt budgeting stays cheap and
        model-agnostic.

        Args:
            text: Text whose approximate token count is needed.

        Returns:
            An integer token estimate using a rough 4-characters-per-token
            heuristic.
        """
        return max(1, len(text) // 4)
