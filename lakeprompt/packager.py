from .datalake import DataLake
from .models import JoinedTuple, LakeContext
from .LLM_utilities import _package_with_toon


class ContextPackager:
    """
    Assembles ranked evidence tuples and the user question into a
    token-budgeted, TOON-encoded prompt ready for the LLM.

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

    def build_context(
        self,
        question: str,
        tuples: list[JoinedTuple],
    ) -> LakeContext:
        """
        Package evidence tuples into a LakeContext with a TOON prompt.

        Tuples are assumed to arrive pre-sorted by descending relevance
        score (as produced by TupleExecutor).  The method includes as
        many rows as fit within *max_tokens*, then builds the prompt.

        Args:
            question: The original natural-language question.
            tuples: Ranked evidence tuples from TupleExecutor.

        Returns:
            A LakeContext containing the prompt, the evidence rows that
            were included, and an estimated token count.
        """
        included = self._fit_to_budget(question, tuples)

        evidence_payload = [
            {
                "id": t.evidence_id,
                "sources": t.provenance,
                "data": t.data,
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
    ) -> list[JoinedTuple]:
        """
        Return the longest prefix of *tuples* whose prompt fits within
        *max_tokens*.  Always includes at least one tuple even if it
        exceeds the budget, so the LLM always has something to work with.

        Args:
            question: User question (included in the overhead estimate).
            tuples: Candidate evidence tuples, best-first.

        Returns:
            A (possibly truncated) list of JoinedTuples.
        """
        if not tuples:
            return []

        included: list[JoinedTuple] = []
        for t in tuples:
            candidate = included + [t]
            overhead = self._estimate_tokens(question) + len(candidate) * 10
            data_tokens = sum(
                self._estimate_tokens(str(c.data)) for c in candidate
            )
            if included and overhead + data_tokens > self.max_tokens:
                break
            included = candidate

        return included

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token count: 1 token ≈ 4 characters."""
        return max(1, len(text) // 4)
