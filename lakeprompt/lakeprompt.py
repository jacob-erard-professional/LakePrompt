import json
import os

import anthropic

from .datalake import DataLake
from .LLM_utilities import DEFAULT_CLAUDE_MODEL
from .models import LakeAnswer
from .profiler import LakeProfiler
from .LLM_utilities import generate_table_summaries
from .retrieval import SemanticRetriever
from .executor import TupleExecutor
from .packager import ContextPackager


class LakePrompt:
    """
    Main entry point for the LakePrompt library.

    Initialises the full pipeline from a directory of CSV files —
    profiling tables, generating summaries, building embeddings, and
    indexing — so that natural language queries can be answered with
    grounded evidence from the data lake.

    Args:
        lake_dir: Path to the directory containing CSV files.
        model: Anthropic Claude model string for summary generation and answering.
            Defaults to 'claude-sonnet-4-20250514'.
        cache_path: Optional path to cache generated summaries to disk.

    Example:
        >>> lp = LakePrompt("./csvs")
        >>> answer = lp.query("Which customers spent the most in January?")
        >>> print(answer.text)
        >>> print(answer.evidence)
    """

    def __init__(
        self,
        lake_dir: str,
        model: str = DEFAULT_CLAUDE_MODEL,
        cache_path: str = None
    ):
        # 1. Load the lake
        self.lake = DataLake.load(lake_dir)

        # 2. Profile all tables → ColumnCards
        self.profiler = LakeProfiler(self.lake, .8)
        self.cards_by_table = self.profiler.profile()

        # 3. Generate LLM summaries for each table
        summaries = generate_table_summaries(
            cards_by_table=self.cards_by_table,
            model=model,
            cache_path=cache_path
        )
        for table_name, cards in self.cards_by_table.items():
            for card in cards:
                card.table_summary = summaries[table_name]

        # 4. Embed cards and build HNSW index
        self.retriever = SemanticRetriever(self.lake, self.cards_by_table)
        self.retriever.build_index()

        # 5. Initialise remaining pipeline modules
        self.model = model
        self.executor = TupleExecutor(self.lake)
        self.packager = ContextPackager(self.lake)

    def query(self, question: str) -> LakeAnswer:
        """
        Answer a natural language question grounded in the data lake.

        Runs the full LakePrompt pipeline — retrieving relevant tables,
        discovering join paths, executing joins, packaging context, and
        calling the LLM to produce a cited answer.

        Args:
            question: A natural language question about the data lake.

        Returns:
            A LakeAnswer containing the answer text and the evidence
            tuples that support it.
        """
        cards   = self.retriever.find_columns(question) # Relevant columns
        paths   = self.profiler.get_join_paths(cards) # Join paths with jacard similarity
        tuples  = self.executor.get_tuples(question, paths) # Executes joins
        context = self.packager.build_context(question, tuples)
        answer_text, cited_ids = self._llm_complete(
            context.prompt,
            valid_ids={item.evidence_id for item in context.evidence},
        )
        
        return LakeAnswer(
            text=answer_text,
            evidence=context.evidence,
            cited_ids=cited_ids,
        )

    def _llm_complete(
        self,
        prompt: str,
        valid_ids: set[str] | None = None,
    ) -> tuple[str, list[str]]:
        """
        Send a packaged prompt to the LLM and return the answer text plus citations.

        The prompt is expected to request a JSON response with an
        ``answer`` key and optional ``cited_ids`` list (as produced by
        ContextPackager.build_context).
        If the response cannot be parsed as JSON, the raw content is
        returned with an empty citation list so the caller always
        receives a non-empty answer.

        Args:
            prompt: A fully assembled TOON-encoded prompt string.
            valid_ids: Optional set of evidence IDs allowed in the
                returned citation list.

        Returns:
            A tuple of (answer_text, cited_ids).
        """
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set.")
        client = anthropic.Anthropic(api_key=api_key)

        response = client.messages.create(
            model=self.model,
            max_tokens=1000,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )

        raw = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        ).strip()

        try:
            parsed = json.loads(raw)
            answer_text = parsed.get("answer", raw)
            cited_ids = parsed.get("cited_ids", [])
            if not isinstance(answer_text, str):
                answer_text = raw
            if not isinstance(cited_ids, list):
                cited_ids = []

            normalized_ids = [
                item for item in cited_ids
                if isinstance(item, str) and (valid_ids is None or item in valid_ids)
            ]
            return answer_text, normalized_ids
        except json.JSONDecodeError:
            return raw, []
