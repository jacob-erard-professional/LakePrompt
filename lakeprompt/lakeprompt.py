import json
import os

import anthropic

from .datalake import DataLake
from .ingest import DataLakePreparer
from .LLM_utilities import DEFAULT_CLAUDE_MODEL, plan_llm_query
from .models import LakeAnswer, QueryPlan
from .profiler import LakeProfiler
from .LLM_utilities import generate_table_summaries
from .retrieval import SemanticRetriever
from .executor import TupleExecutor
from .packager import ContextPackager
from .tracing import PipelineLogger


class LakePrompt:
    """
    Main entry point for the LakePrompt library.

    Initialises the full pipeline from a directory of CSV files so natural
    language queries can be answered with grounded evidence from the data
    lake. This class is needed because the project's value comes from the
    composition of multiple stages; `LakePrompt` provides one stable API
    instead of requiring callers to wire profiling, retrieval, planning,
    execution, packaging, and LLM calls manually.

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
        cache_path: str = None,
        logger: bool = False,
    ):
        self.logger = PipelineLogger(enabled=logger)

        # 1. Load the lake
        self.lake = DataLake.load(lake_dir)

        # 2. Profile all tables → ColumnCards
        self.profiler = LakeProfiler(self.lake, .8)
        self.profiler.logger = self.logger
        self.cards_by_table = self.profiler.profile()

        # 3. Generate LLM summaries for each table
        summaries = generate_table_summaries(
            cards_by_table=self.cards_by_table,
            model=model,
            cache_path=cache_path,
            logger=self.logger,
        )
        for table_name, cards in self.cards_by_table.items():
            for card in cards:
                card.table_summary = summaries[table_name]
        self.logger.log("table_summaries", "Generated table summaries.", summaries)

        # 4. Embed cards and build HNSW index
        self.retriever = SemanticRetriever(self.lake, self.cards_by_table, logger=self.logger)
        self.retriever.build_index()

        # 5. Initialise remaining pipeline modules
        self.model = model
        self.executor = TupleExecutor(self.lake, logger=self.logger)
        self.packager = ContextPackager(self.lake)
        self.packager.logger = self.logger

    @classmethod
    def from_url(
        cls,
        source_url: str,
        model: str = DEFAULT_CLAUDE_MODEL,
        cache_path: str = None,
        source_cache_dir: str | None = None,
        logger: bool = False,
    ) -> "LakePrompt":
        """
        Prepare a supported remote data source into a local CSV lake and load it.

        Supported sources:
        - direct CSV links
        - ZIP archives containing CSV files
        - GitHub repository URLs

        This constructor is needed because many public datasets are hosted
        remotely rather than already normalized as a local CSV directory.

        Args:
            source_url: Remote HTTP(S) URL describing the data source.
            model: Model used for summary generation and answering.
            cache_path: Optional path for cached table summaries.
            source_cache_dir: Optional local cache directory for downloaded
                source material.

        Returns:
            A fully initialized `LakePrompt` instance backed by the prepared
            local lake.
        """
        preparer = DataLakePreparer(cache_root=source_cache_dir)
        prepared = preparer.prepare(source_url)
        instance = cls(
            lake_dir=str(prepared.prepared_dir),
            model=model,
            cache_path=cache_path,
            logger=logger,
        )
        instance.source_url = source_url
        instance.prepared_lake_dir = str(prepared.prepared_dir)
        instance.prepared_source_type = prepared.source_type
        return instance

    def query(self, question: str) -> LakeAnswer:
        """
        Answer a natural language question grounded in the data lake.

        Runs the full LakePrompt pipeline — retrieving relevant tables,
        discovering join paths, executing joins, packaging context, and
        calling the LLM to produce a cited answer. This is the main
        user-facing workflow entry point.

        Args:
            question: A natural language question about the data lake.

        Returns:
            A `LakeAnswer` containing the answer text, supporting evidence,
            and cited evidence IDs when available.
        """
        cards = self.retriever.find_columns(question)
        query_plan = self._plan_query(question, cards)
        paths = self.profiler.get_join_paths(cards, query_plan=query_plan)
        tuples = self.executor.get_tuples(question, paths, query_plan=query_plan)
        context = self.packager.build_context(question, tuples, query_plan=query_plan)
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
        receives a non-empty answer. This helper is needed so model-specific
        API handling stays isolated from the rest of the retrieval pipeline.

        Args:
            prompt: A fully assembled TOON-encoded prompt string.
            valid_ids: Optional set of evidence IDs allowed in the
                returned citation list.

        Returns:
            A tuple of `(answer_text, cited_ids)`.
        """
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set.")
        client = anthropic.Anthropic(api_key=api_key)

        self.logger.log("llm_request", "Requesting final answer.", {"prompt": prompt})
        response = client.messages.create(
            model=self.model,
            max_tokens=1000,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )

        raw = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        ).strip()
        self.logger.log("llm_response", "Received final answer response.", {"response": raw})

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

    def _plan_query(self, question: str, cards) -> QueryPlan:
        """
        Extract structured query intent once and reuse it across the pipeline.

        This method is needed so join-path scoring and SQL refinement can
        share one interpretation of the question instead of asking the LLM
        separately at multiple stages.

        Args:
            question: Natural-language question from the user.
            cards: Retrieved `ColumnCard` candidates associated with the
                question.

        Returns:
            A `QueryPlan` describing filters, projections, grouping, and
            ordering implied by the question. Returns an empty plan if
            planning fails.
        """
        try:
            return plan_llm_query(
                question=question,
                sql_query="SELECT * FROM candidate_tables",
                involved_cards=list(cards),
                model=self.model,
                logger=self.logger,
            )
        except Exception:
            return QueryPlan()
