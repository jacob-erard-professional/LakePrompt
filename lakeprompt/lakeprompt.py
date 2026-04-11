import json
import os

from openai import OpenAI

from .datalake import DataLake
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
        model: OpenRouter model string for summary generation.
            Defaults to 'nvidia/nemotron-3-super-120b-a12b:free'.
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
        model: str = "nvidia/nemotron-3-super-120b-a12b:free",
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
        answer  = self._llm_complete(context.prompt)
        
        return LakeAnswer(text=answer, evidence=context.evidence)

    def _llm_complete(self, prompt: str) -> str:
        """
        Send a packaged prompt to the LLM and return the answer text.

        The prompt is expected to request a JSON response with an
        ``answer`` key (as produced by ContextPackager.build_context).
        If the response cannot be parsed as JSON, the raw content is
        returned so the caller always receives a non-empty string.

        Args:
            prompt: A fully assembled TOON-encoded prompt string.

        Returns:
            The answer string extracted from the model's JSON response,
            or the raw response content if JSON parsing fails.
        """
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )

        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )

        raw = response.choices[0].message.content.strip()

        try:
            parsed = json.loads(raw)
            return parsed.get("answer", raw)
        except json.JSONDecodeError:
            return raw
