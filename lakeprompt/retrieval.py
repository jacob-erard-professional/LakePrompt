import numpy as np
import hnswlib
from sentence_transformers import SentenceTransformer

from .datalake import DataLake
from .models import ColumnCard
from .tracing import NULL_LOGGER, PipelineLogger


class SemanticRetriever:
    """
    Builds and queries a semantic index over ColumnCard summaries.

    Uses SBERT to embed table summaries and hnswlib to build an HNSW
    index for fast approximate nearest neighbour search. At query time,
    it embeds the user question and returns the most relevant
    `ColumnCard` objects. This retriever is needed because join planning
    is only tractable if the system first narrows attention to a small,
    high-signal subset of tables and columns.

    It consumes ColumnCards produced by DataProfiler and its output -
    a ranked list of relevant cards — is consumed by
    DataProfiler.get_join_paths() and TupleExecutor.

    Args:
        lake: The shared DataLake instance.
        cards_by_table: Dictionary mapping table name to its ColumnCards,
            as produced by DataProfiler.profile_lake().
        model_name: SBERT model to use for embedding. Defaults to
            'all-MiniLM-L6-v2', a fast and capable general-purpose model.

    Example:
        >>> retriever = SemanticRetriever(lake, cards_by_table)
        >>> retriever.build_index()
        >>> cards = retriever.find_columns("customers in Denver")
    """

    def __init__(
        self,
        lake: DataLake,
        cards_by_table: dict[str, list[ColumnCard]],
        model_name: str = "all-MiniLM-L6-v2",
        logger: PipelineLogger | None = None,
    ):
        self.lake = lake
        self.cards_by_table = cards_by_table
        self.model = SentenceTransformer(model_name)
        self.index = None
        self._indexed_cards: list[ColumnCard] = []
        self.logger = logger or NULL_LOGGER

    def _get_embedding_text(self, card: ColumnCard) -> str:
        """
        Build the text to embed for a given ColumnCard.

        Combines the table summary, column name, and sample values into
        one string. This helper is needed because embeddings are more
        useful when semantic context and concrete value hints are packed
        together instead of embedding only a bare column name.

        Args:
            card: The ColumnCard to build embedding text for.

        Returns:
            A single string representation of the card for embedding.
        """
        samples = ", ".join(str(v) for v in card.sample_values[:5])
        return (
            f"{card.table_summary}. "
            f"Column: {card.column_name}. "
            f"Sample values: {samples}."
        )

    def embed_cards(self) -> None:
        """
        Compute and store SBERT embeddings for all ColumnCards.

        Embeddings are written directly onto each card's embedding field.
        Runs all cards through SBERT in a single batch for efficiency.
        This method is needed so later retrieval and ranking stages can
        reuse precomputed vectors instead of re-encoding cards repeatedly.

        Returns:
            None. The method mutates cards and caches indexed cards.
        """
        all_cards = [
            card
            for cards in self.cards_by_table.values()
            for card in cards
        ]
        texts = [self._get_embedding_text(card) for card in all_cards]
        embeddings = self.model.encode(texts, show_progress_bar=False)

        for card, embedding in zip(all_cards, embeddings):
            card.embedding = embedding.tolist()

        self._indexed_cards = all_cards

    def build_index(self) -> None:
        """
        Embed all cards and build an HNSW index over their embeddings.

        Calls embed_cards() internally so only build_index() needs to be
        called from outside. The index is stored in `self.index` and used
        by `find_columns()` at query time. Building the index once makes
        semantic lookup fast enough for interactive use.

        Returns:
            None. The method stores the built HNSW index on the retriever.

        Raises:
            ValueError: If no cards are available to index.
        """
        self.embed_cards()

        if not self._indexed_cards:
            raise ValueError("No ColumnCards to index.")

        dim = len(self._indexed_cards[0].embedding)
        n = len(self._indexed_cards)

        self.index = hnswlib.Index(space="cosine", dim=dim)
        self.index.init_index(max_elements=n, ef_construction=200, M=16)

        embeddings = np.array(
            [card.embedding for card in self._indexed_cards],
            dtype=np.float32
        )
        self.index.add_items(embeddings, list(range(n)))
        self.index.set_ef(50)

    def find_columns(
        self,
        question: str,
        top_k: int = 10
    ) -> list[ColumnCard]:
        """
        Find the most semantically relevant ColumnCards for a question.

        Embeds the question using SBERT and queries the HNSW index for
        the nearest neighbours. This method is needed because the rest of
        the pipeline should reason over a ranked shortlist of likely
        columns rather than every column in the lake.

        Args:
            question: The user's natural language question.
            top_k: Number of cards to return. Defaults to 10.

        Returns:
            A list of the most relevant `ColumnCard` objects, ranked by
            semantic relevance.

        Example:
        >>> cards = retriever.find_columns("How much did Denver customers spend?")
        """
        if self.index is None:
            self.build_index()

        question_embedding = self.model.encode(
            [question],
            show_progress_bar=False
        )
        question_embedding = np.array(question_embedding, dtype=np.float32)

        k = min(top_k, len(self._indexed_cards))
        labels, distances = self.index.knn_query(question_embedding, k=k)
        results = [self._indexed_cards[i] for i in labels[0]]
        self.logger.log(
            "semantic_columns",
            "Retrieved semantically similar columns.",
            [
                {
                    "table": card.table_name,
                    "column": card.column_name,
                    "table_summary": card.table_summary,
                    "distance": float(distance),
                }
                for card, distance in zip(results, distances[0], strict=False)
            ],
        )
        return results
