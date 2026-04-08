from .datalake import DataLake
from .models import ColumnCard, JoinPath

class LakeProfiler:
    """
    Profiles all tables in a DataLake by extracting column statistics, computing Jaccard
    similarity between colummns across tables, and building a join graph that downstreams
    use for join path discovery.
    """
    def __init__(self, lake: DataLake, jaccard_threshold: float = 0.5):
        """
        Args:
            lake: The DataLake to profile.
            jaccard_threshold: Threshold for Jaccard similarity between columns.
        """
        self.lake = lake
        self._jaccard_threshold = jaccard_threshold
        self.cards_by_table: dict[str, list[ColumnCard]] = {}

    def profile(self) -> dict[str, list[ColumnCard]]:
        """
        Loads every table in the lake, builds a ColumnCard for each column,
        and stores qualifying cross-table Jaccard matches directly on each
        ColumnCard for constant-time lookup at runtime.
        
        Returns:
            Dictionary mapping table name to its list of ColumnCards.
        """
        cards_by_table: dict[str, list[ColumnCard]] = {}

        for table_name in self.lake.tables:
            sample_df = self.lake.get_sample(table_name, n=100)
            cards: list[ColumnCard] = []
            for column_name in sample_df.columns:
                series = sample_df[column_name]
                cards.append(self._build_column_card(table_name, column_name, series))
            cards_by_table[table_name] = cards

        self._populate_jaccard_matches(cards_by_table)
        self.cards_by_table = cards_by_table
        self.lake.cards = [card for cards in cards_by_table.values() for card in cards]
        self.lake.join_graph = self.build_join_graph(cards_by_table)
        return cards_by_table
    
    def _build_column_card(self, table_name: str, column_name: str, series) -> ColumnCard:
        """
        Builds a single ColumnCard from a pandas Series.

        Args:
            table_name: Name of the table the column belongs to.
            col_name: Name of the column.
            series: The pandas Series for that column.

        Returns:
            A populated ColumnCard.
        """
        sample_values = series.drop_nulls().head(5).to_list()

        if sample_values: 
            summary = (
                f"{column_name} in {table_name} stores {series.dtype} values "
                f"such as {sample_values[:3]}."
            )
        else:
            summary = f"{column_name} in {table_name} stores {series.dtype} values."

        return ColumnCard(
            table_name=table_name,
            column_name=column_name,
            dtype=str(series.dtype),
            sample_values=sample_values,
            summary=summary,
        )

    def jaccard_similarity(self, col_a, col_b) -> float:
        """
        Computes Jaccard similarity between the value sets of two columns.

        Args:
            col_a: First column as a pandas Series.
            col_b: Second column as a pandas Series.

        Returns:
            Jaccard score between 0 and 1.
        """
        set_a = {str(v) for v in col_a if v is not None}
        set_b = {str(v) for v in col_b if v is not None}

        if not set_a and not set_b:
            return 0.0

        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union else 0.0

    def _populate_jaccard_matches(
        self,
        cards_by_table: dict[str, list[ColumnCard]],
    ) -> None:
        """
        Compute and store all qualifying cross-table Jaccard matches on
        ColumnCards using compact (table_name, column_name) lookup keys.
        """
        table_names = list(cards_by_table.keys())
        value_cache: dict[tuple[str, str], set[str]] = {}
        card_index: dict[tuple[str, str], ColumnCard] = {}

        for table_name, cards in cards_by_table.items():
            for card in cards:
                card_key = (table_name, card.column_name)
                card_index[card_key] = card
                value_cache[card_key] = self.lake.get_column_values(*card_key)

        for i, left_table in enumerate(table_names):
            for right_table in table_names[i + 1:]:
                for left_card in cards_by_table[left_table]:
                    left_key = (left_table, left_card.column_name)
                    left_values = value_cache[left_key]

                    for right_card in cards_by_table[right_table]:
                        right_key = (right_table, right_card.column_name)
                        score = self.jaccard_similarity(
                            left_values,
                            value_cache[right_key],
                        )

                        if score < self._jaccard_threshold:
                            continue

                        card_index[left_key].jaccard_matches[right_key] = score
                        card_index[right_key].jaccard_matches[left_key] = score

    def build_join_graph(self, cards_by_table: dict[str, list[ColumnCard]]):
        """
        Build a table-level join graph from the Jaccard matches already
        stored on each ColumnCard.

        Args:
            cards_by_table: Output from profile().

        Returns:
            A graph where nodes are table names and edges carry the joinable
            column names and their Jaccard score.
        """
        table_names = list(cards_by_table.keys())
        graph: dict[str, list[dict[str, object]]] = {t: [] for t in table_names}

        for left_table, left_cards in cards_by_table.items():
            for left_card in left_cards:
                for (right_table, right_column), score in left_card.jaccard_matches.items():
                    graph[left_table].append({
                        "to_table": right_table,
                        "left_column": left_card.column_name,
                        "right_column": right_column,
                        "score": score,
                    })

        return graph

    def get_join_paths(
        self,
        relevant_cards: list[ColumnCard],
        max_paths: int = 10,
    ) -> list[JoinPath]:
        """
        Build join-path candidates from relevant cards and the join graph.

        Produces single-table paths and two-table join paths ranked by score.

        Args:
            relevant_cards: ColumnCards relevant to the current prompt.
            max_paths: Maximum number of join paths to return. Defaults to 10.
        
        Returns:
            A list of JoinPaths sorted by relevance score.
        """
        # TODO: Find a way to do this, and ask ppl about what score means in joinpath.#
        # Do u mean join do we execute join path or add to join path to context?
