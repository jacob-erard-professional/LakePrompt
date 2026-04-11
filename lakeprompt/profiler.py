from collections import Counter

from .datalake import DataLake
from .models import ColumnCard, JoinEdge, JoinPath

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
        seen_edges: set[tuple[str, str, str, str]] = set()

        for left_table, left_cards in cards_by_table.items():
            for left_card in left_cards:
                for (right_table, right_column), score in left_card.jaccard_matches.items():
                    edge_key = (left_table, left_card.column_name, right_table, right_column)
                    if edge_key in seen_edges:
                        continue
                    seen_edges.add(edge_key)
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
        max_hops: int = 3,
    ) -> list[JoinPath]:
        """
        Build join-path candidates from relevant cards and the join graph.

        Produces single-table and multi-hop join paths ranked by score.

        Args:
            relevant_cards: ColumnCards relevant to the current prompt.
            max_paths: Maximum number of join paths to return. Defaults to 10.
            max_hops: Maximum join depth in graph edges. Defaults to 3.
        
        Returns:
            A list of JoinPaths sorted by relevance score.
        """
        if not self.lake.join_graph:
            raise ValueError("Join graph not found. Please run profile() first.")

        if not relevant_cards:
            return []

        table_counts = Counter(card.table_name for card in relevant_cards)
        total_relevant_tables = len(table_counts)
        paths_by_signature: dict[
            tuple[tuple[str, ...], tuple[tuple[str, str, str, str], ...]],
            JoinPath,
        ] = {}
        path_counter = 1

        for root_table in sorted(table_counts):
            if self._register_path(
                paths_by_signature=paths_by_signature,
                path_counter=path_counter,
                tables=[root_table],
                join_edges=[],
                table_counts=table_counts,
                total_relevant_tables=total_relevant_tables,
            ):
                path_counter += 1

            stack: list[tuple[list[str], list[JoinEdge]]] = [([root_table], [])]
            while stack:
                tables, join_edges = stack.pop()
                if len(join_edges) >= max_hops:
                    continue

                current_table = tables[-1]
                for raw_edge in self.lake.join_graph.get(current_table, []):
                    next_table = raw_edge["to_table"]
                    if next_table in tables:
                        continue

                    next_edges = join_edges + [
                        JoinEdge(
                            left_table=current_table,
                            left_column=raw_edge["left_column"],
                            right_table=next_table,
                            right_column=raw_edge["right_column"],
                            score=float(raw_edge["score"]),
                        )
                    ]
                    next_tables = tables + [next_table]

                    if self._register_path(
                        paths_by_signature=paths_by_signature,
                        path_counter=path_counter,
                        tables=next_tables,
                        join_edges=next_edges,
                        table_counts=table_counts,
                        total_relevant_tables=total_relevant_tables,
                    ):
                        path_counter += 1

                    stack.append((next_tables, next_edges))

        ranked = sorted(paths_by_signature.values(), key=lambda p: p.score, reverse=True)
        return ranked[:max_paths]

    def _register_path(
        self,
        *,
        paths_by_signature: dict[
            tuple[tuple[str, ...], tuple[tuple[str, str, str, str], ...]],
            JoinPath,
        ],
        path_counter: int,
        tables: list[str],
        join_edges: list[JoinEdge],
        table_counts: Counter,
        total_relevant_tables: int,
    ) -> bool:
        """
        Store a join path when it covers at least one relevant table and,
        for joined paths, connects two or more relevant tables.
        """
        relevant_tables_in_path = [table for table in tables if table in table_counts]
        if join_edges and len(set(relevant_tables_in_path)) < 2:
            return False

        join_keys = [
            (edge.left_table, edge.left_column, edge.right_table, edge.right_column)
            for edge in join_edges
        ]
        signature = (tuple(tables), tuple(join_keys))
        if signature in paths_by_signature:
            return False

        paths_by_signature[signature] = JoinPath(
            path_id=f"P{path_counter}",
            tables=tables,
            join_keys=join_keys,
            score=self._score_path(
                tables=tables,
                join_edges=join_edges,
                table_counts=table_counts,
                total_relevant_tables=total_relevant_tables,
            ),
            estimated_output_rows=self._estimate_output_rows(tables),
            join_edges=list(join_edges),
        )
        return True

    def _score_path(
        self,
        *,
        tables: list[str],
        join_edges: list[JoinEdge],
        table_counts: Counter,
        total_relevant_tables: int,
    ) -> float:
        """
        Score a path using relevant-table coverage, edge quality, and length.
        """
        relevant_tables = [table for table in tables if table in table_counts]
        total_hits = sum(table_counts.values()) or 1
        coverage_score = sum(table_counts[table] for table in relevant_tables) / total_hits
        table_coverage_score = len(set(relevant_tables)) / max(total_relevant_tables, 1)
        edge_score = (
            sum(edge.score for edge in join_edges) / len(join_edges)
            if join_edges
            else 1.0
        )
        length_penalty = 0.05 * len(join_edges)
        return (0.45 * coverage_score) + (0.35 * table_coverage_score) + (0.20 * edge_score) - length_penalty

    def _estimate_output_rows(self, tables: list[str]) -> int | None:
        """
        Estimate output size using the smallest available table sample size.
        """
        sample_sizes: list[int] = []
        for table in tables:
            try:
                sample_sizes.append(self.lake.get_sample(table, n=100).height)
            except Exception:  # noqa: BLE001
                return None
        return min(sample_sizes) if sample_sizes else None
