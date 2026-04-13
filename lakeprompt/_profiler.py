from collections import Counter
import math

from ._datalake import DataLake
from ._models import ColumnCard, JoinEdge, JoinPath, QueryPlan
from ._tracing import NULL_LOGGER, PipelineLogger

class LakeProfiler:
    """
    Profile the lake, infer joinable columns, and construct the join graph.

    This class is needed because LakePrompt's core retrieval unit is a
    joined tuple rather than an isolated row. Profiling converts raw
    tables into reusable metadata and graph edges so later stages can
    discover joins without rescanning the full lake on every query.
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
        self.logger: PipelineLogger = NULL_LOGGER
        self._sample_size_cache: dict[str, int | None] = {}

    def profile(self) -> dict[str, list[ColumnCard]]:
        """
        Loads every table in the lake, builds a ColumnCard for each column,
        and stores qualifying cross-table Jaccard matches directly on each
        ColumnCard for constant-time lookup at runtime. This is the main
        preprocessing step that makes semantic retrieval and join planning
        practical during interactive use.
        
        Returns:
            A dictionary mapping each table name to its list of populated
            `ColumnCard` objects.
        """
        cards_by_table: dict[str, list[ColumnCard]] = {}

        for table_name in self.lake.tables:
            sample_df = self.lake.get_sample(table_name, n=100)
            cards: list[ColumnCard] = []
            for column_name in sample_df.columns:
                series = sample_df[column_name]
                cards.append(self._build_column_card(table_name, column_name, series))
            cards_by_table[table_name] = cards

        self.cards_by_table = cards_by_table
        self._populate_jaccard_matches(cards_by_table)
        self.lake.cards = [card for cards in cards_by_table.values() for card in cards]
        self.lake.join_graph = self.build_join_graph(cards_by_table)
        self.logger.log(
            "column_cards",
            "Profiled column cards.",
            [
                {
                    "table": table_name,
                    "cards": cards,
                }
                for table_name, cards in cards_by_table.items()
            ],
        )
        return cards_by_table
    
    def _build_column_card(self, table_name: str, column_name: str, series) -> ColumnCard:
        """
        Build a single `ColumnCard` from one sampled column.

        This helper is needed so every column is described in the same
        structure before retrieval, scoring, and prompt generation.

        Args:
            table_name: Name of the table the column belongs to.
            col_name: Name of the column.
            series: The pandas Series for that column.

        Returns:
            A populated `ColumnCard`.
        """
        sample_values = series.drop_nulls().head(5).to_list()
        non_null_values = series.drop_nulls().to_list()
        sample_size = len(series)
        non_null_count = len(non_null_values)
        unique_count = len({str(v) for v in non_null_values})
        uniqueness_ratio = unique_count / non_null_count if non_null_count else 0.0
        null_ratio = 1.0 - (non_null_count / sample_size) if sample_size else 0.0
        sequentiality_ratio = self._sequentiality_ratio(non_null_values)
        surrogate_key_score = self._surrogate_key_score(
            column_name=column_name,
            dtype=str(series.dtype),
            uniqueness_ratio=uniqueness_ratio,
            null_ratio=null_ratio,
            sequentiality_ratio=sequentiality_ratio,
        )
        foreign_key_score = self._foreign_key_score(
            column_name=column_name,
            dtype=str(series.dtype),
            uniqueness_ratio=uniqueness_ratio,
            null_ratio=null_ratio,
        )

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
            uniqueness_ratio=uniqueness_ratio,
            null_ratio=null_ratio,
            sequentiality_ratio=sequentiality_ratio,
            surrogate_key_score=surrogate_key_score,
            foreign_key_score=foreign_key_score,
        )

    def jaccard_similarity(self, col_a, col_b) -> float:
        """
        Computes Jaccard similarity between the value sets of two columns.

        This score is the current basis for join discovery. It gives the
        profiler a cheap, data-driven signal for whether two columns may
        belong to the same key domain.

        Args:
            col_a: First column as a pandas Series.
            col_b: Second column as a pandas Series.

        Returns:
            A Jaccard score between 0.0 and 1.0.
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
        `ColumnCard` objects using compact lookup keys.

        This method is needed so expensive cross-table comparison work
        happens once during profiling instead of during every query.

        Returns:
            None. Matching scores are written onto cards in place.
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
                        compatibility = self._column_name_compatibility(
                            left_card.column_name,
                            right_card.column_name,
                        )
                        heuristic_score = self._schema_join_score(
                            left_table=left_table,
                            left_column=left_card.column_name,
                            right_table=right_table,
                            right_column=right_card.column_name,
                        )
                        fallback_score = self._high_overlap_fallback_score(
                            left_card=left_card,
                            right_card=right_card,
                            jaccard_score=score,
                        )
                        effective_score = max(score * compatibility, heuristic_score, fallback_score)

                        if effective_score < self._jaccard_threshold:
                            continue

                        card_index[left_key].jaccard_matches[right_key] = effective_score
                        card_index[right_key].jaccard_matches[left_key] = effective_score
                        self.logger.log(
                            "jaccard_match",
                            "Discovered joinable columns.",
                            {
                                "left_table": left_table,
                                "left_column": left_card.column_name,
                                "right_table": right_table,
                                "right_column": right_card.column_name,
                                "score": effective_score,
                                "jaccard_score": score,
                                "schema_score": heuristic_score,
                                "fallback_score": fallback_score,
                                "compatibility": compatibility,
                            },
                        )

        self._refine_foreign_key_scores(cards_by_table)

    def _schema_join_score(
        self,
        *,
        left_table: str,
        left_column: str,
        right_table: str,
        right_column: str,
    ) -> float:
        """
        Return a heuristic join score for obvious schema-level key matches.

        This is primarily meant to recover same-database foreign-key joins
        like `concert_singer__concert.stadium_id` ->
        `concert_singer__stadium.stadium_id` even when Jaccard overlap is
        diluted by sparse benchmark data.
        """
        if self._table_namespace(left_table) != self._table_namespace(right_table):
            return 0.0
        left_card = self._card_for(left_table, left_column)
        right_card = self._card_for(right_table, right_column)
        if left_card is None or right_card is None:
            return 0.0
        if left_column == right_column and (
            max(left_card.surrogate_key_score, right_card.surrogate_key_score) >= 0.7
        ):
            return 0.75
        pk_fk_score = max(
            min(left_card.surrogate_key_score, right_card.foreign_key_score),
            min(right_card.surrogate_key_score, left_card.foreign_key_score),
        )
        if pk_fk_score >= 0.55 and self._column_name_compatibility(left_column, right_column) > 0.0:
            return 0.65
        return 0.0

    def _high_overlap_fallback_score(
        self,
        *,
        left_card: ColumnCard,
        right_card: ColumnCard,
        jaccard_score: float,
    ) -> float:
        """
        Allow a conservative fallback join score for near-identical
        string/code domains even when column names do not match.

        This is meant for cases like `sourceairport` -> `airportcode`,
        while avoiding accidental joins between unrelated surrogate keys.
        """
        if jaccard_score < 0.95:
            return 0.0
        if not self._cards_are_string_like(left_card, right_card):
            return 0.0
        if self._cards_are_both_surrogate_like(left_card, right_card):
            return 0.0
        if self._cards_are_both_numeric_like(left_card, right_card):
            return 0.0
        if not self._fallback_name_pattern_ok(left_card.column_name, right_card.column_name):
            return 0.0
        if not (
            max(left_card.foreign_key_score, right_card.foreign_key_score) >= 0.45
            or self._name_suggests_reference(left_card.column_name)
            or self._name_suggests_reference(right_card.column_name)
        ):
            return 0.0
        return 0.95

    def _cards_are_string_like(self, left_card: ColumnCard, right_card: ColumnCard) -> bool:
        return "string" in left_card.dtype.lower() and "string" in right_card.dtype.lower()

    def _cards_are_both_surrogate_like(self, left_card: ColumnCard, right_card: ColumnCard) -> bool:
        return left_card.surrogate_key_score >= 0.7 and right_card.surrogate_key_score >= 0.7

    def _cards_are_both_numeric_like(self, left_card: ColumnCard, right_card: ColumnCard) -> bool:
        numeric_markers = ("int", "float", "double", "decimal")
        return (
            any(marker in left_card.dtype.lower() for marker in numeric_markers)
            and any(marker in right_card.dtype.lower() for marker in numeric_markers)
        )

    def _fallback_name_pattern_ok(self, left_column: str, right_column: str) -> bool:
        left = left_column.lower()
        right = right_column.lower()
        if self._name_suggests_reference(left) and self._name_suggests_code_key(right):
            return True
        if self._name_suggests_reference(right) and self._name_suggests_code_key(left):
            return True
        return False

    def _column_name_compatibility(self, left_column: str, right_column: str) -> float:
        """
        Return a lightweight schema-name compatibility score for a pair of
        candidate join columns.
        """
        left = left_column.lower()
        right = right_column.lower()
        if left == right:
            if self._name_suggests_key(left):
                return 1.0
            return 0.3
        if left == "id" and right.endswith("_id"):
            return 0.7
        if right == "id" and left.endswith("_id"):
            return 0.7
        if self._name_suggests_key(left) and self._name_suggests_key(right):
            return 0.0
        return 0.0

    def _name_suggests_reference(self, column_name: str) -> bool:
        normalized = column_name.lower()
        return (
            normalized.startswith("source")
            or normalized.startswith("dest")
            or normalized.startswith("from")
            or normalized.startswith("to")
            or normalized.endswith("airport")
            or normalized.endswith("airline")
            or normalized.endswith("code")
        )

    def _name_suggests_code_key(self, column_name: str) -> bool:
        normalized = column_name.lower()
        return (
            normalized.endswith("code")
            or normalized.endswith("uid")
            or normalized.endswith("abbreviation")
            or normalized == "uid"
            or normalized == "code"
        )

    def _name_suggests_key(self, column_name: str) -> bool:
        normalized = column_name.lower()
        return (
            normalized == "id"
            or normalized.endswith("_id")
            or normalized.endswith("_key")
            or normalized.endswith("_num")
            or normalized.endswith("_no")
            or normalized.endswith("number")
        )

    def _card_for(self, table_name: str, column_name: str) -> ColumnCard | None:
        for card in self.cards_by_table.get(table_name, []):
            if card.column_name == column_name:
                return card
        return None

    def _sequentiality_ratio(self, values: list[object]) -> float:
        numeric_values = self._coerce_numeric_list(values)
        if len(numeric_values) < 3:
            return 0.0
        ordered = sorted(set(numeric_values))
        if len(ordered) < 3:
            return 0.0
        deltas = [ordered[idx + 1] - ordered[idx] for idx in range(len(ordered) - 1)]
        if not deltas:
            return 0.0
        near_one = sum(1 for delta in deltas if math.isclose(delta, 1.0, rel_tol=0.0, abs_tol=1e-9))
        return near_one / len(deltas)

    def _coerce_numeric_list(self, values: list[object]) -> list[float]:
        coerced: list[float] = []
        for value in values:
            if isinstance(value, bool) or value is None:
                continue
            if isinstance(value, (int, float)):
                coerced.append(float(value))
                continue
            try:
                coerced.append(float(str(value)))
            except (TypeError, ValueError):
                return []
        return coerced

    def _surrogate_key_score(
        self,
        *,
        column_name: str,
        dtype: str,
        uniqueness_ratio: float,
        null_ratio: float,
        sequentiality_ratio: float,
    ) -> float:
        dtype_lower = dtype.lower()
        integer_like = 1.0 if "int" in dtype_lower else 0.0
        name_bonus = 0.15 if self._name_suggests_key(column_name) else 0.0
        score = (
            0.35 * integer_like
            + 0.30 * uniqueness_ratio
            + 0.20 * (1.0 - null_ratio)
            + 0.15 * sequentiality_ratio
            + name_bonus
        )
        return min(score, 1.0)

    def _foreign_key_score(
        self,
        *,
        column_name: str,
        dtype: str,
        uniqueness_ratio: float,
        null_ratio: float,
    ) -> float:
        dtype_lower = dtype.lower()
        integer_like = 1.0 if "int" in dtype_lower else 0.0
        name_bonus = 0.15 if self._name_suggests_key(column_name) else 0.0
        duplicated_ratio = 1.0 - uniqueness_ratio
        score = (
            0.35 * integer_like
            + 0.35 * duplicated_ratio
            + 0.15 * (1.0 - null_ratio)
            + name_bonus
        )
        return min(score, 1.0)

    def _refine_foreign_key_scores(
        self,
        cards_by_table: dict[str, list[ColumnCard]],
    ) -> None:
        """
        Increase foreign-key confidence for columns that overlap with likely
        surrogate keys in other tables.
        """
        for table_name, cards in cards_by_table.items():
            for card in cards:
                best_overlap = 0.0
                for (other_table, other_column), score in card.jaccard_matches.items():
                    if other_table == table_name:
                        continue
                    other_card = self._card_for(other_table, other_column)
                    if other_card is None:
                        continue
                    overlap = score * other_card.surrogate_key_score
                    if overlap > best_overlap:
                        best_overlap = overlap
                card.foreign_key_score = min(1.0, max(card.foreign_key_score, best_overlap))

    def build_join_graph(self, cards_by_table: dict[str, list[ColumnCard]]):
        """
        Build a table-level join graph from the Jaccard matches already
        stored on each `ColumnCard`.

        The graph is needed because join planning is a graph-search
        problem. Pairwise column matches alone are not enough to recover
        multi-hop paths through the lake.

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
        query_plan: QueryPlan | None = None,
        max_paths: int = 10,
        max_hops: int = 3,
    ) -> list[JoinPath]:
        """
        Build join-path candidates from relevant cards and the join graph.

        Produces single-table and multi-hop join paths ranked by score.
        This method is needed because relevant columns alone are not
        executable; the executor needs an ordered join chain it can
        compile into SQL.

        Args:
            relevant_cards: ColumnCards relevant to the current prompt.
            query_plan: Optional structured question intent for path scoring.
            max_paths: Maximum number of join paths to return. Defaults to 10.
            max_hops: Maximum join depth in graph edges. Defaults to 3.
        
        Returns:
            A list of `JoinPath` objects sorted by descending score.
        """
        if not self.lake.join_graph:
            raise ValueError("Join graph not found. Please run profile() first.")

        if not relevant_cards:
            return []

        table_counts = Counter(card.table_name for card in relevant_cards)
        allowed_namespaces = self._allowed_namespaces(table_counts, query_plan)
        if allowed_namespaces:
            table_counts = Counter(
                {
                    table: count
                    for table, count in table_counts.items()
                    if self._table_namespace(table) in allowed_namespaces
                }
            )
        target_tables = set(table_counts) | self._plan_tables(query_plan)
        if len(table_counts) == 1 and target_tables == set(table_counts):
            ranked = self._single_table_paths(
                table_counts,
                query_plan=query_plan,
                max_paths=max_paths,
            )
            self.logger.log("join_paths_progress", "Short-circuited to single-table join planning.", ranked)
            self.logger.log("join_paths", "Ranked join paths.", ranked)
            return ranked

        total_relevant_tables = len(table_counts)
        paths_by_signature: dict[
            tuple[tuple[str, ...], tuple[tuple[str, str, str, str], ...]],
            JoinPath,
        ] = {}
        path_counter = 1
        explored_states = 0
        max_expansions = max(200, max_paths * 100)
        self.logger.log(
            "join_paths_progress",
            "Starting join-path exploration.",
            {
                "root_tables": sorted(table_counts),
                "plan_tables": sorted(target_tables - set(table_counts)),
                "allowed_namespaces": sorted(allowed_namespaces),
                "max_hops": max_hops,
                "max_paths": max_paths,
                "max_expansions": max_expansions,
            },
        )

        for root_table in sorted(table_counts):
            if allowed_namespaces and self._table_namespace(root_table) not in allowed_namespaces:
                continue
            if self._register_path(
                paths_by_signature=paths_by_signature,
                path_counter=path_counter,
                tables=[root_table],
                join_edges=[],
                table_counts=table_counts,
                total_relevant_tables=total_relevant_tables,
                query_plan=query_plan,
                target_tables=target_tables,
            ):
                path_counter += 1

            stack: list[tuple[list[str], list[JoinEdge]]] = [([root_table], [])]
            while stack:
                explored_states += 1
                if explored_states % 100 == 0:
                    self.logger.log(
                        "join_paths_progress",
                        "Exploring join-path candidates.",
                        {
                            "root_table": root_table,
                            "explored_states": explored_states,
                            "stack_size": len(stack),
                            "registered_paths": len(paths_by_signature),
                        },
                    )
                if explored_states >= max_expansions:
                    self.logger.log(
                        "join_paths_progress",
                        "Stopping join-path exploration at expansion cap.",
                        {
                            "explored_states": explored_states,
                            "registered_paths": len(paths_by_signature),
                            "max_expansions": max_expansions,
                        },
                    )
                    stack.clear()
                    break

                tables, join_edges = stack.pop()
                if target_tables and target_tables.issubset(set(tables)):
                    continue
                if len(join_edges) >= max_hops:
                    continue

                current_table = tables[-1]
                for raw_edge in self.lake.join_graph.get(current_table, []):
                    next_table = raw_edge["to_table"]
                    if next_table in tables:
                        continue
                    if not self._should_expand_to_table(
                        next_table=next_table,
                        current_tables=tables,
                        table_counts=table_counts,
                        query_plan=query_plan,
                        target_tables=target_tables,
                        remaining_hops=max_hops - len(join_edges),
                        allowed_namespaces=allowed_namespaces,
                    ):
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
                        query_plan=query_plan,
                        target_tables=target_tables,
                    ):
                        path_counter += 1

                    stack.append((next_tables, next_edges))

        ranked = sorted(paths_by_signature.values(), key=lambda p: p.score, reverse=True)
        self.logger.log("join_paths", "Ranked join paths.", ranked[:max_paths])
        return ranked[:max_paths]

    def _single_table_paths(
        self,
        table_counts: Counter,
        *,
        query_plan: QueryPlan | None,
        max_paths: int,
    ) -> list[JoinPath]:
        """
        Build ranked single-table candidates without graph traversal.
        """
        total_relevant_tables = len(table_counts)
        target_tables = set(table_counts) | self._plan_tables(query_plan)
        paths: list[JoinPath] = []
        for idx, table in enumerate(sorted(table_counts), start=1):
            if target_tables and not target_tables.issubset({table}):
                continue
            paths.append(
                JoinPath(
                    path_id=f"P{idx}",
                    tables=[table],
                    join_keys=[],
                    score=self._score_path(
                        tables=[table],
                        join_edges=[],
                        table_counts=table_counts,
                        total_relevant_tables=total_relevant_tables,
                        query_plan=query_plan,
                    ),
                    estimated_output_rows=self._estimate_output_rows([table]),
                    join_edges=[],
                )
            )
        return sorted(paths, key=lambda p: p.score, reverse=True)[:max_paths]

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
        query_plan: QueryPlan | None,
        target_tables: set[str],
    ) -> bool:
        """
        Store a join path when it covers at least one relevant table and,
        for joined paths, connects two or more relevant tables.

        This helper is needed so duplicate suppression, scoring, path IDs,
        and output-size estimation all happen in one place.

        Returns:
            `True` if the path was accepted and stored, otherwise `False`.
        """
        relevant_tables_in_path = [table for table in tables if table in table_counts]
        covered_target_tables = set(tables) & target_tables
        if join_edges and len(covered_target_tables) < 2:
            return False
        if target_tables and not target_tables.issubset(set(tables)):
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
                query_plan=query_plan,
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
        query_plan: QueryPlan | None,
    ) -> float:
        """
        Score a path using relevant-table coverage, edge quality, plan
        alignment, and a light length penalty.

        This score is needed so the executor can spend work first on the
        most plausible join plans rather than exploring arbitrary graph
        paths.

        Returns:
            A floating-point relevance score where larger values are better.
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
        plan_alignment_score = self._score_plan_alignment(tables, query_plan)
        output_size_score = self._score_output_size(tables)
        namespace_score = self._score_namespace_alignment(tables, query_plan)
        missing_relevant_penalty = 0.10 * max(total_relevant_tables - len(set(relevant_tables)), 0)
        plan_tables = self._plan_tables(query_plan)
        missing_plan_penalty = 0.10 * len(plan_tables - set(tables))
        length_penalty = 0.05 * len(join_edges)
        return (
            (0.30 * coverage_score)
            + (0.25 * table_coverage_score)
            + (0.20 * edge_score)
            + (0.15 * plan_alignment_score)
            + (0.05 * output_size_score)
            + (0.05 * namespace_score)
            - length_penalty
            - missing_relevant_penalty
            - missing_plan_penalty
        )

    def _estimate_output_rows(self, tables: list[str]) -> int | None:
        """
        Estimate output size using the smallest available table sample size.

        The estimate is heuristic, but it is still useful as a lightweight
        signal for path ranking and debugging.

        Args:
            tables: Ordered list of table names in the candidate path.

        Returns:
            An integer estimate of output size, or `None` if sampling fails.
        """
        sample_sizes: list[int] = []
        for table in tables:
            if table not in self._sample_size_cache:
                try:
                    self._sample_size_cache[table] = self.lake.get_sample(table, n=100).height
                except Exception:  # noqa: BLE001
                    self._sample_size_cache[table] = None
            sample_size = self._sample_size_cache[table]
            if sample_size is None:
                return None
            sample_sizes.append(sample_size)
        return min(sample_sizes) if sample_sizes else None

    def _score_plan_alignment(self, tables: list[str], query_plan: QueryPlan | None) -> float:
        """
        Reward paths that include tables explicitly referenced by the query plan.

        This improves join ranking by favoring paths that agree with the
        structured intent extracted from the question.

        Args:
            tables: Tables present in the candidate path.
            query_plan: Optional structured question intent.

        Returns:
            A normalized score between 0.0 and 1.0.
        """
        if query_plan is None:
            return 0.0

        plan_tables: set[str] = set()
        for filter_ in query_plan.filters + query_plan.having:
            if filter_.table:
                plan_tables.add(filter_.table)
        for projection in query_plan.projections:
            if projection.table:
                plan_tables.add(projection.table)
        for order in query_plan.order_by:
            if order.table:
                plan_tables.add(order.table)
        for item in query_plan.group_by:
            if "." in item:
                plan_tables.add(item.split(".", 1)[0])

        if not plan_tables:
            return 0.0

        return len(plan_tables & set(tables)) / len(plan_tables)

    def _plan_tables(self, query_plan: QueryPlan | None) -> set[str]:
        """
        Return the set of tables explicitly referenced by the query plan.
        """
        if query_plan is None:
            return set()
        plan_tables: set[str] = set()
        for filter_ in query_plan.filters + query_plan.having:
            if filter_.table:
                plan_tables.add(filter_.table)
        for projection in query_plan.projections:
            if projection.table:
                plan_tables.add(projection.table)
        for order in query_plan.order_by:
            if order.table:
                plan_tables.add(order.table)
        for item in query_plan.group_by:
            if "." in item:
                plan_tables.add(item.split(".", 1)[0])
        return plan_tables

    def _should_expand_to_table(
        self,
        *,
        next_table: str,
        current_tables: list[str],
        table_counts: Counter,
        query_plan: QueryPlan | None,
        target_tables: set[str],
        remaining_hops: int,
        allowed_namespaces: set[str],
    ) -> bool:
        """
        Decide whether expanding to a neighboring table is useful enough.
        """
        if allowed_namespaces and self._table_namespace(next_table) not in allowed_namespaces:
            return False
        if next_table in table_counts:
            return True
        plan_tables = self._plan_tables(query_plan)
        if next_table in plan_tables:
            return True
        missing_targets = target_tables - set(current_tables) - {next_table}
        if not missing_targets or remaining_hops <= 1:
            return False
        return self._can_reach_targets(
            start_table=next_table,
            target_tables=missing_targets,
            blocked_tables=set(current_tables),
            max_steps=remaining_hops - 1,
        )

    def _can_reach_targets(
        self,
        *,
        start_table: str,
        target_tables: set[str],
        blocked_tables: set[str],
        max_steps: int,
    ) -> bool:
        """
        Return whether a bridge table can still reach a missing target table
        within the remaining hop budget.
        """
        if start_table in target_tables:
            return True
        if max_steps <= 0:
            return False

        frontier: list[tuple[str, int]] = [(start_table, 0)]
        visited = set(blocked_tables)
        visited.add(start_table)

        while frontier:
            table, depth = frontier.pop(0)
            if depth >= max_steps:
                continue
            for edge in self.lake.join_graph.get(table, []):
                neighbor = edge["to_table"]
                if neighbor in visited:
                    continue
                if neighbor in target_tables:
                    return True
                visited.add(neighbor)
                frontier.append((neighbor, depth + 1))

        return False

    def _score_namespace_alignment(self, tables: list[str], query_plan: QueryPlan | None) -> float:
        """
        Reward paths that stay inside the dominant database namespace when
        table names use `namespace__table` formatting.
        """
        allowed_namespaces = self._allowed_namespaces(Counter(tables), query_plan)
        if not allowed_namespaces:
            return 0.0
        matching = sum(
            1 for table in tables
            if self._table_namespace(table) in allowed_namespaces
        )
        return matching / max(len(tables), 1)

    def _allowed_namespaces(
        self,
        table_counts: Counter,
        query_plan: QueryPlan | None,
    ) -> set[str]:
        """
        Restrict Spider-style multi-database joins to namespaces anchored by
        the query plan when possible.
        """
        plan_namespaces = {
            namespace
            for table in self._plan_tables(query_plan)
            if (namespace := self._table_namespace(table))
        }
        if plan_namespaces:
            return plan_namespaces

        relevant_namespaces = {
            namespace
            for table in table_counts
            if (namespace := self._table_namespace(table))
        }
        if len(relevant_namespaces) == 1:
            return relevant_namespaces
        return set()

    def _table_namespace(self, table_name: str) -> str | None:
        """
        Return the leading namespace for table names shaped like
        `database__table`.
        """
        if "__" not in table_name:
            return None
        return table_name.split("__", 1)[0]

    def _score_output_size(self, tables: list[str]) -> float:
        """
        Prefer paths with smaller estimated output sizes.

        This keeps the planner biased toward join paths that are less
        likely to explode in size before evidence ranking.

        Args:
            tables: Tables present in the candidate path.

        Returns:
            A normalized preference score where larger values mean smaller
            estimated outputs.
        """
        estimated = self._estimate_output_rows(tables)
        if estimated is None:
            return 0.0
        return 1.0 / max(estimated, 1)
