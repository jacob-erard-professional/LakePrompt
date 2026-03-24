import os
import json
from openai import OpenAI
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
        self.jaccard_threshold = jaccard_threshold
        self.cards_by_table: dict[str, list[ColumnCard]] = {}

    def profile(self) -> dict[str, list[ColumnCard]]:
        """
        Loads every table in the lake and builds a ColumnCard for each column
        
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

        self.cards_by_table = cards_by_table
        self.lake.cards = [card for cards in cards_by_table.values() for card in cards]
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

    def build_join_graph(self, cards_by_table: dict[str, list[ColumnCard]]):
        """
        Compares every column pair across all table combinations and adds
        an edge to the join graph when Jaccard score meets the threshold.

        Args:
            cards_by_table: Output from profile().

        Returns:
            A graph where nodes are table names and edges carry the joinable
            column names and their Jaccard score.
        """
        table_names = list(cards_by_table.keys())
        graph: dict[str, list[dict[str, object]]] = {t: [] for t in table_names}
        value_cache: dict[tuple[str, str], set[str]] = {}

        for i, left_table in enumerate(table_names):
            for right_table in table_names[i + 1:]:
                left_cards = cards_by_table[left_table]
                right_cards = cards_by_table[right_table]

                for left_card in left_cards:
                    left_key = (left_table, left_card.column_name)
                    if left_key not in value_cache:
                        value_cache[left_key] = self.lake.get_column_values(*left_key)

                    for right_card in right_cards:
                        right_key = (right_table, right_card.column_name)
                        if right_key not in value_cache:
                            value_cache[right_key] = self.lake.get_column_values(*right_key)

                        score = self.jaccard_similarity(
                            value_cache[left_key],
                            value_cache[right_key],
                        )

                        if score >= self.jaccard_threshold:
                            left_edge = {
                                "to_table": right_table,
                                "left_column": left_card.column_name,
                                "right_column": right_card.column_name,
                                "score": score,
                            }
                            right_edge = {
                                "to_table": left_table,
                                "left_column": right_card.column_name,
                                "right_column": left_card.column_name,
                                "score": score,
                            }
                            graph[left_table].append(left_edge)
                            graph[right_table].append(right_edge)

        self.lake.join_graph = graph
        return graph

    def profile_lake(self) -> dict[str, list[ColumnCard]]:
        """Compatibility wrapper for older pipeline naming."""
        return self.profile()

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
        # TODO: Find a way to do this, and ask ppl about what score means in joinpath.


class DataProfiler(LakeProfiler):
    """Backward-compatible alias for older references to DataProfiler."""

def generate_table_summaries(
    cards_by_table: dict[str, list[ColumnCard]],
    batch_size: int = 5,
    model: str = "nvidia/nemotron-3-super-120b-a12b:free",
    cache_path: str = None
) -> dict[str, str]:
    """
    Generate natural language summaries for all tables in the lake.

    Sends tables to the LLM in batches to reduce API calls. Results are
    optionally cached to disk so summaries are not regenerated on every run.
    
    If cache_path is provided and the file exists, previously generated
    summaries are loaded from disk and only tables missing from the cache
    are sent to the LLM. If all tables are already cached, no API call
    is made at all. Results are saved back to the cache after each run,
    so adding a new table to the lake only costs one incremental API call.

    Args:
        cards_by_table: Dictionary mapping table name to its ColumnCards.
        batch_size: Number of tables per API call. Defaults to 5.
        model: OpenRouter model string. Defaults to 'nvidia/nemotron-3-super-120b-a12b:free'.
        cache_path: Optional path to a JSON cache file.

    Returns:
        Dictionary mapping table name to its generated summary string.
    """
    if cache_path and os.path.exists(cache_path):
        with open(cache_path) as f:
            summaries = json.load(f)
        remaining = {t: c for t, c in cards_by_table.items() if t not in summaries}
        if not remaining:
            return summaries
    else:
        summaries = {}
        remaining = cards_by_table

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"]
    )

    table_names = list(remaining.keys())
    batches = [table_names[i:i + batch_size] for i in range(0, len(table_names), batch_size)]

    for batch in batches:
        batch_descriptions = ""
        for table_name in batch:
            col_text = "\n".join(
                f"  - {c.column_name} ({c.dtype}): {c.sample_values[:5]}"
                for c in remaining[table_name]
            )
            batch_descriptions += f"Table: '{table_name}'\n{col_text}\n\n"

        prompt = (
            f"""
            You are helping to document a data lake.

        Below are several tables with their column names and sample values.
        For each table, write a single concise sentence describing what it represents
        in plain English. Do not mention column names directly.

        {batch_descriptions}
        Respond in JSON format like this:
        {{
        "table_name_1": "summary here",
        "table_name_2": "summary here"
        }}
        Only include the JSON in your response, nothing else."""
        )

        response = client.chat.completions.create(
            model=model,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )

        summaries.update(json.loads(response.choices[0].message.content.strip()))

    if cache_path:
        with open(cache_path, "w") as f:
            json.dump(summaries, f, indent=2)

    return summaries
