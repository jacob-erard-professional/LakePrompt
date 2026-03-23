import os
import json
from openai import OpenAI
from .datalake import DataLake
from .models import ColumnCard







# Note to James: This is used for generating semantic embeddings necessary for HNSW. 
# Feel free to move it around, but this wasn't your job so I made it
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