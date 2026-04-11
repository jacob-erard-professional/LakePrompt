# TOON stands for Token-Oriented Object Notation, a compact JSON-compatible
# prompt format designed for LLM inputs. This module uses TOON-style
# serialization patterns for prompt packaging. Credit to the TOON creators
# and maintainers at https://github.com/toon-format/toon.
# Codex was used to generate the initial version of this module, with subsequent 
# human edits for style and functionality.

import json
import os
import re

import anthropic

from .models import ColumnCard


_BARE_STRING_PATTERN = re.compile(r"^[A-Za-z0-9_./:-]+$")
DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-20250514"


def _is_scalar(value: object) -> bool:
    """
    Return whether a value is a TOON-serializable scalar.

    Args:
        value: Value to classify.

    Returns:
        True for JSON scalar types, otherwise False.
    """
    return value is None or isinstance(value, (str, int, float, bool))


def _format_scalar(value: object) -> str:
    """
    Format a scalar value using TOON-compatible literal syntax.

    Args:
        value: Scalar JSON-like value.

    Returns:
        A TOON scalar literal.
    """
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if value == "":
        return '""'
    if isinstance(value, str) and _BARE_STRING_PATTERN.match(value):
        return value
    return json.dumps(value, ensure_ascii=True)


def _is_uniform_object_array(value: list[object]) -> bool:
    """
    Check whether a list qualifies for TOON tabular array encoding.

    Args:
        value: Array candidate.

    Returns:
        True if every element is an object with identical keys and scalar
        values, otherwise False.
    """
    if not value or not all(isinstance(item, dict) for item in value):
        return False

    field_names = list(value[0].keys())
    for item in value:
        if list(item.keys()) != field_names:
            return False
        if not all(_is_scalar(cell) for cell in item.values()):
            return False
    return True


def _encode_array(name: str, value: list[object], indent: int) -> list[str]:
    """
    Encode a named array using TOON array syntax.

    Args:
        name: Field name for the array.
        value: Array to encode.
        indent: Current indentation level.

    Returns:
        Encoded TOON lines.
    """
    prefix = " " * indent
    if not value:
        return [f"{prefix}{name}[0]:"]

    if all(_is_scalar(item) for item in value):
        inline_items = ",".join(_format_scalar(item) for item in value)
        return [f"{prefix}{name}[{len(value)}]: {inline_items}"]

    if _is_uniform_object_array(value):
        field_names = list(value[0].keys())
        lines = [f"{prefix}{name}[{len(value)}]{{{','.join(field_names)}}}:"]
        for item in value:
            row = ",".join(_format_scalar(item[field]) for field in field_names)
            lines.append(f"{prefix}  {row}")
        return lines

    lines = [f"{prefix}{name}[{len(value)}]:"]
    for index, item in enumerate(value):
        entry_name = f"[{index}]"
        lines.extend(_encode_named_value(entry_name, item, indent + 2))
    return lines


def _encode_object(value: dict[str, object], indent: int) -> list[str]:
    """
    Encode an object using TOON indentation-based object syntax.

    Args:
        value: Object to encode.
        indent: Current indentation level.

    Returns:
        Encoded TOON lines.
    """
    lines: list[str] = []
    for key, child in value.items():
        lines.extend(_encode_named_value(key, child, indent))
    return lines


def _encode_named_value(name: str, value: object, indent: int) -> list[str]:
    """
    Encode a named JSON value into TOON lines.

    Args:
        name: Field name.
        value: Value to encode.
        indent: Current indentation level.

    Returns:
        Encoded TOON lines.
    """
    prefix = " " * indent
    if _is_scalar(value):
        return [f"{prefix}{name}: {_format_scalar(value)}"]
    if isinstance(value, dict):
        lines = [f"{prefix}{name}:"]
        lines.extend(_encode_object(value, indent + 2))
        return lines
    if isinstance(value, list):
        return _encode_array(name, value, indent)
    return [f"{prefix}{name}: {json.dumps(value, ensure_ascii=True, separators=(',', ':'))}"]


def _encode_to_toon(value: dict[str, object]) -> str:
    """
    Serialize a JSON-like object into TOON.

    This encoder follows the core TOON patterns documented by the official
    TOON project: indentation-based objects, `[N]` array lengths, and
    `{field,...}` tabular headers for uniform arrays of objects.

    Args:
        value: Root object to encode.

    Returns:
        A TOON document string.
    """
    return "\n".join(_encode_object(value, 0))


def _package_with_toon(
    *,
    task: str,
    payload: dict[str, object],
    response_format: str,
) -> str:
    """
    Build a prompt whose structured context is serialized as TOON.

    Args:
        task: Short description of the model task.
        payload: Structured context for the task.
        response_format: Exact response shape the model must return.

    Returns:
        A prompt containing TOON-serialized context.
    """
    prompt_document = {
        "task": task,
        "context": payload,
        "output_format": response_format,
        "rules": [
            "Be concise.",
            "Return only the requested JSON.",
        ],
    }
    toon_document = _encode_to_toon(prompt_document)
    return (
        "The following context is encoded as TOON "
        "(Token-Oriented Object Notation). Use it exactly.\n\n"
        f"{toon_document}"
    )


def _build_anthropic_client() -> anthropic.Anthropic:
    """
    Build an Anthropic client from environment configuration.

    Raises:
        ValueError: If `ANTHROPIC_API_KEY` is not set.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY is not set.")
    return anthropic.Anthropic(api_key=api_key)


def _response_text(message: anthropic.types.Message) -> str:
    """
    Concatenate text blocks from an Anthropic Messages API response.
    """
    return "".join(
        block.text for block in message.content if getattr(block, "type", None) == "text"
    ).strip()


def generate_table_summaries(
    cards_by_table: dict[str, list[ColumnCard]],
    batch_size: int = 5,
    model: str = DEFAULT_CLAUDE_MODEL,
    cache_path: str = None,
) -> dict[str, str]:
    """
    Generate natural-language summaries for all tables in the lake.

    Sends tables to the LLM in batches to reduce API calls. Results are
    optionally cached to disk so summaries are not regenerated on every run.

    Args:
        cards_by_table: Dictionary mapping table name to its ColumnCards.
        batch_size: Number of tables per API call.
        model: Anthropic Claude model string.
        cache_path: Optional path to a JSON cache file.

    Returns:
        Dictionary mapping table name to its generated summary string.
    """
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as handle:
            summaries = json.load(handle)
        remaining = {t: c for t, c in cards_by_table.items() if t not in summaries}
        if not remaining:
            return summaries
    else:
        summaries = {}
        remaining = cards_by_table

    client = _build_anthropic_client()

    table_names = list(remaining.keys())
    batches = [table_names[i:i + batch_size] for i in range(0, len(table_names), batch_size)]

    for batch in batches:
        batch_payload = {
            table_name: [
                {
                    "column": card.column_name,
                    "dtype": card.dtype,
                    "samples": card.sample_values[:5],
                }
                for card in remaining[table_name]
            ]
            for table_name in batch
        }
        prompt = _package_with_toon(
            task="Write one concise plain-English sentence describing each table without naming columns.",
            payload=batch_payload,
            response_format='{"table_name_1":"summary","table_name_2":"summary"}',
        )

        response = client.messages.create(
            model=model,
            max_tokens=500,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        summaries.update(json.loads(_response_text(response)))

    if cache_path:
        with open(cache_path, "w", encoding="utf-8") as handle:
            json.dump(summaries, handle, indent=2)

    return summaries


def apply_llm_filters_to_sql(
    question: str,
    sql_query: str,
    involved_cards: list[ColumnCard],
    model: str = DEFAULT_CLAUDE_MODEL,
) -> str:
    """
    Ask an LLM to refine an existing SQL join query with filters and
    aggregations when needed.

    The model is instructed to preserve the existing join structure and
    may add filtering predicates, grouping, aggregations, HAVING clauses,
    ordering, or limits implied by the user question and the columns
    available in the involved tables.

    Args:
        question: Original natural-language user question.
        sql_query: Existing SQL query whose joins should remain unchanged.
        involved_cards: ColumnCards for the tables referenced by the query.
        model: Anthropic Claude model string.

    Returns:
        The refined SQL query. If parsing fails, returns the original query.
    """
    client = _build_anthropic_client()

    payload = {
        "question": question,
        "sql": sql_query,
        "columns": [
            {
                "table": card.table_name,
                "column": card.column_name,
                "dtype": card.dtype,
                "samples": card.sample_values[:3],
            }
            for card in involved_cards
        ],
    }
    prompt = _package_with_toon(
        task=(
            "Refine the SQL to answer the question. You may add WHERE filters, "
            "filter CTE predicates, aggregations, GROUP BY, HAVING, ORDER BY, "
            "and LIMIT when needed. Do not change the joined tables, join keys, "
            "or join order."
        ),
        payload=payload,
        response_format='{"refined_sql":"SELECT ..."}',
    )

    response = client.messages.create(
        model=model,
        max_tokens=700,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = _response_text(response)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return sql_query

    refined_sql = parsed.get("refined_sql")
    if not isinstance(refined_sql, str) or not refined_sql.strip():
        return sql_query

    return refined_sql.strip()
