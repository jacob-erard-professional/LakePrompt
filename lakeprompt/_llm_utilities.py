# TOON stands for Token-Oriented Object Notation, a compact JSON-compatible
# prompt format designed for LLM inputs. This module uses TOON-style
# serialization patterns for prompt packaging. Credit to the TOON creators
# and maintainers at https://github.com/toon-format/toon.
# Codex was used to generate the initial version of this module, with subsequent 
# human edits for style and functionality.

import json
import os
import re
from pathlib import Path

import anthropic

from ._models import ColumnCard, QueryFilter, QueryOrder, QueryPlan, QuerySelect
from ._tracing import NULL_LOGGER, PipelineLogger


_BARE_STRING_PATTERN = re.compile(r"^[A-Za-z0-9_./:-]+$")
DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-20250514"
_FENCED_JSON_PATTERN = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_VALID_AGGREGATIONS = {
    "COUNT",
    "SUM",
    "AVG",
    "MIN",
    "MAX",
    "DISTINCT",
}


def _is_scalar(value: object) -> bool:
    """
    Return whether a value is a TOON-serializable scalar.

    This helper is needed so the serializer can choose the right encoding
    path before it emits prompt context.

    Args:
        value: Value to classify.

    Returns:
        True for JSON scalar types, otherwise False.
    """
    return value is None or isinstance(value, (str, int, float, bool))


def _format_scalar(value: object) -> str:
    """
    Format a scalar value using TOON-compatible literal syntax.

    This function is needed because prompt serialization has to preserve
    simple values compactly and consistently.

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

    This is needed so arrays of row-like objects can be rendered in a
    denser, easier-to-read tabular form.

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

    This helper exists so prompt payload arrays are serialized using the
    most compact legal TOON representation.

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

    This helper is needed to recursively serialize nested prompt payloads.

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

    This function is needed because TOON output decisions depend on the
    runtime type of each field.

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
        A TOON document string suitable for inclusion in an LLM prompt.
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

    This helper is needed so multiple callers can share one prompt layout
    and one serialization format.

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

    This helper centralizes client creation so environment validation is
    consistent across utility calls.

    Returns:
        An initialized Anthropic client.

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

    This helper is needed because Anthropic responses can be chunked into
    multiple content blocks.

    Args:
        message: Anthropic message response object.

    Returns:
        A single concatenated text string.
    """
    return "".join(
        block.text for block in message.content if getattr(block, "type", None) == "text"
    ).strip()


def _extract_json_text(raw: str) -> str:
    """
    Strip common markdown fencing around JSON responses.

    Args:
        raw: Raw model text.

    Returns:
        Plain JSON text when recoverable, otherwise the original string.
    """
    match = _FENCED_JSON_PATTERN.search(raw.strip())
    if match:
        return match.group(1).strip()
    return raw.strip()


def _fallback_table_summary(table_name: str, cards: list[ColumnCard]) -> str:
    """
    Build a deterministic local fallback summary for a table.
    """
    column_names = [card.column_name for card in cards[:5]]
    sample_fragments = [
        f"{card.column_name}={card.sample_values[0]}"
        for card in cards
        if card.sample_values
    ][:3]

    parts = [f"Table {table_name}"]
    if column_names:
        parts.append(f"with columns {', '.join(column_names)}")
    if sample_fragments:
        parts.append(f"and sample values such as {', '.join(sample_fragments)}")
    return " ".join(parts) + "."


def generate_table_summaries(
    cards_by_table: dict[str, list[ColumnCard]],
    batch_size: int = 5,
    model: str = DEFAULT_CLAUDE_MODEL,
    cache_path: str = None,
    logger: PipelineLogger | None = None,
) -> dict[str, str]:
    """
    Generate natural-language summaries for all tables in the lake.

    Sends tables to the LLM in batches to reduce API calls. Results are
    optionally cached to disk so summaries are not regenerated on every run.
    Table summaries improve retrieval because column names alone are often
    too sparse to capture what a table is about.

    Args:
        cards_by_table: Dictionary mapping table name to its ColumnCards.
        batch_size: Number of tables per API call.
        model: Anthropic Claude model string.
        cache_path: Optional path to a JSON cache file.

    Returns:
        Dictionary mapping table name to its generated summary string.
    """
    logger = logger or NULL_LOGGER

    if cache_path and os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as handle:
            summaries = json.load(handle)
        remaining = {t: c for t, c in cards_by_table.items() if t not in summaries}
        if not remaining:
            return summaries
    else:
        summaries = {}
        remaining = cards_by_table

    cache_file = Path(cache_path) if cache_path else None

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
        logger.log("llm_request", "Requesting table summaries.", {"prompt": prompt, "tables": batch})

        response = client.messages.create(
            model=model,
            max_tokens=500,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = _response_text(response)
        logger.log("llm_response", "Received table summaries.", {"response": raw, "tables": batch})
        try:
            parsed = json.loads(_extract_json_text(raw))
        except json.JSONDecodeError:
            parsed = {}

        if not isinstance(parsed, dict):
            parsed = {}

        for table_name in batch:
            summary = parsed.get(table_name)
            if isinstance(summary, str) and summary.strip():
                summaries[table_name] = summary.strip()
            else:
                fallback = _fallback_table_summary(table_name, remaining[table_name])
                summaries[table_name] = fallback
                logger.log(
                    "summary_fallback",
                    "Used fallback table summary.",
                    {"table": table_name, "summary": fallback},
                )

        if cache_file is not None:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with cache_file.open("w", encoding="utf-8") as handle:
                json.dump(summaries, handle, indent=2)

    return summaries


def _normalize_query_filter(item: object) -> QueryFilter | None:
    """
    Convert a JSON object into a QueryFilter if it is well-formed.

    This helper is needed so LLM output is validated before it influences
    SQL generation.

    Args:
        item: Candidate JSON-like object.

    Returns:
        A `QueryFilter` if the object is valid, otherwise `None`.
    """
    if not isinstance(item, dict):
        return None

    column = item.get("column")
    operator = item.get("operator")
    value = item.get("value")
    table = item.get("table")

    if not isinstance(column, str) or not column.strip():
        return None
    if not isinstance(operator, str) or not operator.strip():
        return None
    if table is not None and not isinstance(table, str):
        table = None
    operator = operator.strip().upper()

    return QueryFilter(
        table=table,
        column=column.strip(),
        operator=operator,
        value=value,
    )


def _normalize_query_select(item: object) -> QuerySelect | None:
    """
    Convert a JSON object into a QuerySelect if it is well-formed.

    Args:
        item: Candidate JSON-like object.

    Returns:
        A `QuerySelect` if the object is valid, otherwise `None`.
    """
    if not isinstance(item, dict):
        return None

    column = item.get("column")
    table = item.get("table")
    aggregation = item.get("aggregation")

    if not isinstance(column, str) or not column.strip():
        return None
    if table is not None and not isinstance(table, str):
        table = None
    if aggregation is not None and not isinstance(aggregation, str):
        aggregation = None
    if isinstance(aggregation, str):
        aggregation = aggregation.strip().upper()
        if aggregation and aggregation not in _VALID_AGGREGATIONS:
            aggregation = None

    return QuerySelect(
        table=table,
        column=column.strip(),
        aggregation=aggregation if isinstance(aggregation, str) and aggregation else None,
    )


def _normalize_query_order(item: object) -> QueryOrder | None:
    """
    Convert a JSON object into a QueryOrder if it is well-formed.

    Args:
        item: Candidate JSON-like object.

    Returns:
        A `QueryOrder` if the object is valid, otherwise `None`.
    """
    if not isinstance(item, dict):
        return None

    column = item.get("column")
    direction = item.get("direction", "asc")
    table = item.get("table")
    aggregation = item.get("aggregation")

    if not isinstance(column, str) or not column.strip():
        return None
    if not isinstance(direction, str) or not direction.strip():
        direction = "asc"
    if table is not None and not isinstance(table, str):
        table = None
    if aggregation is not None and not isinstance(aggregation, str):
        aggregation = None
    if isinstance(aggregation, str):
        aggregation = aggregation.strip().upper()
        if aggregation and aggregation not in _VALID_AGGREGATIONS:
            aggregation = None

    direction = direction.strip().lower()
    if direction not in {"asc", "desc"}:
        direction = "asc"

    return QueryOrder(
        table=table,
        column=column.strip(),
        direction=direction,
        aggregation=aggregation if isinstance(aggregation, str) and aggregation else None,
    )


def _parse_query_plan(raw: str) -> QueryPlan | None:
    """
    Parse an LLM JSON response into a structured QueryPlan.

    This function is needed so free-form model output becomes a validated,
    inspectable structure before SQL is touched.

    Args:
        raw: Raw text returned by the model.

    Returns:
        A parsed `QueryPlan`, or `None` if parsing fails.
    """
    try:
        parsed = json.loads(_extract_json_text(raw))
    except json.JSONDecodeError:
        return None

    if not isinstance(parsed, dict):
        return None

    filters = [
        normalized
        for item in parsed.get("filters", [])
        if (normalized := _normalize_query_filter(item)) is not None
    ] if isinstance(parsed.get("filters", []), list) else []

    projections = [
        normalized
        for item in parsed.get("projections", [])
        if (normalized := _normalize_query_select(item)) is not None
    ] if isinstance(parsed.get("projections", []), list) else []

    having = [
        normalized
        for item in parsed.get("having", [])
        if (normalized := _normalize_query_filter(item)) is not None
    ] if isinstance(parsed.get("having", []), list) else []

    order_by = [
        normalized
        for item in parsed.get("order_by", [])
        if (normalized := _normalize_query_order(item)) is not None
    ] if isinstance(parsed.get("order_by", []), list) else []

    raw_group_by = parsed.get("group_by", [])
    if isinstance(raw_group_by, list):
        group_by = [item.strip() for item in raw_group_by if isinstance(item, str) and item.strip()]
    else:
        group_by = []

    raw_limit = parsed.get("limit")
    limit = raw_limit if isinstance(raw_limit, int) and raw_limit > 0 else None

    return QueryPlan(
        filters=filters,
        projections=projections,
        group_by=group_by,
        having=having,
        order_by=order_by,
        limit=limit,
    )


def plan_llm_query(
    question: str,
    sql_query: str,
    involved_cards: list[ColumnCard],
    model: str = DEFAULT_CLAUDE_MODEL,
    logger: PipelineLogger | None = None,
) -> QueryPlan:
    """
    Ask an LLM to extract structured query intent for an existing SQL query.

    The model must preserve the existing join structure conceptually and
    return only the incremental filters, projections, grouping, ordering,
    and limit implied by the question. This is needed so the model can
    contribute semantic interpretation without taking control of join logic.

    Args:
        question: Original natural-language user question.
        sql_query: Existing SQL query whose joins should remain unchanged.
        involved_cards: ColumnCards for the tables referenced by the query.
        model: Anthropic Claude model string.

    Returns:
        A structured QueryPlan. If parsing fails, returns an empty plan.
    """
    client = _build_anthropic_client()
    logger = logger or NULL_LOGGER

    payload = {
        "question": question,
        "sql": sql_query,
        "columns": [
            {
                "table": card.table_name,
                "column": card.column_name,
                "dtype": card.dtype,
                "table_summary": card.table_summary,
                "samples": card.sample_values[:3],
            }
            for card in involved_cards
        ],
    }
    prompt = _package_with_toon(
        task=(
            "Extract a structured query plan needed to refine the SQL so it "
            "answers the question. Do not change the joined tables, join keys, "
            "or join order. Return only incremental intent: filters, projected "
            "columns, grouping, having, ordering, and limit.\n"
            "The refined query will be executed with Polars SQL, so only use intent "
            "that can be compiled into valid Polars SQL.\n"
            "Do not rely on unsupported SQL constructs, vendor-specific syntax, or "
            "free-form SQL fragments.\n"
            "Use only exact table names and exact column names from the provided schema.\n"
            "Do not invent synonyms, aliases, normalized names, or paraphrases.\n"
            "Every returned `table` and `column` must exactly match a provided schema entry.\n"
            "If the question uses a synonym such as `secretary` but the schema uses `head`, "
            "you must use the schema name `head`.\n"
            "If the question implies a field but the exact column name is different, use the exact "
            "schema column name.\n"
            "If you are not sure which exact schema field matches, omit that filter or projection "
            "instead of inventing a name."
        ),
        payload=payload,
        response_format=(
            '{"filters":[{"table":"table_name","column":"column_name","operator":"=","value":"literal"}],'
            '"projections":[{"table":"table_name","column":"column_name","aggregation":"SUM"}],'
            '"group_by":["table.column"],'
            '"having":[{"table":"table_name","column":"column_name","operator":">","value":1}],'
            '"order_by":[{"table":"table_name","column":"column_name","direction":"desc","aggregation":"COUNT"}],'
            '"limit":5}'
        ),
    )
    logger.log(
        "llm_request",
        "Requesting structured query plan.",
        {"prompt": prompt, "question": question, "sql_query": sql_query},
    )

    response = client.messages.create(
        model=model,
        max_tokens=700,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = _response_text(response)
    logger.log(
        "llm_response",
        "Received structured query plan response.",
        {"response": raw, "question": question, "sql_query": sql_query},
    )

    plan = _parse_query_plan(raw)
    plan = plan if plan is not None else QueryPlan()
    logger.log("query_plan", "Parsed query plan.", plan)
    return plan


def apply_llm_filters_to_sql(
    question: str,
    sql_query: str,
    involved_cards: list[ColumnCard],
    model: str = DEFAULT_CLAUDE_MODEL,
    logger: PipelineLogger | None = None,
) -> str:
    """
    Backward-compatible wrapper that preserves the old function signature.

    This now extracts a structured `QueryPlan` first and leaves SQL assembly
    to the executor so filter intent is inspectable before execution.

    Args:
        question: Original natural-language user question.
        sql_query: Existing SQL query to refine.
        involved_cards: Column metadata for the query's tables.
        model: Anthropic model name.

    Returns:
        A refined SQL query string.
    """
    from ._executor import apply_query_plan_to_sql

    plan = plan_llm_query(
        question=question,
        sql_query=sql_query,
        involved_cards=involved_cards,
        model=model,
        logger=logger,
    )
    return apply_query_plan_to_sql(sql_query, plan)
