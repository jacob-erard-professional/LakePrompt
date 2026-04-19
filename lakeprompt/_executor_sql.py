import logging
import re
from difflib import SequenceMatcher
from dataclasses import dataclass, replace

from ._datalake import DataLake
from ._llm_utilities import plan_llm_query
from ._models import JoinPath, QueryFilter, QueryOrder, QueryPlan, QuerySelect
from ._tracing import NULL_LOGGER, PipelineLogger

logger = logging.getLogger(__name__)

_INT_PATTERN = re.compile(r"^[+-]?\d+$")
_FLOAT_PATTERN = re.compile(r"^[+-]?(?:\d+\.\d*|\.\d+)$")


def _name_similarity(left: str | None, right: str | None) -> float:
    if not left or not right:
        return 0.0
    left_norm = re.sub(r"[^a-z0-9]+", "_", left.lower()).strip("_")
    right_norm = re.sub(r"[^a-z0-9]+", "_", right.lower()).strip("_")
    if not left_norm or not right_norm:
        return 0.0
    if left_norm == right_norm:
        return 1.0
    return SequenceMatcher(None, left_norm, right_norm).ratio()


def _quote_identifier(identifier: str) -> str:
    """
    Quote a SQL identifier for Polars SQL.

    This helper is needed because executor-generated SQL must be robust to
    table and column names that require quoting.
    """
    escaped = identifier.replace('"', '""')
    return f'"{escaped}"'


def _quote_sql_value(value: object) -> str:
    """
    Convert a Python value into a SQL literal.

    This helper is needed because structured query plans must be compiled
    back into executable SQL deterministically.
    """
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return str(value)
    escaped = str(value).replace("'", "''")
    return f"'{escaped}'"


def _qualified_column(column: str, table: str | None = None) -> str:
    """
    Return a stable SQL column reference.

    This helper is needed because query-plan fragments may refer to either
    bare columns or table-qualified columns.
    """
    if "." in column:
        left, right = column.split(".", 1)
        return f'{_quote_identifier(left)}.{_quote_identifier(right)}'
    if table:
        return f"{_quote_identifier(table)}.{_quote_identifier(column)}"
    return _quote_identifier(column)


def _planned_column(column: str, table: str | None = None) -> str:
    """
    Return a column reference against the base path query output.

    Path SQL selects stable aliases like `table__column` so downstream query
    refinement should target those aliases rather than raw joined columns.
    This avoids backend-specific duplicate-column naming after joins.
    """
    if "." in column:
        table_name, column_name = column.split(".", 1)
        return _quote_identifier(f"{table_name}__{column_name}")
    if table:
        return _quote_identifier(f"{table}__{column}")
    return _quote_identifier(column)


def _filter_to_sql(filter_: QueryFilter) -> str:
    """
    Convert a structured filter into a SQL predicate.

    This helper is needed because `QueryPlan` stores filters as data rather
    than executable SQL text.
    """
    column_sql = _planned_column(filter_.column, filter_.table)
    operator = filter_.operator.upper()
    value = filter_.value

    if operator in {"IN", "NOT IN"} and isinstance(value, list):
        values = ", ".join(_quote_sql_value(item) for item in value)
        return f"{column_sql} {operator} ({values})"

    if operator == "BETWEEN" and isinstance(value, list) and len(value) == 2:
        return (
            f"{column_sql} BETWEEN {_quote_sql_value(value[0])} "
            f"AND {_quote_sql_value(value[1])}"
        )

    if value is None and operator in {"=", "IS"}:
        return f"{column_sql} IS NULL"
    if value is None and operator in {"!=", "<>", "IS NOT"}:
        return f"{column_sql} IS NOT NULL"
    return f"{column_sql} {operator} {_quote_sql_value(value)}"


def _select_to_sql(select: QuerySelect) -> str:
    """
    Convert a structured projection into a SQL select expression.

    This helper is needed because projections may include optional
    aggregation and table qualification.
    """
    column_sql = _planned_column(select.column, select.table)
    if select.aggregation:
        return f"{select.aggregation.upper()}({column_sql})"
    return column_sql


def _select_alias(select: QuerySelect, index: int) -> str:
    """
    Return a stable output alias for a projection.

    This helper is needed because grouped queries and ordered projections
    require predictable column names.
    """
    if select.aggregation:
        table = select.table or "value"
        return f"agg_{index}__{select.aggregation.lower()}__{table}__{select.column}"
    return select.column


def _order_to_sql(order: QueryOrder) -> str:
    """
    Convert a structured ordering directive into SQL.

    This helper is needed because sort directives can target either raw or
    aggregated expressions.
    """
    column_sql = _planned_column(order.column, order.table)
    if order.aggregation:
        column_sql = f"{order.aggregation.upper()}({column_sql})"
    return f"{column_sql} {order.direction.upper()}"


def _group_item_to_sql(item: str) -> str:
    """
    Convert a `group_by` item into SQL.

    Query plans encode group-by columns as either `table.column` or a bare
    column name. This helper keeps the conversion logic consistent in one
    place so grouped projections can be reconciled with the GROUP BY clause.
    """
    return _planned_column(item)


def apply_query_plan_to_sql(sql_query: str, plan: QueryPlan) -> str:
    """
    Deterministically apply a structured QueryPlan to a base SQL query.

    This function is needed because the system wants inspectable model
    intent without allowing free-form SQL rewrites.
    """
    if not isinstance(plan, QueryPlan):
        return sql_query
    if not (
        plan.projections
        or plan.filters
        or plan.group_by
        or plan.having
        or plan.order_by
        or plan.limit is not None
    ):
        return sql_query

    sql = sql_query.strip()
    from_clause = f'FROM (\n{sql}\n) AS "__lakeprompt_base"'

    select_clause = "SELECT *"
    projection_aliases: dict[tuple[str | None, str, str | None], str] = {}
    select_items = list(plan.projections)
    projected_keys = {
        (item.table, item.column, item.aggregation)
        for item in select_items
    }
    for item in plan.order_by:
        key = (item.table, item.column, item.aggregation)
        if item.aggregation and key not in projected_keys:
            select_items.append(
                QuerySelect(
                    table=item.table,
                    column=item.column,
                    aggregation=item.aggregation,
                )
            )
            projected_keys.add(key)
    if plan.projections:
        projection_parts: list[str] = []
        for idx, item in enumerate(select_items):
            alias = _select_alias(item, idx)
            projection_aliases[(item.table, item.column, item.aggregation)] = alias
            projection_parts.append(f"{_select_to_sql(item)} AS {_quote_identifier(alias)}")
        projection_sql = ", ".join(projection_parts)
        select_clause = f"SELECT {projection_sql}"
    elif select_items:
        projection_parts = []
        grouped_aliases: set[str] = set()
        for group_item in plan.group_by:
            alias = group_item.split(".", 1)[-1]
            projection_parts.append(
                f"{_group_item_to_sql(group_item)} AS {_quote_identifier(alias)}"
            )
            grouped_aliases.add(alias)
        for idx, item in enumerate(select_items):
            alias = _select_alias(item, idx)
            projection_aliases[(item.table, item.column, item.aggregation)] = alias
            if alias in grouped_aliases:
                continue
            projection_parts.append(f"{_select_to_sql(item)} AS {_quote_identifier(alias)}")
        select_clause = f"SELECT {', '.join(projection_parts)}"
    elif plan.group_by:
        group_projection_parts = [
            f"{_group_item_to_sql(item)} AS {_quote_identifier(item.split('.', 1)[-1])}"
            for item in plan.group_by
        ]
        if group_projection_parts:
            select_clause = f"SELECT {', '.join(group_projection_parts)}"

    parts = [select_clause, from_clause]

    if plan.filters:
        parts.append("WHERE " + " AND ".join(_filter_to_sql(item) for item in plan.filters))

    if plan.group_by:
        group_items = list(plan.group_by)
        for projection in select_items:
            # Polars SQL is strict about grouped selects: every projected
            # non-aggregated column must appear in GROUP BY.
            if projection.aggregation:
                continue
            group_item = (
                f"{projection.table}.{projection.column}"
                if projection.table
                else projection.column
            )
            if group_item not in group_items:
                group_items.append(group_item)
        group_sql = ", ".join(_group_item_to_sql(item) for item in group_items)
        parts.append(f"GROUP BY {group_sql}")

    if plan.having:
        parts.append("HAVING " + " AND ".join(_filter_to_sql(item) for item in plan.having))

    if plan.order_by:
        order_parts: list[str] = []
        for item in plan.order_by:
            alias = projection_aliases.get((item.table, item.column, item.aggregation))
            if alias:
                order_parts.append(f"{_quote_identifier(alias)} {item.direction.upper()}")
            else:
                order_parts.append(_order_to_sql(item))
        parts.append("ORDER BY " + ", ".join(order_parts))

    if plan.limit is not None:
        parts.append(f"LIMIT {plan.limit}")

    return "\n".join(parts)


def query_plan_tables(plan: QueryPlan | None) -> set[str]:
    """
    Return all tables explicitly required by a structured query plan.

    This helper is needed because the executor must skip paths that cannot
    satisfy the tables referenced by the plan.
    """
    if plan is None:
        return set()
    tables: set[str] = set()
    for filter_ in plan.filters + plan.having:
        if filter_.table:
            tables.add(filter_.table)
    for projection in plan.projections:
        if projection.table:
            tables.add(projection.table)
    for order in plan.order_by:
        if order.table:
            tables.add(order.table)
    for item in plan.group_by:
        if "." in item:
            tables.add(item.split(".", 1)[0])
    return tables


@dataclass
class SqlBuilder:
    """
    Compile join paths into executable base SQL.

    This class is needed because join-path compilation is a separate
    concern from query refinement, execution, and ranking.
    """

    lake: DataLake

    def build_path_sql(self, path: JoinPath) -> str:
        """
        Convert a JoinPath into a SQL SELECT ... JOIN ... ON ... string.

        The generated query includes the full join skeleton but no question-
        specific refinement such as filtering, grouping, or ordering.
        This method is needed because the executor must preserve the chosen
        join structure before any question-specific refinement is applied.
        """
        if len(path.tables) == 1:
            table_name = path.tables[0]
            col_aliases = self._build_column_aliases([table_name])
            select_parts: list[str] = []
            for tbl, col, alias in col_aliases:
                if alias != col:
                    select_parts.append(
                        f"{_quote_identifier(tbl)}.{_quote_identifier(col)} AS {_quote_identifier(alias)}"
                    )
                else:
                    select_parts.append(f"{_quote_identifier(tbl)}.{_quote_identifier(col)}")
            return (
                f"SELECT {', '.join(select_parts)} "
                f"FROM {_quote_identifier(table_name)}"
            )

        tables: list[str] = path.tables
        join_keys: list[tuple[str, str, str, str]] = path.join_keys

        if len(tables) < 2:
            raise ValueError(
                f"JoinPath must reference at least two tables, got: {tables}"
            )

        col_aliases = self._build_column_aliases(tables)

        select_parts: list[str] = []
        for tbl, col, alias in col_aliases:
            if alias != col:
                select_parts.append(
                    f"{_quote_identifier(tbl)}.{_quote_identifier(col)} AS {_quote_identifier(alias)}"
                )
            else:
                select_parts.append(f"{_quote_identifier(tbl)}.{_quote_identifier(col)}")

        sql = f"SELECT {', '.join(select_parts)}\nFROM {_quote_identifier(tables[0])}"

        for t1, col1, t2, col2 in join_keys:
            left_join_expr = self._join_operand_sql(t1, col1)
            right_join_expr = self._join_operand_sql(t2, col2)
            sql += (
                f"\nJOIN {_quote_identifier(t2)} "
                f"ON {left_join_expr} = {right_join_expr}"
            )

        return sql

    def _join_operand_sql(self, table_name: str, column_name: str) -> str:
        """
        Return a SQL expression for a join operand.

        This helper is needed because string join keys need normalization
        so execution matches profiling-time assumptions.
        """
        column_sql = f"{_quote_identifier(table_name)}.{_quote_identifier(column_name)}"
        try:
            dtype = str(self.lake.get_sample(table_name, n=1)[column_name].dtype).lower()
        except Exception:  # noqa: BLE001
            return column_sql
        if "string" in dtype:
            return f"TRIM({column_sql})"
        return column_sql

    def _build_column_aliases(
        self, tables: list[str]
    ) -> list[tuple[str, str, str]]:
        """
        Return `(table, column, alias)` triples for all columns across tables.

        This helper is needed because joined rows need stable, non-colliding
        output keys for downstream ranking and packaging.
        """
        table_cols: dict[str, list[str]] = {}

        for tbl in tables:
            if tbl not in self.lake.tables:
                logger.warning("Table '%s' not found in lake; skipping.", tbl)
                table_cols[tbl] = []
                continue
            cols = self.lake.get_sample(tbl, n=1).columns
            table_cols[tbl] = cols

        result: list[tuple[str, str, str]] = []
        for tbl in tables:
            for col in table_cols.get(tbl, []):
                result.append((tbl, col, f"{tbl}__{col}"))

        return result


@dataclass
class QueryPlanApplier:
    """
    Refine base path SQL using a structured QueryPlan.

    This class is needed because plan coercion and SQL refinement are
    distinct from path compilation and row ranking.
    """

    lake: DataLake
    logger: PipelineLogger = NULL_LOGGER

    def apply(
        self,
        question: str,
        sql: str,
        path: JoinPath,
        query_plan: QueryPlan | None = None,
    ) -> str:
        """
        Apply a shared or inferred query plan to a base path SQL query.

        This method is needed because the question's filter, grouping,
        projection, and ordering intent must be pushed down before execution.
        """
        path_table_set = set(path.tables)
        involved_cards = [
            card
            for card in (getattr(self.lake, "cards", None) or [])
            if card.table_name in path_table_set
        ]

        try:
            raw_plan = query_plan if query_plan is not None else plan_llm_query(question, sql, involved_cards)
            coerced_plan = self._coerce_query_plan_types(raw_plan, involved_cards)
            sanitized_plan = self._sanitize_query_plan(coerced_plan, involved_cards)
            refined_sql = apply_query_plan_to_sql(sql, sanitized_plan)
            self.logger.log(
                "query_plan_debug",
                "Transformed query plan before SQL generation.",
                {
                    "path_id": path.path_id,
                    "path_tables": path.tables,
                    "raw_plan": raw_plan,
                    "coerced_plan": coerced_plan,
                    "sanitized_plan": sanitized_plan,
                    "raw_filter_count": len(raw_plan.filters),
                    "coerced_filter_count": len(coerced_plan.filters),
                    "sanitized_filter_count": len(sanitized_plan.filters),
                    "where_present_after_sanitize": bool(sanitized_plan.filters),
                },
            )
            self.logger.log(
                "sql_refinement",
                "Applied query plan to SQL.",
                {"path_id": path.path_id, "query_plan": sanitized_plan, "sql": refined_sql},
            )
            return refined_sql
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Query-plan application failed for path %s: %s", path.tables, exc
            )
            return sql

    def _coerce_query_plan_types(
        self,
        plan: QueryPlan,
        involved_cards,
    ) -> QueryPlan:
        """
        Coerce query-plan predicate literals to the dtypes of the referenced
        columns when those dtypes are known from the profiled cards.

        This method is needed because LLM-produced plans often express
        literals as strings even when the target columns are typed.
        """
        dtype_map = {
            (card.table_name, card.column_name): card.dtype.lower()
            for card in involved_cards
        }
        filters = [self._coerce_filter_value(item, dtype_map) for item in plan.filters]
        having = [self._coerce_filter_value(item, dtype_map) for item in plan.having]
        return replace(plan, filters=filters, having=having)

    def _coerce_filter_value(
        self,
        filter_: QueryFilter,
        dtype_map: dict[tuple[str, str], str],
    ) -> QueryFilter:
        """
        Convert string filter literals into numeric/bool values when the
        target column dtype indicates they should be typed.

        This helper is needed because type coercion is performed one
        predicate at a time.
        """
        dtype = dtype_map.get((filter_.table, filter_.column))
        if dtype is None:
            return filter_
        return replace(filter_, value=self._coerce_value_to_dtype(filter_.value, dtype))

    def _coerce_value_to_dtype(self, value, dtype: str):
        if value is None:
            return None
        if isinstance(value, list):
            return [self._coerce_value_to_dtype(item, dtype) for item in value]
        if "string" in dtype and not isinstance(value, str):
            return str(value)
        if not isinstance(value, str):
            return value
        normalized = value.strip()
        try:
            if "int" in dtype:
                if not _INT_PATTERN.fullmatch(normalized):
                    return value
                return int(normalized)
            if "float" in dtype or "double" in dtype:
                if not (_INT_PATTERN.fullmatch(normalized) or _FLOAT_PATTERN.fullmatch(normalized)):
                    return value
                return float(normalized)
            if "bool" in dtype:
                lowered = normalized.lower()
                if lowered in {"true", "false"}:
                    return lowered == "true"
        except ValueError:
            return value
        return value

    def _sanitize_query_plan(
        self,
        plan: QueryPlan,
        involved_cards,
    ) -> QueryPlan:
        """
        Drop plan fragments that are unsupported or inconsistent with the
        available path columns so malformed LLM plans degrade gracefully.
        """
        valid_refs = {
            (card.table_name, card.column_name)
            for card in involved_cards
        }
        cards_by_ref = {
            (card.table_name, card.column_name): card
            for card in involved_cards
        }

        def resolve_ref(
            table: str | None,
            column: str,
            *,
            value: object | None = None,
        ) -> tuple[str | None, str] | None:
            if "." in column:
                split_table, split_column = column.split(".", 1)
                table = split_table
                column = split_column
            if table is not None and (table, column) in valid_refs:
                return table, column
            if table is None:
                matches = [(tbl, col) for tbl, col in valid_refs if col == column]
                if len(matches) == 1:
                    return matches[0]

            best_ref: tuple[str, str] | None = None
            best_score = 0.0
            for ref, card in cards_by_ref.items():
                score = 0.0
                column_score = _name_similarity(column, card.column_name)
                table_score = _name_similarity(table, card.table_name) if table else 0.0
                score += column_score * 3.0
                score += table_score
                if value is not None and card.sample_values:
                    normalized_value = str(value).strip().lower()
                    sample_values = {str(item).strip().lower() for item in card.sample_values}
                    if normalized_value and normalized_value in sample_values:
                        score += 2.5
                if column.lower() == card.column_name.lower():
                    score += 1.5
                if table and table.lower() == card.table_name.lower():
                    score += 0.5
                if score > best_score:
                    best_score = score
                    best_ref = ref
            if best_ref is None or best_score < 1.75:
                return None
            return best_ref

        def ground_filter(item: QueryFilter) -> QueryFilter:
            resolved = resolve_ref(item.table, item.column, value=item.value)
            if resolved is None:
                return item
            return replace(item, table=resolved[0], column=resolved[1])

        def ground_select(item: QuerySelect) -> QuerySelect:
            resolved = resolve_ref(item.table, item.column)
            if resolved is None:
                return item
            return replace(item, table=resolved[0], column=resolved[1])

        def ground_order(item: QueryOrder) -> QueryOrder:
            resolved = resolve_ref(item.table, item.column)
            if resolved is None:
                return item
            return replace(item, table=resolved[0], column=resolved[1])

        def ground_group_item(item: str) -> str:
            resolved = resolve_ref(None, item)
            if resolved is None:
                return item
            table, column = resolved
            return f"{table}.{column}" if table else column

        filters_grounded = [ground_filter(item) for item in plan.filters]
        projections_grounded = [ground_select(item) for item in plan.projections]
        having_grounded = [ground_filter(item) for item in plan.having]
        order_grounded = [ground_order(item) for item in plan.order_by]
        group_grounded = [ground_group_item(item) for item in plan.group_by]
        grouped_refs = {
            tuple(item.split(".", 1)) if "." in item else (None, item)
            for item in group_grounded
        }
        projected_raw_refs = {
            (item.table, item.column)
            for item in projections_grounded
            if not item.aggregation
        }

        def has_valid_ref(table: str | None, column: str) -> bool:
            if "." in column:
                left, right = column.split(".", 1)
                return (left, right) in valid_refs
            if table is not None:
                return (table, column) in valid_refs
            return any(card_column == column for _, card_column in valid_refs)

        def is_supported_filter(item: QueryFilter) -> bool:
            if not has_valid_ref(item.table, item.column):
                return False
            value = item.value
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered.startswith("select ") or " join " in lowered or " from " in lowered:
                    return False
            if isinstance(value, list):
                for child in value:
                    if isinstance(child, str):
                        lowered = child.strip().lower()
                        if lowered.startswith("select ") or " join " in lowered or " from " in lowered:
                            return False
            return True

        def is_supported_order(item: QueryOrder) -> bool:
            if not has_valid_ref(item.table, item.column):
                return False
            if item.aggregation:
                return True
            if not (plan.group_by or any(select.aggregation for select in plan.projections)):
                return True
            ref = (item.table, item.column)
            if "." in item.column:
                ref = tuple(item.column.split(".", 1))
            return ref in grouped_refs or ref in projected_raw_refs

        filters = [item for item in filters_grounded if is_supported_filter(item)]
        projections = [
            item for item in projections_grounded
            if has_valid_ref(item.table, item.column)
        ]
        having = [
            item for item in having_grounded
            if is_supported_filter(item) and (
                (item.table, item.column) in grouped_refs
                or (item.table, item.column) in projected_raw_refs
                or ("." in item.column and tuple(item.column.split(".", 1)) in grouped_refs)
            )
        ]
        order_by = [item for item in order_grounded if is_supported_order(item)]
        group_by = [
            item for item in group_grounded
            if has_valid_ref(None, item)
        ]
        limit = plan.limit if isinstance(plan.limit, int) and plan.limit > 0 else None
        return replace(
            plan,
            filters=filters,
            projections=projections,
            group_by=group_by,
            having=having,
            order_by=order_by,
            limit=limit,
        )
