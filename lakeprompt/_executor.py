import warnings

from dataclasses import dataclass
from ._datalake import DataLake
from ._executor_ranking import RowRanker
from ._executor_sql import QueryPlanApplier, SqlBuilder, query_plan_tables
from ._models import JoinPath, JoinedTuple, JoinProvenance, OutputColumn, QueryPlan
from ._tracing import NULL_LOGGER, PipelineLogger

"""
TupleExecutor — Stream S3 of the LakePrompt pipeline.

This module coordinates path execution and tuple packaging.
It is needed because the pipeline must turn candidate join paths into
ranked evidence tuples without exposing the helper layers to callers.
"""



DEFAULT_TOP_R = 20
DEFAULT_MAX_PATHS_TO_EXECUTE = 3


def _normalize_sql(sql: str) -> str:
    """
    Normalize SQL to a single-line string.

    This helper is needed because evidence provenance should carry the
    executed SQL without embedded newlines or irregular spacing.
    """
    return " ".join(sql.split())

@dataclass
class TupleExecutor:
    """
    Coordinate path SQL compilation, refinement, execution, and ranking.

    Join discovery only produces candidate execution plans. `TupleExecutor`
    turns those plans into real evidence rows by delegating SQL compilation
    to `SqlBuilder`, query refinement to `QueryPlanApplier`, and row ranking
    to `RowRanker`.
    This class is needed because the rest of the pipeline expects one
    execution entry point instead of managing those helper stages directly.

    Args:
        lake: An initialised DataLake instance.
    """

    lake: DataLake
    logger: PipelineLogger = NULL_LOGGER

    # Public API
    def get_tuples(
        self,
        question: str,
        paths: list[JoinPath],
        query_plan: QueryPlan | None = None,
        top_r: int = DEFAULT_TOP_R,
        max_paths_to_execute: int = DEFAULT_MAX_PATHS_TO_EXECUTE,
    ) -> list[JoinedTuple]:
        """
        Execute join paths and return the top-r most relevant evidence tuples.

        This method is needed because retrieval returns candidate paths,
        while later stages need concrete ranked evidence rows.

        Args:
            question: The natural language question driving retrieval.
            paths: Join paths produced by the profiler/retriever.
            query_plan: Optional structured question intent reused across
                path validation and SQL refinement.
            top_r: Maximum number of JoinedTuple objects to return.

        Returns:
            A list of at most `top_r` `JoinedTuple` objects, sorted by
            descending relevance score.
        """
        ranker = RowRanker(self.lake)
        q_emb = ranker.embed_question(question)
        all_scored: list[tuple[float, dict, JoinPath]] = []
        sql_by_path_id: dict[str, str] = {}
        output_columns_by_path_id: dict[str, list[OutputColumn]] = {}

        for path in paths[:max_paths_to_execute]:
            rows, executed_sql, output_columns = self._execute_path(path, question=question, query_plan=query_plan)
            if not rows:
                continue
            sql_by_path_id[path.path_id] = executed_sql
            output_columns_by_path_id[path.path_id] = output_columns

            for score, row in ranker.score_rows(rows, q_emb, path):
                all_scored.append((score, row, path))

        all_scored.sort(key=lambda x: x[0], reverse=True)
        diverse = ranker.select_diverse_rows(all_scored, top_r=top_r)
        self.logger.log(
            "ranked_rows",
            "Selected ranked evidence rows.",
            [
                {
                    "score": score,
                    "path_id": path.path_id,
                    "tables": path.tables,
                    "row": row,
                }
                for score, row, path in diverse
            ],
        )
        return self._to_joined_tuples(
            diverse,
            sql_by_path_id=sql_by_path_id,
            output_columns_by_path_id=output_columns_by_path_id,
        )

    def _execute_path(
        self,
        path: JoinPath,
        question: str = "",
        query_plan: QueryPlan | None = None,
    ) -> tuple[list[dict], str, list[OutputColumn]]:
        """
        Build and execute SQL for a join path, returning rows plus the
        final executed SQL string.

        The method validates that the path can satisfy any required
        query-plan tables, compiles base SQL, applies question-specific
        query refinement, then executes the resulting statement.
        This method is needed because a join path is only useful after it
        has been materialized into actual rows.

        Args:
            path: The join path to execute.
            question: Optional natural language question used when a
                query plan must be inferred locally.
            query_plan: Optional structured query intent, reused instead of
                planning again from the question.

        Returns:
            A tuple of `(rows, normalized_sql)`. `rows` may be empty when
            execution fails or produces no results.
        """
        if query_plan is not None:
            required_tables = query_plan_tables(query_plan)
            if required_tables and not required_tables.issubset(set(path.tables)):
                warnings.warn(
                    (
                        f"Skipping path {path.tables} because it does not cover "
                        f"required query-plan tables {sorted(required_tables)}."
                    ),
                    stacklevel=2,
                )
                return [], "", []

        sql_builder = SqlBuilder(self.lake)
        try:
            sql = sql_builder.build_path_sql(path)
        except ValueError as exc:
            warnings.warn(f"Skipping malformed JoinPath: {exc}", stacklevel=2)
            return [], "", []

        output_columns: list[OutputColumn] = []
        if question or query_plan is not None:
            plan_applier = QueryPlanApplier(self.lake, logger=self.logger)
            sql, output_columns = plan_applier.apply(question, sql, path, query_plan=query_plan)

        self.logger.log("sql_query", "Executing SQL query.", {"path_id": path.path_id, "sql": sql})

        try:
            result_df = self.lake.query(sql)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"Query failed for path {path.tables}: {exc}", stacklevel=2)
            return [], _normalize_sql(sql), output_columns

        rows = result_df.to_dicts()

        if not rows:
            warnings.warn(f"Join path {path.tables} produced zero rows.", stacklevel=2)

        return rows, _normalize_sql(sql), output_columns

    def _to_joined_tuples(
        self,
        ranked: list[tuple[float, dict, JoinPath]],
        sql_by_path_id: dict[str, str],
        output_columns_by_path_id: dict[str, list[OutputColumn]],
    ) -> list[JoinedTuple]:
        """
        Wrap scored rows into `JoinedTuple` objects with sequential evidence IDs.

        IDs are assigned E1, E2, … in score-descending order and are stable
        within a single get_tuples() call.
        This method is needed because downstream packaging and citation
        logic expects evidence rows to carry IDs and provenance metadata.

        Args:
            ranked: List of (score, row_dict, join_path) triples.
            sql_by_path_id: Final executed SQL keyed by source path ID.

        Returns:
            A list of fully populated `JoinedTuple` objects.
        """
        return [
            JoinedTuple(
                evidence_id=f"E{idx}",  # E1, E2, E3, … used by the packager to cite evidence
                data=row,
                output_columns=list(output_columns_by_path_id.get(path.path_id, [])),
                provenance=JoinProvenance(
                    path_id=path.path_id,
                    tables=list(path.tables),
                    join_keys=list(path.join_keys),
                    path_score=path.score,
                    sql=sql_by_path_id.get(path.path_id, ""),
                ),
                join_path=path,
                relevance_score=score,
            )
            for idx, (score, row, path) in enumerate(ranked, start=1)
        ]
