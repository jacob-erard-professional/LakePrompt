from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sqlite3
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

import anthropic
import polars as pl


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lakeprompt._packager import prettify_row_data_keys


DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-20250514"


def _tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric-ish tokens for answer scoring."""
    normalized = re.sub(r"(?<=\d),(?=\d)", "", text.lower())
    return re.findall(r"[a-z0-9_.-]+", normalized)


def _extract_answer_atoms(value: Any) -> list[str]:
    """
    Flatten a structured answer into comparable primitive atoms.

    Dict keys are ignored so structured SQL ground truth compares on
    returned values rather than JSON field names.
    """
    if value is None:
        return []
    if isinstance(value, dict):
        atoms: list[str] = []
        for child in value.values():
            atoms.extend(_extract_answer_atoms(child))
        return atoms
    if isinstance(value, list):
        atoms: list[str] = []
        for child in value:
            atoms.extend(_extract_answer_atoms(child))
        return atoms
    if isinstance(value, bool):
        return ["true" if value else "false"]
    return _tokenize(str(value))


def _parse_answer_payload(text: str) -> Any | None:
    """
    Parse a response as JSON when it looks like structured output.
    """
    stripped = text.strip()
    if not stripped:
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return None


def _f1_from_counters(pred_counter: Counter[str], gt_counter: Counter[str]) -> float:
    """
    Compute multiset F1 from token/atom counters.
    """
    pred_total = sum(pred_counter.values())
    gt_total = sum(gt_counter.values())
    if not pred_total or not gt_total:
        return 0.0
    common = sum((pred_counter & gt_counter).values())
    if not common:
        return 0.0
    precision = common / pred_total
    recall = common / gt_total
    return 2 * precision * recall / (precision + recall)


def _token_f1(prediction: str, ground_truth: str) -> float:
    """
    Compute answer F1 between a predicted answer and ground truth.

    For structured SQL-result ground truth, compare flattened primitive
    values rather than raw JSON punctuation and keys. For plain text,
    fall back to token overlap.

    Args:
        prediction: Model-generated answer string.
        ground_truth: Reference answer string.

    Returns:
        F1 score between 0.0 and 1.0.
    """
    gt_payload = _parse_answer_payload(ground_truth)
    pred_payload = _parse_answer_payload(prediction)

    if gt_payload is not None:
        gt_counter = Counter(_extract_answer_atoms(gt_payload))
        pred_counter = Counter(
            _extract_answer_atoms(pred_payload)
            if pred_payload is not None
            else _tokenize(prediction)
        )
        return _f1_from_counters(pred_counter, gt_counter)

    return _f1_from_counters(Counter(_tokenize(prediction)), Counter(_tokenize(ground_truth)))


def _exact_match(prediction: str, ground_truth: str) -> bool:
    """Case-insensitive, stripped exact match."""
    return prediction.strip().lower() == ground_truth.strip().lower()


def _faithfulness_score(answer: str, cited_ids: list[str], evidence_ids: set[str]) -> float:
    """
    Estimate faithfulness as the fraction of valid evidence IDs that are cited.

    A fully faithful answer cites at least one valid evidence item. An answer
    with no evidence available is scored 0.0.

    Args:
        answer: The model-generated answer text.
        cited_ids: Evidence IDs the model explicitly cited.
        evidence_ids: All valid evidence IDs available in the context.

    Returns:
        Score between 0.0 and 1.0.
    """
    if not evidence_ids:
        return 0.0
    valid_cited = {eid for eid in cited_ids if eid in evidence_ids}
    # Also count IDs mentioned inline in the answer text (E1, E2, …)
    for eid in evidence_ids:
        if eid in answer:
            valid_cited.add(eid)
    if not valid_cited:
        return 0.0
    return len(valid_cited) / len(evidence_ids)


def _context_token_count(evidence_lines: str) -> int:
    """Approximate token count by whitespace splitting."""
    return len(evidence_lines.split())


def _join_count(evidence: list) -> int:
    """Count distinct join paths used across all evidence items."""
    path_ids: set[str] = set()
    for item in evidence:
        provenance = getattr(item, "provenance", None)
        if provenance is not None:
            path_ids.add(getattr(provenance, "path_id", ""))
    return len(path_ids)


@dataclass
class ConditionResult:
    """Stores the answer and metrics for one evaluation condition."""
    condition: str
    prompt: str
    answer: str
    latency_seconds: float
    evidence_count: int
    join_count: int
    context_tokens: int  # whitespace-token count of serialized evidence lines only
    prompt_tokens: int = 0  # whitespace-token count of the full prompt sent to the LLM
    cited_ids: list[str] = field(default_factory=list)
    generated_sql: list[str] = field(default_factory=list)
    error: str | None = None
    # Filled in after ground truth is available
    exact_match: bool = False
    token_f1: float = 0.0
    faithfulness: float = 0.0


@dataclass
class EvaluationExample:
    schema_id: str
    question: str
    join_metadata: dict[str, Any]
    ground_truth: str  # populated from question generation when available
    ground_truth_error: str | None = None
    expected_sql: str = ""
    conditions: list[ConditionResult] = field(default_factory=list)


class ClaudeClient:
    """Thin wrapper around the Anthropic Messages API."""

    def __init__(self, model: str = DEFAULT_CLAUDE_MODEL):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set.")
        self.model = model
        self.client = anthropic.Anthropic(api_key=api_key)

    def complete(
        self,
        *,
        system: str,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.2,
    ) -> str:
        message = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(
            block.text for block in message.content if getattr(block, "type", None) == "text"
        ).strip()


class SpiderJoinEvaluation:
    """
    Generates join-requiring questions for Spider Join schemas and records
    answers across four baseline conditions plus the full LakePrompt
    condition, with metrics computed per example.
    """

    def __init__(
        self,
        dataset_root: str,
        metadata_csv: str,
        output_json: str,
        output_txt: str,
        claude_model: str = DEFAULT_CLAUDE_MODEL,
    ):
        self.dataset_root = Path(dataset_root)
        self.metadata_csv = Path(metadata_csv)
        self.output_json = Path(output_json)
        self.output_txt = Path(output_txt)
        self.claude = ClaudeClient(model=claude_model)
        self._lakeprompt_cache: dict[tuple[str, str, str | None], Any] = {}

    def run(
        self,
        *,
        schema_limit: int = 3,
        questions_per_schema: int = 3,
        sample_rows: int = 3,
        lakeprompt_model: str = DEFAULT_CLAUDE_MODEL,
        cache_path: str | None = None,
        questions_file: str | None = None,
        max_questions: int | None = None,
    ) -> dict[str, Any]:
        examples: list[EvaluationExample] = []
        run_metadata = {
            "dataset_root": str(self.dataset_root),
            "metadata_csv": str(self.metadata_csv),
            "claude_model": self.claude.model,
        }
        if max_questions is not None:
            run_metadata["max_questions"] = max_questions
        if questions_file:
            run_metadata["questions_file"] = str(Path(questions_file))
            self._run_from_questions_file(
                questions_file=questions_file,
                sample_rows=sample_rows,
                lakeprompt_model=lakeprompt_model,
                cache_path=cache_path,
                examples=examples,
                run_metadata=run_metadata,
                max_questions=max_questions,
            )
        else:
            metadata_rows = self._load_metadata()
            schema_rows = self._group_metadata_by_schema(metadata_rows, schema_limit)

            for schema_id, rows in schema_rows.items():
                schema_dir = self._resolve_schema_dir(schema_id)
                schema_description = self._build_schema_description(schema_dir, sample_rows)
                questions, ground_truths = self._generate_questions(
                    schema_id=schema_id,
                    schema_description=schema_description,
                    join_rows=rows,
                    question_count=questions_per_schema,
                )

                for question, ground_truth in zip(questions, ground_truths):
                    if max_questions is not None and len(examples) >= max_questions:
                        break
                    examples.append(
                        self._run_example(
                            schema_id=schema_id,
                            schema_dir=schema_dir,
                            schema_description=schema_description,
                            question=question,
                            ground_truth=ground_truth,
                            join_metadata=rows[0],
                            sample_rows=sample_rows,
                            lakeprompt_model=lakeprompt_model,
                            cache_path=cache_path,
                        )
                    )
                    self._write_progress(examples, run_metadata)
                if max_questions is not None and len(examples) >= max_questions:
                    break

        payload = self._build_payload(examples)
        self._write_json(payload)
        self._write_txt(payload)
        return payload

    def _run_from_questions_file(
        self,
        *,
        questions_file: str,
        sample_rows: int,
        lakeprompt_model: str,
        cache_path: str | None,
        examples: list[EvaluationExample],
        run_metadata: dict[str, Any],
        max_questions: int | None,
    ) -> None:
        questions_path = Path(questions_file)
        raw_items = self._load_question_items(questions_path)
        schema_descriptions: dict[str, str] = {}

        for item in raw_items:
            if max_questions is not None and len(examples) >= max_questions:
                break
            schema_id = item["schema_id"]
            schema_dir = self._resolve_schema_dir(schema_id)
            if schema_id not in schema_descriptions:
                schema_descriptions[schema_id] = self._build_schema_description(
                    schema_dir, sample_rows
                )
            examples.append(
                self._run_example(
                    schema_id=schema_id,
                    schema_dir=schema_dir,
                    schema_description=schema_descriptions[schema_id],
                    question=item["question"],
                    expected_sql=item.get("query", ""),
                    ground_truth=item.get("ground_truth", ""),
                    join_metadata={},
                    sample_rows=sample_rows,
                    lakeprompt_model=lakeprompt_model,
                    cache_path=cache_path,
                )
            )
            self._write_progress(examples, run_metadata)

    def _run_example(
        self,
        *,
        schema_id: str,
        schema_dir: Path,
        schema_description: str,
        question: str,
        expected_sql: str,
        ground_truth: str,
        join_metadata: dict[str, Any],
        sample_rows: int,
        lakeprompt_model: str,
        cache_path: str | None,
    ) -> EvaluationExample:
        example = EvaluationExample(
            schema_id=schema_id,
            question=question,
            expected_sql=expected_sql,
            join_metadata=join_metadata,
            ground_truth=ground_truth,
        )
        if expected_sql.strip():
            try:
                example.ground_truth = self._execute_expected_sql(schema_dir, expected_sql)
                example.ground_truth_error = None
            except Exception as exc:  # noqa: BLE001
                example.ground_truth_error = str(exc)

        example.conditions.append(self._run_no_context(question))
        example.conditions.append(
            self._run_single_table(schema_dir, schema_description, question, sample_rows)
        )
        example.conditions.append(
            self._run_naive_multitable(schema_dir, schema_description, question, sample_rows)
        )
        example.conditions.append(self._run_schema_baseline(schema_description, question))

        lp_result, join_no_ranking_result = self._run_lakeprompt_conditions(
            schema_dir, question, lakeprompt_model, cache_path
        )
        example.conditions.append(lp_result)
        example.conditions.append(join_no_ranking_result)

        evidence_ids = self._evidence_ids_from_condition(lp_result)
        for cond in example.conditions:
            cond.exact_match = _exact_match(cond.answer, example.ground_truth)
            cond.token_f1 = _token_f1(cond.answer, example.ground_truth)
            cond.faithfulness = _faithfulness_score(
                cond.answer, cond.cited_ids, evidence_ids
            )

        return example

    def _load_csvs_into_sqlite(
        self,
        conn: sqlite3.Connection,
        schema_dir: Path,
    ) -> None:
        for csv_path in sorted(schema_dir.glob("*.csv")):
            table_name = csv_path.stem
            with csv_path.open(newline="", encoding="utf-8") as handle:
                reader = csv.reader(handle)
                try:
                    headers = next(reader)
                except StopIteration:
                    continue
                quoted_columns = [f'"{header}" TEXT' for header in headers]
                conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')
                conn.execute(f'CREATE TABLE "{table_name}" ({", ".join(quoted_columns)})')
                placeholders = ", ".join("?" for _ in headers)
                insert_sql = f'INSERT INTO "{table_name}" VALUES ({placeholders})'
                rows = [tuple(row) for row in reader]
                if rows:
                    conn.executemany(insert_sql, rows)
        conn.commit()

    def _canonical_ground_truth(self, cursor: sqlite3.Cursor) -> str:
        columns = [desc[0] for desc in cursor.description or []]
        rows = cursor.fetchall()

        if not columns:
            return ""
        if len(columns) == 1:
            values = [row[0] for row in rows]
            if not values:
                return ""
            if len(values) == 1:
                return str(values[0])
            return json.dumps(values, ensure_ascii=True)

        payload = [
            {column: value for column, value in zip(columns, row, strict=False)}
            for row in rows
        ]
        if not payload:
            return ""
        return json.dumps(payload, ensure_ascii=True, sort_keys=True)

    def _execute_expected_sql(self, schema_dir: Path, sql: str) -> str:
        conn = sqlite3.connect(":memory:")
        try:
            self._load_csvs_into_sqlite(conn, schema_dir)
            cursor = conn.execute(sql)
            return self._canonical_ground_truth(cursor)
        finally:
            conn.close()

    def _load_question_items(self, questions_path: Path) -> list[dict[str, str]]:
        if not questions_path.exists():
            raise FileNotFoundError(f"Questions file not found: {questions_path}")

        if questions_path.suffix.lower() == ".json":
            payload = json.loads(questions_path.read_text(encoding="utf-8"))
            if not isinstance(payload, list):
                raise ValueError("Questions JSON must be a list of question objects.")
            items = payload
        else:
            items = []
            for line in questions_path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                parts = [part.strip() for part in stripped.split("\t")]
                if len(parts) < 2:
                    raise ValueError(
                        "Questions text file must use tab-separated lines: "
                        "<schema_id>\\t<question>[\\t<ground_truth>]"
                    )
                item: dict[str, str] = {
                    "schema_id": parts[0],
                    "question": parts[1],
                }
                if len(parts) > 2:
                    item["ground_truth"] = parts[2]
                items.append(item)

        normalized: list[dict[str, str]] = []
        for item in items:
            if not isinstance(item, dict):
                raise ValueError("Each question entry must be an object.")
            schema_id = str(
                item.get("schema_id", "") or item.get("db_id", "")
            ).strip()
            question = str(item.get("question", "")).strip()
            if not schema_id or not question:
                raise ValueError(
                    "Each question entry must include non-empty 'schema_id' (or 'db_id') and 'question'."
                )
            normalized_item = {
                "schema_id": schema_id,
                "question": question,
            }
            query = str(item.get("query", "")).strip()
            if query:
                normalized_item["query"] = query
            ground_truth = str(item.get("ground_truth", "")).strip()
            if ground_truth:
                normalized_item["ground_truth"] = ground_truth
            normalized.append(normalized_item)
        return normalized

    def _run_no_context(self, question: str) -> ConditionResult:
        """Condition 1: LLM answers with zero context."""
        prompt = f"Answer the following question as best you can.\n\nQuestion:\n{question}"
        start = time.monotonic()
        try:
            answer = self.claude.complete(
                system="Answer questions about data. If you cannot answer, say so.",
                prompt=prompt,
            )
            error = None
        except Exception as exc:  # noqa: BLE001
            answer = ""
            error = str(exc)
        return ConditionResult(
            condition="no_context",
            prompt=prompt,
            answer=answer,
            latency_seconds=round(time.monotonic() - start, 3),
            evidence_count=0,
            join_count=0,
            context_tokens=0,
            prompt_tokens=_context_token_count(prompt),
            error=error,
        )

    def _run_single_table(
        self,
        schema_dir: Path,
        schema_description: str,
        question: str,
        sample_rows: int,
    ) -> ConditionResult:
        """
        Condition 2: top rows from the single best-matching table, no join.

        The best table is chosen by counting how many question tokens appear
        in its column names — a lightweight heuristic that avoids an LLM call.
        """
        best_table_rows = self._best_single_table_rows(schema_dir, question, sample_rows)
        prompt = (
            f"Use only the following rows from a single table to answer the question.\n\n"
            f"Rows:\n{best_table_rows}\n\nQuestion:\n{question}"
        )
        context_tokens = _context_token_count(best_table_rows)
        start = time.monotonic()
        try:
            answer = self.claude.complete(
                system="Answer questions about data using only the provided rows. If insufficient, say so.",
                prompt=prompt,
            )
            error = None
        except Exception as exc:  # noqa: BLE001
            answer = ""
            error = str(exc)
        return ConditionResult(
            condition="single_table",
            prompt=prompt,
            answer=answer,
            latency_seconds=round(time.monotonic() - start, 3),
            evidence_count=sample_rows,
            join_count=0,
            context_tokens=context_tokens,
            prompt_tokens=_context_token_count(prompt),
            error=error,
        )

    def _run_naive_multitable(
        self,
        schema_dir: Path,
        schema_description: str,
        question: str,
        sample_rows: int,
    ) -> ConditionResult:
        """
        Condition 3: top rows from every table concatenated, no join performed.
        """
        all_rows_text = self._all_table_rows_text(schema_dir, sample_rows)
        prompt = (
            f"Use only the following rows from multiple tables to answer the question. "
            f"Tables are not joined.\n\nRows:\n{all_rows_text}\n\nQuestion:\n{question}"
        )
        context_tokens = _context_token_count(all_rows_text)
        start = time.monotonic()
        try:
            answer = self.claude.complete(
                system="Answer questions about data using only the provided rows. If insufficient, say so.",
                prompt=prompt,
            )
            error = None
        except Exception as exc:
            answer = ""
            error = str(exc)
        return ConditionResult(
            condition="naive_multitable",
            prompt=prompt,
            answer=answer,
            latency_seconds=round(time.monotonic() - start, 3),
            evidence_count=sample_rows,
            join_count=0,
            context_tokens=context_tokens,
            prompt_tokens=_context_token_count(prompt),
            error=error,
        )

    def _run_schema_baseline(self, schema_description: str, question: str) -> ConditionResult:
        """Condition 4: schema + sample rows context (original baseline)."""
        prompt = self._build_baseline_prompt(schema_description, question)
        context_tokens = _context_token_count(schema_description)
        start = time.monotonic()
        try:
            answer = self.claude.complete(
                system="Answer questions about relational data. If the schema context is insufficient, say so.",
                prompt=prompt,
            )
            error = None
        except Exception as exc:
            answer = ""
            error = str(exc)
        return ConditionResult(
            condition="schema_baseline",
            prompt=prompt,
            answer=answer,
            latency_seconds=round(time.monotonic() - start, 3),
            evidence_count=0,
            join_count=0,
            context_tokens=context_tokens,
            prompt_tokens=_context_token_count(prompt),
            error=error,
        )

    def _get_lakeprompt(
        self,
        schema_dir: Path,
        lakeprompt_model: str,
        cache_path: str | None,
    ):
        from lakeprompt import LakePrompt

        cache_key = (str(schema_dir.resolve()), lakeprompt_model, cache_path)
        if cache_key not in self._lakeprompt_cache:
            self._lakeprompt_cache[cache_key] = LakePrompt(
                str(schema_dir),
                model=lakeprompt_model,
                cache_path=cache_path,
            )
        return self._lakeprompt_cache[cache_key]

    def _run_lakeprompt_conditions(
        self,
        schema_dir: Path,
        question: str,
        lakeprompt_model: str,
        cache_path: str | None,
    ) -> tuple[ConditionResult, ConditionResult]:
        """Run LakePrompt once and derive both ranked and shuffled-evidence conditions."""
        start = time.monotonic()
        try:
            lp = self._get_lakeprompt(schema_dir, lakeprompt_model, cache_path)
            result = lp.query(question)
            generated_sql = self._extract_generated_sql(result.evidence)

            evidence_lines = self._serialize_evidence(result.evidence)
            context_tokens = _context_token_count(evidence_lines)
            joins = _join_count(result.evidence)
            ranked_prompt = self._build_lakeprompt_prompt(question, evidence_lines)
            ranked_condition = ConditionResult(
                condition="lakeprompt_ranked",
                prompt=result.prompt or ranked_prompt,
                answer=result.text,
                latency_seconds=round(time.monotonic() - start, 3),
                evidence_count=len(result.evidence),
                join_count=joins,
                context_tokens=context_tokens,
                prompt_tokens=_context_token_count(ranked_prompt),
                cited_ids=result.cited_ids,
                generated_sql=generated_sql,
                error=None,
            )

            shuffled_evidence = list(result.evidence)
            random.shuffle(shuffled_evidence)
            shuffled_lines = self._serialize_evidence(shuffled_evidence)
            shuffled_tokens = _context_token_count(shuffled_lines)
            shuffled_joins = _join_count(shuffled_evidence)
            shuffled_prompt = self._build_lakeprompt_prompt(question, shuffled_lines)
            shuffled_answer = self.claude.complete(
                system=(
                    "Answer questions using only the provided evidence rows. "
                    "If the evidence is insufficient, say so explicitly."
                ),
                prompt=shuffled_prompt,
            )
            shuffled_cited_ids = [
                item.evidence_id
                for item in shuffled_evidence
                if item.evidence_id in shuffled_answer
            ]
            shuffled_condition = ConditionResult(
                condition="join_no_ranking",
                prompt=shuffled_prompt,
                answer=shuffled_answer,
                latency_seconds=round(time.monotonic() - start, 3),
                evidence_count=len(shuffled_evidence),
                join_count=shuffled_joins,
                context_tokens=shuffled_tokens,
                prompt_tokens=_context_token_count(shuffled_prompt),
                cited_ids=shuffled_cited_ids,
                generated_sql=generated_sql,
                error=None,
            )
        except Exception as exc:  # noqa: BLE001
            elapsed = round(time.monotonic() - start, 3)
            ranked_condition = ConditionResult(
                condition="lakeprompt_ranked",
                prompt="",
                answer="",
                latency_seconds=elapsed,
                evidence_count=0,
                join_count=0,
                context_tokens=0,
                prompt_tokens=0,
                cited_ids=[],
                generated_sql=[],
                error=str(exc),
            )
            shuffled_condition = ConditionResult(
                condition="join_no_ranking",
                prompt="",
                answer="",
                latency_seconds=elapsed,
                evidence_count=0,
                join_count=0,
                context_tokens=0,
                prompt_tokens=0,
                cited_ids=[],
                generated_sql=[],
                error=str(exc),
            )

        return ranked_condition, shuffled_condition

    def _extract_generated_sql(self, evidence: list) -> list[str]:
        seen: set[str] = set()
        queries: list[str] = []
        for item in evidence:
            provenance = getattr(item, "provenance", None)
            sql = getattr(provenance, "sql", "") if provenance is not None else ""
            if sql and sql not in seen:
                seen.add(sql)
                queries.append(sql)
        return queries

    def _serialize_evidence(self, evidence: list) -> str:
        return "\n".join(
            f"{item.evidence_id}: {json.dumps(prettify_row_data_keys(item.data), default=str, sort_keys=True)}"
            for item in evidence
        )

    def _evidence_ids_from_condition(self, cond: ConditionResult) -> set[str]:
        return {f"E{i}" for i in range(1, cond.evidence_count + 1)}

    def _best_single_table_rows(
        self, schema_dir: Path, question: str, sample_rows: int
    ) -> str:
        """Return sample rows from the table whose columns best match the question."""
        question_tokens = set(question.lower().split())
        best_score = -1
        best_text = ""
        for csv_path in sorted(schema_dir.glob("*.csv")):
            df = pl.read_csv(csv_path, n_rows=sample_rows)
            col_tokens = set(" ".join(df.columns).lower().split())
            score = len(question_tokens & col_tokens)
            if score > best_score:
                best_score = score
                rows_text = f"Table: {csv_path.stem}\n"
                rows_text += "\n".join(
                    json.dumps(row, default=str, sort_keys=True) for row in df.to_dicts()
                )
                best_text = rows_text
        return best_text or "[no tables found]"

    def _all_table_rows_text(self, schema_dir: Path, sample_rows: int) -> str:
        """Return sample rows from every table in the schema, concatenated."""
        sections: list[str] = []
        for csv_path in sorted(schema_dir.glob("*.csv")):
            df = pl.read_csv(csv_path, n_rows=sample_rows)
            section = f"Table: {csv_path.stem}\n"
            section += "\n".join(
                json.dumps(row, default=str, sort_keys=True) for row in df.to_dicts()
            )
            sections.append(section)
        return "\n\n".join(sections) or "[no tables found]"

    def _load_metadata(self) -> list[dict[str, str]]:
        with self.metadata_csv.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    def _group_metadata_by_schema(
        self,
        rows: list[dict[str, str]],
        schema_limit: int,
    ) -> dict[str, list[dict[str, str]]]:
        grouped: dict[str, list[dict[str, str]]] = {}
        for row in rows:
            schema_id = self._infer_schema_id(row)
            if schema_id is None:
                continue
            if schema_id not in grouped and len(grouped) >= schema_limit:
                continue
            grouped.setdefault(schema_id, []).append(row)
        selected_ids = list(grouped.keys())[:schema_limit]
        return {schema_id: grouped[schema_id] for schema_id in selected_ids}

    def _infer_schema_id(self, row: dict[str, str]) -> str | None:
        for key in ("db_id", "database_id", "schema_id", "schema", "db"):
            value = row.get(key)
            if value:
                return value
        return None

    def _resolve_schema_dir(self, schema_id: str) -> Path:
        direct = self.dataset_root / schema_id
        if direct.exists():
            return direct
        nested = self.dataset_root / "database" / schema_id
        if nested.exists():
            return nested
        raise FileNotFoundError(
            f"Could not find schema directory for '{schema_id}' in {self.dataset_root}"
        )

    def _build_schema_description(self, schema_dir: Path, sample_rows: int) -> str:
        sections: list[str] = []
        for csv_path in sorted(schema_dir.glob("*.csv")):
            df = pl.read_csv(csv_path, n_rows=sample_rows)
            sections.append(f"Table: {csv_path.stem}")
            sections.append(f"Columns: {', '.join(df.columns)}")
            if df.height:
                sections.append("Sample rows:")
                for row in df.to_dicts():
                    sections.append(json.dumps(row, default=str, sort_keys=True))
            sections.append("")
        return "\n".join(sections).strip()

    def _generate_questions(
        self,
        *,
        schema_id: str,
        schema_description: str,
        join_rows: list[dict[str, str]],
        question_count: int,
    ) -> tuple[list[str], list[str]]:
        """
        Ask Claude to generate join-requiring questions and reference answers.

        Returns a tuple of (questions, ground_truths). Ground truths are
        short expected answer strings used for exact match and F1 scoring.
        """
        join_summary = json.dumps(join_rows[:10], indent=2)
        prompt = f"""
            Generate exactly {question_count} natural-language questions for the Spider schema "{schema_id}".

            Requirements:
            - Every question must require joining at least two tables.
            - Prefer questions that clearly depend on the known join metadata.
            - For each question, provide a concise expected answer (1-10 words).
            - Return JSON only in this format:
            {{
              "items": [
                {{"question": "q1", "answer": "expected answer 1"}},
                {{"question": "q2", "answer": "expected answer 2"}}
              ]
            }}

            Schema summary:
            {schema_description}

            Join metadata examples:
            {join_summary}
            """.strip()

        raw = self.claude.complete(
            system=(
                "You generate evaluation questions for relational multi-table reasoning. "
                "Return strict JSON only."
            ),
            prompt=prompt,
            max_tokens=1400,
            temperature=0.4,
        )
        parsed = json.loads(raw)
        items = parsed.get("items", [])
        if not isinstance(items, list):
            raise ValueError(f"Claude returned invalid question payload: {raw}")
        items = items[:question_count]
        questions = [item["question"] for item in items if isinstance(item, dict)]
        ground_truths = [item.get("answer", "") for item in items if isinstance(item, dict)]
        return questions, ground_truths

    def _build_baseline_prompt(self, schema_description: str, question: str) -> str:
        return (
            f"Use the following schema and sample rows to answer the question.\n"
            f"If you cannot answer confidently from this information alone, say so.\n\n"
            f"Schema:\n{schema_description}\n\nQuestion:\n{question}"
        )

    def _build_lakeprompt_prompt(self, question: str, lakeprompt_context: str) -> str:
        return (
            f"Use only the following LakePrompt evidence rows to answer the question.\n"
            f"If the evidence is insufficient, say so.\n\n"
            f"Evidence:\n{lakeprompt_context or '[no evidence returned]'}\n\n"
            f"Question:\n{question}"
        )

    def _build_payload(self, examples: list[EvaluationExample]) -> dict[str, Any]:
        return self._build_payload_with_metadata(examples, {})

    def _serialize_condition(self, condition: ConditionResult) -> dict[str, Any]:
        return {
            "baseline_name": condition.condition,
            "exact_prompt_sent_via_api": condition.prompt,
            "response_from_claude": condition.answer,
            "generated_sql": condition.generated_sql,
            "scoring_metrics": {
                "exact_match": condition.exact_match,
                "token_f1": condition.token_f1,
                "faithfulness": condition.faithfulness,
                "latency_seconds": condition.latency_seconds,
                "evidence_count": condition.evidence_count,
                "join_count": condition.join_count,
                "context_tokens": condition.context_tokens,
                "prompt_tokens": condition.prompt_tokens,
                "cited_ids": condition.cited_ids,
                "error": condition.error,
            },
        }

    def _build_payload_with_metadata(
        self,
        examples: list[EvaluationExample],
        extra_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        baseline_order = {
            "no_context": 0,
            "single_table": 1,
            "naive_multitable": 2,
            "schema_baseline": 3,
            "join_no_ranking": 4,
            "lakeprompt_ranked": 5,
        }
        serialized_examples = []
        for ex in examples:
            serialized_examples.append({
                "schema_id": ex.schema_id,
                "question": ex.question,
                "expected_sql": ex.expected_sql,
                "ground_truth": ex.ground_truth,
                "ground_truth_error": ex.ground_truth_error,
                "join_metadata": ex.join_metadata,
                "baselines": [self._serialize_condition(c) for c in ex.conditions],
            })

        # Aggregate metrics per condition across all examples
        condition_names = [c.condition for c in examples[0].conditions] if examples else []
        condition_names = sorted(
            condition_names,
            key=lambda name: (baseline_order.get(name, 999), name),
        )
        aggregate: dict[str, dict[str, Any]] = {}
        for name in condition_names:
            cond_results = [
                c for ex in examples for c in ex.conditions if c.condition == name
            ]
            n = len(cond_results)
            aggregate[name] = {
                "n": n,
                "exact_match_rate": round(sum(c.exact_match for c in cond_results) / n, 4) if n else 0,
                "mean_token_f1": round(sum(c.token_f1 for c in cond_results) / n, 4) if n else 0,
                "mean_faithfulness": round(sum(c.faithfulness for c in cond_results) / n, 4) if n else 0,
                "mean_latency_seconds": round(sum(c.latency_seconds for c in cond_results) / n, 4) if n else 0,
                "mean_context_tokens": round(sum(c.context_tokens for c in cond_results) / n, 1) if n else 0,
                "mean_prompt_tokens": round(sum(c.prompt_tokens for c in cond_results) / n, 1) if n else 0,
                "mean_join_count": round(sum(c.join_count for c in cond_results) / n, 2) if n else 0,
                "error_rate": round(sum(1 for c in cond_results if c.error) / n, 4) if n else 0,
            }

        return {
            "generated_at": datetime.now(UTC).isoformat(),
            "dataset_root": str(self.dataset_root),
            "metadata_csv": str(self.metadata_csv),
            "claude_model": self.claude.model,
            **extra_metadata,
            "aggregate_metrics": aggregate,
            "examples": serialized_examples,
        }

    def _write_progress(
        self,
        examples: list[EvaluationExample],
        run_metadata: dict[str, Any],
    ) -> None:
        payload = self._build_payload_with_metadata(examples, run_metadata)
        self._write_json(payload)
        self._write_txt(payload)

    def _write_json(self, payload: dict[str, Any]) -> None:
        self.output_json.parent.mkdir(parents=True, exist_ok=True)
        with self.output_json.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def _write_txt(self, payload: dict[str, Any]) -> None:
        baseline_order = {
            "no_context": 0,
            "single_table": 1,
            "naive_multitable": 2,
            "schema_baseline": 3,
            "join_no_ranking": 4,
            "lakeprompt_ranked": 5,
        }

        lines = [
            "# LakePrompt Evaluation Report",
            "",
            "## Run Metadata",
            f"- Generated at: `{payload['generated_at']}`",
            f"- Dataset root: `{payload['dataset_root']}`",
            f"- Metadata CSV: `{payload['metadata_csv']}`",
            f"- Claude model: `{payload['claude_model']}`",
        ]

        if "questions_file" in payload:
            lines.append(f"- Questions file: `{payload['questions_file']}`")
        if "max_questions" in payload:
            lines.append(f"- Max questions: `{payload['max_questions']}`")

        lines.extend([
            "## Metric Glossary",
            "",
            "- `exact_match_rate`: Fraction of examples where the answer text exactly matches the ground truth after lowercasing and trimming.",
            "- `mean_token_f1`: Average token-overlap F1 between the answer text and ground truth.",
            "- `mean_faithfulness`: Average fraction of valid evidence IDs cited or mentioned in the answer.",
            "- `mean_latency_seconds`: Average wall-clock runtime for that baseline per example.",
            "- `mean_context_tokens`: Average whitespace-token count of the evidence/context portion only.",
            "- `mean_prompt_tokens`: Average whitespace-token count of the full prompt sent to Claude.",
            "- `mean_join_count`: Average number of distinct join paths used by the evidence for that baseline.",
            "- `error_rate`: Fraction of examples where that baseline recorded an execution or API error.",
            "- `exact_match`: Whether one example's answer exactly matched the ground truth after lowercasing and trimming.",
            "- `token_f1`: Token-overlap F1 for one example.",
            "- `faithfulness`: Fraction of valid evidence IDs cited or explicitly mentioned for one example.",
            "- `latency_seconds`: Runtime for one example.",
            "- `evidence_count`: Number of evidence rows returned for one example.",
            "- `join_count`: Number of distinct join paths represented in those evidence rows.",
            "- `context_tokens`: Approximate whitespace-token count of the evidence/context only for one example.",
            "- `prompt_tokens`: Approximate whitespace-token count of the full prompt for one example.",
            "- `cited_ids`: Evidence IDs cited by the answer when present.",
            "- `error`: Execution or API error captured for that example, if any.",
            "",
            "## Aggregate Metrics",
            "",
        ])

        for cond_name, metrics in payload.get("aggregate_metrics", {}).items():
            lines.extend([
                f"### {cond_name}",
                "",
                "| Metric | Value |",
                "| --- | --- |",
            ])
            for key, value in metrics.items():
                lines.append(f"| `{key}` | `{value}` |")
            lines.append("")

        lines.extend(["## Examples", ""])

        for index, example in enumerate(payload["examples"], start=1):
            ground_truth = example["ground_truth"] or "[none]"
            lines.extend([
                f"## Example {index}",
                "",
                f"- Schema: `{example['schema_id']}`",
                f"- Question: {example['question']}",
                f"- Ground Truth: {ground_truth}",
                "",
            ])
            if example.get("ground_truth_error"):
                lines.extend([
                    f"- Ground Truth Error: `{example['ground_truth_error']}`",
                    "",
                ])
            if example.get("expected_sql"):
                lines.extend([
                    "**Expected SQL Query**",
                    "",
                    "```sql",
                    example["expected_sql"],
                    "```",
                    "",
                ])
            shared_generated_sql: list[str] = []
            for baseline in example["baselines"]:
                if baseline["baseline_name"] in {"join_no_ranking", "lakeprompt_ranked"}:
                    shared_generated_sql = baseline.get("generated_sql", [])
                    if shared_generated_sql:
                        break
            if shared_generated_sql:
                lines.extend([
                    "**LakePrompt Generated Query**",
                    "",
                ])
                for sql in shared_generated_sql:
                    lines.extend([
                        "```sql",
                        sql,
                        "```",
                        "",
                    ])
            ordered_baselines = sorted(
                example["baselines"],
                key=lambda item: (baseline_order.get(item["baseline_name"], 999), item["baseline_name"]),
            )
            for cond in ordered_baselines:
                metrics = cond["scoring_metrics"]
                cited_ids = ", ".join(metrics["cited_ids"]) if metrics["cited_ids"] else "[none]"
                lines.extend([
                    f"### {cond['baseline_name']}",
                    "",
                    "**Prompt**",
                    "",
                    "```text",
                    cond["exact_prompt_sent_via_api"] or "[no prompt]",
                    "```",
                    "",
                    "**Response**",
                    "",
                    "```text",
                    cond["response_from_claude"] or "[no answer]",
                    "```",
                    "",
                ])
                lines.extend([
                    "**Scoring Metrics**",
                    "",
                    "| Metric | Value |",
                    "| --- | --- |",
                    f"| `exact_match` | `{metrics['exact_match']}` |",
                    f"| `token_f1` | `{metrics['token_f1']:.3f}` |",
                    f"| `faithfulness` | `{metrics['faithfulness']:.3f}` |",
                    f"| `latency_seconds` | `{metrics['latency_seconds']}` |",
                    f"| `evidence_count` | `{metrics['evidence_count']}` |",
                    f"| `join_count` | `{metrics['join_count']}` |",
                    f"| `context_tokens` | `{metrics['context_tokens']}` |",
                    f"| `prompt_tokens` | `{metrics['prompt_tokens']}` |",
                    f"| `cited_ids` | `{cited_ids}` |",
                    f"| `error` | `{metrics['error'] or '[none]'}` |",
                    "",
                ])

        self.output_txt.parent.mkdir(parents=True, exist_ok=True)
        self.output_txt.write_text("\n".join(lines), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Spider join evaluation for LakePrompt.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--metadata-csv", default=None)
    parser.add_argument("--questions-file", default=None)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-txt", required=True)
    parser.add_argument("--schema-limit", type=int, default=3)
    parser.add_argument("--questions-per-schema", type=int, default=3)
    parser.add_argument("--sample-rows", type=int, default=3)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--claude-model", default=DEFAULT_CLAUDE_MODEL)
    parser.add_argument("--lakeprompt-model", default=DEFAULT_CLAUDE_MODEL)
    parser.add_argument("--cache-path", default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if not args.questions_file and not args.metadata_csv:
        raise ValueError("Provide either --questions-file or --metadata-csv.")
    if args.questions_file and args.metadata_csv:
        raise ValueError("Provide only one of --questions-file or --metadata-csv.")
    runner = SpiderJoinEvaluation(
        dataset_root=args.dataset_root,
        metadata_csv=args.metadata_csv or "",
        output_json=args.output_json,
        output_txt=args.output_txt,
        claude_model=args.claude_model,
    )
    runner.run(
        schema_limit=args.schema_limit,
        questions_per_schema=args.questions_per_schema,
        sample_rows=args.sample_rows,
        lakeprompt_model=args.lakeprompt_model,
        cache_path=args.cache_path,
        questions_file=args.questions_file,
        max_questions=args.max_questions,
    )


if __name__ == "__main__":
    main()
