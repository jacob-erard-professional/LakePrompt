from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

import anthropic
import polars as pl


DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-20250514"


def _tokenize(text: str) -> list[str]:
    """Lowercase whitespace-split tokens for F1 computation."""
    return text.lower().split()


def _token_f1(prediction: str, ground_truth: str) -> float:
    """
    Compute token-overlap F1 between a predicted answer and ground truth.

    Args:
        prediction: Model-generated answer string.
        ground_truth: Reference answer string.

    Returns:
        F1 score between 0.0 and 1.0.
    """
    pred_tokens = _tokenize(prediction)
    gt_tokens = _tokenize(ground_truth)
    if not pred_tokens or not gt_tokens:
        return 0.0
    pred_set = set(pred_tokens)
    gt_set = set(gt_tokens)
    common = pred_set & gt_set
    if not common:
        return 0.0
    precision = len(common) / len(pred_set)
    recall = len(common) / len(gt_set)
    return 2 * precision * recall / (precision + recall)


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
    context_tokens: int
    cited_ids: list[str] = field(default_factory=list)
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

    def run(
        self,
        *,
        schema_limit: int = 3,
        questions_per_schema: int = 3,
        sample_rows: int = 3,
        lakeprompt_model: str = "nvidia/nemotron-3-super-120b-a12b:free",
        cache_path: str | None = None,
    ) -> dict[str, Any]:
        metadata_rows = self._load_metadata()
        schema_rows = self._group_metadata_by_schema(metadata_rows, schema_limit)

        examples: list[EvaluationExample] = []

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
                example = EvaluationExample(
                    schema_id=schema_id,
                    question=question,
                    join_metadata=rows[0],
                    ground_truth=ground_truth,
                )

                # Condition 1: no context (LLM only)
                example.conditions.append(
                    self._run_no_context(question)
                )

                # Condition 2: single-table retrieval
                example.conditions.append(
                    self._run_single_table(schema_dir, schema_description, question, sample_rows)
                )

                # Condition 3: naive multi-table concatenation (no join)
                example.conditions.append(
                    self._run_naive_multitable(schema_dir, schema_description, question, sample_rows)
                )

                # Condition 4: schema baseline (original condition)
                example.conditions.append(
                    self._run_schema_baseline(schema_description, question)
                )

                # Condition 5: LakePrompt full pipeline
                lp_result = self._run_lakeprompt(
                    schema_dir, question, lakeprompt_model, cache_path
                )
                example.conditions.append(lp_result)

                # Condition 6: join without ranking (shuffled evidence)
                example.conditions.append(
                    self._run_join_no_ranking(
                        schema_dir, question, lakeprompt_model, cache_path
                    )
                )

                # Compute metrics against ground truth
                evidence_ids = self._evidence_ids_from_condition(lp_result)
                for cond in example.conditions:
                    cond.exact_match = _exact_match(cond.answer, ground_truth)
                    cond.token_f1 = _token_f1(cond.answer, ground_truth)
                    cond.faithfulness = _faithfulness_score(
                        cond.answer, cond.cited_ids, evidence_ids
                    )

                examples.append(example)

        payload = self._build_payload(examples)
        self._write_json(payload)
        self._write_txt(payload)
        return payload

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
            error=error,
        )

    def _run_lakeprompt(
        self,
        schema_dir: Path,
        question: str,
        lakeprompt_model: str,
        cache_path: str | None,
    ) -> ConditionResult:
        """Condition 5: full LakePrompt pipeline with ranked evidence."""
        start = time.monotonic()
        try:
            from lakeprompt import LakePrompt

            lp = LakePrompt(str(schema_dir), model=lakeprompt_model, cache_path=cache_path)
            result = lp.query(question)

            evidence_lines = self._serialize_evidence(result.evidence)
            context_tokens = _context_token_count(evidence_lines)
            joins = _join_count(result.evidence)

            prompt = self._build_lakeprompt_prompt(question, evidence_lines)
            answer = result.text
            cited_ids = result.cited_ids
            error = None
        except Exception as exc:  # noqa: BLE001
            evidence_lines = ""
            context_tokens = 0
            joins = 0
            prompt = ""
            answer = ""
            cited_ids = []
            error = str(exc)

        return ConditionResult(
            condition="lakeprompt_ranked",
            prompt=prompt,
            answer=answer,
            latency_seconds=round(time.monotonic() - start, 3),
            evidence_count=len(cited_ids),
            join_count=joins,
            context_tokens=context_tokens,
            cited_ids=cited_ids,
            error=error,
        )

    def _run_join_no_ranking(
        self,
        schema_dir: Path,
        question: str,
        lakeprompt_model: str,
        cache_path: str | None,
    ) -> ConditionResult:
        """
        Condition 6: join executed but evidence order is shuffled before prompting.

        This isolates the contribution of relevance ranking from the rest of
        the pipeline.
        """
        start = time.monotonic()
        try:
            from lakeprompt import LakePrompt

            lp = LakePrompt(str(schema_dir), model=lakeprompt_model, cache_path=cache_path)
            result = lp.query(question)

            shuffled_evidence = list(result.evidence)
            random.shuffle(shuffled_evidence)

            evidence_lines = self._serialize_evidence(shuffled_evidence)
            context_tokens = _context_token_count(evidence_lines)
            joins = _join_count(shuffled_evidence)

            prompt = self._build_lakeprompt_prompt(question, evidence_lines)
            answer = self.claude.complete(
                system=(
                    "Answer questions using only the provided evidence rows. "
                    "If the evidence is insufficient, say so explicitly."
                ),
                prompt=prompt,
            )
            cited_ids = [
                item.evidence_id
                for item in shuffled_evidence
                if item.evidence_id in answer
            ]
            error = None
        except Exception as exc:  # noqa: BLE001
            evidence_lines = ""
            context_tokens = 0
            joins = 0
            prompt = ""
            answer = ""
            cited_ids = []
            error = str(exc)

        return ConditionResult(
            condition="join_no_ranking",
            prompt=prompt,
            answer=answer,
            latency_seconds=round(time.monotonic() - start, 3),
            evidence_count=len(cited_ids),
            join_count=joins,
            context_tokens=context_tokens,
            cited_ids=cited_ids,
            error=error,
        )

    def _serialize_evidence(self, evidence: list) -> str:
        return "\n".join(
            f"{item.evidence_id}: {json.dumps(item.data, default=str, sort_keys=True)}"
            for item in evidence
        )

    def _evidence_ids_from_condition(self, cond: ConditionResult) -> set[str]:
        return set(cond.cited_ids)

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
        serialized_examples = []
        for ex in examples:
            serialized_examples.append({
                "schema_id": ex.schema_id,
                "question": ex.question,
                "ground_truth": ex.ground_truth,
                "join_metadata": ex.join_metadata,
                "conditions": [asdict(c) for c in ex.conditions],
            })

        # Aggregate metrics per condition across all examples
        condition_names = [c.condition for c in examples[0].conditions] if examples else []
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
                "mean_join_count": round(sum(c.join_count for c in cond_results) / n, 2) if n else 0,
                "error_rate": round(sum(1 for c in cond_results if c.error) / n, 4) if n else 0,
            }

        return {
            "generated_at": datetime.now(UTC).isoformat(),
            "dataset_root": str(self.dataset_root),
            "metadata_csv": str(self.metadata_csv),
            "claude_model": self.claude.model,
            "aggregate_metrics": aggregate,
            "examples": serialized_examples,
        }

    def _write_json(self, payload: dict[str, Any]) -> None:
        self.output_json.parent.mkdir(parents=True, exist_ok=True)
        with self.output_json.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def _write_txt(self, payload: dict[str, Any]) -> None:
        lines = [
            "LakePrompt Evaluation Report",
            f"Generated at: {payload['generated_at']}",
            f"Dataset root: {payload['dataset_root']}",
            f"Metadata CSV: {payload['metadata_csv']}",
            f"Claude model: {payload['claude_model']}",
            "",
            "=== Aggregate Metrics ===",
        ]

        for cond_name, metrics in payload.get("aggregate_metrics", {}).items():
            lines.append(f"\nCondition: {cond_name}")
            for k, v in metrics.items():
                lines.append(f"  {k}: {v}")

        lines.extend(["", "=" * 80, ""])

        for index, example in enumerate(payload["examples"], start=1):
            lines.extend([
                f"Example {index}",
                f"Schema: {example['schema_id']}",
                f"Question: {example['question']}",
                f"Ground Truth: {example['ground_truth']}",
                "",
            ])
            for cond in example["conditions"]:
                lines.extend([
                    f"  [{cond['condition']}]",
                    f"  Answer: {cond['answer'] or '[no answer]'}",
                    f"  Exact Match: {cond['exact_match']}  Token F1: {cond['token_f1']:.3f}  Faithfulness: {cond['faithfulness']:.3f}",
                    f"  Latency: {cond['latency_seconds']}s  Context Tokens: {cond['context_tokens']}  Joins: {cond['join_count']}",
                    f"  Error: {cond['error'] or '[none]'}",
                    "",
                ])
            lines.extend(["-" * 80, ""])

        self.output_txt.parent.mkdir(parents=True, exist_ok=True)
        self.output_txt.write_text("\n".join(lines), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Spider join evaluation for LakePrompt.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-txt", required=True)
    parser.add_argument("--schema-limit", type=int, default=3)
    parser.add_argument("--questions-per-schema", type=int, default=3)
    parser.add_argument("--sample-rows", type=int, default=3)
    parser.add_argument("--claude-model", default=DEFAULT_CLAUDE_MODEL)
    parser.add_argument("--lakeprompt-model", default="nvidia/nemotron-3-super-120b-a12b:free")
    parser.add_argument("--cache-path", default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    runner = SpiderJoinEvaluation(
        dataset_root=args.dataset_root,
        metadata_csv=args.metadata_csv,
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
    )


if __name__ == "__main__":
    main()
