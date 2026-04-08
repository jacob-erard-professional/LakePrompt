from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

import anthropic
import polars as pl


DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-20250514"


@dataclass
class EvaluationExample:
    schema_id: str
    question: str
    join_metadata: dict[str, Any]
    baseline_prompt: str
    baseline_answer: str
    lakeprompt_context: str
    lakeprompt_answer: str
    lakeprompt_evidence_count: int
    lakeprompt_error: str | None = None


class ClaudeClient:
    """Thin wrapper around the Anthropic Messages API."""

    def __init__(self, model: str = DEFAULT_CLAUDE_MODEL):
        """
        Initialise an Anthropic client for evaluation-time prompting.

        Args:
            model: Claude model name to use for question generation and
                answer completion.

        Raises:
            ValueError: If `ANTHROPIC_API_KEY` is not set.
        """
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
        """
        Send a single prompt to Claude and return concatenated text output.

        Args:
            system: System instruction for the request.
            prompt: User-visible prompt content.
            max_tokens: Maximum response length.
            temperature: Sampling temperature.

        Returns:
            The text content returned by Claude.
        """
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
    baseline vs LakePrompt answers to JSON and TXT outputs.
    """

    def __init__(
        self,
        dataset_root: str,
        metadata_csv: str,
        output_json: str,
        output_txt: str,
        claude_model: str = DEFAULT_CLAUDE_MODEL,
    ):
        """
        Configure an evaluation run over Spider join schemas.

        Args:
            dataset_root: Root directory containing schema CSV folders.
            metadata_csv: Path to Spider join metadata CSV.
            output_json: Destination for structured JSON results.
            output_txt: Destination for human-readable TXT results.
            claude_model: Claude model used for generation and answering.
        """
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
        """
        Execute the full evaluation loop and persist both output formats.

        For each selected schema, this method generates join-requiring
        questions with Claude, records a baseline Claude answer from raw
        schema context, then attempts the LakePrompt workflow and records
        the evidence-grounded Claude answer.

        Args:
            schema_limit: Maximum number of schemas to evaluate.
            questions_per_schema: Number of generated questions per schema.
            sample_rows: Number of sample rows per table to include in
                schema summaries sent to Claude.
            lakeprompt_model: Model string passed into LakePrompt for table
                summary generation.
            cache_path: Optional summary-cache path forwarded to LakePrompt.

        Returns:
            A dictionary matching the JSON payload written to disk.
        """
        metadata_rows = self._load_metadata()
        schema_rows = self._group_metadata_by_schema(metadata_rows, schema_limit)

        examples: list[EvaluationExample] = []

        for schema_id, rows in schema_rows.items():
            schema_dir = self._resolve_schema_dir(schema_id)
            schema_description = self._build_schema_description(schema_dir, sample_rows)
            questions = self._generate_questions(
                schema_id=schema_id,
                schema_description=schema_description,
                join_rows=rows,
                question_count=questions_per_schema,
            )

            for question in questions:
                baseline_prompt = self._build_baseline_prompt(schema_description, question)
                baseline_answer = self.claude.complete(
                    system=(
                        "Answer questions about relational data. "
                        "If the schema context is insufficient, say so."
                    ),
                    prompt=baseline_prompt,
                )

                lakeprompt_context = ""
                lakeprompt_answer = ""
                lakeprompt_evidence_count = 0
                lakeprompt_error = None

                try:
                    from .lakeprompt import LakePrompt

                    lp = LakePrompt(
                        str(schema_dir),
                        model=lakeprompt_model,
                        cache_path=cache_path,
                    )
                    result = lp.query(question)
                    evidence_lines = [
                        f"{item.evidence_id}: {json.dumps(item.data, default=str, sort_keys=True)}"
                        for item in result.evidence
                    ]
                    lakeprompt_context = "\n".join(evidence_lines)
                    lakeprompt_evidence_count = len(result.evidence)
                    lakeprompt_prompt = self._build_lakeprompt_prompt(question, lakeprompt_context)
                    lakeprompt_answer = self.claude.complete(
                        system=(
                            "Answer questions using only the provided evidence rows. "
                            "If the evidence is insufficient, say so explicitly."
                        ),
                        prompt=lakeprompt_prompt,
                    )
                except Exception as exc:  # noqa: BLE001
                    lakeprompt_error = str(exc)

                examples.append(
                    EvaluationExample(
                        schema_id=schema_id,
                        question=question,
                        join_metadata=rows[0],
                        baseline_prompt=baseline_prompt,
                        baseline_answer=baseline_answer,
                        lakeprompt_context=lakeprompt_context,
                        lakeprompt_answer=lakeprompt_answer,
                        lakeprompt_evidence_count=lakeprompt_evidence_count,
                        lakeprompt_error=lakeprompt_error,
                    )
                )

        payload = {
            "generated_at": datetime.now(UTC).isoformat(),
            "dataset_root": str(self.dataset_root),
            "metadata_csv": str(self.metadata_csv),
            "claude_model": self.claude.model,
            "examples": [asdict(example) for example in examples],
        }

        self._write_json(payload)
        self._write_txt(payload)
        return payload

    def _load_metadata(self) -> list[dict[str, str]]:
        """
        Load Spider join metadata rows from CSV.

        Returns:
            A list of metadata records keyed by CSV header.
        """
        with self.metadata_csv.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    def _group_metadata_by_schema(
        self,
        rows: list[dict[str, str]],
        schema_limit: int,
    ) -> dict[str, list[dict[str, str]]]:
        """
        Group metadata rows by schema identifier up to a configured limit.

        Args:
            rows: Raw metadata records from the CSV file.
            schema_limit: Maximum number of distinct schemas to keep.

        Returns:
            A mapping from schema id to all metadata rows for that schema.
        """
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
        """
        Extract the schema identifier from a metadata row.

        Args:
            row: One metadata record from `dev_metadata.csv`.

        Returns:
            The detected schema identifier, or `None` if no known key is
            present.
        """
        for key in ("db_id", "database_id", "schema_id", "schema", "db"):
            value = row.get(key)
            if value:
                return value
        return None

    def _resolve_schema_dir(self, schema_id: str) -> Path:
        """
        Resolve a schema id to its directory of CSV files.

        Args:
            schema_id: Spider schema/database identifier.

        Returns:
            Path to the directory containing that schema's CSV files.

        Raises:
            FileNotFoundError: If the schema directory cannot be found.
        """
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
        """
        Build a compact natural-language schema summary from CSV files.

        Args:
            schema_dir: Directory containing a schema's CSV tables.
            sample_rows: Number of example rows to include per table.

        Returns:
            A plain-text schema description suitable for prompting Claude.
        """
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
    ) -> list[str]:
        """
        Ask Claude to generate join-requiring questions for one schema.

        Args:
            schema_id: Schema/database identifier.
            schema_description: Prompt-ready description of schema tables.
            join_rows: Metadata rows describing known join paths.
            question_count: Number of questions to request.

        Returns:
            A list of generated natural-language questions.

        Raises:
            ValueError: If Claude does not return the expected JSON shape.
        """
        join_summary = json.dumps(join_rows[:10], indent=2)
        prompt = f"""
            Generate exactly {question_count} natural-language questions for the Spider schema "{schema_id}".

            Requirements:
            - Every question must require joining at least two tables.
            - Prefer questions that clearly depend on the known join metadata.
            - Do not answer the questions.
            - Return JSON only in this format:
            {{
            "questions": ["q1", "q2"]
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
            max_tokens=1200,
            temperature=0.4,
        )
        parsed = json.loads(raw)
        questions = parsed.get("questions", [])
        if not isinstance(questions, list) or not all(isinstance(q, str) for q in questions):
            raise ValueError(f"Claude returned invalid question payload: {raw}")
        return questions[:question_count]

    def _build_baseline_prompt(self, schema_description: str, question: str) -> str:
        """
        Build the baseline prompt without LakePrompt evidence.

        Args:
            schema_description: Raw schema and sample-row context.
            question: Natural-language evaluation question.

        Returns:
            Prompt text for the baseline Claude condition.
        """
        return f"""
            Use the following schema and sample rows to answer the question.
            If you cannot answer confidently from this information alone, say so.

            Schema:
            {schema_description}

            Question:
            {question}
            """.strip()

    def _build_lakeprompt_prompt(self, question: str, lakeprompt_context: str) -> str:
        """
        Build the evidence-grounded prompt for the LakePrompt condition.

        Args:
            question: Natural-language evaluation question.
            lakeprompt_context: Serialized evidence rows returned by LakePrompt.

        Returns:
            Prompt text instructing Claude to answer from evidence only.
        """
        return f"""
            Use only the following LakePrompt evidence rows to answer the question.
            If the evidence is insufficient, say so.

            Evidence:
            {lakeprompt_context or "[no evidence returned]"}

            Question:
            {question}
            """.strip()

    def _write_json(self, payload: dict[str, Any]) -> None:
        """
        Write the structured evaluation payload to a JSON file.

        Args:
            payload: Evaluation results dictionary to persist.
        """
        self.output_json.parent.mkdir(parents=True, exist_ok=True)
        with self.output_json.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def _write_txt(self, payload: dict[str, Any]) -> None:
        """
        Write a human-readable evaluation report to a TXT file.

        Args:
            payload: Evaluation results dictionary to render.
        """
        lines = [
            "LakePrompt Evaluation Report",
            f"Generated at: {payload['generated_at']}",
            f"Dataset root: {payload['dataset_root']}",
            f"Metadata CSV: {payload['metadata_csv']}",
            f"Claude model: {payload['claude_model']}",
            "",
        ]

        for index, example in enumerate(payload["examples"], start=1):
            lines.extend(
                [
                    f"Example {index}",
                    f"Schema: {example['schema_id']}",
                    f"Question: {example['question']}",
                    "",
                    "Baseline Answer:",
                    example["baseline_answer"] or "[no answer]",
                    "",
                    "LakePrompt Evidence:",
                    example["lakeprompt_context"] or "[no evidence]",
                    "",
                    "LakePrompt Answer:",
                    example["lakeprompt_answer"] or "[no answer]",
                    "",
                    f"LakePrompt Evidence Count: {example['lakeprompt_evidence_count']}",
                    f"LakePrompt Error: {example['lakeprompt_error'] or '[none]'}",
                    "",
                    "-" * 80,
                    "",
                ]
            )

        self.output_txt.parent.mkdir(parents=True, exist_ok=True)
        self.output_txt.write_text("\n".join(lines), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    """
    Build the CLI parser for the evaluation runner.

    Returns:
        An `ArgumentParser` configured with evaluation options.
    """
    parser = argparse.ArgumentParser(description="Run Spider join evaluation for LakePrompt.")
    parser.add_argument("--dataset-root", required=True, help="Directory containing Spider CSV schemas.")
    parser.add_argument("--metadata-csv", required=True, help="Path to Spider dev_metadata.csv.")
    parser.add_argument("--output-json", required=True, help="Output path for machine-readable JSON.")
    parser.add_argument("--output-txt", required=True, help="Output path for human-readable TXT.")
    parser.add_argument("--schema-limit", type=int, default=3, help="Number of schemas to evaluate.")
    parser.add_argument(
        "--questions-per-schema",
        type=int,
        default=3,
        help="Number of generated questions per schema.",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=3,
        help="Number of sample rows per table in schema prompts.",
    )
    parser.add_argument(
        "--claude-model",
        default=DEFAULT_CLAUDE_MODEL,
        help="Anthropic model for question generation and answering.",
    )
    parser.add_argument(
        "--lakeprompt-model",
        default="nvidia/nemotron-3-super-120b-a12b:free",
        help="Model string passed into LakePrompt for table summarization.",
    )
    parser.add_argument(
        "--cache-path",
        default=None,
        help="Optional cache path for LakePrompt table summaries.",
    )
    return parser


def main() -> None:
    """Parse CLI arguments and run the evaluation workflow."""
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
