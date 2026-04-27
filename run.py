from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lakeprompt import LakePrompt
from lakeprompt._tracing import PipelineLogger


class CliPipelineTracker(PipelineLogger):
    """Compact interactive tracker for LakePrompt pipeline progress."""

    _SECTION_LABELS = {
        "table_summaries": "Prepared table summaries",
        "retrieval": "Retrieved relevant columns",
        "join_paths_progress": "Planning join paths",
        "join_paths": "Ranked join paths",
        "sql_refinement": "Refined SQL",
        "sql_query": "Executing SQL",
        "ranked_rows": "Ranked evidence rows",
        "prompt_context": "Packaged prompt context",
        "llm_request": "Requesting final answer",
        "llm_response": "Received final answer",
    }

    def __init__(self) -> None:
        super().__init__(enabled=True)
        self._last_line: str | None = None

    def log(self, section: str, message: str, payload: Any | None = None) -> None:
        label = self._SECTION_LABELS.get(section)
        if label is None:
            return
        line = f"[stage] {label}"
        if line == self._last_line:
            return
        self._last_line = line
        print(line, flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interactive CLI for querying a GitHub/CSV/ZIP-backed LakePrompt database."
    )
    parser.add_argument(
        "--database_link",
        required=True,
        help="GitHub repository URL, direct CSV URL, ZIP URL, or local CSV directory path.",
    )
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-20250514",
        help="Anthropic model to use for LakePrompt.",
    )
    parser.add_argument(
        "--cache_path",
        default=None,
        help="Optional path for cached table summaries.",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        help="Optional cache directory for local artifacts.",
    )
    parser.add_argument(
        "--source_cache_dir",
        default=None,
        help="Optional cache directory for downloaded remote sources.",
    )
    return parser


def _looks_like_url(value: str) -> bool:
    return value.startswith("http://") or value.startswith("https://")


def _build_lakeprompt(args: argparse.Namespace) -> LakePrompt:
    source = args.database_link
    print("[stage] Initializing LakePrompt", flush=True)
    if _looks_like_url(source):
        lp = LakePrompt.from_url(
            source,
            model=args.model,
            cache_path=args.cache_path,
            cache_dir=args.cache_dir,
            source_cache_dir=args.source_cache_dir,
            logger=False,
        )
    else:
        lp = LakePrompt(
            source,
            model=args.model,
            cache_path=args.cache_path,
            cache_dir=args.cache_dir,
            logger=False,
        )

    tracker = CliPipelineTracker()
    lp.logger = tracker
    lp.profiler.logger = tracker
    lp.retriever.logger = tracker
    lp.executor.logger = tracker
    lp.packager.logger = tracker
    print("[stage] LakePrompt ready", flush=True)
    table_names = sorted(lp.lake.tables)
    print("\nLoaded tables:")
    for table_name in table_names:
        print(f"- {table_name}")
    print()
    return lp


def _print_answer(answer: Any) -> None:
    print("\nAnswer:")
    print(answer.text or "[no answer]")

    if getattr(answer, "cited_ids", None):
        print(f"\nCited evidence: {', '.join(answer.cited_ids)}")

    input_tokens = getattr(answer, "api_input_tokens", None)
    output_tokens = getattr(answer, "api_output_tokens", None)
    if input_tokens is not None or output_tokens is not None:
        print(
            "\nToken usage:"
            f" input={input_tokens if input_tokens is not None else '[unknown]'}"
            f", output={output_tokens if output_tokens is not None else '[unknown]'}"
        )

    evidence = getattr(answer, "evidence", []) or []
    if evidence:
        print("\nEvidence:")
        for item in evidence:
            print(f"- {item.evidence_id}: {item.data}")


def repl(lp: LakePrompt) -> None:
    print("\nInteractive mode. Enter a question, or type `exit` / `quit`.\n")
    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break

        print("[stage] Starting query", flush=True)
        try:
            answer = lp.query(question)
        except Exception as exc:  # noqa: BLE001
            print(f"\nError: {exc}\n", flush=True)
            continue

        _print_answer(answer)
        print()


def main() -> int:
    args = build_parser().parse_args()
    lp = _build_lakeprompt(args)
    repl(lp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
