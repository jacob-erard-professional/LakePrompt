from __future__ import annotations

import argparse
import json
import sys

from lakeprompt import LakePrompt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a simple LakePrompt demo against a local CSV lake.",
    )
    parser.add_argument(
        "--lake-dir",
        required=True,
        type=str,
        help="Path to a local directory containing CSV files.",
    )
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="Optional first natural-language question to ask before entering interactive mode.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-sonnet-4-20250514",
        help="Anthropic model to use.",
    )
    parser.add_argument(
        "--cache-path",
        type=str,
        default=None,
        help="Optional explicit JSON path for cached table summaries.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Optional directory for default LakePrompt caches and saved artifacts.",
    )
    parser.add_argument(
        "--no-save-artifacts",
        action="store_true",
        help="Disable default summary-cache persistence for this run.",
    )
    parser.add_argument(
        "--logger",
        action="store_true",
        help="Print LakePrompt pipeline traces to stdout.",
    )
    parser.add_argument(
        "--show-prompt",
        action="store_true",
        help="Print the final packaged prompt sent to the answer model.",
    )
    parser.add_argument(
        "--show-sql",
        action="store_true",
        help="Print the executed SQL for each evidence path.",
    )
    return parser


def _build_lakeprompt(args: argparse.Namespace) -> LakePrompt:
    return LakePrompt(
        lake_dir=args.lake_dir,
        model=args.model,
        cache_path=args.cache_path,
        cache_dir=args.cache_dir,
        save_artifacts=not args.no_save_artifacts,
        logger=args.logger,
    )


def _print_answer(answer, *, show_prompt: bool, show_sql: bool) -> None:
    print("\nAnswer:")
    print(answer.text)

    print("\nRaw Response:")
    print(answer.raw_response or "(none)")

    print("\nCited IDs:")
    if answer.cited_ids:
        print(", ".join(answer.cited_ids))
    else:
        print("(none)")

    if show_prompt:
        print("\nPrompt:")
        print(answer.prompt or "(none)")

    print("\nEvidence:")
    if not answer.evidence:
        print("(no evidence rows returned)")
        return

    for item in answer.evidence:
        print(
            f"{item.evidence_id} | "
            f"score={item.relevance_score:.4f} | "
            f"path={item.provenance.path_id} | "
            f"tables={','.join(item.provenance.tables)}"
        )
        if show_sql:
            print("SQL:")
            print(item.provenance.sql or "(none)")
        print("Row:")
        print(json.dumps(item.data, default=str, sort_keys=True))


def _run_question(lakeprompt: LakePrompt, question: str, *, show_prompt: bool, show_sql: bool) -> None:
    answer = lakeprompt.query(question)
    _print_answer(answer, show_prompt=show_prompt, show_sql=show_sql)


def main() -> None:
    args = build_parser().parse_args()
    lakeprompt = _build_lakeprompt(args)

    print("LakePrompt interactive CLI")
    print("Enter a question, or type 'q' to quit.")

    if args.question:
        _run_question(
            lakeprompt,
            args.question,
            show_prompt=args.show_prompt,
            show_sql=args.show_sql,
        )

    try:
        while True:
            try:
                question = input("\n> ").strip()
            except EOFError:
                print("\nExiting.")
                return
            if not question:
                continue
            if question.lower() == "q":
                print("Exiting.")
                return
            _run_question(
                lakeprompt,
                question,
                show_prompt=args.show_prompt,
                show_sql=args.show_sql,
            )
    except KeyboardInterrupt:
        print("\nExiting.")
        return


if __name__ == "__main__":
    main()
