from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = REPO_ROOT / "eval" / "spider_join_questions.json"
DEFAULT_DATASET_ROOT = REPO_ROOT / "data"
DEFAULT_OUTPUT = REPO_ROOT / "eval" / "spider_join_questions_with_ground_truth.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Execute Spider SQL against local CSV schemas and attach ground truth answers.",
    )
    parser.add_argument(
        "--input-json",
        default=str(DEFAULT_INPUT),
        help="Path to the Spider join questions JSON file.",
    )
    parser.add_argument(
        "--dataset-root",
        default=str(DEFAULT_DATASET_ROOT),
        help="Root directory containing per-schema CSV folders.",
    )
    parser.add_argument(
        "--output-json",
        default=str(DEFAULT_OUTPUT),
        help="Path for the enriched JSON output.",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Optional limit for debugging a subset.",
    )
    return parser


def _resolve_schema_dir(dataset_root: Path, schema_id: str) -> Path:
    direct = dataset_root / schema_id
    if direct.exists():
        return direct
    nested = dataset_root / "database" / schema_id
    if nested.exists():
        return nested
    raise FileNotFoundError(f"Could not find schema directory for '{schema_id}' in {dataset_root}")


def _load_csvs_into_sqlite(conn: sqlite3.Connection, schema_dir: Path) -> None:
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


def _canonical_ground_truth(cursor: sqlite3.Cursor) -> str:
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
        return json.dumps([value for value in values], ensure_ascii=True)

    payload = [
        {column: value for column, value in zip(columns, row, strict=False)}
        for row in rows
    ]
    if not payload:
        return ""
    return json.dumps(payload, ensure_ascii=True, sort_keys=True)


def _execute_query(dataset_root: Path, schema_id: str, sql: str) -> str:
    schema_dir = _resolve_schema_dir(dataset_root, schema_id)
    conn = sqlite3.connect(":memory:")
    try:
        _load_csvs_into_sqlite(conn, schema_dir)
        cursor = conn.execute(sql)
        return _canonical_ground_truth(cursor)
    finally:
        conn.close()


def main() -> None:
    args = build_parser().parse_args()
    input_json = Path(args.input_json)
    dataset_root = Path(args.dataset_root)
    output_json = Path(args.output_json)

    items = json.loads(input_json.read_text(encoding="utf-8"))
    if not isinstance(items, list):
        raise ValueError("Input JSON must be a list of question objects.")

    enriched: list[dict] = []
    for index, item in enumerate(items, start=1):
        if args.max_questions is not None and len(enriched) >= args.max_questions:
            break
        if not isinstance(item, dict):
            continue
        schema_id = str(item.get("schema_id") or item.get("db_id") or "").strip()
        query = str(item.get("query") or "").strip()
        if not schema_id or not query:
            continue
        record = dict(item)
        try:
            record["ground_truth"] = _execute_query(dataset_root, schema_id, query)
            record["ground_truth_error"] = None
        except Exception as exc:  # noqa: BLE001
            record["ground_truth"] = ""
            record["ground_truth_error"] = str(exc)
        enriched.append(record)
        if index % 100 == 0:
            print(f"Processed {index} questions...")

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(enriched, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"Wrote {len(enriched)} questions to {output_json}")


if __name__ == "__main__":
    main()
