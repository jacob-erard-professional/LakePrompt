import csv
import sqlite3
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_INFER_SCHEMA_LENGTH = 10_000
DEFAULT_NULL_VALUES = ["-"]


@dataclass
class DataLake:
    """
    Central representation of a data lake for use across LakePrompt.

    Scans a directory of CSV files and registers each as a named table
    in SQLite. This class exists so every pipeline stage
    works from one consistent abstraction instead of passing raw file
    paths or ad hoc DataFrames around. Centralizing loading, sampling,
    and SQL execution reduces duplication and makes later stages easier
    to test and reason about.

    Args:
        lake_dir: Path to the directory containing CSV files.

    Example:
        >>> lake = DataLake.load("./data/csvs")
        >>> lake.query("SELECT * FROM customers WHERE city = 'Denver'")
    """

    lake_dir: Path
    tables: dict[str, object] = field(default_factory=dict)    
    cards: list = field(default_factory=list)
    join_graph: dict = field(default_factory=dict)
    connection: sqlite3.Connection | None = field(default=None, init=False, repr=False)
    db_path: Path | None = field(default=None, init=False)

    @classmethod
    def load(cls, lake_dir: str) -> "DataLake":
        """
        Scan a directory of CSV files and return an initialised DataLake.

        Each CSV becomes a named table using the filename without extension
        (e.g. 'customers.csv' → 'customers'). Tables are loaded lazily —
        no data is read into memory until a query or sample is requested.

        This factory is needed so the rest of the system can assume the
        lake is fully registered and ready for profiling, retrieval, and
        query execution from a single call.

        Args:
            lake_dir: Path to the directory containing CSV files.

        Returns:
            A `DataLake` populated with one SQLite-backed table per CSV.

        Raises:
            FileNotFoundError: If lake_dir does not exist.
            ValueError: If no CSV files are found.
        """
        lake = cls(lake_dir=Path(lake_dir))
        lake._load_tables()
        return lake

    def _load_tables(self) -> None:
        """
        Load all CSV files in `lake_dir` into a local SQLite database.

        This internal helper keeps filesystem validation and registration
        logic in one place, which makes initialization behavior easier to
        maintain.

        Returns:
            None. The method mutates `self.tables` in place.
        """
        if not self.lake_dir.exists():
            raise FileNotFoundError(
                f"Lake directory not found: {self.lake_dir}"
            )
        
        csv_files = sorted(self.lake_dir.rglob("*.csv"))
        if not csv_files:
            raise ValueError(f"No CSV files found in: {self.lake_dir}")

        temp_dir = Path(tempfile.mkdtemp(prefix="lakeprompt_sqlite_"))
        self.db_path = temp_dir / "lake.sqlite3"
        self.connection = sqlite3.connect(str(self.db_path))
        self.connection.row_factory = sqlite3.Row

        used_names: set[str] = set()
        for path in csv_files:
            table_name = self._table_name_from_path(path, used_names)
            self._load_csv_into_sqlite(path, table_name)
            self.tables[table_name] = {"name": table_name}
        self.connection.commit()

    def _load_csv_into_sqlite(self, csv_path: Path, table_name: str) -> None:
        """
        Create a SQLite table for one CSV and bulk insert its rows.
        """
        if self.connection is None:
            raise RuntimeError("SQLite connection is not initialized.")

        with csv_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            try:
                header = next(reader)
            except StopIteration:
                return

            columns = [column.strip() for column in header]
            sample_rows: list[list[object | None]] = []
            inferred_types = ["INTEGER"] * len(columns)

            for row_index, row in enumerate(reader):
                normalized = self._normalize_csv_row(row, len(columns))
                if row_index < DEFAULT_INFER_SCHEMA_LENGTH:
                    sample_rows.append(normalized)
                    for idx, value in enumerate(normalized):
                        inferred_types[idx] = self._merge_sqlite_types(
                            inferred_types[idx],
                            self._sqlite_type_for_value(value),
                        )
                else:
                    break

        create_columns = ", ".join(
            f'{self._quote_identifier(column)} {dtype}'
            for column, dtype in zip(columns, inferred_types, strict=False)
        )
        self.connection.execute(
            f'CREATE TABLE {self._quote_identifier(table_name)} ({create_columns})'
        )

        placeholders = ", ".join("?" for _ in columns)
        insert_sql = (
            f'INSERT INTO {self._quote_identifier(table_name)} '
            f'VALUES ({placeholders})'
        )

        if sample_rows:
            self.connection.executemany(insert_sql, sample_rows)

        with csv_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            next(reader, None)
            skipped = 0
            batch: list[list[object | None]] = []
            for row in reader:
                normalized = self._normalize_csv_row(row, len(columns))
                if skipped < len(sample_rows):
                    skipped += 1
                    continue
                batch.append(normalized)
                if len(batch) >= 1000:
                    self.connection.executemany(insert_sql, batch)
                    batch.clear()
            if batch:
                self.connection.executemany(insert_sql, batch)

    def _normalize_csv_row(self, row: list[str], width: int) -> list[object | None]:
        """
        Normalize one CSV row for SQLite insertion and type inference.
        """
        padded = list(row[:width]) + [""] * max(0, width - len(row))
        normalized: list[object | None] = []
        for value in padded:
            stripped = value.strip()
            if stripped in {"", *DEFAULT_NULL_VALUES}:
                normalized.append(None)
                continue
            if self._is_int_like(stripped):
                normalized.append(int(stripped))
                continue
            if self._is_float_like(stripped):
                normalized.append(float(stripped))
                continue
            normalized.append(stripped)
        return normalized

    @staticmethod
    def _is_int_like(value: str) -> bool:
        if value.startswith(("+", "-")):
            value = value[1:]
        return value.isdigit()

    @staticmethod
    def _is_float_like(value: str) -> bool:
        try:
            float(value)
        except ValueError:
            return False
        return True

    @staticmethod
    def _sqlite_type_for_value(value: object | None) -> str:
        if value is None:
            return "INTEGER"
        if isinstance(value, bool):
            return "INTEGER"
        if isinstance(value, int):
            return "INTEGER"
        if isinstance(value, float):
            return "REAL"
        return "TEXT"

    @staticmethod
    def _merge_sqlite_types(current: str, observed: str) -> str:
        if current == observed:
            return current
        if "TEXT" in {current, observed}:
            return "TEXT"
        if "REAL" in {current, observed}:
            return "REAL"
        return "INTEGER"

    @staticmethod
    def _quote_identifier(identifier: str) -> str:
        escaped = identifier.replace('"', '""')
        return f'"{escaped}"'

    def _table_name_from_path(self, csv_path: Path, used_names: set[str]) -> str:
        """
        Build a stable table name for a CSV path, including nested paths when needed.

        Nested directory components are preserved to avoid collisions between
        files that share the same basename in different folders.
        """
        relative = csv_path.relative_to(self.lake_dir)
        parts = list(relative.parts)
        stemmed = "__".join(Path(part).stem for part in parts[:-1] + [relative.name])
        candidate = stemmed
        counter = 2
        while candidate in used_names:
            candidate = f"{stemmed}_{counter}"
            counter += 1
        used_names.add(candidate)
        return candidate

    def query(self, sql: str) -> list[dict[str, Any]]:
        """
        Execute a SQL query across all registered tables.

        Tables are referenced by their name (filename without extension).

        Supports SELECT, JOIN, WHERE, GROUP BY, ORDER BY, aggregations,
        CTEs, and subqueries. This method is needed because later stages
        produce SQL-shaped plans; keeping execution here gives the project
        one consistent bridge between planning and actual table access.

        Args:
            sql: SQL query referencing one or more table names.

        Returns:
            A list of row dicts keyed by output column name.
        """
        if self.connection is None:
            raise RuntimeError("SQLite connection is not initialized.")
        cursor = self.connection.execute(sql)
        rows = cursor.fetchall()
        if not cursor.description:
            return []
        columns = [item[0] for item in cursor.description]
        return [
            {column: row[idx] for idx, column in enumerate(columns)}
            for row in rows
        ]

    def get_sample(self, table_name: str, n: int = 1000) -> list[dict[str, Any]]:
        """
        Return a small sample from a table.

        Preferred over accessing `lake.tables` directly. Sampling is needed
        by profiling, summary generation, and join discovery because those
        stages need representative values without materializing full tables.

        Args:
            table_name: Name of the table to sample.
            n: Maximum number of rows to return. Defaults to 1000.

        Returns:
            A list of row dicts with at most n rows.

        Raises:
            KeyError: If table_name is not found in self.tables.
        """
        if table_name not in self.tables:
            raise KeyError(
                f"Table '{table_name}' not found. "
                f"Available: {list(self.tables.keys())}"
            )
        return self.query(
            f"SELECT * FROM {self._quote_identifier(table_name)} LIMIT {int(n)}"
        )

    def get_table_columns(self, table_name: str) -> list[str]:
        """
        Return the declared column names for a table.
        """
        if table_name not in self.tables:
            raise KeyError(
                f"Table '{table_name}' not found. "
                f"Available: {list(self.tables.keys())}"
            )
        cursor = self.connection.execute(
            f"PRAGMA table_info({self._quote_identifier(table_name)})"
        )
        return [row[1] for row in cursor.fetchall()]

    def get_column_dtype(self, table_name: str, col: str) -> str:
        """
        Return the SQLite dtype for one table column.
        """
        cursor = self.connection.execute(
            f"PRAGMA table_info({self._quote_identifier(table_name)})"
        )
        for row in cursor.fetchall():
            if row[1] == col:
                dtype = str(row[2]).lower()
                if "int" in dtype:
                    return "integer"
                if "real" in dtype or "float" in dtype or "double" in dtype:
                    return "float"
                return "string"
        raise ValueError(
            f"Column '{col}' not found in '{table_name}'. "
            f"Available: {self.get_table_columns(table_name)}"
        )

    def get_column_values(self, table_name: str, col: str) -> set:
        """
        Return the unique string-cast values of a column from a sample.

        Used by DataProfiler.jaccard_similarity() to compute value-set
        overlap between columns across tables. This helper is needed so
        join detection compares columns in a uniform, null-safe way even
        when source files use different physical dtypes.

        Args:
            table_name: Name of the table containing the column.
            col: Name of the column.

        Returns:
            A set of unique, null-dropped, string-cast values drawn from
            the sampled table data.

        Raises:
            KeyError: If table_name is not found.
            ValueError: If col is not found in the table.
        """
        if table_name not in self.tables:
            raise KeyError(
                f"Table '{table_name}' not found. "
                f"Available: {list(self.tables.keys())}"
            )
        if col not in self.get_table_columns(table_name):
            raise ValueError(
                f"Column '{col}' not found in '{table_name}'. "
                f"Available: {self.get_table_columns(table_name)}"
            )
        sample = self.query(
            " ".join(
                [
                    f"SELECT DISTINCT {self._quote_identifier(col)} AS {self._quote_identifier(col)}",
                    f"FROM {self._quote_identifier(table_name)}",
                    f"WHERE {self._quote_identifier(col)} IS NOT NULL",
                    "LIMIT 1000",
                ]
            )
        )
        values = [str(row[col]).strip() for row in sample if row.get(col) is not None]
        return {value for value in values if value}

    def __del__(self) -> None:
        if self.connection is not None:
            self.connection.close()

    def __repr__(self) -> str:
        return f"DataLake(tables={list(self.tables.keys())})"
