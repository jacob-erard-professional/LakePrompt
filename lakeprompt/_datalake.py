from dataclasses import dataclass, field
from pathlib import Path
import polars as pl


DEFAULT_INFER_SCHEMA_LENGTH = 10_000
DEFAULT_NULL_VALUES = ["-"]


@dataclass
class DataLake:
    """
    Central representation of a data lake for use across LakePrompt.

    Scans a directory of CSV files and registers each as a named table
    using Polars lazy frames. This class exists so every pipeline stage
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
            A `DataLake` populated with one lazy Polars frame per CSV.

        Raises:
            FileNotFoundError: If lake_dir does not exist.
            ValueError: If no CSV files are found.
        """
        lake = cls(lake_dir=Path(lake_dir))
        lake._load_tables()
        return lake

    def _load_tables(self) -> None:
        """
        Load all CSV files in `lake_dir` as named lazy tables.

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

        used_names: set[str] = set()
        for path in csv_files:
            table_name = self._table_name_from_path(path, used_names)
            lazy_frame = pl.scan_csv(
                str(path),
                infer_schema_length=DEFAULT_INFER_SCHEMA_LENGTH,
                null_values=DEFAULT_NULL_VALUES,
                ignore_errors=True,
            )
            schema = lazy_frame.collect_schema()
            string_columns = [
                name
                for name, dtype in schema.items()
                if str(dtype).lower() == "string"
            ]
            if string_columns:
                lazy_frame = lazy_frame.with_columns(
                    [pl.col(name).str.strip_chars().alias(name) for name in string_columns]
                )
            self.tables[table_name] = lazy_frame

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

    def query(self, sql: str) -> pl.DataFrame:
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
            An eager Polars DataFrame containing the query result.
        """
        ctx = pl.SQLContext()
        for name, lf in self.tables.items():
            ctx.register(name, lf)
        return ctx.execute(sql).collect()

    def get_sample(self, table_name: str, n: int = 1000) -> pl.DataFrame:
        """
        Return a small sample from a table.

        Preferred over accessing `lake.tables` directly. Sampling is needed
        by profiling, summary generation, and join discovery because those
        stages need representative values without materializing full tables.

        Args:
            table_name: Name of the table to sample.
            n: Maximum number of rows to return. Defaults to 1000.

        Returns:
            An eager Polars DataFrame with at most n rows.

        Raises:
            KeyError: If table_name is not found in self.tables.
        """
        if table_name not in self.tables:
            raise KeyError(
                f"Table '{table_name}' not found. "
                f"Available: {list(self.tables.keys())}"
            )
        return self.tables[table_name].collect().head(n)

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
        sample = self.get_sample(table_name)
        if col not in sample.columns:
            raise ValueError(
                f"Column '{col}' not found in '{table_name}'. "
                f"Available: {sample.columns}"
            )
        values = (
            sample[col]
            .drop_nulls()
            .cast(pl.Utf8)
            .str.strip_chars()
            .unique()
            .to_list()
        )
        return {value for value in values if value}

    def __repr__(self) -> str:
        return f"DataLake(tables={list(self.tables.keys())})"
