from dataclasses import dataclass, field
from pathlib import Path
import polars as pl

@dataclass
class DataLake:
    """
    Central representation of a data lake for use across LakePrompt.

    Scans a directory of CSV files and registers each as a named table.
    Supports two backends — polars for single-machine workloads (default),
    and spark for distributed environments.

    Args:
        lake_dir: Path to the directory containing CSV files.
        backend: Query engine to use. Either 'polars' (default) or 'spark'.
            Use polars unless you are working with very large datasets on a
            distributed cluster. Spark requires pyspark to be installed.

    Example:
        >>> lake = DataLake.load("./data/csvs")
        >>> lake = DataLake.load("./data/csvs", backend="spark")
        >>> lake.query("SELECT * FROM customers WHERE city = 'Denver'")
    """

    lake_dir: Path
    backend: str = "polars"
    tables: dict[str, object] = field(default_factory=dict)    
    cards: list = field(default_factory=list)
    join_graph: dict = field(default_factory=dict)
    _spark: object = field(default=None, repr=False)

    @classmethod
    def load(cls, lake_dir: str, backend: str = "polars") -> "DataLake":
        """
        Scan a directory of CSV files and return an initialised DataLake.

        Each CSV becomes a named table using the filename without extension
        (e.g. 'customers.csv' → 'customers'). Tables are loaded lazily —
        no data is read into memory until a query or sample is requested.

        Args:
            lake_dir: Path to the directory containing CSV files.
            backend: Either 'polars' (default) or 'spark'.

        Raises:
            FileNotFoundError: If lake_dir does not exist.
            ValueError: If no CSV files are found, or if an invalid
                backend is specified.
            ImportError: If backend='spark' and pyspark is not installed.
        """
        if backend not in ("polars", "spark"):
            raise ValueError(
                f"Invalid backend '{backend}'. Choose 'polars' or 'spark'."
            )
        if backend == "spark":
            try:
                from pyspark.sql import SparkSession, DataFrame as SparkDataFrame
            except ImportError:
                raise ImportError(
                    "PySpark is not installed. Run `pip install pyspark` "
                    "or use the default polars backend."
                )
                
        lake = cls(lake_dir=Path(lake_dir), backend=backend)
        lake._load_tables()
        return lake

    def _load_tables(self) -> None:
        """Load all CSV files in lake_dir as named tables."""
        if not self.lake_dir.exists():
            raise FileNotFoundError(
                f"Lake directory not found: {self.lake_dir}"
            )
        
        csv_files = list(self.lake_dir.glob("*.csv"))
        if not csv_files:
            raise ValueError(f"No CSV files found in: {self.lake_dir}")

        if self.backend == "spark":
            from pyspark.sql import SparkSession
            self._spark = SparkSession.builder \
                .appName("LakePrompt") \
                .getOrCreate()
            for path in csv_files:
                name = path.stem
                df = self._spark.read \
                    .option("header", "true") \
                    .option("inferSchema", "true") \
                    .csv(str(path))
                df.createOrReplaceTempView(name)
                self.tables[name] = df
        else:
            for path in csv_files:
                self.tables[path.stem] = pl.scan_csv(str(path))

    def query(self, sql: str) -> pl.DataFrame:
        """
        Execute a SQL query across all registered tables.

        Tables are referenced by their name (filename without extension).

        For the polars backend, supports SELECT, JOIN, WHERE, GROUP BY,
        ORDER BY, aggregations, CTEs, and subqueries. Window functions
        are not supported — use the Polars expression API directly for those.

        For the spark backend, supports full Spark SQL which is close to
        ANSI SQL and includes window functions.

        Args:
            sql: SQL query referencing one or more table names.

        Returns:
            An eager Polars DataFrame (polars backend) or a Spark DataFrame
            (spark backend).
        """
        if self.backend == "spark":
            return self._spark.sql(sql)
        ctx = pl.SQLContext()
        for name, lf in self.tables.items():
            ctx.register(name, lf)
        return ctx.execute(sql).collect()

    def get_sample(self, table_name: str, n: int = 10) -> pl.DataFrame:
        """
        Return a small sample from a table.

        Preferred over accessing lake.tables directly — always returns a
        Polars DataFrame regardless of backend, so module code stays
        backend-agnostic.

        Args:
            table_name: Name of the table to sample.
            n: Number of rows to return. Defaults to 100.

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
        if self.backend == "spark":
            return pl.from_pandas(
                self.tables[table_name].limit(n).toPandas()
            )
        return self.tables[table_name].collect().head(n)

    def get_column_values(self, table_name: str, col: str) -> set:
        """
        Return the unique string-cast values of a column from a sample.

        Used by DataProfiler.jaccard_similarity() to compute value-set
        overlap between columns across tables.

        Args:
            table_name: Name of the table containing the column.
            col: Name of the column.

        Returns:
            A set of unique, null-dropped, string-cast values.

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
        return set(sample[col].drop_nulls().cast(pl.Utf8).unique().to_list())

    def __repr__(self) -> str:
        return (
            f"DataLake(tables={list(self.tables.keys())}, "
            f"backend={self.backend})"
        )