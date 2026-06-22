"""
Profiling engine. DuckDB does all the work out-of-core - we never pull the
full file into pandas. Only a reservoir sample (small, fixed size) ever
becomes an in-memory DataFrame.
"""
import os
import duckdb

from .data_context import DataContextObject, ColumnProfile
from ..config import CONFIG

NUMERIC_TYPES = {
    "BIGINT", "DOUBLE", "INTEGER", "FLOAT", "HUGEINT", "SMALLINT",
    "TINYINT", "DECIMAL", "UBIGINT", "UINTEGER", "USMALLINT", "UTINYINT", "REAL",
}


def _read_expr(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".csv", ".tsv", ".txt"):
        return f"read_csv_auto('{path}')"
    if ext == ".parquet":
        return f"read_parquet('{path}')"
    if ext == ".json":
        return f"read_json_auto('{path}')"
    raise ValueError(f"Unsupported file type: {ext}")


def profile_dataset(
    path: str,
    sample_size: int = None,
    sample_dir: str = None,
    con: "duckdb.DuckDBPyConnection | None" = None,
) -> DataContextObject:
    """
    Build a DataContextObject for `path` without ever materializing the full
    file in Python memory. DuckDB computes per-column null/distinct/min/max/
    mean/std stats as out-of-core aggregations; only a small reservoir
    sample (sample_size rows) is ever pulled into a pandas-readable form,
    and that happens via a COPY straight to parquet, not via .df() on the
    full table. sample_size/sample_dir default to CONFIG.ingestion if not
    passed explicitly.
    """
    sample_size = CONFIG.ingestion.sample_size if sample_size is None else sample_size
    sample_dir = CONFIG.ingestion.sample_dir if sample_dir is None else sample_dir
    own_con = con is None
    con = con or duckdb.connect()
    try:
        rel = _read_expr(path)

        schema_df = con.sql(f"DESCRIBE SELECT * FROM {rel}").df()
        n_rows = con.sql(f"SELECT COUNT(*) FROM {rel}").fetchone()[0]

        columns: dict[str, ColumnProfile] = {}
        for _, row in schema_df.iterrows():
            col, dtype = row["column_name"], row["column_type"]
            is_numeric = dtype.upper() in NUMERIC_TYPES
            qcol = f'"{col}"'

            agg = [
                f"COUNT(*) FILTER (WHERE {qcol} IS NULL) AS null_count",
                f"approx_count_distinct({qcol}) AS distinct_count",
            ]
            if is_numeric:
                agg += [
                    f"MIN({qcol}) AS min_val",
                    f"MAX({qcol}) AS max_val",
                    f"AVG({qcol})::DOUBLE AS mean_val",
                    f"STDDEV({qcol})::DOUBLE AS std_val",
                    f"SKEWNESS({qcol})::DOUBLE AS skew_val",
                ]
            stats = con.sql(f"SELECT {', '.join(agg)} FROM {rel}").fetchone()
            null_count = stats[0]
            distinct_count = stats[1]

            prof = ColumnProfile(
                name=col,
                dtype=dtype,
                null_count=null_count,
                null_pct=(null_count / n_rows) if n_rows else 0.0,
                distinct_count=distinct_count,
                distinct_is_approx=True,
            )
            if is_numeric:
                prof.min_val, prof.max_val, prof.mean, prof.std, prof.skew = stats[2], stats[3], stats[4], stats[5], stats[6]
            columns[col] = prof

        os.makedirs(sample_dir, exist_ok=True)
        sample_path = os.path.join(sample_dir, f"{os.path.basename(path)}.sample.parquet")
        effective_sample = min(sample_size, n_rows) if n_rows else sample_size
        if effective_sample > 0:
            con.sql(
                f"COPY (SELECT * FROM {rel} USING SAMPLE {effective_sample} ROWS (reservoir)) "
                f"TO '{sample_path}' (FORMAT PARQUET)"
            )
        else:
            sample_path = None

        dco = DataContextObject(
            source_name=path,
            n_rows=n_rows,
            n_cols=len(columns),
            columns=columns,
            reservoir_sample_path=sample_path,
        )

        if n_rows == 0:
            dco.add_flag("empty_dataset", "critical", "Dataset has zero rows.")
        for col, prof in columns.items():
            if prof.null_pct >= 0.99:
                dco.add_flag("near_empty_column", "warning", f"{prof.null_pct:.1%} null", column=col)
            if prof.distinct_count == 1:
                dco.add_flag("constant_column", "info", "Only one distinct value", column=col)

        return dco
    finally:
        if own_con:
            con.close()


def query_full_data(path: str, sql_select_clause: str, con: "duckdb.DuckDBPyConnection | None" = None):
    """
    Push an aggregation/query down to DuckDB against the FULL file - never
    pulls the whole file into Python. sql_select_clause should reference
    the table as `t`, e.g. "region, SUM(revenue) FROM t GROUP BY region".
    Returns a pandas DataFrame (expected to be small - aggregated/grouped results).
    """
    own_con = con is None
    con = con or duckdb.connect()
    try:
        rel = _read_expr(path)
        query = f"SELECT {sql_select_clause.replace('FROM t', f'FROM {rel} AS t')}"
        return con.sql(query).df()
    finally:
        if own_con:
            con.close()


def get_class_counts(path: str, column: str, con: "duckdb.DuckDBPyConnection | None" = None) -> dict:
    """Full, exact value_counts on one column via DuckDB - used by the target health audit."""
    own_con = con is None
    con = con or duckdb.connect()
    try:
        rel = _read_expr(path)
        df = con.sql(f'SELECT "{column}" AS v, COUNT(*) AS c FROM {rel} WHERE "{column}" IS NOT NULL GROUP BY "{column}"').df()
        return dict(zip(df["v"], df["c"]))
    finally:
        if own_con:
            con.close()
