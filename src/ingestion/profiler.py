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
    Build a DataContextObject using a single-pass mega-query.
    Scans the dataset exactly ONCE regardless of column count.
    """
    sample_size = CONFIG.ingestion.sample_size if sample_size is None else sample_size
    sample_dir = CONFIG.ingestion.sample_dir if sample_dir is None else sample_dir
    own_con = con is None
    con = con or duckdb.connect()
    
    try:
        rel = _read_expr(path)

        schema_df = con.sql(f"DESCRIBE SELECT * FROM {rel}").df()
        n_rows = con.sql(f"SELECT COUNT(*) FROM {rel}").fetchone()[0]

        # 1. Build the Mega-Query
        select_exprs = []
        col_meta = []
        
        for _, row in schema_df.iterrows():
            col, dtype = row["column_name"], row["column_type"]
            is_numeric = dtype.upper() in NUMERIC_TYPES
            qcol = f'"{col}"'
            
            col_meta.append((col, dtype, is_numeric))
            
            # Base stats for all columns
            select_exprs.append(f"COUNT(*) FILTER (WHERE {qcol} IS NULL)")
            select_exprs.append(f"approx_count_distinct({qcol})")
            
            # Additional stats for numeric columns
            if is_numeric:
                select_exprs.extend([
                    f"MIN({qcol})",
                    f"MAX({qcol})",
                    f"AVG({qcol})::DOUBLE",
                    f"STDDEV({qcol})::DOUBLE",
                    f"SKEWNESS({qcol})::DOUBLE"
                ])

        # 2. Execute the single query
        if select_exprs:
            mega_query = f"SELECT {', '.join(select_exprs)} FROM {rel}"
            row_data = con.sql(mega_query).fetchone()
        else:
            row_data = []

        # 3. Unpack the results into the DataContextObject
        columns: dict[str, ColumnProfile] = {}
        idx = 0
        
        for col, dtype, is_numeric in col_meta:
            null_count = row_data[idx]; idx += 1
            distinct_count = row_data[idx]; idx += 1

            prof = ColumnProfile(
                name=col,
                dtype=dtype,
                null_count=null_count,
                null_pct=(null_count / n_rows) if n_rows else 0.0,
                distinct_count=distinct_count,
                distinct_is_approx=True,
            )
            
            if is_numeric:
                prof.min_val = row_data[idx]; idx += 1
                prof.max_val = row_data[idx]; idx += 1
                prof.mean = row_data[idx]; idx += 1
                prof.std = row_data[idx]; idx += 1
                prof.skew = row_data[idx]; idx += 1

            columns[col] = prof

        # 4. Generate the Reservoir Sample
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
