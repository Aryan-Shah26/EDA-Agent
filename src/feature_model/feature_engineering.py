"""
Rule-based feature engineering suggestions. Deterministic on purpose (not
LLM-narrated) - a fixed rule given the same skew/cardinality/null% always
produces the same suggestion, which is what makes this stress-testable
(tests/test_feature_suggestions.py asserts exact outcomes, not "looks
reasonable"). The chat agent can narrate these in natural language; it
should not be inventing the underlying numbers.
"""
from ..ingestion.data_context import DataContextObject
from ..config import CONFIG
from ..tools.registry import process, ProcessCost

NUMERIC_TYPES = {"BIGINT", "DOUBLE", "INTEGER", "FLOAT", "DECIMAL", "HUGEINT", "SMALLINT", "TINYINT", "REAL"}
DATETIME_TYPES = {"DATE", "TIMESTAMP", "TIMESTAMP WITH TIME ZONE", "TIME"}


@process(
    name="feature_engineering_suggestions",
    description="Rule-based feature engineering suggestions (skew transforms, encoding strategy, datetime decomposition, missingness handling).",
    cost=ProcessCost.FREE,
    category="model_suggestion",
)
def suggest_feature_engineering(dco: DataContextObject, **_) -> dict:
    """
    Walks every column once and emits at most one suggestion category per
    column. Sorted by precedence: Missingness -> Identifiers -> Datetime -> Skew -> Encoding.
    """
    cfg = CONFIG.feature_model
    suggestions = []

    for name, prof in dco.columns.items():
        if name == dco.target.column:
            continue

        dtype = prof.dtype.upper()
        distinct = prof.distinct_count or 0
        distinct_ratio = distinct / dco.n_rows if dco.n_rows else 0

        # 1. PRIORITY: Missing Data
        # (Moved to the top so columns like 'Cabin' get flagged for nulls first)
        if prof.null_pct > cfg.high_null_threshold:
            suggestions.append({
                "column": name, "kind": "missingness",
                "detail": f"{prof.null_pct:.1%} null",
                "suggestion": "median/mode imputation with a missingness indicator column, or "
                              "consider dropping if this rate persists in the full data",
            })
            continue

        # 2. IDENTIFIER CHECK (New Rule)
        # If a text column is almost entirely unique, it's an ID, Name, or raw text.
        if dtype not in NUMERIC_TYPES and dtype not in DATETIME_TYPES:
            if distinct_ratio > 0.85 and dco.n_rows > 50:
                suggestions.append({
                    "column": name, "kind": "drop_identifier",
                    "detail": f"{distinct_ratio:.0%} unique",
                    "suggestion": "Drop column. High uniqueness indicates a primary key, name, or raw string that will cause overfitting.",
                })
                continue

        # 3. Datetime Extraction
        if dtype in DATETIME_TYPES:
            suggestions.append({
                "column": name, "kind": "datetime_decomposition",
                "detail": f"{name} is {prof.dtype}",
                "suggestion": "extract hour/day/month components; consider cyclical "
                              "(sin/cos) encoding for hour/day instead of raw integers",
            })
            continue

        # 4. Continuous Numeric Skew
        is_continuous_numeric = dtype in NUMERIC_TYPES and distinct > 20
        if is_continuous_numeric and prof.skew is not None and abs(prof.skew) > cfg.skew_threshold:
            transform = "log1p" if prof.skew > 0 else "square or Box-Cox (left-skewed)"
            suggestions.append({
                "column": name, "kind": "skew_transform",
                "detail": f"skew={prof.skew:.2f}",
                "suggestion": f"apply {transform} transform (Yeo-Johnson if column can be <=0)",
            })
            continue

        # 5. Standard High Cardinality Categorical
        if dtype not in NUMERIC_TYPES and distinct > cfg.high_cardinality_threshold:
            suggestions.append({
                "column": name, "kind": "high_cardinality_encoding",
                "detail": f"distinct≈{distinct}",
                "suggestion": "use target or frequency encoding, not one-hot (would create "
                              f"{distinct}+ sparse columns)",
            })
            continue

    return {"type": "table", "data": suggestions}