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
    column (skew transform, encoding strategy, datetime decomposition, or
    missingness handling) - whichever applies. Each suggestion carries the
    actual statistic that triggered it, not just a generic recommendation.
    """
    cfg = CONFIG.feature_model
    suggestions = []

    for name, prof in dco.columns.items():
        if name == dco.target.column:
            continue  # the target isn't a feature to transform - audited separately in health_audit.py

        dtype = prof.dtype.upper()

        if dtype in DATETIME_TYPES:
            suggestions.append({
                "column": name, "kind": "datetime_decomposition",
                "detail": f"{name} is {prof.dtype}",
                "suggestion": "extract hour/day/day-of-week/month components; consider cyclical "
                              "(sin/cos) encoding for hour and day-of-week instead of raw integers",
            })
            continue

        # Skew is only meaningful for genuinely continuous columns - a binary/low-cardinality
        # numeric (e.g. a 0/1 flag stored as BIGINT) can show extreme "skew" purely from class
        # imbalance, and log-transforming a flag column is meaningless.
        is_continuous_numeric = dtype in NUMERIC_TYPES and (prof.distinct_count or 0) > 20
        if is_continuous_numeric and prof.skew is not None and abs(prof.skew) > cfg.skew_threshold:
            transform = "log1p" if prof.skew > 0 else "square or Box-Cox (left-skewed)"
            suggestions.append({
                "column": name, "kind": "skew_transform",
                "detail": f"skew={prof.skew:.2f}",
                "suggestion": f"apply {transform} transform (Yeo-Johnson if column can be <=0)",
            })
            continue

        if dtype not in NUMERIC_TYPES and (prof.distinct_count or 0) > cfg.high_cardinality_threshold:
            suggestions.append({
                "column": name, "kind": "high_cardinality_encoding",
                "detail": f"distinct≈{prof.distinct_count}",
                "suggestion": "use target or frequency encoding, not one-hot (would create "
                              f"{prof.distinct_count}+ sparse columns)",
            })
            continue

        if prof.null_pct > cfg.high_null_threshold:
            suggestions.append({
                "column": name, "kind": "missingness",
                "detail": f"{prof.null_pct:.1%} null",
                "suggestion": "median/mode imputation with a missingness indicator column, or "
                              "consider dropping if this rate persists in the full (non-sample) data",
            })

    return {"type": "table", "data": suggestions}
