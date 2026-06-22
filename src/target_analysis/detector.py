"""
Stage 1 of target analysis: candidate detection.

Deliberately decoupled from health/balance. A column's candidacy score is a
function of its NAME and its SHAPE (cardinality), never its class balance.
This is the direct fix for V3 silently skipping sparse targets: a 0.1%
minority binary column scores identically to a 50/50 binary column here.
Balance is evaluated separately and later, in health_audit.py.
"""
import re
from ..ingestion.data_context import DataContextObject

TARGET_KEYWORDS = {
    "target", "label", "y", "outcome", "churn", "fraud", "default", "class",
    "response", "flag", "status", "result", "diagnosis", "survived",
    "converted", "purchase", "click",
}


def _tokens(name: str) -> set[str]:
    """Split on non-alphanumeric boundaries so single-letter/short keywords
    (e.g. 'y') match whole tokens only, not substrings ('country' must not
    match 'y')."""
    return set(re.split(r"[^a-z0-9]+", name.lower())) - {""}


def detect_target_candidates(dco: DataContextObject) -> list[dict]:
    """
    Score every column as a potential target. Returns a list of
    {column, score, reasons} sorted descending by score. Score is built
    only from name keywords, position, and cardinality SHAPE - never from
    class_counts/balance, which isn't even passed in here. That's what
    keeps a 0.1%-positive binary column scoring the same as a 50/50 one;
    balance is audited separately and later, in health_audit.py.
    """
    col_names = list(dco.columns.keys())
    n = len(col_names)
    candidates = []

    for idx, (name, prof) in enumerate(dco.columns.items()):
        score = 0.0
        reasons = []
        lname = name.lower()
        tokens = _tokens(name)

        if (tokens & TARGET_KEYWORDS) or lname.startswith("is_"):
            score += 0.5
            reasons.append("name matches a common target keyword")

        if idx == n - 1 and n > 1:
            score += 0.15
            reasons.append("last column in dataset")

        distinct = prof.distinct_count
        if distinct is not None:
            if distinct == 2:
                score += 0.3
                reasons.append("binary column (shape-based; balance is not a factor here)")
            elif 2 < distinct <= 20:
                score += 0.15
                reasons.append("low-cardinality categorical column")

        if prof.null_pct >= 0.5:
            score -= 0.2
            reasons.append("high null rate reduces target plausibility")

        if prof.dtype.upper() in {"VARCHAR", "TEXT", "BLOB"} and (distinct or 0) > 1000:
            score -= 0.15
            reasons.append("high-cardinality text column, unlikely target")

        score = max(0.0, min(1.0, score))
        if score > 0:
            candidates.append({"column": name, "score": round(score, 3), "reasons": reasons})

    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates
