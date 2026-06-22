"""
Stage 2 of target analysis: health audit.

Runs unconditionally on whichever column is selected (user-confirmed or
top candidate) - this is what actually catches sparsity, and it runs
regardless of how the column was chosen. class_counts must come from an
exact full-data GROUP BY (see profiler.get_class_counts), not the
reservoir sample - a 0.1% minority class can be entirely absent from a
10k-row sample of a multi-million row file.
"""
import math
from ..ingestion.data_context import DataContextObject
from ..config import CONFIG


def audit_target_health(dco: DataContextObject, column: str, class_counts: dict | None = None) -> dict:
    """
    Unconditional health check on `column`, regardless of how it was chosen
    (user-confirmed or top candidate). This is deliberately separate from
    detector.py's scoring - that function ignores balance on purpose, so
    this is the only place sparsity actually gets caught.

    class_counts must come from an EXACT full-data GROUP BY (see
    profiler.get_class_counts), not the reservoir sample - a 0.1% minority
    class can be entirely absent from a 10k-row sample of a multi-million
    row file, which would make this audit miss exactly what it's for.
    """
    sparse_threshold = CONFIG.target_analysis.sparse_threshold
    extreme_sparse_threshold = CONFIG.target_analysis.extreme_sparse_threshold
    null_warning_threshold = CONFIG.target_analysis.null_warning_threshold

    prof = dco.columns[column]
    result = {
        "column": column,
        "null_pct": prof.null_pct,
        "distinct_count": prof.distinct_count,
        "flags": [],
        "recommendations": [],
    }

    if prof.null_pct >= null_warning_threshold:
        result["flags"].append({
            "code": "target_nulls",
            "severity": "warning",
            "detail": f"{prof.null_pct:.1%} of target values are null",
        })

    if class_counts is not None:
        total = sum(class_counts.values())
        if total == 0:
            result["flags"].append({
                "code": "target_empty",
                "severity": "critical",
                "detail": "Target has no non-null values - cannot be used as-is.",
            })
            return result

        n_classes = len(class_counts)
        sorted_counts = sorted(class_counts.items(), key=lambda x: x[1])
        minority_label, minority_count = sorted_counts[0]
        minority_ratio = minority_count / total

        result["n_classes"] = n_classes
        result["minority_ratio"] = round(minority_ratio, 4)
        probs = [c / total for c in class_counts.values()]
        result["entropy"] = round(-sum(p * math.log2(p) for p in probs if p > 0), 3)

        if n_classes == 1:
            result["flags"].append({
                "code": "single_class",
                "severity": "critical",
                "detail": "Target has only one distinct value - not usable for supervised learning as-is.",
            })
        elif minority_ratio < extreme_sparse_threshold:
            result["flags"].append({
                "code": "sparse_target",
                "severity": "warning",
                "detail": (
                    f"Minority class '{minority_label}' is {minority_ratio:.2%} of data "
                    f"({minority_count}/{total}) - extreme imbalance."
                ),
            })
            result["recommendations"] += [
                "evaluate with PR-AUC / F1 / recall-at-precision, not accuracy or ROC-AUC alone",
                "use class_weight='balanced' or scale_pos_weight as a baseline",
                "use StratifiedKFold for any cross-validation",
                "consider framing as anomaly detection (Isolation Forest, one-class SVM) given <1% positive rate",
                "if resampling, try SMOTE/ADASYN only after a class-weighted baseline, on train folds only",
            ]
        elif minority_ratio < sparse_threshold:
            result["flags"].append({
                "code": "sparse_target",
                "severity": "warning",
                "detail": (
                    f"Minority class '{minority_label}' is {minority_ratio:.2%} of data "
                    f"({minority_count}/{total})."
                ),
            })
            result["recommendations"] += [
                "evaluate with PR-AUC / F1, not accuracy or ROC-AUC alone",
                "use class_weight='balanced' or scale_pos_weight",
                "use StratifiedKFold for cross-validation",
            ]

    return result
