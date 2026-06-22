"""
Model selection suggestions. Two-step, same separation-of-concerns as
target_analysis: infer_problem_type() is a pure function of the target
column's shape; suggest_models() then branches on problem type AND the
target health audit's sparsity flags (from health_audit.py) to decide
whether to lead with anomaly-detection framing or a standard supervised
shortlist. Each suggested model ships an Optuna search space TEMPLATE, not
a single fixed hyperparameter set - the actual tuning run is the user's/
agent's job, this just scopes it sensibly for the problem at hand.
"""
from ..ingestion.data_context import DataContextObject
from ..config import CONFIG
from ..tools.registry import process, ProcessCost

NUMERIC_TYPES = {"BIGINT", "DOUBLE", "INTEGER", "FLOAT", "DECIMAL", "HUGEINT", "SMALLINT", "TINYINT", "REAL"}


def infer_problem_type(dco: DataContextObject) -> str:
    """Returns 'binary_classification' | 'multiclass_classification' |
    'regression' | 'unknown' based only on the confirmed/candidate target
    column's dtype and cardinality - no class_counts needed here."""
    col = dco.target.column
    if not col or col not in dco.columns:
        return "unknown"
    prof = dco.columns[col]
    distinct = prof.distinct_count or 0
    if distinct == 2:
        return "binary_classification"
    if prof.dtype.upper() not in NUMERIC_TYPES and 2 < distinct <= 20:
        return "multiclass_classification"
    if prof.dtype.upper() in NUMERIC_TYPES and distinct > 20:
        return "regression"
    if 2 < distinct <= 20:
        return "multiclass_classification"
    return "unknown"


def _sparsity_tier(dco: DataContextObject) -> str:
    """'extreme' | 'moderate' | 'none' based on health_audit.py's flags,
    if the audit has been run. 'none' (not 'unknown') if no audit is
    present yet, since absence of a sparsity flag is the safe default to
    suggest standard models on."""
    health = dco.target.health or {}
    codes = [f["code"] for f in health.get("flags", [])]
    if "sparse_target" not in codes:
        return "none"
    ratio = health.get("minority_ratio")
    extreme_cutoff = CONFIG.feature_model.extreme_sparse_threshold_for_anomaly
    return "extreme" if (ratio is not None and ratio < extreme_cutoff) else "moderate"


@process(
    name="model_selection_suggestions",
    description="Ranked model shortlist with Optuna search space templates, sparsity-aware (requires a confirmed target).",
    cost=ProcessCost.FREE,
    requires_target=True,
    category="model_suggestion",
)
def suggest_models(dco: DataContextObject, **_) -> dict:
    """
    Requires dco.target.column to be set (run target_analysis first - see
    detector.py/health_audit.py). Returns {"type": "table", "data": [...]}
    where each entry is {model, rationale, optuna_search_space}, ranked
    with the recommended-first model at index 0. Returns a clear
    'not_ready' note instead of raising if no target is set yet.
    """
    if not dco.target.column:
        return {"type": "table", "data": [], "note": "no target set - run target detection/confirmation first"}

    problem_type = infer_problem_type(dco)
    n_rows = dco.n_rows or 0
    small_data = n_rows < CONFIG.feature_model.small_dataset_rows
    tier = _sparsity_tier(dco) if "classification" in problem_type else "none"

    shortlist = []

    if problem_type in {"binary_classification", "multiclass_classification"}:
        if tier == "extreme":
            shortlist.append({
                "model": "IsolationForest",
                "rationale": "minority class <1% of data - frame as anomaly detection rather than "
                             "balanced classification",
                "optuna_search_space": {
                    "n_estimators": {"type": "int", "low": 50, "high": 300},
                    "contamination": {"type": "float", "low": 0.001, "high": 0.05, "log": True},
                    "max_features": {"type": "float", "low": 0.5, "high": 1.0},
                },
            })
        if tier in {"extreme", "moderate"}:
            shortlist.append({
                "model": "LightGBM" if not small_data else "LogisticRegression",
                "rationale": f"sparsity tier={tier} - use class_weight/scale_pos_weight, evaluate "
                             "with PR-AUC/F1 not accuracy",
                "optuna_search_space": {
                    "max_depth": {"type": "int", "low": 3, "high": 12},
                    "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
                    "min_child_weight": {"type": "int", "low": 1, "high": 20},
                    "scale_pos_weight": {"type": "float", "low": 1.0, "high": float(1 / max(
                        dco.target.health.get("minority_ratio", 0.05), 0.001))},
                } if not small_data else {
                    "C": {"type": "float", "low": 1e-3, "high": 1e2, "log": True},
                    "class_weight": {"type": "categorical", "choices": ["balanced"]},
                },
            })
        else:
            shortlist += [
                {
                    "model": "LightGBM",
                    "rationale": "balanced target, default strong baseline for tabular data",
                    "optuna_search_space": {
                        "num_leaves": {"type": "int", "low": 15, "high": 255},
                        "max_depth": {"type": "int", "low": 3, "high": 12},
                        "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
                        "n_estimators": {"type": "int", "low": 100, "high": 1000},
                    },
                },
                {
                    "model": "RandomForest",
                    "rationale": "robust baseline, fewer hyperparameters to tune than boosting",
                    "optuna_search_space": {
                        "n_estimators": {"type": "int", "low": 100, "high": 600},
                        "max_depth": {"type": "int", "low": 4, "high": 30},
                        "min_samples_leaf": {"type": "int", "low": 1, "high": 20},
                    },
                },
                {
                    "model": "LogisticRegression",
                    "rationale": "linear baseline - cheap sanity check against the tree models above",
                    "optuna_search_space": {"C": {"type": "float", "low": 1e-3, "high": 1e2, "log": True}},
                },
            ]

    elif problem_type == "regression":
        shortlist = [
            {
                "model": "LightGBM",
                "rationale": "default strong baseline for tabular regression",
                "optuna_search_space": {
                    "num_leaves": {"type": "int", "low": 15, "high": 255},
                    "max_depth": {"type": "int", "low": 3, "high": 12},
                    "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
                    "n_estimators": {"type": "int", "low": 100, "high": 1000},
                },
            },
            {
                "model": "RandomForestRegressor",
                "rationale": "robust baseline, fewer hyperparameters than boosting",
                "optuna_search_space": {
                    "n_estimators": {"type": "int", "low": 100, "high": 600},
                    "max_depth": {"type": "int", "low": 4, "high": 30},
                },
            },
            {
                "model": "Ridge",
                "rationale": "linear baseline - cheap sanity check against the tree models above",
                "optuna_search_space": {"alpha": {"type": "float", "low": 1e-3, "high": 1e2, "log": True}},
            },
        ]
    else:
        return {"type": "table", "data": [], "note": f"could not infer a usable problem type for target '{dco.target.column}'"}

    return {"type": "table", "data": shortlist, "problem_type": problem_type, "sparsity_tier": tier}
