"""Window-size + score-threshold ablation for automata feature branches.

This script intentionally does NOT modify runner.py. It imports the current
branch's ProbabilisticAutomata implementation and only passes constructor
parameters that the current branch supports.

Main workflow:
  1) Run a broad ablation with original + gaussian_noise only.
  2) Rank candidates by original F1 per dataset.
  3) Run unseen_data only for the best N candidates per dataset.
  4) Save outputs in organized branch/dataset-specific folders.

Default v5 protocol:
  - window_size values: 3..10
  - broader low-threshold coverage for both BATADAL and SKAB

Usage from project root:
  python src/experiments/window_threshold_ablation.py --dry-run
  python src/experiments/window_threshold_ablation.py --datasets batadal
  python src/experiments/window_threshold_ablation.py --datasets skab
  python src/experiments/window_threshold_ablation.py --datasets both
"""

from __future__ import annotations

import argparse
import inspect
import functools
import json
import os
import re
import subprocess
import sys
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

# Allow execution from src/experiments or project root.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.data_pipeline.sax_paa import SaxPaaTransformer
from src.experiments.evaluator import calculate_metrics
from src.models.automata_model import ProbabilisticAutomata


DEFAULT_CONFIG: dict[str, Any] = {
    # Shared automata defaults
    "window_size": 4,
    "alphabet_size": 3,
    "anomaly_threshold": 0.05,
    "skab_anomaly_threshold": 0.90,
    "batadal_anomaly_threshold": 0.05,
    "noise_level": 0.1,
    "seeds": [42, 123, 2026, 7, 999],
    "learning_rate": 0.0,

    # This ablation's search space
    "window_threshold_window_sizes": list(range(3, 11)),

    # BATADAL threshold range is intentionally wider than the old [3.5066, 3.912, 4.2, 4.5]
    # because non-4 window sizes produced all-zero F1 under that narrow high-threshold range.
    "batadal_window_threshold_score_thresholds": [
        # Low thresholds: useful when larger window sizes compress the score range.
        0.0,
        0.001,
        0.005,
        0.01,
        0.05,
        0.1054,
        0.2231,
        0.3567,
        0.5108,
        0.6931,
        0.9163,
        # Mid/high thresholds: keep the known BATADAL baseline region.
        1.2040,
        1.6094,
        2.3026,
        2.9957,
        3.5066,
        3.9120,
        4.2,
    ],
    "skab_window_threshold_score_thresholds": [
        # Add lower thresholds for larger windows, but keep the known SKAB baseline region.
        0.0,
        0.001,
        0.005,
        0.01,
        0.05,
        0.1054,
        0.2231,
        0.3567,
        0.5108,
        0.6931,
        0.9163,
    ],

    # Broad ablation scenarios. unseen_data is deliberately excluded here.
    "window_threshold_main_scenarios": ["original", "gaussian_noise"],
    # After the broad run, unseen_data is evaluated only for the best N candidates per dataset.
    "window_threshold_unseen_top_n": 3,
    # Use calculate_scores-based prediction for speed. This avoids expensive explainability logs in model.predict().
    "window_threshold_fast_predict": True,
    # Cache nearest-pattern lookup inside a single trained model run. Useful for state-similarity branches.
    "window_threshold_cache_nearest": True,

    # Base/final Markov settings used as the fixed part of the ablation
    "batadal_final_order": 3,
    "batadal_final_smoothing_alpha": 0.1,
    "batadal_final_decision_mode": "avg_negative_log",
    "batadal_final_score_window": 5,
    "batadal_final_score_threshold": 3.9120,
    "batadal_final_max_mapping_distance": None,

    "skab_final_order": 3,
    "skab_final_smoothing_alpha": 0.5,
    "skab_final_decision_mode": "avg_negative_log",
    "skab_final_score_window": 10,
    "skab_final_score_threshold": 0.2231,
    "skab_final_max_mapping_distance": None,

    # feature/state-similarity defaults
    "distance_penalty_alpha": 1.0,
    "distance_tolerance": 1,
    "batadal_final_distance_penalty_alpha": 1.0,
    "skab_final_distance_penalty_alpha": 1.0,
    "batadal_final_distance_tolerance": 1,
    "skab_final_distance_tolerance": 1,

    # feature/entropy-threshold defaults
    "entropy_threshold_enabled": True,
    "entropy_enabled": True,
    "entropy_threshold": 0.85,
    "entropy_mode": "normalized",
    "batadal_final_entropy_threshold_enabled": True,
    "skab_final_entropy_threshold_enabled": True,
    "batadal_final_entropy_enabled": True,
    "skab_final_entropy_enabled": True,
    "batadal_final_entropy_threshold": 0.85,
    "skab_final_entropy_threshold": 0.85,
    "batadal_final_entropy_mode": "normalized",
    "skab_final_entropy_mode": "normalized",

    # feature/transition-confidence defaults
    "transition_confidence_enabled": True,
    "transition_confidence_k": 10.0,
    "transition_confidence_weight": 0.5,
    "transition_confidence_mode": "additive",
    "transition_confidence_use_transition_count": True,
    "use_transition_count": True,
    "batadal_final_transition_confidence_enabled": True,
    "skab_final_transition_confidence_enabled": True,
    "batadal_final_transition_confidence_k": 10.0,
    "skab_final_transition_confidence_k": 10.0,
    "batadal_final_transition_confidence_weight": 0.5,
    "skab_final_transition_confidence_weight": 0.5,
    "batadal_final_transition_confidence_mode": "additive",
    "skab_final_transition_confidence_mode": "additive",
    "batadal_final_transition_confidence_use_transition_count": True,
    "skab_final_transition_confidence_use_transition_count": True,
    "batadal_final_use_transition_count": True,
    "skab_final_use_transition_count": True,

    # feature/dirichlet-smoothing defaults
    "dirichlet_smoothing_enabled": True,
    "dirichlet_alpha": 5.0,
    "dirichlet_prior_mode": "uniform",
    "batadal_final_dirichlet_smoothing_enabled": True,
    "batadal_final_dirichlet_alpha": 7.5,
    "batadal_final_dirichlet_prior_mode": "uniform",
    "skab_final_dirichlet_smoothing_enabled": True,
    "skab_final_dirichlet_alpha": 30.0,
    "skab_final_dirichlet_prior_mode": "uniform",

    # feature/ensemble-markov defaults
    "ensemble_enabled": False,
    "ensemble_orders": [2, 3, 4],
    "ensemble_weights": None,
    "ensemble_aggregation": "mean",
    "batadal_final_ensemble_enabled": False,
    "batadal_final_ensemble_orders": [2, 3, 4],
    "batadal_final_ensemble_weights": None,
    "batadal_final_ensemble_aggregation": "mean",
    "skab_final_ensemble_enabled": True,
    "skab_final_ensemble_orders": [2, 3, 4],
    "skab_final_ensemble_weights": None,
    "skab_final_ensemble_aggregation": "max",

    # feature/time-decay defaults
    "time_decay_enabled": False,
    "time_decay_rate": 0.99,
    "time_decay_min_weight": 0.01,
    "batadal_final_time_decay_enabled": False,
    "batadal_final_time_decay_rate": 0.99,
    "batadal_final_time_decay_min_weight": 0.01,
    "skab_final_time_decay_enabled": True,
    "skab_final_time_decay_rate": 0.999,
    "skab_final_time_decay_min_weight": 0.01,
}

# Candidate constructor parameters used across branches. Only parameters accepted
# by the current branch's ProbabilisticAutomata.__init__ are passed.
MODEL_PARAM_KEYS = [
    "smoothing",
    "order",
    "learning_rate",
    "smoothing_alpha",
    "distance_penalty_alpha",
    "distance_tolerance",
    "entropy_threshold_enabled",
    "entropy_enabled",
    "entropy_threshold",
    "entropy_mode",
    "transition_confidence_enabled",
    "transition_confidence_k",
    "transition_confidence_weight",
    "transition_confidence_mode",
    "transition_confidence_use_transition_count",
    "use_transition_count",
    "dirichlet_smoothing_enabled",
    "dirichlet_alpha",
    "dirichlet_prior_mode",
    "ensemble_enabled",
    "ensemble_orders",
    "ensemble_weights",
    "ensemble_aggregation",
    "time_decay_enabled",
    "time_decay_rate",
    "time_decay_min_weight",
]

PREDICT_PARAM_KEYS = [
    "anomaly_threshold",
    "decision_mode",
    "score_threshold",
    "score_window",
    "max_mapping_distance",
]

SUMMARY_PARAM_COLS = [
    "dataset",
    "window_size",
    "score_threshold",
    "order",
    "smoothing_alpha",
    "decision_mode",
    "score_window",
    "max_mapping_distance",
    "distance_penalty_alpha",
    "distance_tolerance",
    "entropy_threshold_enabled",
    "entropy_enabled",
    "entropy_threshold",
    "entropy_mode",
    "transition_confidence_enabled",
    "transition_confidence_k",
    "transition_confidence_weight",
    "transition_confidence_mode",
    "transition_confidence_use_transition_count",
    "use_transition_count",
    "dirichlet_smoothing_enabled",
    "dirichlet_alpha",
    "dirichlet_prior_mode",
    "ensemble_enabled",
    "ensemble_orders",
    "ensemble_weights",
    "ensemble_aggregation",
    "time_decay_enabled",
    "time_decay_rate",
    "time_decay_min_weight",
]


def parse_number_list(raw: str, cast=float) -> list[Any]:
    values: list[Any] = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            values.append(cast(item))
    return values


def parse_str_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def load_automata_config(config_path: str = "src/config/settings.json") -> dict[str, Any]:
    config = DEFAULT_CONFIG.copy()
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            user_config = json.load(f)
        automata_config = user_config.get("automata", {})
        # Keep unknown keys too, because feature branches may add their own names.
        config.update(automata_config)
    return config


def get_branch_slug() -> str:
    try:
        completed = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        branch = completed.stdout.strip() or "unknown_branch"
    except Exception:
        branch = "unknown_branch"

    if branch.startswith("feature/"):
        branch = branch.split("/", 1)[1]
    return re.sub(r"[^a-zA-Z0-9]+", "_", branch).strip("_").lower() or "unknown_branch"


def filter_kwargs(callable_obj: Any, candidate_kwargs: Mapping[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(callable_obj)
    params = signature.parameters
    accepts_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    if accepts_var_kwargs:
        return dict(candidate_kwargs)
    return {k: v for k, v in candidate_kwargs.items() if k in params}


def output_value(value: Any) -> Any:
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def apply_dataset_final_config(config: dict[str, Any], dataset: str, window_size: int, score_threshold: float) -> dict[str, Any]:
    prefix = "batadal" if dataset.upper() == "BATADAL" else "skab"
    dataset_config = config.copy()

    dataset_config.update({
        "window_size": int(window_size),
        "order": config.get(f"{prefix}_final_order", config.get("order", 3)),
        "smoothing_alpha": config.get(f"{prefix}_final_smoothing_alpha", config.get("smoothing_alpha", 1.0)),
        "decision_mode": config.get(f"{prefix}_final_decision_mode", config.get("decision_mode", "avg_negative_log")),
        "score_window": config.get(f"{prefix}_final_score_window", config.get("score_window", 1)),
        "score_threshold": float(score_threshold),
        "max_mapping_distance": config.get(f"{prefix}_final_max_mapping_distance", config.get("max_mapping_distance", None)),
        "auto_score_percentile": None,
        "anomaly_threshold": config.get(f"{prefix}_anomaly_threshold", config.get("anomaly_threshold", 0.05)),
    })

    # Apply dataset-specific final feature values when present. Otherwise keep global/default value.
    for key in MODEL_PARAM_KEYS:
        final_key = f"{prefix}_final_{key}"
        if final_key in config:
            dataset_config[key] = config[final_key]

    return dataset_config


def make_model(config: dict[str, Any]) -> tuple[ProbabilisticAutomata, dict[str, Any]]:
    candidate_kwargs = {"smoothing": True}
    for key in MODEL_PARAM_KEYS:
        if key in config:
            candidate_kwargs[key] = config[key]

    model_kwargs = filter_kwargs(ProbabilisticAutomata.__init__, candidate_kwargs)
    model = ProbabilisticAutomata(**model_kwargs)
    return model, model_kwargs


def inject_gaussian_noise(series: np.ndarray, noise_level: float = 0.1, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, noise_level, series.shape)
    return series + noise


def enable_nearest_pattern_cache(model: ProbabilisticAutomata, config: dict[str, Any]) -> None:
    """Cache nearest-pattern lookups for the lifetime of a trained model instance.

    State-similarity branches may call _find_nearest_pattern many times. The
    model.predict() method can also generate expensive explainability distance
    reports. This cache is safe for ablation metrics because the trained pattern
    set is fixed after fit() when learning_rate=0.
    """
    if not config.get("window_threshold_cache_nearest", True):
        return
    if getattr(model, "_window_threshold_nearest_cache_enabled", False):
        return
    if not hasattr(model, "_find_nearest_pattern"):
        return

    original_find = model._find_nearest_pattern

    @functools.lru_cache(maxsize=200000)
    def cached_find(pattern: str):
        return original_find(pattern)

    model._find_nearest_pattern = cached_find  # type: ignore[method-assign]
    model._window_threshold_nearest_cache_enabled = True


def _predictions_from_scores(model: ProbabilisticAutomata, patterns: list[str], scores: list[float], config: dict[str, Any]) -> list[int]:
    order = int(getattr(model, "order", config.get("order", 1)) or 1)
    if len(patterns) < order:
        return [0] * max(0, len(patterns) - 1)

    decision_mode = config.get("decision_mode", "probability")
    eps = 1e-12

    if decision_mode in {"negative_log", "avg_negative_log"}:
        score_threshold = config.get("score_threshold")
        if score_threshold is None:
            score_threshold = -np.log(float(config.get("anomaly_threshold", 0.05)) + eps)
        core_predictions = [1 if float(score) > float(score_threshold) else 0 for score in scores]
    else:
        anomaly_threshold = float(config.get("anomaly_threshold", 0.05))
        core_predictions = [1 if float(score) < anomaly_threshold else 0 for score in scores]

    padding_count = max(0, order - 1)
    return [0] * padding_count + core_predictions


def fast_predict_with_scores(model: ProbabilisticAutomata, patterns: list[str], config: dict[str, Any]):
    """Metric-only prediction path.

    Prefer calculate_scores() over predict() when available, because predict() in
    several branches creates explainability logs. In state-similarity, those logs
    require Levenshtein distances to every trained pattern and can dominate the
    runtime for large window_size values.
    """
    if not hasattr(model, "calculate_scores"):
        raise AttributeError("Model does not expose calculate_scores().")

    enable_nearest_pattern_cache(model, config)

    score_candidate_kwargs = {
        "decision_mode": config.get("decision_mode"),
        "score_window": config.get("score_window"),
        "max_mapping_distance": config.get("max_mapping_distance"),
        "score_threshold": config.get("score_threshold"),
        "anomaly_threshold": config.get("anomaly_threshold"),
    }
    score_kwargs = filter_kwargs(model.calculate_scores, score_candidate_kwargs)
    scores = model.calculate_scores(patterns, **score_kwargs)
    preds = _predictions_from_scores(model, patterns, scores, config)
    return preds, []


def predict_with_supported_kwargs(model: ProbabilisticAutomata, patterns: list[str], config: dict[str, Any]):
    if config.get("window_threshold_fast_predict", True):
        try:
            return fast_predict_with_scores(model, patterns, config)
        except Exception as exc:
            if config.get("window_threshold_fast_predict_strict", False):
                raise
            print(f"[UYARI] Fast predict başarısız, model.predict fallback kullanılacak: {exc}")

    candidate_kwargs = {key: config.get(key) for key in PREDICT_PARAM_KEYS if key in config}
    predict_kwargs = filter_kwargs(model.predict, candidate_kwargs)
    return model.predict(patterns, **predict_kwargs)


def run_experiment_pipeline(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: dict[str, Any],
    dataset_name: str,
    fold_name: str,
    seed: int,
    scenarios: set[str],
) -> list[dict[str, Any]]:
    transformer = SaxPaaTransformer(alphabet_size=config["alphabet_size"])
    train_patterns = transformer.transform(X_train, window_size=config["window_size"])

    model, model_kwargs = make_model(config)
    model.fit(train_patterns)

    num_states = len(getattr(model, "total_exits", {}))
    trained_patterns = getattr(model, "trained_patterns", set())
    transitions = getattr(model, "transitions", {})
    vocab_size = len(trained_patterns)
    num_transitions = sum(len(targets) for targets in transitions.values()) if transitions else 0
    transition_density = num_transitions / (num_states * vocab_size) if num_states > 0 and vocab_size > 0 else 0.0

    common_fields = {
        "dataset": dataset_name,
        "fold": fold_name,
        "seed": seed,
        "window_size": config["window_size"],
        "alphabet_size": config["alphabet_size"],
        "order": config.get("order"),
        "smoothing_alpha": config.get("smoothing_alpha"),
        "decision_mode": config.get("decision_mode"),
        "score_window": config.get("score_window"),
        "score_threshold": config.get("score_threshold"),
        "anomaly_threshold": config.get("anomaly_threshold"),
        "max_mapping_distance": config.get("max_mapping_distance"),
        "num_states": num_states,
        "vocab_size": vocab_size,
        "num_transitions": num_transitions,
        "transition_density": transition_density,
    }
    for key, value in model_kwargs.items():
        if key not in common_fields:
            common_fields[key] = output_value(value)

    results: list[dict[str, Any]] = []
    test_patterns_orig = transformer.transform(X_test, window_size=config["window_size"])

    if "original" in scenarios:
        preds_orig, _ = predict_with_supported_kwargs(model, test_patterns_orig, config)
        y_test_aligned_orig = y_test[: len(preds_orig)]
        metrics_orig = calculate_metrics(y_test_aligned_orig, preds_orig)
        metrics_orig.update({"scenario": "original", **common_fields})
        results.append(metrics_orig)

    if "gaussian_noise" in scenarios:
        X_test_noisy = inject_gaussian_noise(X_test, noise_level=config.get("noise_level", 0.1), seed=seed)
        test_patterns_noisy = transformer.transform(X_test_noisy, window_size=config["window_size"])
        preds_noisy, _ = predict_with_supported_kwargs(model, test_patterns_noisy, config)
        y_test_aligned_noisy = y_test[: len(preds_noisy)]
        metrics_noisy = calculate_metrics(y_test_aligned_noisy, preds_noisy)
        metrics_noisy.update({"scenario": "gaussian_noise", **common_fields})
        results.append(metrics_noisy)

    if "unseen_data" in scenarios:
        unseen_test_patterns: list[str] = []
        y_test_unseen: list[int] = []
        y_test_sliding = y_test[: len(test_patterns_orig)]
        for idx, pattern in enumerate(test_patterns_orig):
            if pattern not in trained_patterns:
                unseen_test_patterns.append(pattern)
                y_test_unseen.append(y_test_sliding[idx])

        if len(unseen_test_patterns) > 1:
            preds_unseen, _ = predict_with_supported_kwargs(model, unseen_test_patterns, config)
            y_test_aligned_unseen = y_test_unseen[-len(preds_unseen):]
            metrics_unseen = calculate_metrics(y_test_aligned_unseen, preds_unseen)
        else:
            metrics_unseen = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0}

        metrics_unseen.update({"scenario": "unseen_data", **common_fields})
        results.append(metrics_unseen)

    return results


def load_batadal_data():
    required = [
        "data/processed/batadal_X_train_adasyn_pc1.csv",
        "data/processed/batadal_X_test_pc1.csv",
        "data/processed/batadal_y_test.csv",
    ]
    if not all(os.path.exists(path) for path in required):
        return None

    X_train = pd.read_csv(required[0]).values.flatten()
    X_test = pd.read_csv(required[1]).values.flatten()
    y_test = pd.read_csv(required[2]).values.flatten()
    y_test = np.where(y_test == -999, 0, y_test)
    return X_train, X_test, y_test


def iter_skab_folds():
    for fold in range(1, 6):
        train_file = f"data/processed/skab_fold{fold}_X_train_pc1.csv"
        test_file = f"data/processed/skab_fold{fold}_X_test_pc1.csv"
        y_file = f"data/processed/skab_fold{fold}_y_test.csv"
        if not (os.path.exists(train_file) and os.path.exists(test_file) and os.path.exists(y_file)):
            continue
        X_train = pd.read_csv(train_file).values.flatten()
        X_test = pd.read_csv(test_file).values.flatten()
        y_test = pd.read_csv(y_file).values.flatten()
        yield fold, X_train, X_test, y_test


def dataset_label_from_choice(datasets: str) -> str:
    return {"both": "both", "batadal": "batadal", "skab": "skab"}[datasets]


def make_output_paths(output_dir: str, branch_slug: str, dataset_label: str) -> dict[str, str]:
    base_dir = os.path.join(output_dir, branch_slug)
    paths = {
        "base": base_dir,
        "main": os.path.join(base_dir, "01_main_ablation"),
        "unseen": os.path.join(base_dir, "02_unseen_top_candidates"),
        "reports": os.path.join(base_dir, "03_reports"),
        "partials": os.path.join(base_dir, "00_partials"),
    }
    for path in paths.values():
        os.makedirs(path, exist_ok=True)

    prefix = f"{branch_slug}_{dataset_label}"
    paths.update({
        "run_plan": os.path.join(paths["reports"], f"{prefix}_run_plan.json"),
        "main_metrics": os.path.join(paths["main"], f"{prefix}_main_metrics.csv"),
        "main_summary": os.path.join(paths["main"], f"{prefix}_main_summary.csv"),
        "main_best": os.path.join(paths["main"], f"{prefix}_main_best_candidates.csv"),
        "top3_candidates": os.path.join(paths["reports"], f"{prefix}_selected_top_candidates.csv"),
        "unseen_metrics": os.path.join(paths["unseen"], f"{prefix}_unseen_top_candidates_metrics.csv"),
        "unseen_summary": os.path.join(paths["unseen"], f"{prefix}_unseen_top_candidates_summary.csv"),
        "final_comparison": os.path.join(paths["reports"], f"{prefix}_final_comparison.csv"),
        "main_partial": os.path.join(paths["partials"], f"{prefix}_main_partial_metrics.csv"),
        "unseen_partial": os.path.join(paths["partials"], f"{prefix}_unseen_partial_metrics.csv"),
    })
    return paths


def build_summary(df: pd.DataFrame, scenario: str | None = "original") -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    working = df.copy()
    if scenario is not None:
        working = working[working["scenario"] == scenario].copy()
    if working.empty:
        return pd.DataFrame()

    preferred_group_cols = ["scenario"] + SUMMARY_PARAM_COLS if scenario is None else SUMMARY_PARAM_COLS
    group_cols = [col for col in preferred_group_cols if col in working.columns]

    summary = working.groupby(group_cols, dropna=False).agg(
        n_runs=("f1_score", "count"),
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        precision_mean=("precision", "mean"),
        precision_std=("precision", "std"),
        recall_mean=("recall", "mean"),
        recall_std=("recall", "std"),
        f1_score_mean=("f1_score", "mean"),
        f1_score_std=("f1_score", "std"),
        transition_density_mean=("transition_density", "mean"),
        num_states_mean=("num_states", "mean"),
        num_transitions_mean=("num_transitions", "mean"),
    ).reset_index()

    sort_cols = [col for col in ["dataset", "f1_score_mean", "recall_mean", "precision_mean"] if col in summary.columns]
    ascending = [True, False, False, False][:len(sort_cols)]
    if sort_cols:
        summary = summary.sort_values(sort_cols, ascending=ascending)
    return summary


def select_top_candidates(summary: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if summary.empty or top_n <= 0:
        return pd.DataFrame()
    sort_cols = ["dataset", "f1_score_mean", "recall_mean", "precision_mean"]
    summary_sorted = summary.sort_values(sort_cols, ascending=[True, False, False, False]).copy()
    return summary_sorted.groupby("dataset", group_keys=False).head(top_n).reset_index(drop=True)


def save_partial_results(all_results: list[dict[str, Any]], path: str) -> None:
    if not all_results:
        return
    pd.DataFrame(all_results).to_csv(path, index=False)


def save_run_plan(paths: dict[str, str], plan: dict[str, Any]) -> None:
    with open(paths["run_plan"], "w", encoding="utf-8") as f:
        json.dump(plan, f, ensure_ascii=False, indent=2)


def run_main_ablation(
    config: dict[str, Any],
    datasets: str,
    window_sizes: list[int],
    batadal_thresholds: list[float],
    skab_thresholds: list[float],
    seeds: list[int],
    scenarios: set[str],
    partial_path: str,
) -> pd.DataFrame:
    all_results: list[dict[str, Any]] = []

    if datasets in {"both", "batadal"}:
        batadal_data = load_batadal_data()
        if batadal_data is None:
            print("\n[UYARI] BATADAL processed dosyaları bulunamadı, BATADAL atlanıyor.")
        else:
            X_train_b, X_test_b, y_test_b = batadal_data
            print("\n>> BATADAL main ablation başlıyor...")
            for window_size in window_sizes:
                for threshold in batadal_thresholds:
                    combo_results: list[dict[str, Any]] = []
                    for seed in seeds:
                        cc = apply_dataset_final_config(config, "BATADAL", int(window_size), float(threshold))
                        res = run_experiment_pipeline(
                            X_train_b,
                            X_test_b,
                            y_test_b,
                            cc,
                            "BATADAL",
                            "single",
                            seed=int(seed),
                            scenarios=scenarios,
                        )
                        all_results.extend(res)
                        combo_results.extend(res)

                    temp_df = pd.DataFrame([r for r in combo_results if r["scenario"] == "original"])
                    if not temp_df.empty:
                        print(
                            f"BATADAL -> window_size={window_size}, threshold={threshold} | "
                            f"P={temp_df['precision'].mean():.4f}, "
                            f"R={temp_df['recall'].mean():.4f}, "
                            f"F1={temp_df['f1_score'].mean():.4f}"
                        )
                    save_partial_results(all_results, partial_path)

    if datasets in {"both", "skab"}:
        skab_folds = list(iter_skab_folds())
        if not skab_folds:
            print("\n[UYARI] SKAB fold dosyaları bulunamadı, SKAB atlanıyor.")
        else:
            print("\n>> SKAB main ablation başlıyor...")
            for window_size in window_sizes:
                for threshold in skab_thresholds:
                    combo_results = []
                    for fold, X_train_s, X_test_s, y_test_s in skab_folds:
                        for seed in seeds:
                            cc = apply_dataset_final_config(config, "SKAB", int(window_size), float(threshold))
                            res = run_experiment_pipeline(
                                X_train_s,
                                X_test_s,
                                y_test_s,
                                cc,
                                "SKAB",
                                f"fold_{fold}",
                                seed=int(seed),
                                scenarios=scenarios,
                            )
                            all_results.extend(res)
                            combo_results.extend(res)

                    temp_df = pd.DataFrame([r for r in combo_results if r["scenario"] == "original"])
                    if not temp_df.empty:
                        print(
                            f"SKAB -> window_size={window_size}, threshold={threshold} | "
                            f"P={temp_df['precision'].mean():.4f}, "
                            f"R={temp_df['recall'].mean():.4f}, "
                            f"F1={temp_df['f1_score'].mean():.4f}"
                        )
                    save_partial_results(all_results, partial_path)

    return pd.DataFrame(all_results)


def run_unseen_for_top_candidates(
    config: dict[str, Any],
    top_candidates: pd.DataFrame,
    seeds: list[int],
    partial_path: str,
) -> pd.DataFrame:
    if top_candidates.empty:
        return pd.DataFrame()

    all_results: list[dict[str, Any]] = []
    scenarios = {"unseen_data"}

    print("\n>> En iyi adaylar için unseen_data değerlendirmesi başlıyor...")
    for _, candidate in top_candidates.iterrows():
        dataset = str(candidate["dataset"]).upper()
        window_size = int(candidate["window_size"])
        threshold = float(candidate["score_threshold"])
        print(f"\n[UNSEEN] {dataset} -> window_size={window_size}, threshold={threshold}")

        if dataset == "BATADAL":
            batadal_data = load_batadal_data()
            if batadal_data is None:
                print("[UYARI] BATADAL processed dosyaları bulunamadı, unseen atlanıyor.")
                continue
            X_train_b, X_test_b, y_test_b = batadal_data
            for seed in seeds:
                cc = apply_dataset_final_config(config, "BATADAL", window_size, threshold)
                res = run_experiment_pipeline(
                    X_train_b,
                    X_test_b,
                    y_test_b,
                    cc,
                    "BATADAL",
                    "single",
                    seed=int(seed),
                    scenarios=scenarios,
                )
                all_results.extend(res)
                save_partial_results(all_results, partial_path)

        elif dataset == "SKAB":
            skab_folds = list(iter_skab_folds())
            if not skab_folds:
                print("[UYARI] SKAB fold dosyaları bulunamadı, unseen atlanıyor.")
                continue
            for fold, X_train_s, X_test_s, y_test_s in skab_folds:
                for seed in seeds:
                    cc = apply_dataset_final_config(config, "SKAB", window_size, threshold)
                    res = run_experiment_pipeline(
                        X_train_s,
                        X_test_s,
                        y_test_s,
                        cc,
                        "SKAB",
                        f"fold_{fold}",
                        seed=int(seed),
                        scenarios=scenarios,
                    )
                    all_results.extend(res)
                    save_partial_results(all_results, partial_path)

    return pd.DataFrame(all_results)


def merge_final_comparison(main_summary: pd.DataFrame, unseen_summary: pd.DataFrame, top_candidates: pd.DataFrame) -> pd.DataFrame:
    if top_candidates.empty:
        return pd.DataFrame()

    key_cols = ["dataset", "window_size", "score_threshold"]
    main_cols = key_cols + [
        "precision_mean",
        "recall_mean",
        "f1_score_mean",
        "accuracy_mean",
        "n_runs",
    ]
    main_existing = [col for col in main_cols if col in main_summary.columns]
    main_part = top_candidates[main_existing].copy()
    rename_main = {
        "precision_mean": "original_precision_mean",
        "recall_mean": "original_recall_mean",
        "f1_score_mean": "original_f1_score_mean",
        "accuracy_mean": "original_accuracy_mean",
        "n_runs": "original_n_runs",
    }
    main_part = main_part.rename(columns=rename_main)

    if unseen_summary.empty:
        return main_part

    unseen_cols = key_cols + [
        "precision_mean",
        "recall_mean",
        "f1_score_mean",
        "accuracy_mean",
        "n_runs",
    ]
    unseen_existing = [col for col in unseen_cols if col in unseen_summary.columns]
    unseen_part = unseen_summary[unseen_existing].copy().rename(columns={
        "precision_mean": "unseen_precision_mean",
        "recall_mean": "unseen_recall_mean",
        "f1_score_mean": "unseen_f1_score_mean",
        "accuracy_mean": "unseen_accuracy_mean",
        "n_runs": "unseen_n_runs",
    })

    return main_part.merge(unseen_part, on=key_cols, how="left")


def run_ablation(
    config: dict[str, Any],
    datasets: str,
    output_dir: str,
    dry_run: bool = False,
    main_scenarios: set[str] | None = None,
    skip_unseen_top: bool = False,
    top_n_unseen: int | None = None,
) -> pd.DataFrame:
    branch_slug = get_branch_slug()
    dataset_label = dataset_label_from_choice(datasets)
    paths = make_output_paths(output_dir, branch_slug, dataset_label)

    seeds = [int(seed) for seed in config.get("seeds", [42, 123, 2026, 7, 999])]
    window_sizes = [int(x) for x in (config.get("window_threshold_window_sizes") or config.get("window_size_values") or list(range(3, 11)))]
    batadal_thresholds = [float(x) for x in config.get("batadal_window_threshold_score_thresholds", DEFAULT_CONFIG["batadal_window_threshold_score_thresholds"])]
    skab_thresholds = [float(x) for x in config.get("skab_window_threshold_score_thresholds", DEFAULT_CONFIG["skab_window_threshold_score_thresholds"])]
    main_scenarios = main_scenarios or set(config.get("window_threshold_main_scenarios", ["original", "gaussian_noise"]))
    top_n = int(top_n_unseen if top_n_unseen is not None else config.get("window_threshold_unseen_top_n", 3))

    if "unseen_data" in main_scenarios:
        raise ValueError(
            "unseen_data geniş main ablation içinde çalıştırılmamalı. "
            "Bu script unseen_data'yı otomatik olarak en iyi adaylar için ayrı çalıştırır."
        )

    supported_model_params = list(filter_kwargs(ProbabilisticAutomata.__init__, {key: None for key in MODEL_PARAM_KEYS}).keys())
    plan = {
        "branch_slug": branch_slug,
        "datasets": datasets,
        "supported_model_params": supported_model_params,
        "window_sizes": window_sizes,
        "batadal_thresholds": batadal_thresholds,
        "skab_thresholds": skab_thresholds,
        "seeds": seeds,
        "main_scenarios": sorted(main_scenarios),
        "unseen_top_n_per_dataset": top_n,
        "skip_unseen_top": skip_unseen_top,
        "fast_predict": bool(config.get("window_threshold_fast_predict", True)),
        "cache_nearest_pattern": bool(config.get("window_threshold_cache_nearest", True)),
        "output_base": paths["base"],
    }
    save_run_plan(paths, plan)

    print("\n--- WINDOW SIZE + SCORE THRESHOLD ABLATION ---")
    print(f"Branch slug: {branch_slug}")
    print(f"Desteklenen model parametreleri: {supported_model_params}")
    print(f"Window size aralığı: {window_sizes}")
    print(f"BATADAL threshold aralığı: {batadal_thresholds}")
    print(f"SKAB threshold aralığı: {skab_thresholds}")
    print(f"Seeds: {seeds}")
    print(f"Main senaryolar: {sorted(main_scenarios)}")
    print(f"Fast predict: {bool(config.get('window_threshold_fast_predict', True))}")
    print(f"Nearest-pattern cache: {bool(config.get('window_threshold_cache_nearest', True))}")
    print(f"Unseen-data: main bittikten sonra dataset başına en iyi {top_n} aday")
    print(f"Output klasörü: {paths['base']}")

    if dry_run:
        print("\n[DRY-RUN] Çalıştırma yapılmadı.")
        print(f"[OK] Run plan kaydedildi: {paths['run_plan']}")
        return pd.DataFrame()

    main_df = run_main_ablation(
        config=config,
        datasets=datasets,
        window_sizes=window_sizes,
        batadal_thresholds=batadal_thresholds,
        skab_thresholds=skab_thresholds,
        seeds=seeds,
        scenarios=main_scenarios,
        partial_path=paths["main_partial"],
    )

    if main_df.empty:
        print("\n[UYARI] Hiç main sonuç üretilemedi.")
        return pd.DataFrame()

    main_df.to_csv(paths["main_metrics"], index=False)
    main_summary = build_summary(main_df, scenario="original")
    main_summary.to_csv(paths["main_summary"], index=False)
    main_best = select_top_candidates(main_summary, top_n=20)
    main_best.to_csv(paths["main_best"], index=False)

    print(f"\n[OK] Main metrics kaydedildi: {paths['main_metrics']}")
    print(f"[OK] Main summary kaydedildi: {paths['main_summary']}")
    print(f"[OK] Main en iyi adaylar kaydedildi: {paths['main_best']}")

    top_candidates = select_top_candidates(main_summary, top_n=top_n)
    top_candidates.to_csv(paths["top3_candidates"], index=False)
    print(f"[OK] Unseen için seçilen adaylar kaydedildi: {paths['top3_candidates']}")

    print("\n--- Original F1'e göre seçilen en iyi adaylar ---")
    display_cols = ["dataset", "window_size", "score_threshold", "precision_mean", "recall_mean", "f1_score_mean"]
    display_cols = [col for col in display_cols if col in top_candidates.columns]
    print(top_candidates[display_cols].to_string(index=False))

    unseen_df = pd.DataFrame()
    unseen_summary = pd.DataFrame()
    if not skip_unseen_top and top_n > 0 and not top_candidates.empty:
        unseen_df = run_unseen_for_top_candidates(
            config=config,
            top_candidates=top_candidates,
            seeds=seeds,
            partial_path=paths["unseen_partial"],
        )
        if not unseen_df.empty:
            unseen_df.to_csv(paths["unseen_metrics"], index=False)
            unseen_summary = build_summary(unseen_df, scenario="unseen_data")
            unseen_summary.to_csv(paths["unseen_summary"], index=False)
            print(f"\n[OK] Unseen top-candidates metrics kaydedildi: {paths['unseen_metrics']}")
            print(f"[OK] Unseen top-candidates summary kaydedildi: {paths['unseen_summary']}")
        else:
            print("\n[UYARI] Unseen-data sonucu üretilemedi.")
    else:
        print("\n[INFO] Unseen-data top-candidate adımı atlandı.")

    final_comparison = merge_final_comparison(main_summary, unseen_summary, top_candidates)
    if not final_comparison.empty:
        final_comparison.to_csv(paths["final_comparison"], index=False)
        print(f"[OK] Final karşılaştırma kaydedildi: {paths['final_comparison']}")
        print("\n--- Final karşılaştırma ---")
        print(final_comparison.to_string(index=False))

    return main_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Window-size + score-threshold ablation for automata branches.")
    parser.add_argument("--config", default="src/config/settings.json", help="settings.json path")
    parser.add_argument("--datasets", choices=["both", "batadal", "skab"], default="both", help="which dataset(s) to run")
    parser.add_argument("--output-dir", default="results/outputs/window_threshold_ablation", help="where CSV outputs are written")
    parser.add_argument("--window-sizes", default=None, help="comma-separated integers, e.g. 3,4,5,6,7,8,9,10")
    parser.add_argument("--batadal-thresholds", default=None, help="comma-separated floats")
    parser.add_argument("--skab-thresholds", default=None, help="comma-separated floats")
    parser.add_argument("--dry-run", action="store_true", help="print plan and supported params without running experiments")
    parser.add_argument(
        "--main-scenarios",
        default=None,
        help="comma-separated main scenario names: original,gaussian_noise. Do not include unseen_data.",
    )
    parser.add_argument("--top-n-unseen", type=int, default=None, help="run unseen_data for best N candidates per dataset")
    parser.add_argument("--skip-unseen-top", action="store_true", help="skip unseen_data top-candidate phase")
    args = parser.parse_args()

    config = load_automata_config(args.config)

    if args.window_sizes:
        config["window_threshold_window_sizes"] = parse_number_list(args.window_sizes, cast=int)
    if args.batadal_thresholds:
        config["batadal_window_threshold_score_thresholds"] = parse_number_list(args.batadal_thresholds, cast=float)
    if args.skab_thresholds:
        config["skab_window_threshold_score_thresholds"] = parse_number_list(args.skab_thresholds, cast=float)

    main_scenarios = None
    if args.main_scenarios:
        main_scenarios = set(parse_str_list(args.main_scenarios))
        allowed = {"original", "gaussian_noise"}
        invalid = main_scenarios - allowed
        if invalid:
            raise ValueError(f"Geçersiz main senaryo: {sorted(invalid)}. Geçerli main senaryolar: {sorted(allowed)}")

    run_ablation(
        config=config,
        datasets=args.datasets,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        main_scenarios=main_scenarios,
        skip_unseen_top=args.skip_unseen_top,
        top_n_unseen=args.top_n_unseen,
    )


if __name__ == "__main__":
    main()