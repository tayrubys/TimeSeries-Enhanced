import json
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..")
    )
)

from src.data_pipeline.sax_paa import SaxPaaTransformer
from src.experiments.evaluator import calculate_metrics
from src.experiments.run_skab_alergia_state_merging_experiment_temporal_persistence import (
    apply_temporal_persistence,
    choose_inner_split,
    combine_statistics,
    evaluate_unseen,
    fit_dual_models,
    inject_noise,
    load_config,
    load_source_files,
    load_vector,
    prepare_grouped_patterns,
    score_dual,
    score_grouped,
    select_threshold,
    select_threshold_with_grouped_persistence,
    summarize_scores,
    to_binary,
)


OUTPUT_DIR = "results/outputs"
REFERENCE_METRICS_PATH = os.path.join(
    OUTPUT_DIR,
    "skab_dual_alergia_fold_seed_metrics.csv",
)

PREFIX = "skab_block_confidence_exact_pipeline"

METRICS_OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    f"{PREFIX}_fold_seed_metrics.csv",
)
VALIDATION_OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    f"{PREFIX}_validation_search.csv",
)
BEST_OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    f"{PREFIX}_best_configs.csv",
)
PARITY_OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    f"{PREFIX}_baseline_parity.csv",
)
OVERALL_OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    f"{PREFIX}_overall_summary.csv",
)
REPORT_OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    f"{PREFIX}_report.json",
)

SUM_MARGIN_QUANTILES = [
    0.00,
    0.25,
    0.50,
    0.60,
    0.70,
    0.75,
    0.80,
    0.90,
]

PARITY_TOLERANCE = 1e-9


def load_reference_metrics(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            "Temporal Persistence reference metrik dosyası bulunamadı:\n"
            f"{path}\n\n"
            "Önce şu runner çalıştırılmalıdır:\n"
            "python src/experiments/"
            "run_skab_alergia_state_merging_experiment_temporal_persistence.py"
        )

    frame = pd.read_csv(path)

    required = {
        "fold",
        "seed",
        "scenario",
        "precision",
        "recall",
        "f1_score",
    }
    missing = required.difference(frame.columns)

    if missing:
        raise ValueError(
            "Reference metrik dosyasında eksik sütunlar var: "
            f"{sorted(missing)}"
        )

    original = frame[
        frame["scenario"].astype(str) == "original"
    ].copy()

    if original.duplicated(
        ["fold", "seed"],
        keep=False,
    ).any():
        raise ValueError(
            "Reference metrik dosyasında original senaryosu için "
            "aynı fold/seed birden fazla kez bulunuyor."
        )

    return original


def get_reference_row(
    reference_metrics,
    fold_id,
    seed,
):
    selected = reference_metrics[
        (
            reference_metrics["fold"].astype(int)
            == int(fold_id)
        )
        & (
            reference_metrics["seed"].astype(int)
            == int(seed)
        )
    ]

    if len(selected) != 1:
        raise ValueError(
            f"Fold={fold_id}, seed={seed} için tek reference satırı "
            "bulunamadı."
        )

    return selected.iloc[0]


def find_anomaly_blocks(predictions):
    predictions = np.asarray(
        predictions,
        dtype=int,
    ).flatten()

    blocks = []
    start = None

    for index, value in enumerate(predictions):
        if value == 1 and start is None:
            start = index
        elif value == 0 and start is not None:
            blocks.append((start, index))
            start = None

    if start is not None:
        blocks.append((start, len(predictions)))

    return blocks


def score_grouped_items(
    normal_model,
    anomaly_model,
    grouped_data,
):
    items = []

    for item in grouped_data:
        scores, logs = score_dual(
            normal_model,
            anomaly_model,
            item["patterns"],
        )
        labels = np.asarray(
            item["transition_labels"],
            dtype=int,
        )

        if len(scores) != len(labels):
            raise ValueError(
                f"{item['source_file']} skor/etiket uzunluğu farklı: "
                f"{len(scores)}/{len(labels)}"
            )

        copied_logs = []

        for log in logs:
            copied = dict(log)
            copied["source_file"] = item["source_file"]
            copied_logs.append(copied)

        items.append(
            {
                "source_file": item["source_file"],
                "scores": np.asarray(
                    scores,
                    dtype=float,
                ),
                "labels": labels,
                "logs": copied_logs,
            }
        )

    return items


def create_persistence_baseline(
    scores,
    threshold,
    min_anomaly_run,
):
    scores = np.asarray(
        scores,
        dtype=float,
    )

    raw_predictions = (
        scores >= float(threshold)
    ).astype(int)

    baseline_predictions = apply_temporal_persistence(
        raw_predictions,
        int(min_anomaly_run),
    )

    return raw_predictions, baseline_predictions


def collect_baseline_sum_margins(
    scored_items,
    threshold,
    min_anomaly_run,
):
    values = []

    for item in scored_items:
        _, baseline = create_persistence_baseline(
            item["scores"],
            threshold,
            min_anomaly_run,
        )
        margins = (
            item["scores"] - float(threshold)
        )

        for start, end in find_anomaly_blocks(
            baseline
        ):
            values.append(
                float(
                    np.sum(margins[start:end])
                )
            )

    return np.asarray(values, dtype=float)


def build_sum_margin_candidates(
    scored_items,
    threshold,
    min_anomaly_run,
):
    values = collect_baseline_sum_margins(
        scored_items,
        threshold,
        min_anomaly_run,
    )

    if len(values) == 0:
        return [0.0]

    candidates = {
        0.0,
        *(
            max(float(value), 0.0)
            for value in np.quantile(
                values,
                SUM_MARGIN_QUANTILES,
            )
        ),
    }

    return sorted(candidates)


def build_length_candidates(min_anomaly_run):
    min_anomaly_run = int(min_anomaly_run)

    return sorted(
        {
            min_anomaly_run,
            min_anomaly_run + 1,
            min_anomaly_run + 2,
            min_anomaly_run + 3,
            min_anomaly_run + 5,
            min_anomaly_run + 8,
        }
    )


def apply_block_filter_to_file(
    scores,
    threshold,
    baseline_min_anomaly_run,
    minimum_block_length,
    minimum_sum_margin,
):
    scores = np.asarray(
        scores,
        dtype=float,
    )

    raw, baseline = create_persistence_baseline(
        scores,
        threshold,
        baseline_min_anomaly_run,
    )

    filtered = np.zeros_like(baseline)
    margins = scores - float(threshold)
    block_logs = []

    for block_id, (start, end) in enumerate(
        find_anomaly_blocks(baseline),
        start=1,
    ):
        block_length = int(end - start)
        sum_margin = float(
            np.sum(margins[start:end])
        )

        length_pass = (
            block_length >= int(minimum_block_length)
        )
        margin_pass = (
            sum_margin >= float(minimum_sum_margin)
        )
        kept = bool(
            length_pass and margin_pass
        )

        if kept:
            filtered[start:end] = 1

        block_logs.append(
            {
                "block_id": int(block_id),
                "start_index": int(start),
                "end_index": int(end - 1),
                "block_length": block_length,
                "sum_margin": sum_margin,
                "length_pass": length_pass,
                "margin_pass": margin_pass,
                "kept": kept,
            }
        )

    return {
        "raw_predictions": raw,
        "baseline_predictions": baseline,
        "filtered_predictions": filtered,
        "block_logs": block_logs,
    }


def apply_grouped_block_filter(
    scored_items,
    threshold,
    baseline_min_anomaly_run,
    minimum_block_length,
    minimum_sum_margin,
):
    raw_parts = []
    baseline_parts = []
    filtered_parts = []
    label_parts = []
    logs = []
    block_logs = []

    for item in scored_items:
        result = apply_block_filter_to_file(
            item["scores"],
            threshold,
            baseline_min_anomaly_run,
            minimum_block_length,
            minimum_sum_margin,
        )

        raw_parts.append(
            result["raw_predictions"]
        )
        baseline_parts.append(
            result["baseline_predictions"]
        )
        filtered_parts.append(
            result["filtered_predictions"]
        )
        label_parts.append(
            item["labels"]
        )

        for index, log in enumerate(
            item["logs"]
        ):
            copied = dict(log)
            copied["raw_decision"] = (
                "anomaly"
                if result["raw_predictions"][index]
                else "normal"
            )
            copied["baseline_decision"] = (
                "anomaly"
                if result["baseline_predictions"][index]
                else "normal"
            )
            copied["decision"] = (
                "anomaly"
                if result["filtered_predictions"][index]
                else "normal"
            )
            logs.append(copied)

        for block_log in result["block_logs"]:
            block_logs.append(
                {
                    "source_file": item["source_file"],
                    **block_log,
                }
            )

    def concatenate(parts):
        if not parts:
            return np.array([], dtype=int)
        return np.concatenate(parts)

    return {
        "raw_predictions": concatenate(
            raw_parts
        ),
        "baseline_predictions": concatenate(
            baseline_parts
        ),
        "filtered_predictions": concatenate(
            filtered_parts
        ),
        "labels": concatenate(
            label_parts
        ),
        "logs": logs,
        "block_logs": block_logs,
    }


def select_exact_persistence_model(
    split_info,
    config,
    fold_id,
    seed,
):
    """
    Başarılı SKAB Temporal Persistence runner'ındaki seçim sırasını
    birebir tekrarlar ve seçilen model nesnelerini aynı süreçte tutar.

    Aşama 1:
      merge_alpha + min_state_count + smoothing_alpha

    Aşama 2:
      distance_penalty + threshold + min_anomaly_run
    """
    search_rows = []
    best = None

    print("\n--- EXACT STATE-MERGING VALIDATION TARAMASI ---")

    for merge_alpha in config["merge_alphas"]:
        for min_count in config["min_counts"]:
            for smoothing_alpha in (
                config["smoothing_alphas"]
            ):
                kwargs = {
                    "merge_alpha": float(
                        merge_alpha
                    ),
                    "min_state_count": int(
                        min_count
                    ),
                    "smoothing_alpha": float(
                        smoothing_alpha
                    ),
                    "max_pattern_distance": config[
                        "max_pattern_distance"
                    ],
                    "distance_penalty": 0.0,
                }

                normal_model, anomaly_model = (
                    fit_dual_models(
                        split_info[
                            "normal_sequences"
                        ],
                        split_info[
                            "anomaly_sequences"
                        ],
                        kwargs,
                    )
                )

                (
                    val_scores,
                    val_labels,
                    val_logs,
                ) = score_grouped(
                    normal_model,
                    anomaly_model,
                    split_info["grouped_val"],
                )

                threshold, metrics = (
                    select_threshold(
                        val_labels,
                        val_scores,
                    )
                )
                stats = combine_statistics(
                    normal_model,
                    anomaly_model,
                )
                diagnostics = summarize_scores(
                    val_labels,
                    val_scores,
                    val_logs,
                    threshold,
                )

                search_rows.append(
                    {
                        "search_stage": (
                            "state_merging"
                        ),
                        "fold": int(fold_id),
                        "seed": int(seed),
                        **kwargs,
                        "threshold": float(
                            threshold
                        ),
                        **stats,
                        **{
                            f"validation_{key}": value
                            for key, value
                            in metrics.items()
                        },
                        **diagnostics,
                    }
                )

                key = (
                    metrics["f1_score"],
                    metrics["precision"],
                    metrics["recall"],
                    -stats[
                        "total_states_after"
                    ],
                )

                if (
                    best is None
                    or key > best["key"]
                ):
                    best = {
                        "key": key,
                        "normal_model": normal_model,
                        "anomaly_model": anomaly_model,
                        "structure_stats": stats,
                        "threshold": threshold,
                        "metrics": metrics,
                        "config": {
                            "window_size": config[
                                "window_size"
                            ],
                            "alphabet_size": config[
                                "alphabet_size"
                            ],
                            "merge_alpha": float(
                                merge_alpha
                            ),
                            "min_state_count": int(
                                min_count
                            ),
                            "smoothing_alpha": float(
                                smoothing_alpha
                            ),
                            "distance_penalty": 0.0,
                            "min_anomaly_run": 1,
                            "min_validation_recall": (
                                config[
                                    "min_validation_recall"
                                ]
                            ),
                        },
                    }

    if best is None:
        raise RuntimeError(
            "State-merging validation taramasında "
            "geçerli model bulunamadı."
        )

    print(
        "State-merging best -> "
        f"alpha={best['config']['merge_alpha']} | "
        f"min={best['config']['min_state_count']} | "
        f"smooth={best['config']['smoothing_alpha']} | "
        f"F1={best['metrics']['f1_score']:.4f}"
    )

    print(
        "\n--- EXACT SOURCE_FILE TEMPORAL "
        "PERSISTENCE TARAMASI ---"
    )

    penalty_best = None

    for distance_penalty in (
        config["distance_penalties"]
    ):
        distance_penalty = float(
            distance_penalty
        )

        best[
            "normal_model"
        ].distance_penalty = distance_penalty
        best[
            "anomaly_model"
        ].distance_penalty = distance_penalty

        (
            val_scores,
            val_labels,
            val_logs,
        ) = score_grouped(
            best["normal_model"],
            best["anomaly_model"],
            split_info["grouped_val"],
        )

        selection = (
            select_threshold_with_grouped_persistence(
                val_labels,
                val_scores,
                split_info["grouped_val"],
                config["min_anomaly_runs"],
                config[
                    "min_validation_recall"
                ],
            )
        )

        metrics = selection["metrics"]
        raw_predictions = selection[
            "raw_predictions"
        ]
        predictions = selection[
            "predictions"
        ]

        search_rows.append(
            {
                "search_stage": (
                    "source_file_temporal_persistence"
                ),
                "fold": int(fold_id),
                "seed": int(seed),
                "merge_alpha": best[
                    "config"
                ]["merge_alpha"],
                "min_state_count": best[
                    "config"
                ]["min_state_count"],
                "smoothing_alpha": best[
                    "config"
                ]["smoothing_alpha"],
                "distance_penalty": (
                    distance_penalty
                ),
                "min_anomaly_run": int(
                    selection[
                        "min_anomaly_run"
                    ]
                ),
                "threshold": float(
                    selection["threshold"]
                ),
                "recall_constraint_met": bool(
                    selection[
                        "recall_constraint_met"
                    ]
                ),
                "selection_mode": selection[
                    "selection_mode"
                ],
                "validation_raw_predicted_anomaly_count": int(
                    raw_predictions.sum()
                ),
                "validation_predicted_anomaly_count": int(
                    predictions.sum()
                ),
                **best["structure_stats"],
                **{
                    f"validation_{key}": value
                    for key, value
                    in metrics.items()
                },
            }
        )

        key = (
            int(
                selection[
                    "recall_constraint_met"
                ]
            ),
            metrics["f1_score"],
            metrics["precision"],
            metrics["recall"],
            -int(predictions.sum()),
            -int(
                selection[
                    "min_anomaly_run"
                ]
            ),
        )

        if (
            penalty_best is None
            or key > penalty_best["key"]
        ):
            penalty_best = {
                "key": key,
                "distance_penalty": (
                    distance_penalty
                ),
                "min_anomaly_run": int(
                    selection[
                        "min_anomaly_run"
                    ]
                ),
                "threshold": float(
                    selection["threshold"]
                ),
                "metrics": metrics,
                "recall_constraint_met": bool(
                    selection[
                        "recall_constraint_met"
                    ]
                ),
                "selection_mode": selection[
                    "selection_mode"
                ],
            }

    if penalty_best is None:
        raise RuntimeError(
            "Temporal Persistence taramasında "
            "geçerli ayar bulunamadı."
        )

    best[
        "normal_model"
    ].distance_penalty = penalty_best[
        "distance_penalty"
    ]
    best[
        "anomaly_model"
    ].distance_penalty = penalty_best[
        "distance_penalty"
    ]
    best["threshold"] = penalty_best[
        "threshold"
    ]
    best["metrics"] = penalty_best[
        "metrics"
    ]
    best[
        "recall_constraint_met"
    ] = penalty_best[
        "recall_constraint_met"
    ]
    best["selection_mode"] = penalty_best[
        "selection_mode"
    ]
    best["config"][
        "distance_penalty"
    ] = penalty_best[
        "distance_penalty"
    ]
    best["config"][
        "min_anomaly_run"
    ] = penalty_best[
        "min_anomaly_run"
    ]

    print(
        "Persistence best -> "
        f"distance="
        f"{best['config']['distance_penalty']} | "
        f"min_run="
        f"{best['config']['min_anomaly_run']} | "
        f"threshold="
        f"{best['threshold']:.6f} | "
        f"P={best['metrics']['precision']:.4f} | "
        f"R={best['metrics']['recall']:.4f} | "
        f"F1={best['metrics']['f1_score']:.4f}"
    )

    return best, search_rows


def select_block_filter(
    scored_items,
    threshold,
    baseline_min_anomaly_run,
    min_validation_recall,
):
    length_candidates = build_length_candidates(
        baseline_min_anomaly_run
    )
    sum_margin_candidates = (
        build_sum_margin_candidates(
            scored_items,
            threshold,
            baseline_min_anomaly_run,
        )
    )

    rows = []
    best_eligible = None
    best_fallback = None

    for minimum_block_length in (
        length_candidates
    ):
        for minimum_sum_margin in (
            sum_margin_candidates
        ):
            result = apply_grouped_block_filter(
                scored_items,
                threshold,
                baseline_min_anomaly_run,
                minimum_block_length,
                minimum_sum_margin,
            )

            metrics = calculate_metrics(
                result["labels"],
                result[
                    "filtered_predictions"
                ],
            )

            baseline_selected = bool(
                int(minimum_block_length)
                == int(
                    baseline_min_anomaly_run
                )
                and math.isclose(
                    float(minimum_sum_margin),
                    0.0,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            )

            total_blocks = len(
                result["block_logs"]
            )
            kept_blocks = sum(
                block["kept"]
                for block in result[
                    "block_logs"
                ]
            )

            row = {
                "minimum_block_length": int(
                    minimum_block_length
                ),
                "minimum_sum_margin": float(
                    minimum_sum_margin
                ),
                "baseline_selected": (
                    baseline_selected
                ),
                "validation_total_block_count": int(
                    total_blocks
                ),
                "validation_kept_block_count": int(
                    kept_blocks
                ),
                "validation_removed_block_count": int(
                    total_blocks - kept_blocks
                ),
                "validation_baseline_predicted_anomaly_count": int(
                    result[
                        "baseline_predictions"
                    ].sum()
                ),
                "validation_filtered_predicted_anomaly_count": int(
                    result[
                        "filtered_predictions"
                    ].sum()
                ),
                **{
                    f"validation_{key}": value
                    for key, value
                    in metrics.items()
                },
            }
            rows.append(row)

            candidate = {
                "minimum_block_length": int(
                    minimum_block_length
                ),
                "minimum_sum_margin": float(
                    minimum_sum_margin
                ),
                "baseline_selected": (
                    baseline_selected
                ),
                "metrics": metrics,
                "result": result,
            }

            eligible = (
                metrics["recall"]
                >= float(min_validation_recall)
            )

            if eligible:
                eligible_key = (
                    metrics["f1_score"],
                    metrics["precision"],
                    metrics["recall"],
                    -int(
                        result[
                            "filtered_predictions"
                        ].sum()
                    ),
                    -int(minimum_block_length),
                    -float(minimum_sum_margin),
                )

                if (
                    best_eligible is None
                    or eligible_key
                    > best_eligible["key"]
                ):
                    best_eligible = {
                        **candidate,
                        "key": eligible_key,
                        "recall_constraint_met": True,
                    }

            fallback_key = (
                metrics["recall"],
                metrics["f1_score"],
                metrics["precision"],
                -int(
                    result[
                        "filtered_predictions"
                    ].sum()
                ),
                -int(minimum_block_length),
                -float(minimum_sum_margin),
            )

            if (
                best_fallback is None
                or fallback_key
                > best_fallback["key"]
            ):
                best_fallback = {
                    **candidate,
                    "key": fallback_key,
                    "recall_constraint_met": (
                        eligible
                    ),
                }

    selected = (
        best_eligible
        if best_eligible is not None
        else best_fallback
    )

    return (
        selected,
        rows,
        length_candidates,
        sum_margin_candidates,
    )


def count_removed_points(
    labels,
    baseline_predictions,
    filtered_predictions,
):
    labels = np.asarray(labels, dtype=int)
    baseline_predictions = np.asarray(
        baseline_predictions,
        dtype=int,
    )
    filtered_predictions = np.asarray(
        filtered_predictions,
        dtype=int,
    )

    removed = (
        (baseline_predictions == 1)
        & (filtered_predictions == 0)
    )

    return {
        "removed_true_positive_count": int(
            np.sum(
                removed & (labels == 1)
            )
        ),
        "removed_false_positive_count": int(
            np.sum(
                removed & (labels == 0)
            )
        ),
    }


def compare_with_reference(
    fold_id,
    seed,
    regenerated_metrics,
    reference_row,
):
    row = {
        "fold": int(fold_id),
        "seed": int(seed),
    }
    all_match = True

    for metric in [
        "precision",
        "recall",
        "f1_score",
    ]:
        regenerated = float(
            regenerated_metrics[metric]
        )
        reference = float(
            reference_row[metric]
        )
        difference = (
            regenerated - reference
        )
        match = math.isclose(
            regenerated,
            reference,
            rel_tol=0.0,
            abs_tol=PARITY_TOLERANCE,
        )

        row[
            f"regenerated_{metric}"
        ] = regenerated
        row[
            f"reference_{metric}"
        ] = reference
        row[
            f"{metric}_difference"
        ] = difference
        row[f"{metric}_match"] = match
        all_match = all_match and match

    row["all_metrics_match"] = all_match
    return row


def evaluate_scored_items(
    scored_items,
    threshold,
    baseline_min_anomaly_run,
    selected_filter,
):
    result = apply_grouped_block_filter(
        scored_items,
        threshold,
        baseline_min_anomaly_run,
        selected_filter[
            "minimum_block_length"
        ],
        selected_filter[
            "minimum_sum_margin"
        ],
    )

    baseline_metrics = calculate_metrics(
        result["labels"],
        result["baseline_predictions"],
    )
    filtered_metrics = calculate_metrics(
        result["labels"],
        result["filtered_predictions"],
    )

    removed = count_removed_points(
        result["labels"],
        result["baseline_predictions"],
        result["filtered_predictions"],
    )

    return {
        "result": result,
        "baseline_metrics": baseline_metrics,
        "filtered_metrics": filtered_metrics,
        "removed": removed,
    }


def make_metric_row(
    fold_id,
    seed,
    method,
    scenario,
    metrics,
    best,
    selected_filter,
    sample_count,
    anomaly_count,
    unseen_count,
):
    return {
        "dataset": "SKAB",
        "fold": int(fold_id),
        "seed": int(seed),
        "method": method,
        "scenario": scenario,
        **best["config"],
        "selected_threshold": float(
            best["threshold"]
        ),
        "minimum_block_length": int(
            selected_filter[
                "minimum_block_length"
            ]
        ),
        "minimum_sum_margin": float(
            selected_filter[
                "minimum_sum_margin"
            ]
        ),
        "baseline_selected": bool(
            selected_filter[
                "baseline_selected"
            ]
        ),
        "sample_count": int(sample_count),
        "actual_anomaly_count": int(
            anomaly_count
        ),
        "unseen_count": int(unseen_count),
        **metrics,
    }


def run_fold_seed(
    fold_id,
    seed,
    config,
    transformer,
    reference_metrics,
):
    print("\n" + "=" * 82)
    print(
        f"SKAB FOLD {fold_id} | SEED {seed} | "
        "EXACT-PIPELINE BLOCK CONFIDENCE"
    )
    print("=" * 82)

    paths = {
        "X_train": (
            f"data/processed/"
            f"skab_fold{fold_id}_X_train_pc1.csv"
        ),
        "y_train": (
            f"data/processed/"
            f"skab_fold{fold_id}_y_train.csv"
        ),
        "train_source": (
            f"data/processed/"
            f"skab_fold{fold_id}_train_source_file.csv"
        ),
        "X_test": (
            f"data/processed/"
            f"skab_fold{fold_id}_X_test_pc1.csv"
        ),
        "y_test": (
            f"data/processed/"
            f"skab_fold{fold_id}_y_test.csv"
        ),
        "test_source": (
            f"data/processed/"
            f"skab_fold{fold_id}_test_source_file.csv"
        ),
    }

    X_train = np.asarray(
        load_vector(paths["X_train"]),
        dtype=float,
    )
    y_train = to_binary(
        load_vector(paths["y_train"])
    )
    train_source = load_source_files(
        paths["train_source"]
    )

    X_test = np.asarray(
        load_vector(paths["X_test"]),
        dtype=float,
    )
    y_test = to_binary(
        load_vector(paths["y_test"])
    )
    test_source = load_source_files(
        paths["test_source"]
    )

    split_info = choose_inner_split(
        X_train,
        y_train,
        train_source,
        transformer,
        config,
        fold_id,
        seed,
    )

    best, base_search_rows = (
        select_exact_persistence_model(
            split_info,
            config,
            fold_id,
            seed,
        )
    )

    validation_items = score_grouped_items(
        best["normal_model"],
        best["anomaly_model"],
        split_info["grouped_val"],
    )

    (
        selected_filter,
        block_rows,
        length_candidates,
        sum_margin_candidates,
    ) = select_block_filter(
        validation_items,
        best["threshold"],
        best["config"][
            "min_anomaly_run"
        ],
        config["min_validation_recall"],
    )

    for row in base_search_rows:
        row["experiment_stage"] = (
            "baseline_model_selection"
        )

    for row in block_rows:
        row.update(
            {
                "experiment_stage": (
                    "block_confidence_filter"
                ),
                "fold": int(fold_id),
                "seed": int(seed),
                "merge_alpha": best[
                    "config"
                ]["merge_alpha"],
                "min_state_count": best[
                    "config"
                ]["min_state_count"],
                "smoothing_alpha": best[
                    "config"
                ]["smoothing_alpha"],
                "distance_penalty": best[
                    "config"
                ]["distance_penalty"],
                "selected_threshold": float(
                    best["threshold"]
                ),
                "baseline_min_anomaly_run": int(
                    best["config"][
                        "min_anomaly_run"
                    ]
                ),
            }
        )

    print("\n--- BLOCK FILTER INNER VALIDATION ---")
    print(
        "Persistence baseline -> "
        f"P={best['metrics']['precision']:.4f} | "
        f"R={best['metrics']['recall']:.4f} | "
        f"F1={best['metrics']['f1_score']:.4f}"
    )
    print(
        "Selected filter -> "
        f"min_length="
        f"{selected_filter['minimum_block_length']} | "
        f"min_sum_margin="
        f"{selected_filter['minimum_sum_margin']:.6f} | "
        f"P={selected_filter['metrics']['precision']:.4f} | "
        f"R={selected_filter['metrics']['recall']:.4f} | "
        f"F1={selected_filter['metrics']['f1_score']:.4f} | "
        f"baseline_selected="
        f"{selected_filter['baseline_selected']}"
    )

    grouped_test = prepare_grouped_patterns(
        X_test,
        y_test,
        test_source,
        transformer,
        config["window_size"],
        float(split_info["train_mean"]),
        float(split_info["train_std"]),
    )
    test_items = score_grouped_items(
        best["normal_model"],
        best["anomaly_model"],
        grouped_test,
    )
    original = evaluate_scored_items(
        test_items,
        best["threshold"],
        best["config"][
            "min_anomaly_run"
        ],
        selected_filter,
    )

    reference_row = get_reference_row(
        reference_metrics,
        fold_id,
        seed,
    )
    parity_row = compare_with_reference(
        fold_id,
        seed,
        original["baseline_metrics"],
        reference_row,
    )

    if not parity_row[
        "all_metrics_match"
    ]:
        raise RuntimeError(
            "Exact-pipeline baseline parity kontrolü başarısız.\n"
            f"Fold={fold_id}, seed={seed}\n"
            f"Regenerated F1="
            f"{original['baseline_metrics']['f1_score']:.12f}\n"
            f"Reference F1="
            f"{float(reference_row['f1_score']):.12f}\n\n"
            "Temporal Persistence reference dosyasının bu kodla aynı "
            "runner/model sürümünden üretildiğini kontrol et."
        )

    noisy_test = inject_noise(
        X_test,
        config["noise_level"],
        seed,
    )
    grouped_noisy = prepare_grouped_patterns(
        noisy_test,
        y_test,
        test_source,
        transformer,
        config["window_size"],
        float(split_info["train_mean"]),
        float(split_info["train_std"]),
    )
    noisy_items = score_grouped_items(
        best["normal_model"],
        best["anomaly_model"],
        grouped_noisy,
    )
    noisy = evaluate_scored_items(
        noisy_items,
        best["threshold"],
        best["config"][
            "min_anomaly_run"
        ],
        selected_filter,
    )

    baseline_unseen = evaluate_unseen(
        original["result"][
            "baseline_predictions"
        ],
        original["result"]["labels"],
        original["result"]["logs"],
    )
    filtered_unseen = evaluate_unseen(
        original["result"][
            "filtered_predictions"
        ],
        original["result"]["labels"],
        original["result"]["logs"],
    )

    metric_rows = [
        make_metric_row(
            fold_id,
            seed,
            "persistence_baseline",
            "original",
            original["baseline_metrics"],
            best,
            selected_filter,
            len(
                original["result"][
                    "labels"
                ]
            ),
            int(
                original["result"][
                    "labels"
                ].sum()
            ),
            baseline_unseen["count"],
        ),
        make_metric_row(
            fold_id,
            seed,
            "persistence_plus_block_confidence",
            "original",
            original["filtered_metrics"],
            best,
            selected_filter,
            len(
                original["result"][
                    "labels"
                ]
            ),
            int(
                original["result"][
                    "labels"
                ].sum()
            ),
            filtered_unseen["count"],
        ),
        make_metric_row(
            fold_id,
            seed,
            "persistence_baseline",
            "gaussian_noise",
            noisy["baseline_metrics"],
            best,
            selected_filter,
            len(
                noisy["result"]["labels"]
            ),
            int(
                noisy["result"][
                    "labels"
                ].sum()
            ),
            sum(
                log["overall_status"]
                == "unseen"
                for log in noisy["result"][
                    "logs"
                ]
            ),
        ),
        make_metric_row(
            fold_id,
            seed,
            "persistence_plus_block_confidence",
            "gaussian_noise",
            noisy["filtered_metrics"],
            best,
            selected_filter,
            len(
                noisy["result"]["labels"]
            ),
            int(
                noisy["result"][
                    "labels"
                ].sum()
            ),
            sum(
                log["overall_status"]
                == "unseen"
                for log in noisy["result"][
                    "logs"
                ]
            ),
        ),
        make_metric_row(
            fold_id,
            seed,
            "persistence_baseline",
            "unseen_data",
            baseline_unseen["metrics"],
            best,
            selected_filter,
            baseline_unseen["count"],
            baseline_unseen[
                "anomaly_count"
            ],
            baseline_unseen["count"],
        ),
        make_metric_row(
            fold_id,
            seed,
            "persistence_plus_block_confidence",
            "unseen_data",
            filtered_unseen["metrics"],
            best,
            selected_filter,
            filtered_unseen["count"],
            filtered_unseen[
                "anomaly_count"
            ],
            filtered_unseen["count"],
        ),
    ]

    removed = original["removed"]

    best_row = {
        "fold": int(fold_id),
        "seed": int(seed),
        **best["config"],
        "selected_threshold": float(
            best["threshold"]
        ),
        "minimum_block_length": int(
            selected_filter[
                "minimum_block_length"
            ]
        ),
        "minimum_sum_margin": float(
            selected_filter[
                "minimum_sum_margin"
            ]
        ),
        "baseline_selected": bool(
            selected_filter[
                "baseline_selected"
            ]
        ),
        "block_validation_precision": float(
            selected_filter[
                "metrics"
            ]["precision"]
        ),
        "block_validation_recall": float(
            selected_filter[
                "metrics"
            ]["recall"]
        ),
        "block_validation_f1": float(
            selected_filter[
                "metrics"
            ]["f1_score"]
        ),
        "test_removed_true_positive_count": (
            removed[
                "removed_true_positive_count"
            ]
        ),
        "test_removed_false_positive_count": (
            removed[
                "removed_false_positive_count"
            ]
        ),
        "length_candidates": json.dumps(
            length_candidates
        ),
        "sum_margin_candidates": json.dumps(
            sum_margin_candidates
        ),
    }

    print("\n--- OUTER TEST ---")
    print(
        "Persistence baseline -> "
        f"P={original['baseline_metrics']['precision']:.4f} | "
        f"R={original['baseline_metrics']['recall']:.4f} | "
        f"F1={original['baseline_metrics']['f1_score']:.4f}"
    )
    print(
        "Persistence + Block Confidence -> "
        f"P={original['filtered_metrics']['precision']:.4f} | "
        f"R={original['filtered_metrics']['recall']:.4f} | "
        f"F1={original['filtered_metrics']['f1_score']:.4f}"
    )
    print(
        "Kaldırılan test noktaları -> "
        f"FP="
        f"{removed['removed_false_positive_count']} | "
        f"TP="
        f"{removed['removed_true_positive_count']}"
    )
    print("Baseline parity: PASS")

    return {
        "metric_rows": metric_rows,
        "validation_rows": (
            base_search_rows + block_rows
        ),
        "best_row": best_row,
        "parity_row": parity_row,
    }


def create_overall_summary(metrics_df):
    columns = [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "unseen_count",
    ]

    summary = metrics_df.groupby(
        ["method", "scenario"]
    )[columns].agg(["mean", "std"])

    summary.columns = [
        f"{column}_{stat}"
        for column, stat in summary.columns
    ]

    return summary.reset_index()


def main():
    print(
        "\n--- SKAB EXACT-PIPELINE BLOCK "
        "CONFIDENCE DENEYİ ---"
    )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    config = load_config()
    reference_metrics = (
        load_reference_metrics(
            REFERENCE_METRICS_PATH
        )
    )
    transformer = SaxPaaTransformer(
        alphabet_size=config[
            "alphabet_size"
        ]
    )

    metric_rows = []
    validation_rows = []
    best_rows = []
    parity_rows = []

    for fold_id in range(
        1,
        config["n_folds"] + 1,
    ):
        for seed in config["seeds"]:
            result = run_fold_seed(
                fold_id,
                seed,
                config,
                transformer,
                reference_metrics,
            )

            metric_rows.extend(
                result["metric_rows"]
            )
            validation_rows.extend(
                result["validation_rows"]
            )
            best_rows.append(
                result["best_row"]
            )
            parity_rows.append(
                result["parity_row"]
            )

    metrics_df = pd.DataFrame(
        metric_rows
    )
    validation_df = pd.DataFrame(
        validation_rows
    )
    best_df = pd.DataFrame(
        best_rows
    )
    parity_df = pd.DataFrame(
        parity_rows
    )
    overall_df = create_overall_summary(
        metrics_df
    )

    if not parity_df[
        "all_metrics_match"
    ].all():
        raise RuntimeError(
            "En az bir fold/seed baseline "
            "parity kontrolünü geçemedi."
        )

    metrics_df.to_csv(
        METRICS_OUTPUT_PATH,
        index=False,
    )
    validation_df.to_csv(
        VALIDATION_OUTPUT_PATH,
        index=False,
    )
    best_df.to_csv(
        BEST_OUTPUT_PATH,
        index=False,
    )
    parity_df.to_csv(
        PARITY_OUTPUT_PATH,
        index=False,
    )
    overall_df.to_csv(
        OVERALL_OUTPUT_PATH,
        index=False,
    )

    report = {
        "pipeline_replayed": (
            "state_merging_search -> "
            "distance_threshold_persistence_search -> "
            "block_confidence_search"
        ),
        "selected_model_reconstructed_from_csv": False,
        "outer_test_used_for_selection": False,
        "all_baseline_parity_checks_passed": True,
        "fold_seed_count": int(
            len(parity_df)
        ),
        "block_filter_baseline_selected_count": int(
            best_df[
                "baseline_selected"
            ].sum()
        ),
    }

    with open(
        REPORT_OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print(
        "\n--- BASELINE PARITY SONUCU ---"
    )
    print(
        f"PASS: "
        f"{int(parity_df['all_metrics_match'].sum())}"
        f"/{len(parity_df)} fold/seed"
    )

    print(
        "\n--- SKAB EXACT-PIPELINE BLOCK "
        "CONFIDENCE GENEL ÖZET ---"
    )
    print(
        overall_df.to_string(
            index=False
        )
    )

    print("\nSonuç dosyaları:")
    print(f"- {METRICS_OUTPUT_PATH}")
    print(f"- {VALIDATION_OUTPUT_PATH}")
    print(f"- {BEST_OUTPUT_PATH}")
    print(f"- {PARITY_OUTPUT_PATH}")
    print(f"- {OVERALL_OUTPUT_PATH}")
    print(f"- {REPORT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()