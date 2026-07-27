import json
import os
import sys
from itertools import product

import numpy as np
import pandas as pd

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
)

from src.experiments.evaluator import calculate_metrics
from src.experiments.run_skab_alergia_state_merging_experiment import (
    build_grouped_class_sequences,
    choose_inner_split,
    combine_statistics,
    evaluate_unseen,
    inject_noise,
    load_config as load_base_config,
    load_source_files,
    load_vector,
    prepare_grouped_patterns,
    score_dual,
    to_binary,
)
from src.data_pipeline.sax_paa import SaxPaaTransformer
from src.models.alergia_state_merging import AlergiaStateMergingAutomata


CONFIG_PATH = "src/config/settings.json"
OUTPUT_DIR = "results/outputs"


def load_config():
    """
    Mevcut SKAB ALERGIA ayarlarını kullanır ve bu deneye özel
    persistence/recall ayarlarını ekler.
    """
    config = load_base_config()

    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        automata = json.load(file).get("automata", {})

    config.update(
        {
            "min_anomaly_runs": [
                int(value)
                for value in automata.get(
                    "alergia_min_anomaly_run_values",
                    [1, 2, 3, 4, 5, 6, 7, 8, 10, 12],
                )
            ],
            "min_validation_recall": float(
                automata.get("alergia_min_validation_recall", 0.70)
            ),
            # Final threshold+persistence taramasında validation skorlarının
            # quantile değerleri kullanılır. Böylece arama kontrollü kalır.
            "threshold_quantile_count": int(
                automata.get(
                    "alergia_class_specific_threshold_quantile_count",
                    41,
                )
            ),
        }
    )

    return config


def create_model_configs(config):
    """27 ALERGIA parametre kombinasyonunu oluşturur."""
    model_configs = []

    for merge_alpha, min_count, smoothing_alpha in product(
        config["merge_alphas"],
        config["min_counts"],
        config["smoothing_alphas"],
    ):
        model_configs.append(
            {
                "merge_alpha": float(merge_alpha),
                "min_state_count": int(min_count),
                "smoothing_alpha": float(smoothing_alpha),
                "max_pattern_distance": config["max_pattern_distance"],
                "distance_penalty": 0.0,
            }
        )

    return model_configs


def config_signature(model_config):
    """Bir model ayarını karşılaştırılabilir tuple biçimine dönüştürür."""
    return (
        float(model_config["merge_alpha"]),
        int(model_config["min_state_count"]),
        float(model_config["smoothing_alpha"]),
    )


def fit_model_bank(sequences, model_configs, class_name):
    """
    Bir sınıf için bütün ALERGIA modellerini yalnızca bir kez eğitir.

    Daha sonra model çiftleri değerlendirilirken yeniden fit yapılmaz.
    """
    bank = []

    for model_id, model_config in enumerate(model_configs):
        model = AlergiaStateMergingAutomata(**model_config)
        model.fit_sequences(sequences)
        statistics = model.get_model_statistics()

        bank.append(
            {
                "model_id": int(model_id),
                "class_name": class_name,
                "config": dict(model_config),
                "model": model,
                "statistics": statistics,
            }
        )

        print(
            f"{class_name} model {model_id + 1:02d}/{len(model_configs)} | "
            f"alpha={model_config['merge_alpha']} | "
            f"min={model_config['min_state_count']} | "
            f"smooth={model_config['smoothing_alpha']} | "
            f"states={statistics['original_state_count']}"
            f"->{statistics['merged_state_count']}"
        )

    return bank


def score_single_model_grouped(model, grouped_data):
    """Bir modeli bütün validation source_file gruplarında skorlar."""
    grouped_scores = []
    all_scores = []

    for item in grouped_data:
        scores, _ = model.score_patterns(item["patterns"])

        if len(scores) != len(item["transition_labels"]):
            raise ValueError(
                f"{item['source_file']} skor/etiket sayısı farklı: "
                f"{len(scores)}/{len(item['transition_labels'])}"
            )

        grouped_scores.append(np.asarray(scores, dtype=float))
        all_scores.extend(np.asarray(scores, dtype=float).tolist())

    return grouped_scores, np.asarray(all_scores, dtype=float)


def cache_validation_scores(model_bank, grouped_validation):
    """Her modelin validation surprise skorunu bir kez hesaplar."""
    for model_info in model_bank:
        grouped_scores, flat_scores = score_single_model_grouped(
            model_info["model"],
            grouped_validation,
        )
        model_info["validation_grouped_scores"] = grouped_scores
        model_info["validation_scores"] = flat_scores


def fast_select_threshold(labels, scores):
    """
    Tek-threshold için en iyi validation eşiğini sıralama ve kümülatif
    sayımlar kullanarak seçer. Her aday için baştan tahmin üretmez.
    """
    labels = np.asarray(labels, dtype=int).flatten()
    scores = np.asarray(scores, dtype=float).flatten()

    if len(labels) != len(scores) or len(scores) == 0:
        raise ValueError("Validation etiket ve skorları uygun değil.")

    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]

    total_positive = int(sorted_labels.sum())
    total_negative = int(len(sorted_labels) - total_positive)

    best = None

    # Hiç anomaly tahmin etmeyen aday.
    empty_threshold = float(sorted_scores[0] + 1e-12)
    empty_metrics = calculate_metrics(
        labels,
        np.zeros(len(labels), dtype=int),
    )
    best = {
        "key": (
            empty_metrics["f1_score"],
            empty_metrics["precision"],
            empty_metrics["recall"],
            0,
        ),
        "threshold": empty_threshold,
        "metrics": empty_metrics,
        "predicted_anomaly_count": 0,
    }

    true_positive = 0
    false_positive = 0
    index = 0

    while index < len(sorted_scores):
        score_value = sorted_scores[index]
        group_end = index

        while (
            group_end < len(sorted_scores)
            and sorted_scores[group_end] == score_value
        ):
            if sorted_labels[group_end] == 1:
                true_positive += 1
            else:
                false_positive += 1
            group_end += 1

        false_negative = total_positive - true_positive
        true_negative = total_negative - false_positive
        predicted_count = true_positive + false_positive

        precision = (
            true_positive / predicted_count
            if predicted_count > 0
            else 0.0
        )
        recall = (
            true_positive / total_positive
            if total_positive > 0
            else 0.0
        )
        f1_score = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )
        accuracy = (
            (true_positive + true_negative) / len(labels)
            if len(labels) > 0
            else 0.0
        )

        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1_score),
        }
        key = (
            metrics["f1_score"],
            metrics["precision"],
            metrics["recall"],
            -int(predicted_count),
        )

        if key > best["key"]:
            best = {
                "key": key,
                "threshold": float(score_value),
                "metrics": metrics,
                "predicted_anomaly_count": int(predicted_count),
            }

        index = group_end

    return best


def evaluate_model_pairs(normal_bank, anomaly_bank, validation_labels):
    """
    27 normal × 27 anomaly = 729 model çiftini validation üzerinde
    değerlendirir. Model fit işlemi burada tekrar yapılmaz.
    """
    pair_rows = []
    best_pair = None
    best_shared_pair = None

    for normal_info in normal_bank:
        for anomaly_info in anomaly_bank:
            dual_scores = (
                normal_info["validation_scores"]
                - anomaly_info["validation_scores"]
            )
            threshold_selection = fast_select_threshold(
                validation_labels,
                dual_scores,
            )
            metrics = threshold_selection["metrics"]
            total_states_after = (
                normal_info["statistics"]["merged_state_count"]
                + anomaly_info["statistics"]["merged_state_count"]
            )
            shared_parameters = (
                config_signature(normal_info["config"])
                == config_signature(anomaly_info["config"])
            )

            row = {
                "search_stage": "class_specific_model_pair",
                "normal_model_id": normal_info["model_id"],
                "anomaly_model_id": anomaly_info["model_id"],
                "shared_parameters": bool(shared_parameters),
                "normal_merge_alpha": normal_info["config"]["merge_alpha"],
                "normal_min_state_count": normal_info["config"][
                    "min_state_count"
                ],
                "normal_smoothing_alpha": normal_info["config"][
                    "smoothing_alpha"
                ],
                "anomaly_merge_alpha": anomaly_info["config"][
                    "merge_alpha"
                ],
                "anomaly_min_state_count": anomaly_info["config"][
                    "min_state_count"
                ],
                "anomaly_smoothing_alpha": anomaly_info["config"][
                    "smoothing_alpha"
                ],
                "normal_states_before": normal_info["statistics"][
                    "original_state_count"
                ],
                "normal_states_after": normal_info["statistics"][
                    "merged_state_count"
                ],
                "anomaly_states_before": anomaly_info["statistics"][
                    "original_state_count"
                ],
                "anomaly_states_after": anomaly_info["statistics"][
                    "merged_state_count"
                ],
                "total_states_after": total_states_after,
                "threshold": threshold_selection["threshold"],
                "predicted_anomaly_count": threshold_selection[
                    "predicted_anomaly_count"
                ],
                **{
                    f"validation_{key}": value
                    for key, value in metrics.items()
                },
            }
            pair_rows.append(row)

            key = (
                metrics["f1_score"],
                metrics["precision"],
                metrics["recall"],
                -total_states_after,
            )
            candidate = {
                "key": key,
                "normal_info": normal_info,
                "anomaly_info": anomaly_info,
                "threshold": threshold_selection["threshold"],
                "metrics": metrics,
                "shared_parameters": bool(shared_parameters),
            }

            if best_pair is None or key > best_pair["key"]:
                best_pair = candidate

            if shared_parameters and (
                best_shared_pair is None or key > best_shared_pair["key"]
            ):
                best_shared_pair = candidate

    return pair_rows, best_pair, best_shared_pair


def score_grouped_items(normal_model, anomaly_model, grouped_data):
    """Dual skorları source_file sınırlarını koruyarak döndürür."""
    scored_items = []

    for item in grouped_data:
        scores, logs = score_dual(
            normal_model,
            anomaly_model,
            item["patterns"],
        )
        labels = np.asarray(item["transition_labels"], dtype=int)

        if len(scores) != len(labels):
            raise ValueError(
                f"{item['source_file']} skor/etiket sayısı farklı: "
                f"{len(scores)}/{len(labels)}"
            )

        for log in logs:
            log["source_file"] = item["source_file"]

        scored_items.append(
            {
                "source_file": item["source_file"],
                "scores": np.asarray(scores, dtype=float),
                "labels": labels,
                "logs": logs,
            }
        )

    return scored_items


def flatten_scored_items(scored_items):
    scores = np.concatenate(
        [item["scores"] for item in scored_items]
    ) if scored_items else np.array([], dtype=float)
    labels = np.concatenate(
        [item["labels"] for item in scored_items]
    ) if scored_items else np.array([], dtype=int)
    logs = [
        log
        for item in scored_items
        for log in item["logs"]
    ]
    return scores, labels, logs


def apply_temporal_persistence(predictions, min_anomaly_run):
    """Kısa anomaly bloklarını source_file içinde kaldırır."""
    predictions = np.asarray(predictions, dtype=int).flatten()
    min_anomaly_run = int(min_anomaly_run)

    if min_anomaly_run < 1:
        raise ValueError("min_anomaly_run en az 1 olmalıdır.")

    if min_anomaly_run == 1 or len(predictions) == 0:
        return predictions.copy()

    filtered = np.zeros_like(predictions)
    start = 0

    while start < len(predictions):
        if predictions[start] == 0:
            start += 1
            continue

        end = start
        while end < len(predictions) and predictions[end] == 1:
            end += 1

        if end - start >= min_anomaly_run:
            filtered[start:end] = 1

        start = end

    return filtered


def predict_scored_items(scored_items, threshold, min_anomaly_run):
    """
    Threshold ve persistence işlemini her source_file için ayrı uygular.
    """
    all_predictions = []
    all_raw_predictions = []
    all_labels = []
    all_logs = []
    file_metrics = []

    for item in scored_items:
        raw_predictions = (
            item["scores"] >= threshold
        ).astype(int)
        predictions = apply_temporal_persistence(
            raw_predictions,
            min_anomaly_run,
        )

        metrics = calculate_metrics(item["labels"], predictions)
        file_metrics.append(
            {
                "source_file": item["source_file"],
                **metrics,
            }
        )

        for raw_prediction, prediction, log in zip(
            raw_predictions,
            predictions,
            item["logs"],
        ):
            log_copy = dict(log)
            log_copy["threshold"] = float(threshold)
            log_copy["min_anomaly_run"] = int(min_anomaly_run)
            log_copy["raw_decision"] = (
                "anomaly" if raw_prediction == 1 else "normal"
            )
            log_copy["decision"] = (
                "anomaly" if prediction == 1 else "normal"
            )
            all_logs.append(log_copy)

        all_predictions.extend(predictions.tolist())
        all_raw_predictions.extend(raw_predictions.tolist())
        all_labels.extend(item["labels"].tolist())

    return {
        "predictions": np.asarray(all_predictions, dtype=int),
        "raw_predictions": np.asarray(all_raw_predictions, dtype=int),
        "labels": np.asarray(all_labels, dtype=int),
        "logs": all_logs,
        "file_metrics": file_metrics,
    }


def summarize_file_stability(file_metrics):
    if not file_metrics:
        return {
            "mean_file_f1": 0.0,
            "min_file_f1": 0.0,
            "std_file_f1": 0.0,
        }

    f1_values = np.asarray(
        [row["f1_score"] for row in file_metrics],
        dtype=float,
    )
    return {
        "mean_file_f1": float(np.mean(f1_values)),
        "min_file_f1": float(np.min(f1_values)),
        "std_file_f1": float(np.std(f1_values)),
    }


def make_threshold_candidates(scores, quantile_count, extra_values=None):
    scores = np.asarray(scores, dtype=float).flatten()

    if len(scores) == 0:
        raise ValueError("Threshold adayı için skorlar boş olamaz.")

    quantile_count = max(int(quantile_count), 3)
    candidates = list(
        np.quantile(
            scores,
            np.linspace(0.0, 1.0, quantile_count),
        )
    )
    candidates.extend(
        [
            float(scores.min() - 1e-12),
            float(scores.max() + 1e-12),
        ]
    )

    if extra_values:
        candidates.extend(float(value) for value in extra_values)

    return np.asarray(sorted(set(candidates)), dtype=float)


def select_threshold_with_persistence(
    scored_items,
    min_anomaly_runs,
    min_validation_recall,
    threshold_quantile_count,
    extra_thresholds=None,
):
    """
    Threshold ve min_anomaly_run değerlerini yalnızca inner validation
    üzerinde seçer. Persistence her source_file içinde ayrı uygulanır.
    """
    scores, labels, _ = flatten_scored_items(scored_items)
    fast_baseline = fast_select_threshold(labels, scores)
    extras = [fast_baseline["threshold"]]

    if extra_thresholds:
        extras.extend(extra_thresholds)

    thresholds = make_threshold_candidates(
        scores,
        threshold_quantile_count,
        extra_values=extras,
    )
    run_values = sorted(set(int(value) for value in min_anomaly_runs))

    best_eligible = None
    best_fallback = None
    rows = []

    for threshold in thresholds:
        for min_anomaly_run in run_values:
            result = predict_scored_items(
                scored_items,
                threshold,
                min_anomaly_run,
            )
            metrics = calculate_metrics(
                result["labels"],
                result["predictions"],
            )
            stability = summarize_file_stability(
                result["file_metrics"]
            )
            eligible = (
                metrics["recall"] >= min_validation_recall
            )

            row = {
                "search_stage": "threshold_temporal_persistence",
                "threshold": float(threshold),
                "min_anomaly_run": int(min_anomaly_run),
                "recall_constraint_met": bool(eligible),
                "raw_predicted_anomaly_count": int(
                    result["raw_predictions"].sum()
                ),
                "predicted_anomaly_count": int(
                    result["predictions"].sum()
                ),
                **{
                    f"validation_{key}": value
                    for key, value in metrics.items()
                },
                **{
                    f"validation_{key}": value
                    for key, value in stability.items()
                },
            }
            rows.append(row)

            candidate = {
                "threshold": float(threshold),
                "min_anomaly_run": int(min_anomaly_run),
                "metrics": metrics,
                "stability": stability,
                "result": result,
                "recall_constraint_met": bool(eligible),
            }

            eligible_key = (
                metrics["f1_score"],
                metrics["precision"],
                metrics["recall"],
                stability["min_file_f1"],
                stability["mean_file_f1"],
                -stability["std_file_f1"],
                -int(result["predictions"].sum()),
                -int(min_anomaly_run),
            )
            fallback_key = (
                metrics["recall"],
                metrics["f1_score"],
                metrics["precision"],
                stability["min_file_f1"],
                stability["mean_file_f1"],
                -stability["std_file_f1"],
                -int(result["predictions"].sum()),
                -int(min_anomaly_run),
            )

            if eligible and (
                best_eligible is None
                or eligible_key > best_eligible["key"]
            ):
                best_eligible = {
                    **candidate,
                    "key": eligible_key,
                    "selection_mode": "recall_constrained",
                }

            if (
                best_fallback is None
                or fallback_key > best_fallback["key"]
            ):
                best_fallback = {
                    **candidate,
                    "key": fallback_key,
                    "selection_mode": "fallback_highest_recall",
                }

    return (
        best_eligible if best_eligible is not None else best_fallback,
        rows,
    )


def scan_class_specific_distance_penalties(
    normal_model,
    anomaly_model,
    grouped_validation,
    config,
):
    """
    Seçilen model çiftinde normal ve anomaly Levenshtein cezalarını
    birbirinden bağımsız olarak validation üzerinde tarar.
    """
    rows = []
    best_eligible = None
    best_fallback = None

    for normal_penalty, anomaly_penalty in product(
        config["distance_penalties"],
        config["distance_penalties"],
    ):
        normal_model.distance_penalty = float(normal_penalty)
        anomaly_model.distance_penalty = float(anomaly_penalty)

        scored_items = score_grouped_items(
            normal_model,
            anomaly_model,
            grouped_validation,
        )
        scores, labels, _ = flatten_scored_items(scored_items)
        threshold_selection = fast_select_threshold(labels, scores)
        metrics = threshold_selection["metrics"]
        eligible = (
            metrics["recall"] >= config["min_validation_recall"]
        )

        row = {
            "search_stage": "class_specific_distance_penalty",
            "normal_distance_penalty": float(normal_penalty),
            "anomaly_distance_penalty": float(anomaly_penalty),
            "threshold": threshold_selection["threshold"],
            "recall_constraint_met": bool(eligible),
            "predicted_anomaly_count": threshold_selection[
                "predicted_anomaly_count"
            ],
            **{
                f"validation_{key}": value
                for key, value in metrics.items()
            },
        }
        rows.append(row)

        candidate = {
            "normal_distance_penalty": float(normal_penalty),
            "anomaly_distance_penalty": float(anomaly_penalty),
            "threshold": threshold_selection["threshold"],
            "metrics": metrics,
            "scored_items": scored_items,
            "recall_constraint_met": bool(eligible),
        }
        eligible_key = (
            metrics["f1_score"],
            metrics["precision"],
            metrics["recall"],
            -threshold_selection["predicted_anomaly_count"],
        )
        fallback_key = (
            metrics["recall"],
            metrics["f1_score"],
            metrics["precision"],
            -threshold_selection["predicted_anomaly_count"],
        )

        if eligible and (
            best_eligible is None
            or eligible_key > best_eligible["key"]
        ):
            best_eligible = {
                **candidate,
                "key": eligible_key,
                "selection_mode": "recall_constrained",
            }

        if (
            best_fallback is None
            or fallback_key > best_fallback["key"]
        ):
            best_fallback = {
                **candidate,
                "key": fallback_key,
                "selection_mode": "fallback_highest_recall",
            }

    return (
        best_eligible if best_eligible is not None else best_fallback,
        rows,
    )


def make_result_row(
    fold_id,
    seed,
    scenario,
    metrics,
    best,
    split_info,
    sample_count,
    anomaly_count,
    unseen_count,
    test_file_count,
):
    return {
        "dataset": "SKAB",
        "model": "class_specific_dual_alergia",
        "fold": int(fold_id),
        "seed": int(seed),
        "scenario": scenario,
        "window_size": best["window_size"],
        "alphabet_size": best["alphabet_size"],
        "normal_merge_alpha": best["normal_config"]["merge_alpha"],
        "normal_min_state_count": best["normal_config"][
            "min_state_count"
        ],
        "normal_smoothing_alpha": best["normal_config"][
            "smoothing_alpha"
        ],
        "anomaly_merge_alpha": best["anomaly_config"]["merge_alpha"],
        "anomaly_min_state_count": best["anomaly_config"][
            "min_state_count"
        ],
        "anomaly_smoothing_alpha": best["anomaly_config"][
            "smoothing_alpha"
        ],
        "normal_distance_penalty": best["normal_distance_penalty"],
        "anomaly_distance_penalty": best["anomaly_distance_penalty"],
        "selected_threshold": best["threshold"],
        "min_anomaly_run": best["min_anomaly_run"],
        "validation_f1": best["metrics"]["f1_score"],
        "validation_recall": best["metrics"]["recall"],
        "validation_recall_constraint_met": best[
            "recall_constraint_met"
        ],
        "selected_pair_shared_parameters": best[
            "selected_pair_shared_parameters"
        ],
        "final_training_scope": "inner_train_calibrated",
        "inner_splitter": split_info["splitter"],
        "inner_split_id": split_info["split_id"],
        "inner_train_file_count": len(split_info["train_groups"]),
        "inner_validation_file_count": len(split_info["val_groups"]),
        "test_file_count": int(test_file_count),
        "sample_count": int(sample_count),
        "actual_anomaly_count": int(anomaly_count),
        "unseen_count": int(unseen_count),
        **best["structure_stats"],
        **metrics,
    }


def run_fold_seed(fold_id, seed, config, transformer):
    print("\n" + "=" * 84)
    print(
        f"SKAB FOLD {fold_id} | SEED {seed} | "
        "CLASS-SPECIFIC DUAL ALERGIA"
    )
    print("=" * 84)

    paths = {
        "X_train": f"data/processed/skab_fold{fold_id}_X_train_pc1.csv",
        "y_train": f"data/processed/skab_fold{fold_id}_y_train.csv",
        "train_source": (
            f"data/processed/skab_fold{fold_id}_train_source_file.csv"
        ),
        "X_test": f"data/processed/skab_fold{fold_id}_X_test_pc1.csv",
        "y_test": f"data/processed/skab_fold{fold_id}_y_test.csv",
        "test_source": (
            f"data/processed/skab_fold{fold_id}_test_source_file.csv"
        ),
    }

    X_train = np.asarray(load_vector(paths["X_train"]), dtype=float)
    y_train = to_binary(load_vector(paths["y_train"]))
    train_source = load_source_files(paths["train_source"])
    X_test = np.asarray(load_vector(paths["X_test"]), dtype=float)
    y_test = to_binary(load_vector(paths["y_test"]))
    test_source = load_source_files(paths["test_source"])

    if not (len(X_train) == len(y_train) == len(train_source)):
        raise ValueError(
            f"Fold {fold_id} train X/y/source farklı: "
            f"{len(X_train)}/{len(y_train)}/{len(train_source)}"
        )
    if not (len(X_test) == len(y_test) == len(test_source)):
        raise ValueError(
            f"Fold {fold_id} test X/y/source farklı: "
            f"{len(X_test)}/{len(y_test)}/{len(test_source)}"
        )

    train_files = set(train_source)
    test_files = set(test_source)
    overlap = train_files.intersection(test_files)
    if overlap:
        raise RuntimeError(
            f"Fold {fold_id} outer train/test source_file çakışması: "
            f"{overlap}"
        )

    print(
        f"Train X/y/source: {len(X_train)}/{len(y_train)}/"
        f"{len(train_source)} | Test: {len(X_test)}/{len(y_test)}/"
        f"{len(test_source)}"
    )
    print(
        f"Train source_file={len(train_files)} | "
        f"Test source_file={len(test_files)}"
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

    print(
        f"Inner {split_info['splitter']} split={split_info['split_id']} | "
        f"train file={len(split_info['train_groups'])} | "
        f"validation file={len(split_info['val_groups'])} | "
        f"validation transition={len(split_info['val_labels'])} | "
        f"anomaly={int(split_info['val_labels'].sum())}"
    )

    model_configs = create_model_configs(config)
    print(
        f"\nModel adayı: normal={len(model_configs)} | "
        f"anomaly={len(model_configs)} | "
        f"çift={len(model_configs) ** 2}"
    )

    print("\n--- NORMAL MODEL BANKASI EĞİTİLİYOR ---")
    normal_bank = fit_model_bank(
        split_info["normal_sequences"],
        model_configs,
        "normal",
    )
    print("\n--- ANOMALY MODEL BANKASI EĞİTİLİYOR ---")
    anomaly_bank = fit_model_bank(
        split_info["anomaly_sequences"],
        model_configs,
        "anomaly",
    )

    print("\n--- MODEL SKORLARI ÖNBELLEĞE ALINIYOR ---")
    cache_validation_scores(normal_bank, split_info["grouped_val"])
    cache_validation_scores(anomaly_bank, split_info["grouped_val"])

    print("\n--- 27 × 27 MODEL ÇİFTİ VALIDATION TARAMASI ---")
    pair_rows, best_pair, best_shared_pair = evaluate_model_pairs(
        normal_bank,
        anomaly_bank,
        split_info["val_labels"],
    )

    pair_rows_sorted = sorted(
        pair_rows,
        key=lambda row: (
            row["validation_f1_score"],
            row["validation_precision"],
            row["validation_recall"],
            -row["total_states_after"],
        ),
        reverse=True,
    )

    print("Validation en iyi 5 model çifti:")
    for row in pair_rows_sorted[:5]:
        print(
            f"N(alpha={row['normal_merge_alpha']}, "
            f"min={row['normal_min_state_count']}, "
            f"smooth={row['normal_smoothing_alpha']}) | "
            f"A(alpha={row['anomaly_merge_alpha']}, "
            f"min={row['anomaly_min_state_count']}, "
            f"smooth={row['anomaly_smoothing_alpha']}) | "
            f"shared={row['shared_parameters']} | "
            f"P={row['validation_precision']:.4f} | "
            f"R={row['validation_recall']:.4f} | "
            f"F1={row['validation_f1_score']:.4f}"
        )

    print("\nEn iyi ortak-parametre çifti:")
    print(
        f"alpha={best_shared_pair['normal_info']['config']['merge_alpha']} | "
        f"min={best_shared_pair['normal_info']['config']['min_state_count']} | "
        f"smooth={best_shared_pair['normal_info']['config']['smoothing_alpha']} | "
        f"val F1={best_shared_pair['metrics']['f1_score']:.4f}"
    )

    normal_model = best_pair["normal_info"]["model"]
    anomaly_model = best_pair["anomaly_info"]["model"]

    print("\n--- CLASS-SPECIFIC LEVENSHTEIN CEZASI TARAMASI ---")
    distance_best, distance_rows = scan_class_specific_distance_penalties(
        normal_model,
        anomaly_model,
        split_info["grouped_val"],
        config,
    )

    normal_model.distance_penalty = distance_best[
        "normal_distance_penalty"
    ]
    anomaly_model.distance_penalty = distance_best[
        "anomaly_distance_penalty"
    ]

    print(
        f"Distance seçimi -> normal={distance_best['normal_distance_penalty']} | "
        f"anomaly={distance_best['anomaly_distance_penalty']} | "
        f"threshold={distance_best['threshold']:.6f} | "
        f"P={distance_best['metrics']['precision']:.4f} | "
        f"R={distance_best['metrics']['recall']:.4f} | "
        f"F1={distance_best['metrics']['f1_score']:.4f}"
    )

    print("\n--- THRESHOLD + TEMPORAL PERSISTENCE TARAMASI ---")
    selected_scored_validation = score_grouped_items(
        normal_model,
        anomaly_model,
        split_info["grouped_val"],
    )
    persistence_best, persistence_rows = (
        select_threshold_with_persistence(
            selected_scored_validation,
            config["min_anomaly_runs"],
            config["min_validation_recall"],
            config["threshold_quantile_count"],
            extra_thresholds=[distance_best["threshold"]],
        )
    )

    structure_stats = combine_statistics(normal_model, anomaly_model)
    selected_pair_shared = best_pair["shared_parameters"]

    best = {
        "window_size": config["window_size"],
        "alphabet_size": config["alphabet_size"],
        "normal_config": best_pair["normal_info"]["config"],
        "anomaly_config": best_pair["anomaly_info"]["config"],
        "normal_distance_penalty": distance_best[
            "normal_distance_penalty"
        ],
        "anomaly_distance_penalty": distance_best[
            "anomaly_distance_penalty"
        ],
        "threshold": persistence_best["threshold"],
        "min_anomaly_run": persistence_best["min_anomaly_run"],
        "metrics": persistence_best["metrics"],
        "stability": persistence_best["stability"],
        "recall_constraint_met": persistence_best[
            "recall_constraint_met"
        ],
        "selection_mode": persistence_best["selection_mode"],
        "selected_pair_shared_parameters": selected_pair_shared,
        "structure_stats": structure_stats,
    }

    print("\n--- FOLD/SEED İÇİN EN İYİ CLASS-SPECIFIC AYAR ---")
    print(
        "Normal -> "
        f"alpha={best['normal_config']['merge_alpha']} | "
        f"min={best['normal_config']['min_state_count']} | "
        f"smooth={best['normal_config']['smoothing_alpha']} | "
        f"distance={best['normal_distance_penalty']}"
    )
    print(
        "Anomaly -> "
        f"alpha={best['anomaly_config']['merge_alpha']} | "
        f"min={best['anomaly_config']['min_state_count']} | "
        f"smooth={best['anomaly_config']['smoothing_alpha']} | "
        f"distance={best['anomaly_distance_penalty']}"
    )
    print(
        f"threshold={best['threshold']:.6f} | "
        f"min_run={best['min_anomaly_run']} | "
        f"P={best['metrics']['precision']:.4f} | "
        f"R={best['metrics']['recall']:.4f} | "
        f"F1={best['metrics']['f1_score']:.4f} | "
        f"constraint_met={best['recall_constraint_met']} | "
        f"pair_shared={selected_pair_shared}"
    )

    # Mevcut SKAB runner ile aynı şekilde threshold ölçeğini korumak için
    # inner-train üzerinde eğitilmiş modeller outer testte kullanılır.
    final_train_mean = split_info["train_mean"]
    final_train_std = split_info["train_std"]

    grouped_test = prepare_grouped_patterns(
        X_test,
        y_test,
        test_source,
        transformer,
        config["window_size"],
        final_train_mean,
        final_train_std,
    )
    test_scored_items = score_grouped_items(
        normal_model,
        anomaly_model,
        grouped_test,
    )
    test_result = predict_scored_items(
        test_scored_items,
        best["threshold"],
        best["min_anomaly_run"],
    )
    original_metrics = calculate_metrics(
        test_result["labels"],
        test_result["predictions"],
    )
    original_unseen_count = sum(
        log["overall_status"] == "unseen"
        for log in test_result["logs"]
    )

    noisy_test = inject_noise(X_test, config["noise_level"], seed)
    grouped_noisy_test = prepare_grouped_patterns(
        noisy_test,
        y_test,
        test_source,
        transformer,
        config["window_size"],
        final_train_mean,
        final_train_std,
    )
    noisy_scored_items = score_grouped_items(
        normal_model,
        anomaly_model,
        grouped_noisy_test,
    )
    noisy_result = predict_scored_items(
        noisy_scored_items,
        best["threshold"],
        best["min_anomaly_run"],
    )
    noise_metrics = calculate_metrics(
        noisy_result["labels"],
        noisy_result["predictions"],
    )
    noise_unseen_count = sum(
        log["overall_status"] == "unseen"
        for log in noisy_result["logs"]
    )

    unseen = evaluate_unseen(
        test_result["predictions"],
        test_result["labels"],
        test_result["logs"],
    )

    metric_rows = [
        make_result_row(
            fold_id,
            seed,
            "original",
            original_metrics,
            best,
            split_info,
            len(test_result["labels"]),
            int(test_result["labels"].sum()),
            original_unseen_count,
            len(test_files),
        ),
        make_result_row(
            fold_id,
            seed,
            "gaussian_noise",
            noise_metrics,
            best,
            split_info,
            len(noisy_result["labels"]),
            int(noisy_result["labels"].sum()),
            noise_unseen_count,
            len(test_files),
        ),
        make_result_row(
            fold_id,
            seed,
            "unseen_data",
            unseen["metrics"],
            best,
            split_info,
            unseen["count"],
            unseen["anomaly_count"],
            unseen["count"],
            len(test_files),
        ),
    ]

    common_validation_fields = {
        "dataset": "SKAB",
        "fold": int(fold_id),
        "seed": int(seed),
        "inner_splitter": split_info["splitter"],
        "inner_split_id": split_info["split_id"],
        "validation_sample_count": len(split_info["val_labels"]),
        "validation_anomaly_count": int(split_info["val_labels"].sum()),
    }

    for row in pair_rows:
        row.update(common_validation_fields)

    for row in distance_rows:
        row.update(
            {
                **common_validation_fields,
                "normal_merge_alpha": best_pair["normal_info"]["config"][
                    "merge_alpha"
                ],
                "normal_min_state_count": best_pair["normal_info"][
                    "config"
                ]["min_state_count"],
                "normal_smoothing_alpha": best_pair["normal_info"][
                    "config"
                ]["smoothing_alpha"],
                "anomaly_merge_alpha": best_pair["anomaly_info"][
                    "config"
                ]["merge_alpha"],
                "anomaly_min_state_count": best_pair["anomaly_info"][
                    "config"
                ]["min_state_count"],
                "anomaly_smoothing_alpha": best_pair["anomaly_info"][
                    "config"
                ]["smoothing_alpha"],
            }
        )

    for row in persistence_rows:
        row.update(
            {
                **common_validation_fields,
                "normal_merge_alpha": best["normal_config"]["merge_alpha"],
                "normal_min_state_count": best["normal_config"][
                    "min_state_count"
                ],
                "normal_smoothing_alpha": best["normal_config"][
                    "smoothing_alpha"
                ],
                "anomaly_merge_alpha": best["anomaly_config"][
                    "merge_alpha"
                ],
                "anomaly_min_state_count": best["anomaly_config"][
                    "min_state_count"
                ],
                "anomaly_smoothing_alpha": best["anomaly_config"][
                    "smoothing_alpha"
                ],
                "normal_distance_penalty": best[
                    "normal_distance_penalty"
                ],
                "anomaly_distance_penalty": best[
                    "anomaly_distance_penalty"
                ],
            }
        )

    best_config_row = {
        "dataset": "SKAB",
        "fold": int(fold_id),
        "seed": int(seed),
        "window_size": config["window_size"],
        "alphabet_size": config["alphabet_size"],
        "normal_merge_alpha": best["normal_config"]["merge_alpha"],
        "normal_min_state_count": best["normal_config"]["min_state_count"],
        "normal_smoothing_alpha": best["normal_config"][
            "smoothing_alpha"
        ],
        "anomaly_merge_alpha": best["anomaly_config"]["merge_alpha"],
        "anomaly_min_state_count": best["anomaly_config"][
            "min_state_count"
        ],
        "anomaly_smoothing_alpha": best["anomaly_config"][
            "smoothing_alpha"
        ],
        "normal_distance_penalty": best["normal_distance_penalty"],
        "anomaly_distance_penalty": best["anomaly_distance_penalty"],
        "selected_threshold": best["threshold"],
        "min_anomaly_run": best["min_anomaly_run"],
        "selected_pair_shared_parameters": selected_pair_shared,
        "best_shared_pair_validation_f1": best_shared_pair["metrics"][
            "f1_score"
        ],
        "best_any_pair_validation_f1": best_pair["metrics"]["f1_score"],
        "final_validation_f1": best["metrics"]["f1_score"],
        "final_validation_precision": best["metrics"]["precision"],
        "final_validation_recall": best["metrics"]["recall"],
        "validation_recall_constraint_met": best[
            "recall_constraint_met"
        ],
        "validation_selection_mode": best["selection_mode"],
        "validation_mean_file_f1": best["stability"]["mean_file_f1"],
        "validation_min_file_f1": best["stability"]["min_file_f1"],
        "validation_std_file_f1": best["stability"]["std_file_f1"],
        "inner_splitter": split_info["splitter"],
        "inner_split_id": split_info["split_id"],
        "inner_train_files": "|".join(split_info["train_groups"]),
        "inner_validation_files": "|".join(split_info["val_groups"]),
        **structure_stats,
    }

    print("\n--- FOLD TEST SONUÇLARI ---")
    print(
        f"Original -> Precision={original_metrics['precision']:.4f} | "
        f"Recall={original_metrics['recall']:.4f} | "
        f"F1={original_metrics['f1_score']:.4f}"
    )
    print(
        f"Gaussian Noise -> Precision={noise_metrics['precision']:.4f} | "
        f"Recall={noise_metrics['recall']:.4f} | "
        f"F1={noise_metrics['f1_score']:.4f}"
    )
    print(
        f"Unseen -> n={unseen['count']} | "
        f"F1={unseen['metrics']['f1_score']:.4f}"
    )

    return {
        "metrics_rows": metric_rows,
        "pair_rows": pair_rows,
        "post_selection_rows": distance_rows + persistence_rows,
        "best_config_row": best_config_row,
        "explainability": {
            "original": test_result["logs"][:100],
            "gaussian_noise": noisy_result["logs"][:100],
        },
    }


def create_summary(metrics_df):
    columns = ["accuracy", "precision", "recall", "f1_score", "unseen_count"]
    summary = metrics_df.groupby("scenario")[columns].agg(["mean", "std"])
    summary.columns = [
        f"{column}_{stat}" for column, stat in summary.columns
    ]
    return summary.reset_index()


def create_fold_summary(metrics_df):
    columns = ["accuracy", "precision", "recall", "f1_score", "unseen_count"]
    summary = metrics_df.groupby(["fold", "scenario"])[columns].agg(
        ["mean", "std"]
    )
    summary.columns = [
        f"{column}_{stat}" for column, stat in summary.columns
    ]
    return summary.reset_index()


def main():
    print(
        "\n--- SKAB 5-FOLD × 5-SEED CLASS-SPECIFIC "
        "DUAL ALERGIA DENEYİ BAŞLATILIYOR ---"
    )

    config = load_config()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    transformer = SaxPaaTransformer(
        alphabet_size=config["alphabet_size"]
    )

    all_metric_rows = []
    all_pair_rows = []
    all_post_selection_rows = []
    all_best_config_rows = []
    explainability = {}

    for fold_id in range(1, config["n_folds"] + 1):
        for seed in config["seeds"]:
            result = run_fold_seed(
                fold_id,
                seed,
                config,
                transformer,
            )
            all_metric_rows.extend(result["metrics_rows"])
            all_pair_rows.extend(result["pair_rows"])
            all_post_selection_rows.extend(
                result["post_selection_rows"]
            )
            all_best_config_rows.append(result["best_config_row"])
            explainability[f"fold_{fold_id}_seed_{seed}"] = result[
                "explainability"
            ]

    metrics_df = pd.DataFrame(all_metric_rows)
    pair_df = pd.DataFrame(all_pair_rows)
    post_selection_df = pd.DataFrame(all_post_selection_rows)
    best_config_df = pd.DataFrame(all_best_config_rows)
    fold_summary_df = create_fold_summary(metrics_df)
    overall_summary_df = create_summary(metrics_df)

    metrics_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "skab_class_specific_dual_alergia_fold_seed_metrics.csv",
        ),
        index=False,
    )
    pair_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "skab_class_specific_dual_alergia_pair_validation_search.csv",
        ),
        index=False,
    )
    post_selection_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "skab_class_specific_dual_alergia_post_selection_search.csv",
        ),
        index=False,
    )
    best_config_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "skab_class_specific_dual_alergia_best_configs.csv",
        ),
        index=False,
    )
    fold_summary_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "skab_class_specific_dual_alergia_fold_summary.csv",
        ),
        index=False,
    )
    overall_summary_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "skab_class_specific_dual_alergia_overall_summary.csv",
        ),
        index=False,
    )

    with open(
        os.path.join(
            OUTPUT_DIR,
            "skab_class_specific_dual_alergia_explainability.json",
        ),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(explainability, file, indent=4, ensure_ascii=False)

    print("\n--- SKAB CLASS-SPECIFIC GENEL ÖZET ---")
    print(overall_summary_df.to_string(index=False))
    print("\nSonuç dosyaları results/outputs klasörüne kaydedildi.")


if __name__ == "__main__":
    main()
