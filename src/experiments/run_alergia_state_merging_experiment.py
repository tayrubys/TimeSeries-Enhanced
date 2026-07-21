import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data_pipeline.sax_paa import SaxPaaTransformer
from src.experiments.evaluator import calculate_metrics
from src.models.alergia_state_merging import AlergiaStateMergingAutomata


CONFIG_PATH = "src/config/settings.json"
OUTPUT_DIR = "results/outputs"

#csv dosyalarını numpy dizisine cevirir
def load_vector(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dosya bulunamadı: {path}")
    return pd.read_csv(path).values.flatten()

#-999 u 0, 0 dan buyuklerı 1
def to_binary(labels):
    labels = np.asarray(labels).flatten()
    labels = np.where(labels == -999, 0, labels)
    return np.where(labels > 0, 1, 0).astype(int)


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        automata = json.load(file).get("automata", {})

    return {
        "window_size": int(automata.get("window_size", 4)),
        "alphabet_size": int(automata.get("alphabet_size", 3)),
        "noise_level": float(automata.get("noise_level", 0.1)),
        "seed": int(automata.get("seeds", [42])[0]),
        "merge_alphas": automata.get(
            "alergia_merge_alpha_values", [0.01, 0.05, 0.1]
        ),
        "min_counts": automata.get(
            "alergia_min_state_count_values", [2, 5, 10]
        ),
        "smoothing_alphas": automata.get(
            "alergia_smoothing_alpha_values", [0.1, 0.5, 1.0]
        ),
        "max_pattern_distance": automata.get(
            "alergia_max_pattern_distance", None
        ),
        "distance_penalties": automata.get(
            "alergia_distance_penalty_values",
            [0.0, 0.25, 0.5, 1.0, 2.0],
        ),
        "min_anomaly_runs": [
            int(value)
            for value in automata.get(
                "alergia_min_anomaly_run_values",
                [1, 2, 3, 4, 5, 6, 7, 8, 10, 12],
            )
        ],
        "validation_block_count": int(
            automata.get(
                "alergia_validation_block_count",
                3,
            )
        ),
        "min_validation_recall": float(
            automata.get(
                "alergia_min_validation_recall",
                0.70,
            )
        ),
    }

#sax patternlerine cevirir
def transform_patterns(transformer, series, window_size, train_mean, train_std):
    # Her bölümü train istatistikleriyle normalize ediyoruz.
    normalized = (np.asarray(series, dtype=float) - train_mean) / train_std
    paa = transformer.apply_paa(normalized, window_size)
    sax = transformer.convert_to_sax(paa)
    return transformer.get_sliding_windows(sax, window_size)


def align_labels(labels, window_size):
    #Ham etiketleri önce PAA bloğuna, sonra pattern seviyesine taşır
    labels = to_binary(labels)
    usable = (len(labels) // window_size) * window_size

    if usable == 0:
        return np.array([], dtype=int)

    paa_labels = labels[:usable].reshape(-1, window_size).max(axis=1)
    pattern_count = len(paa_labels) - window_size + 1

    return np.asarray(
        [
            int(paa_labels[i : i + window_size].max())
            for i in range(max(pattern_count, 0))
        ],
        dtype=int,
    )


def build_class_sequences(patterns, transition_labels):
    """
    Her eğitim geçişini bağımsız iki pattern'lık sequence yapar.
    Bu model fit sırasında yalnızca komşu geçiş sayılarını kullandığı için
    uzun sequence kurmak ek hafıza oluşturmaz. Bağımsız çiftler ayrıca
    sınıf değişimlerinde sequence'lerin birbirine bağlanmasını engeller.
    """
    transition_labels = np.asarray(transition_labels, dtype=int)

    if len(patterns) - 1 != len(transition_labels):
        raise ValueError(
            "Pattern ve geçiş etiketi uzunlukları uyuşmuyor."
        )
    
    normal_sequences = []
    anomaly_sequences = []

    for index, label in enumerate(transition_labels):
        sequence = [patterns[index], patterns[index + 1]]

        if int(label) == 1:
            anomaly_sequences.append(sequence)
        else:
            normal_sequences.append(sequence)

    return normal_sequences, anomaly_sequences


def score_dual(normal_model, anomaly_model, patterns):
    #gelen veri için hem normal hem anomali modelınden supriz skoru alır
    normal_scores, normal_logs = normal_model.score_patterns(patterns)
    anomaly_scores, anomaly_logs = anomaly_model.score_patterns(patterns)

    if len(normal_scores) != len(anomaly_scores):
        raise ValueError("Normal ve anomaly skor sayıları farklı.")

    #yüksek değer anomaly modelinin geçişi daha iyi açıkladığını gösterir
    dual_scores = normal_scores - anomaly_scores
    trained_union = normal_model.trained_patterns | anomaly_model.trained_patterns
    logs = []

    for index, score in enumerate(dual_scores):
        incoming = normal_logs[index]["incoming_pattern"]

        logs.append(
            {
                "time_step": int(normal_logs[index]["time_step"]),
                "incoming_pattern": incoming,
                "overall_status": (
                    "seen" if incoming in trained_union else "unseen"
                ),
                "normal_mapped_to": normal_logs[index]["mapped_to"],
                "normal_probability": normal_logs[index][
                    "transition_probability"
                ],
                "normal_levenshtein_distance": normal_logs[index][
                    "levenshtein_distance"
                ],
                "normal_mapping_penalty": normal_logs[index][
                    "mapping_penalty"
                ],
                "normal_surprise": float(normal_scores[index]),
                "anomaly_mapped_to": anomaly_logs[index]["mapped_to"],
                "anomaly_probability": anomaly_logs[index][
                    "transition_probability"
                ],
                "anomaly_levenshtein_distance": anomaly_logs[index][
                    "levenshtein_distance"
                ],
                "anomaly_mapping_penalty": anomaly_logs[index][
                    "mapping_penalty"
                ],
                "anomaly_surprise": float(anomaly_scores[index]),
                "dual_score": float(score),
            }
        )

    return dual_scores, logs


def apply_temporal_persistence(predictions, min_anomaly_run):
    """
    Yalnızca en az min_anomaly_run uzunluğundaki ardışık anomaly
    tahminlerini korur. Daha kısa anomaly blokları normal yapılır.
    """
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


#belirli bir esik değere gore sonucların genel istatistiklerini hesaplar
def summarize_scores(
    labels,
    scores,
    logs,
    threshold,
    predictions=None,
):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    raw_predictions = (scores >= threshold).astype(int)

    if predictions is None:
        predictions = raw_predictions
    else:
        predictions = np.asarray(predictions, dtype=int).flatten()

    if len(predictions) != len(labels):
        raise ValueError(
            "Filtrelenmiş tahmin ve validation etiketi sayıları farklı."
        )

    normal_scores = scores[labels == 0]
    anomaly_scores = scores[labels == 1]

    return {
        "validation_unique_score_count": int(len(np.unique(scores))),
        "validation_raw_predicted_anomaly_count": int(
            raw_predictions.sum()
        ),
        "validation_predicted_anomaly_count": int(predictions.sum()),
        "validation_normal_score_mean": (
            float(np.mean(normal_scores)) if len(normal_scores) else 0.0
        ),
        "validation_anomaly_score_mean": (
            float(np.mean(anomaly_scores)) if len(anomaly_scores) else 0.0
        ),
        "validation_score_mean_gap": (
            float(np.mean(anomaly_scores) - np.mean(normal_scores))
            if len(normal_scores) and len(anomaly_scores)
            else 0.0
        ),
        "validation_mapping_disagreement_count": int(
            sum(
                log["normal_mapped_to"] != log["anomaly_mapped_to"]
                for log in logs
            )
        ),
        "validation_normal_distance_mean": float(
            np.mean(
                [log["normal_levenshtein_distance"] for log in logs]
            )
        ) if logs else 0.0,
        "validation_anomaly_distance_mean": float(
            np.mean(
                [log["anomaly_levenshtein_distance"] for log in logs]
            )
        ) if logs else 0.0,
    }

#yeni gelen datada belirlenen en ıyı esige gore anomalı normal tahmını yapar
def predict_dual(
    normal_model,
    anomaly_model,
    patterns,
    threshold,
    min_anomaly_run,
):
    scores, logs = score_dual(
        normal_model,
        anomaly_model,
        patterns,
    )

    raw_predictions = (scores >= threshold).astype(int)
    predictions = apply_temporal_persistence(
        raw_predictions,
        min_anomaly_run,
    )

    for raw_prediction, prediction, log in zip(
        raw_predictions,
        predictions,
        logs,
    ):
        log["threshold"] = float(threshold)
        log["min_anomaly_run"] = int(min_anomaly_run)
        log["raw_decision"] = (
            "anomaly" if raw_prediction == 1 else "normal"
        )
        log["decision"] = (
            "anomaly" if prediction == 1 else "normal"
        )

    return predictions, logs

#en iyi threshold değerini arar
def select_threshold(labels, scores):
    candidates = np.concatenate(
        ([scores.min() - 1e-12], np.unique(scores), [scores.max() + 1e-12])
    )
    best = None

    for threshold in candidates:
        predictions = (scores >= threshold).astype(int)
        metrics = calculate_metrics(labels, predictions)
        key = (
            metrics["f1_score"],
            metrics["precision"],
            metrics["recall"],
            -int(predictions.sum()),
        )

        if best is None or key > best["key"]:
            best = {
                "key": key,
                "threshold": float(threshold),
                "metrics": metrics,
            }

    return best["threshold"], best["metrics"]


def build_validation_blocks(labels, requested_block_count):
    """
    Validation verisini etiketlere bakmadan eşit kronolojik bloklara ayırır.

    Blok sınırları yalnızca örnek sırasına ve toplam örnek sayısına göre
    belirlenir. Böylece anomaly etiketleri validation yapısını şekillendirmez.
    """
    labels = np.asarray(labels, dtype=int).flatten()

    if len(labels) == 0:
        raise ValueError("Validation etiketleri boş olamaz.")

    requested_block_count = int(requested_block_count)

    if requested_block_count < 2:
        raise ValueError(
            "validation_block_count en az 2 olmalıdır."
        )

    effective_count = min(
        requested_block_count,
        len(labels),
    )

    return [
        np.asarray(block, dtype=int)
        for block in np.array_split(
            np.arange(len(labels)),
            effective_count,
        )
        if len(block) > 0
    ]


def calculate_block_stability(
    labels,
    predictions,
    block_count,
):
    labels = np.asarray(labels, dtype=int).flatten()
    predictions = np.asarray(
        predictions,
        dtype=int,
    ).flatten()

    if len(labels) != len(predictions):
        raise ValueError(
            "Validation etiketi ve tahmin sayıları farklı."
        )

    blocks = build_validation_blocks(
        labels,
        block_count,
    )

    block_rows = []
    block_f1_values = []

    for block_id, indices in enumerate(blocks, start=1):
        block_labels = labels[indices]
        block_predictions = predictions[indices]

        block_metrics = calculate_metrics(
            block_labels,
            block_predictions,
        )

        block_f1 = float(
            block_metrics["f1_score"]
        )
        block_f1_values.append(block_f1)

        block_rows.append(
            {
                "block": block_id,
                "start_index": int(indices[0]),
                "end_index": int(indices[-1]),
                "sample_count": int(len(indices)),
                "anomaly_count": int(
                    block_labels.sum()
                ),
                "predicted_anomaly_count": int(
                    block_predictions.sum()
                ),
                "precision": float(
                    block_metrics["precision"]
                ),
                "recall": float(
                    block_metrics["recall"]
                ),
                "f1_score": block_f1,
            }
        )

    block_f1_values = np.asarray(
        block_f1_values,
        dtype=float,
    )

    return {
        "block_rows": block_rows,
        "block_count": int(len(block_rows)),
        "block_f1_values": block_f1_values,
        "mean_block_f1": float(
            np.mean(block_f1_values)
        ),
        "min_block_f1": float(
            np.min(block_f1_values)
        ),
        "std_block_f1": float(
            np.std(block_f1_values)
        ),
    }


def select_threshold_with_persistence(
    labels,
    scores,
    min_anomaly_runs,
    validation_block_count,
    min_validation_recall,
):
    """
    Threshold ve minimum ardışık anomaly uzunluğunu yalnızca validation
    üzerinde birlikte seçer.

    Önce validation recall değeri belirlenen alt sınırı sağlayan ayarlar
    tutulur. Bu ayarlar arasında global validation F1 değeri en yüksek
    olan seçilir. Kronolojik blok metrikleri yalnızca eşitlik bozucu
    olarak kullanılır.

    Hiçbir ayar recall sınırını sağlayamazsa en yüksek recall değerine
    sahip ayar güvenli fallback olarak seçilir.
    """
    labels = np.asarray(labels, dtype=int).flatten()
    scores = np.asarray(scores, dtype=float).flatten()
    min_validation_recall = float(min_validation_recall)

    if len(labels) != len(scores) or len(scores) == 0:
        raise ValueError(
            "Validation etiket ve skorları uygun değil."
        )

    if not 0.0 <= min_validation_recall <= 1.0:
        raise ValueError(
            "min_validation_recall 0 ile 1 arasında olmalıdır."
        )

    run_values = sorted(
        set(int(value) for value in min_anomaly_runs)
    )

    if not run_values or run_values[0] < 1:
        raise ValueError(
            "min_anomaly_run değerleri en az 1 olmalıdır."
        )

    candidates = np.concatenate(
        (
            [scores.min() - 1e-12],
            np.unique(scores),
            [scores.max() + 1e-12],
        )
    )

    best_eligible = None
    best_fallback = None

    for threshold in candidates:
        raw_predictions = (
            scores >= threshold
        ).astype(int)

        for min_anomaly_run in run_values:
            filtered_predictions = apply_temporal_persistence(
                raw_predictions,
                min_anomaly_run,
            )

            metrics = calculate_metrics(
                labels,
                filtered_predictions,
            )

            stability = calculate_block_stability(
                labels,
                filtered_predictions,
                validation_block_count,
            )

            candidate = {
                "threshold": float(threshold),
                "min_anomaly_run": int(min_anomaly_run),
                "metrics": metrics,
                "stability": stability,
                "raw_predictions": raw_predictions.copy(),
                "predictions": filtered_predictions.copy(),
            }

            eligible = (
                metrics["recall"]
                >= min_validation_recall
            )

            if eligible:
                eligible_key = (
                    metrics["f1_score"],
                    metrics["precision"],
                    metrics["recall"],
                    stability["min_block_f1"],
                    stability["mean_block_f1"],
                    -stability["std_block_f1"],
                    -int(filtered_predictions.sum()),
                    -int(min_anomaly_run),
                )

                if (
                    best_eligible is None
                    or eligible_key > best_eligible["key"]
                ):
                    best_eligible = {
                        **candidate,
                        "key": eligible_key,
                        "recall_constraint_met": True,
                        "selection_mode": (
                            "recall_constrained"
                        ),
                    }

            fallback_key = (
                metrics["recall"],
                metrics["f1_score"],
                metrics["precision"],
                stability["min_block_f1"],
                stability["mean_block_f1"],
                -stability["std_block_f1"],
                -int(filtered_predictions.sum()),
                -int(min_anomaly_run),
            )

            if (
                best_fallback is None
                or fallback_key > best_fallback["key"]
            ):
                best_fallback = {
                    **candidate,
                    "key": fallback_key,
                    "recall_constraint_met": eligible,
                    "selection_mode": (
                        "fallback_highest_recall"
                    ),
                }

    if best_eligible is not None:
        return best_eligible

    return best_fallback


#gurultu ekler
def inject_noise(series, level, seed):
    rng = np.random.default_rng(seed)
    return np.asarray(series, dtype=float) + rng.normal(
        0.0, level, size=len(series)
    )

#unseen verilerdeki performansına bakar
def evaluate_unseen(predictions, labels, logs):
    indices = [
        index
        for index, log in enumerate(logs)
        if log.get("overall_status") == "unseen"
    ]

    if not indices:
        return {
            "metrics": {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
            },
            "count": 0,
            "anomaly_count": 0,
        }

    unseen_predictions = np.asarray(predictions)[indices]
    unseen_labels = np.asarray(labels)[indices]

    return {
        "metrics": calculate_metrics(unseen_labels, unseen_predictions),
        "count": len(indices),
        "anomaly_count": int(unseen_labels.sum()),
    }


def make_result(
    scenario,
    metrics,
    best,
    sample_count,
    anomaly_count,
    unseen_count,
):
    return {
        "dataset": "BATADAL",
        "model": "dual_alergia_state_merging",
        "scenario": scenario,
        **best["config"],
        "selected_threshold": best["threshold"],
        "validation_f1": best["metrics"]["f1_score"],
        "validation_strategy": (
            "recall_constrained_equal_chronological_blocks"
        ),
        "validation_selection_mode": best.get(
            "selection_mode",
            "",
        ),
        "validation_recall_constraint_met": best.get(
            "recall_constraint_met",
            False,
        ),
        "validation_mean_block_f1": best.get(
            "stability",
            {},
        ).get("mean_block_f1", 0.0),
        "validation_min_block_f1": best.get(
            "stability",
            {},
        ).get("min_block_f1", 0.0),
        "validation_std_block_f1": best.get(
            "stability",
            {},
        ).get("std_block_f1", 0.0),
        "normal_states_before": best["normal_stats"]["original_state_count"],
        "normal_states_after": best["normal_stats"]["merged_state_count"],
        "anomaly_states_before": best["anomaly_stats"]["original_state_count"],
        "anomaly_states_after": best["anomaly_stats"]["merged_state_count"],
        "sample_count": sample_count,
        "actual_anomaly_count": anomaly_count,
        "unseen_count": unseen_count,
        **metrics,
    }


def main():
    print("\n--- BATADAL DUAL ALERGIA DENEYİ BAŞLATILIYOR ---")

    config = load_config()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    X_train = load_vector(
        "data/processed/batadal_X_train_adasyn_pc1.csv"
    )
    y_train = to_binary(
        load_vector("data/processed/batadal_y_train_adasyn.csv")
    )
    X_val = load_vector("data/processed/batadal_X_val_pc1.csv")
    y_val = to_binary(load_vector("data/processed/batadal_y_val.csv"))
    X_test = load_vector("data/processed/batadal_X_test_pc1.csv")
    y_test = to_binary(load_vector("data/processed/batadal_y_test.csv"))

    print(
        f"Train X/y: {len(X_train)}/{len(y_train)} | "
        f"Val: {len(X_val)}/{len(y_val)} | "
        f"Test: {len(X_test)}/{len(y_test)}"
    )
    print(
        f"ADASYN normal/anomaly: "
        f"{int((y_train == 0).sum())}/{int((y_train == 1).sum())}"
    )

    window_size = config["window_size"]
    transformer = SaxPaaTransformer(alphabet_size=config["alphabet_size"])

    train_mean = float(np.mean(X_train))
    train_std = float(np.std(X_train)) or 1.0

    train_patterns = transform_patterns(
        transformer, X_train, window_size, train_mean, train_std
    )
    val_patterns = transform_patterns(
        transformer, X_val, window_size, train_mean, train_std
    )
    test_patterns = transform_patterns(
        transformer, X_test, window_size, train_mean, train_std
    )

    train_labels = align_labels(y_train, window_size)
    val_labels = align_labels(y_val, window_size)
    test_labels = align_labels(y_test, window_size)
    #pattern ve etiket sayıları eslesiyor mu kontrolu yapar
    for name, patterns, labels in [
        ("Train", train_patterns, train_labels),
        ("Validation", val_patterns, val_labels),
        ("Test", test_patterns, test_labels),
    ]:
        if len(patterns) != len(labels):
            raise ValueError(
                f"{name} pattern/etiket sayısı farklı: "
                f"{len(patterns)}/{len(labels)}"
            )

    train_transition_labels = train_labels[1:]
    val_transition_labels = val_labels[1:]
    test_transition_labels = test_labels[1:]

    normal_sequences, anomaly_sequences = build_class_sequences(
        train_patterns, train_transition_labels
    )

    print(
        f"Pattern train/val/test: "
        f"{len(train_patterns)}/{len(val_patterns)}/{len(test_patterns)}"
    )
    print(
        f"Normal/anomaly sequence: "
        f"{len(normal_sequences)}/{len(anomaly_sequences)}"
    )

    search_rows = []
    best = None

    print("\n--- VALIDATION PARAMETRE TARAMASI ---")
    #hiperparametre optimizasyonu
    for merge_alpha in config["merge_alphas"]:
        for min_count in config["min_counts"]:
            for smoothing_alpha in config["smoothing_alphas"]:
                kwargs = {
                    "merge_alpha": float(merge_alpha),
                    "min_state_count": int(min_count),
                    "smoothing_alpha": float(smoothing_alpha),
                    "max_pattern_distance": config["max_pattern_distance"],
                    "distance_penalty": 0.0,
                }

                normal_model = AlergiaStateMergingAutomata(**kwargs)
                anomaly_model = AlergiaStateMergingAutomata(**kwargs)

                normal_model.fit_sequences(normal_sequences)
                anomaly_model.fit_sequences(anomaly_sequences)

                val_scores, val_logs = score_dual(
                    normal_model, anomaly_model, val_patterns
                )
                threshold, metrics = select_threshold(
                    val_transition_labels, val_scores
                )
                diagnostics = summarize_scores(
                    val_transition_labels,
                    val_scores,
                    val_logs,
                    threshold,
                )

                normal_stats = normal_model.get_model_statistics()
                anomaly_stats = anomaly_model.get_model_statistics()
                total_states = (
                    normal_stats["merged_state_count"]
                    + anomaly_stats["merged_state_count"]
                )

                search_rows.append(
                    {
                        "search_stage": "state_merging",
                        **kwargs,
                        "min_anomaly_run": 1,
                        "threshold": threshold,
                        "normal_states_before": normal_stats[
                            "original_state_count"
                        ],
                        "normal_states_after": normal_stats[
                            "merged_state_count"
                        ],
                        "anomaly_states_before": anomaly_stats[
                            "original_state_count"
                        ],
                        "anomaly_states_after": anomaly_stats[
                            "merged_state_count"
                        ],
                        **{f"validation_{k}": v for k, v in metrics.items()},
                        **diagnostics,
                    }
                )

                print(
                    f"alpha={merge_alpha} | min={min_count} | "
                    f"smooth={smoothing_alpha} | "
                    f"normal={normal_stats['original_state_count']}"
                    f"->{normal_stats['merged_state_count']} | "
                    f"anomaly={anomaly_stats['original_state_count']}"
                    f"->{anomaly_stats['merged_state_count']} | "
                    f"F1={metrics['f1_score']:.4f}"
                )

                key = (
                    metrics["f1_score"],
                    metrics["precision"],
                    metrics["recall"],
                    -total_states,
                )
                #yeni parametreler oncekılerden ıyıse best objesini gunceller
                if best is None or key > best["key"]:
                    best = {
                        "key": key,
                        "normal_model": normal_model,
                        "anomaly_model": anomaly_model,
                        "threshold": threshold,
                        "metrics": metrics,
                        "normal_stats": normal_stats,
                        "anomaly_stats": anomaly_stats,
                        "config": {
                            "window_size": config["window_size"],
                            "alphabet_size": config["alphabet_size"],
                            "merge_alpha": float(merge_alpha),
                            "min_state_count": int(min_count),
                            "smoothing_alpha": float(smoothing_alpha),
                            "distance_penalty": 0.0,
                            "min_anomaly_run": 1,
                        },
                    }

    print(
        "\n--- RECALL KISITLI TEMPORAL PERSISTENCE TARAMASI ---"
    )

    penalty_best = None

    for distance_penalty in config["distance_penalties"]:
        distance_penalty = float(distance_penalty)

        best["normal_model"].distance_penalty = distance_penalty
        best["anomaly_model"].distance_penalty = distance_penalty

        val_scores, val_logs = score_dual(
            best["normal_model"],
            best["anomaly_model"],
            val_patterns,
        )

        selection = select_threshold_with_persistence(
            val_transition_labels,
            val_scores,
            config["min_anomaly_runs"],
            config["validation_block_count"],
            config["min_validation_recall"],
        )

        threshold = selection["threshold"]
        min_anomaly_run = selection["min_anomaly_run"]
        metrics = selection["metrics"]
        stability = selection["stability"]
        recall_constraint_met = selection[
            "recall_constraint_met"
        ]
        selection_mode = selection["selection_mode"]
        filtered_predictions = selection["predictions"]
        raw_predictions = selection["raw_predictions"]

        diagnostics = summarize_scores(
            val_transition_labels,
            val_scores,
            val_logs,
            threshold,
            predictions=filtered_predictions,
        )

        block_f1_text = "|".join(
            f"{value:.6f}"
            for value in stability["block_f1_values"]
        )

        search_rows.append(
            {
                "search_stage": (
                    "recall_constrained_temporal_persistence"
                ),
                "merge_alpha": best["config"]["merge_alpha"],
                "min_state_count": best["config"]["min_state_count"],
                "smoothing_alpha": best["config"]["smoothing_alpha"],
                "max_pattern_distance": config["max_pattern_distance"],
                "distance_penalty": distance_penalty,
                "min_anomaly_run": min_anomaly_run,
                "threshold": threshold,
                "min_validation_recall": config[
                    "min_validation_recall"
                ],
                "recall_constraint_met": (
                    recall_constraint_met
                ),
                "selection_mode": selection_mode,
                "validation_block_count": stability[
                    "block_count"
                ],
                "validation_mean_block_f1": stability[
                    "mean_block_f1"
                ],
                "validation_min_block_f1": stability[
                    "min_block_f1"
                ],
                "validation_std_block_f1": stability[
                    "std_block_f1"
                ],
                "validation_block_f1_values": block_f1_text,
                "validation_block_details": json.dumps(
                    stability["block_rows"],
                    ensure_ascii=False,
                ),
                "normal_states_before": best["normal_stats"][
                    "original_state_count"
                ],
                "normal_states_after": best["normal_stats"][
                    "merged_state_count"
                ],
                "anomaly_states_before": best["anomaly_stats"][
                    "original_state_count"
                ],
                "anomaly_states_after": best["anomaly_stats"][
                    "merged_state_count"
                ],
                **{
                    f"validation_{key}": value
                    for key, value in metrics.items()
                },
                **diagnostics,
            }
        )

        print(
            f"distance={distance_penalty} | "
            f"min_run={min_anomaly_run} | "
            f"threshold={threshold:.6f} | "
            f"raw_anomaly={int(raw_predictions.sum())} | "
            f"filtered_anomaly={int(filtered_predictions.sum())} | "
            f"global_F1={metrics['f1_score']:.4f} | "
            f"recall={metrics['recall']:.4f} | "
            f"constraint_met={recall_constraint_met} | "
            f"min_block_F1="
            f"{stability['min_block_f1']:.4f} | "
            f"mean_block_F1="
            f"{stability['mean_block_f1']:.4f} | "
            f"blocks={block_f1_text}"
        )

        key = (
            int(recall_constraint_met),
            metrics["f1_score"],
            metrics["precision"],
            metrics["recall"],
            stability["min_block_f1"],
            stability["mean_block_f1"],
            -stability["std_block_f1"],
            -int(filtered_predictions.sum()),
            -int(min_anomaly_run),
        )

        if penalty_best is None or key > penalty_best["key"]:
            penalty_best = {
                "key": key,
                "distance_penalty": distance_penalty,
                "min_anomaly_run": min_anomaly_run,
                "threshold": threshold,
                "metrics": metrics,
                "stability": stability,
                "recall_constraint_met": (
                    recall_constraint_met
                ),
                "selection_mode": selection_mode,
            }

    best["normal_model"].distance_penalty = penalty_best[
        "distance_penalty"
    ]
    best["anomaly_model"].distance_penalty = penalty_best[
        "distance_penalty"
    ]
    best["threshold"] = penalty_best["threshold"]
    best["metrics"] = penalty_best["metrics"]
    best["stability"] = penalty_best["stability"]
    best["recall_constraint_met"] = penalty_best[
        "recall_constraint_met"
    ]
    best["selection_mode"] = penalty_best[
        "selection_mode"
    ]
    best["config"]["min_validation_recall"] = config[
        "min_validation_recall"
    ]
    best["config"]["distance_penalty"] = penalty_best[
        "distance_penalty"
    ]
    best["config"]["min_anomaly_run"] = penalty_best[
        "min_anomaly_run"
    ]

    pd.DataFrame(search_rows).to_csv(
        os.path.join(
            OUTPUT_DIR, "batadal_dual_alergia_validation_search.csv"
        ),
        index=False,
    )

    print("\n--- EN İYİ AYAR ---")
    print(
        f"alpha={best['config']['merge_alpha']} | "
        f"min={best['config']['min_state_count']} | "
        f"smooth={best['config']['smoothing_alpha']} | "
        f"distance_penalty={best['config']['distance_penalty']} | "
        f"min_anomaly_run={best['config']['min_anomaly_run']} | "
        f"threshold={best['threshold']:.6f} | "
        f"val F1={best['metrics']['f1_score']:.4f} | "
        f"val recall={best['metrics']['recall']:.4f} | "
        f"recall_constraint_met="
        f"{best['recall_constraint_met']} | "
        f"mean_block_F1="
        f"{best['stability']['mean_block_f1']:.4f} | "
        f"min_block_F1="
        f"{best['stability']['min_block_f1']:.4f}"
    )

    test_predictions, test_logs = predict_dual(
        best["normal_model"],
        best["anomaly_model"],
        test_patterns,
        best["threshold"],
        best["config"]["min_anomaly_run"],
    )
    original_metrics = calculate_metrics(
        test_transition_labels, test_predictions
    )
    original_unseen_count = sum(
        log["overall_status"] == "unseen" for log in test_logs
    )

    noisy_test = inject_noise(
        X_test, config["noise_level"], config["seed"]
    )
    noisy_patterns = transform_patterns(
        transformer, noisy_test, window_size, train_mean, train_std
    )
    noisy_predictions, noisy_logs = predict_dual(
        best["normal_model"],
        best["anomaly_model"],
        noisy_patterns,
        best["threshold"],
        best["config"]["min_anomaly_run"],
    )
    noise_metrics = calculate_metrics(
        test_transition_labels, noisy_predictions
    )
    noise_unseen_count = sum(
        log["overall_status"] == "unseen" for log in noisy_logs
    )

    unseen = evaluate_unseen(
        test_predictions, test_transition_labels, test_logs
    )

    rows = [
        make_result(
            "original",
            original_metrics,
            best,
            len(test_transition_labels),
            int(test_transition_labels.sum()),
            original_unseen_count,
        ),
        make_result(
            "gaussian_noise",
            noise_metrics,
            best,
            len(test_transition_labels),
            int(test_transition_labels.sum()),
            noise_unseen_count,
        ),
        make_result(
            "unseen_data",
            unseen["metrics"],
            best,
            unseen["count"],
            unseen["anomaly_count"],
            unseen["count"],
        ),
    ]

    pd.DataFrame(rows).to_csv(
        os.path.join(
            OUTPUT_DIR, "batadal_dual_alergia_state_merging_metrics.csv"
        ),
        index=False,
    )

    with open(
        os.path.join(
            OUTPUT_DIR, "batadal_dual_alergia_explainability.json"
        ),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(test_logs[:100], file, indent=4, ensure_ascii=False)

    with open(
        os.path.join(
            OUTPUT_DIR, "batadal_dual_alergia_state_mapping.json"
        ),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "best_config": best["config"],
                "threshold": best["threshold"],
                "validation_stability": {
                    "strategy": (
                        "recall_constrained_equal_chronological_blocks"
                    ),
                    "primary_selection_metric": (
                        "global_validation_f1_with_recall_constraint"
                    ),
                    "minimum_validation_recall": best[
                        "config"
                    ]["min_validation_recall"],
                    "recall_constraint_met": best[
                        "recall_constraint_met"
                    ],
                    "selection_mode": best[
                        "selection_mode"
                    ],
                    "block_count": best["stability"][
                        "block_count"
                    ],
                    "mean_block_f1": best["stability"][
                        "mean_block_f1"
                    ],
                    "min_block_f1": best["stability"][
                        "min_block_f1"
                    ],
                    "std_block_f1": best["stability"][
                        "std_block_f1"
                    ],
                    "blocks": best["stability"][
                        "block_rows"
                    ],
                },
                "normal_model": {
                    "statistics": best["normal_stats"],
                    "state_mapping": best[
                        "normal_model"
                    ].get_state_mapping(),
                    "merged_members": best[
                        "normal_model"
                    ].get_merged_members(),
                },
                "anomaly_model": {
                    "statistics": best["anomaly_stats"],
                    "state_mapping": best[
                        "anomaly_model"
                    ].get_state_mapping(),
                    "merged_members": best[
                        "anomaly_model"
                    ].get_merged_members(),
                },
            },
            file,
            indent=4,
            ensure_ascii=False,
        )

    print(
        f"Test predicted anomaly: "
        f"{int(test_predictions.sum())}/{len(test_predictions)} | "
        f"actual anomaly: {int(test_transition_labels.sum())}"
    )

    print("\n--- TEST SONUÇLARI ---")
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


if __name__ == "__main__":
    main()