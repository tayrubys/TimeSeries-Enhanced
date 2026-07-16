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


def load_vector(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dosya bulunamadı: {path}")
    return pd.read_csv(path).values.flatten()


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
    }


def transform_patterns(transformer, series, window_size, train_mean, train_std):
    # Her bölümü train istatistikleriyle normalize ediyoruz.
    normalized = (np.asarray(series, dtype=float) - train_mean) / train_std
    paa = transformer.apply_paa(normalized, window_size)
    sax = transformer.convert_to_sax(paa)
    return transformer.get_sliding_windows(sax, window_size)


def align_labels(labels, window_size):
    # Ham etiketleri önce PAA bloğuna, sonra pattern seviyesine taşıyoruz.
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
    Ardışık aynı sınıfa ait geçişleri uzun sequence'lere ayırır.
    Böylece ALERGIA yalnızca ikili geçişleri değil,
    devam eden pattern davranışlarını da öğrenebilir.
    """

    if len(patterns) - 1 != len(transition_labels):
        raise ValueError(
            "Pattern ve geçiş etiketi uzunlukları uyuşmuyor."
        )

    normal_sequences = []
    anomaly_sequences = []

    current_label = int(transition_labels[0])
    current_sequence = [patterns[0], patterns[1]]

    for index in range(1, len(transition_labels)):
        label = int(transition_labels[index])

        if label == current_label:
            current_sequence.append(patterns[index + 1])
        else:
            if len(current_sequence) >= 2:
                if current_label == 1:
                    anomaly_sequences.append(current_sequence)
                else:
                    normal_sequences.append(current_sequence)

            # Sınıf değişiminde yeni sequence başlatılıyor.
            current_label = label
            current_sequence = [
                patterns[index],
                patterns[index + 1]
            ]

    # Son sequence'i de ekliyoruz.
    if len(current_sequence) >= 2:
        if current_label == 1:
            anomaly_sequences.append(current_sequence)
        else:
            normal_sequences.append(current_sequence)

    return normal_sequences, anomaly_sequences


def score_dual(normal_model, anomaly_model, patterns):
    normal_scores, normal_logs = normal_model.score_patterns(patterns)
    anomaly_scores, anomaly_logs = anomaly_model.score_patterns(patterns)

    if len(normal_scores) != len(anomaly_scores):
        raise ValueError("Normal ve anomaly skor sayıları farklı.")

    # Yüksek değer, anomaly modelinin geçişi daha iyi açıkladığını gösterir.
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
                "normal_surprise": float(normal_scores[index]),
                "anomaly_mapped_to": anomaly_logs[index]["mapped_to"],
                "anomaly_probability": anomaly_logs[index][
                    "transition_probability"
                ],
                "anomaly_surprise": float(anomaly_scores[index]),
                "dual_score": float(score),
            }
        )

    return dual_scores, logs


def predict_dual(normal_model, anomaly_model, patterns, threshold):
    scores, logs = score_dual(normal_model, anomaly_model, patterns)
    predictions = (scores >= threshold).astype(int)

    for prediction, log in zip(predictions, logs):
        log["threshold"] = float(threshold)
        log["decision"] = "anomaly" if prediction == 1 else "normal"

    return predictions, logs


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


def inject_noise(series, level, seed):
    rng = np.random.default_rng(seed)
    return np.asarray(series, dtype=float) + rng.normal(
        0.0, level, size=len(series)
    )


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

    for merge_alpha in config["merge_alphas"]:
        for min_count in config["min_counts"]:
            for smoothing_alpha in config["smoothing_alphas"]:
                kwargs = {
                    "merge_alpha": float(merge_alpha),
                    "min_state_count": int(min_count),
                    "smoothing_alpha": float(smoothing_alpha),
                    "max_pattern_distance": config["max_pattern_distance"],
                }

                normal_model = AlergiaStateMergingAutomata(**kwargs)
                anomaly_model = AlergiaStateMergingAutomata(**kwargs)

                normal_model.fit_sequences(normal_sequences)
                anomaly_model.fit_sequences(anomaly_sequences)

                val_scores, _ = score_dual(
                    normal_model, anomaly_model, val_patterns
                )
                threshold, metrics = select_threshold(
                    val_transition_labels, val_scores
                )

                normal_stats = normal_model.get_model_statistics()
                anomaly_stats = anomaly_model.get_model_statistics()
                total_states = (
                    normal_stats["merged_state_count"]
                    + anomaly_stats["merged_state_count"]
                )

                search_rows.append(
                    {
                        **kwargs,
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
                        },
                    }

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
        f"threshold={best['threshold']:.6f} | "
        f"val F1={best['metrics']['f1_score']:.4f}"
    )

    test_predictions, test_logs = predict_dual(
        best["normal_model"],
        best["anomaly_model"],
        test_patterns,
        best["threshold"],
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