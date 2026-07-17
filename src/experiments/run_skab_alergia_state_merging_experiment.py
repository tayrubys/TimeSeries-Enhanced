import json
import os
import sys
from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:  # Eski sklearn sürümleri için yedek.
    StratifiedGroupKFold = None

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


def load_source_files(path):
    values = load_vector(path)
    return np.asarray([str(value) for value in values])


def to_binary(labels):
    labels = np.asarray(labels).flatten()
    labels = np.where(labels == -999, 0, labels)
    return np.where(labels > 0, 1, 0).astype(int)


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        full_config = json.load(file)

    automata = full_config.get("automata", {})
    deep_learning = full_config.get("deep_learning", {})

    return {
        "window_size": int(automata.get("window_size", 4)),
        "alphabet_size": int(automata.get("alphabet_size", 3)),
        "noise_level": float(automata.get("noise_level", 0.1)),
        "seeds": [int(seed) for seed in automata.get(
            "seeds", [42, 123, 2026, 7, 999]
        )],
        "n_folds": int(
            automata.get("skab_n_folds", deep_learning.get("n_folds", 5))
        ),
        "inner_n_splits": int(
            automata.get("alergia_skab_inner_n_splits", 5)
        ),
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
    # Validation ve test yalnızca ilgili train istatistikleriyle normalize edilir.
    normalized = (np.asarray(series, dtype=float) - train_mean) / train_std
    paa = transformer.apply_paa(normalized, window_size)
    sax = transformer.convert_to_sax(paa)
    return transformer.get_sliding_windows(sax, window_size)


def align_labels(labels, window_size):
    # Etiketleri önce PAA bloklarına, sonra sliding-window pattern seviyesine taşır.
    labels = to_binary(labels)
    usable = (len(labels) // window_size) * window_size

    if usable == 0:
        return np.array([], dtype=int)

    paa_labels = labels[:usable].reshape(-1, window_size).max(axis=1)
    pattern_count = len(paa_labels) - window_size + 1

    if pattern_count <= 0:
        return np.array([], dtype=int)

    return np.asarray(
        [
            int(paa_labels[index:index + window_size].max())
            for index in range(pattern_count)
        ],
        dtype=int,
    )


def grouped_indices(source_files):
    # Satır sırasını koruyarak her CSV dosyasını ayrı bir sequence olarak tutar.
    grouped = OrderedDict()

    for index, source_file in enumerate(source_files):
        grouped.setdefault(str(source_file), []).append(index)

    for source_file, indices in grouped.items():
        yield source_file, np.asarray(indices, dtype=int)


def prepare_grouped_patterns(
    values,
    labels,
    source_files,
    transformer,
    window_size,
    train_mean,
    train_std,
):
    values = np.asarray(values, dtype=float)
    labels = to_binary(labels)
    source_files = np.asarray(source_files)

    if not (len(values) == len(labels) == len(source_files)):
        raise ValueError(
            "X, y ve source_file satır sayıları farklı: "
            f"{len(values)}/{len(labels)}/{len(source_files)}"
        )

    grouped_data = []

    for source_file, indices in grouped_indices(source_files):
        patterns = transform_patterns(
            transformer,
            values[indices],
            window_size,
            train_mean,
            train_std,
        )
        pattern_labels = align_labels(labels[indices], window_size)

        if len(patterns) != len(pattern_labels):
            raise ValueError(
                f"{source_file} pattern/etiket sayısı farklı: "
                f"{len(patterns)}/{len(pattern_labels)}"
            )

        # En az iki pattern yoksa transition üretilemez.
        if len(patterns) < 2:
            continue

        grouped_data.append(
            {
                "source_file": source_file,
                "patterns": patterns,
                "pattern_labels": pattern_labels,
                "transition_labels": pattern_labels[1:],
            }
        )

    return grouped_data


def build_class_sequences(patterns, transition_labels):
    """
    Aynı sınıfa ait ardışık geçişleri sequence olarak toplar.
    Sınıf değişiminde normal ile anomaly arasında sahte geçiş kurulmaz.
    """
    transition_labels = np.asarray(transition_labels, dtype=int)

    if len(patterns) - 1 != len(transition_labels):
        raise ValueError("Pattern ve geçiş etiketi uzunlukları uyuşmuyor.")

    if len(transition_labels) == 0:
        return [], []

    normal_sequences = []
    anomaly_sequences = []
    current_label = int(transition_labels[0])
    current_sequence = [patterns[0], patterns[1]]

    for index in range(1, len(transition_labels)):
        label = int(transition_labels[index])

        if label == current_label:
            current_sequence.append(patterns[index + 1])
            continue

        if current_label == 1:
            anomaly_sequences.append(current_sequence)
        else:
            normal_sequences.append(current_sequence)

        current_label = label
        current_sequence = [patterns[index], patterns[index + 1]]

    if current_label == 1:
        anomaly_sequences.append(current_sequence)
    else:
        normal_sequences.append(current_sequence)

    return normal_sequences, anomaly_sequences


def build_grouped_class_sequences(grouped_data):
    normal_sequences = []
    anomaly_sequences = []

    for item in grouped_data:
        file_normal, file_anomaly = build_class_sequences(
            item["patterns"], item["transition_labels"]
        )
        normal_sequences.extend(file_normal)
        anomaly_sequences.extend(file_anomaly)

    return normal_sequences, anomaly_sequences


def fit_dual_models(normal_sequences, anomaly_sequences, kwargs):
    normal_model = AlergiaStateMergingAutomata(**kwargs)
    anomaly_model = AlergiaStateMergingAutomata(**kwargs)
    normal_model.fit_sequences(normal_sequences)
    anomaly_model.fit_sequences(anomaly_sequences)
    return normal_model, anomaly_model


def score_dual(normal_model, anomaly_model, patterns):
    normal_scores, normal_logs = normal_model.score_patterns(patterns)
    anomaly_scores, anomaly_logs = anomaly_model.score_patterns(patterns)

    if len(normal_scores) != len(anomaly_scores):
        raise ValueError("Normal ve anomaly skor sayıları farklı.")

    # Pozitif skor, geçişi anomaly modelinin daha iyi açıkladığını gösterir.
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
                "normal_probability": normal_logs[index]["transition_probability"],
                "normal_surprise": float(normal_scores[index]),
                "anomaly_mapped_to": anomaly_logs[index]["mapped_to"],
                "anomaly_probability": anomaly_logs[index]["transition_probability"],
                "anomaly_surprise": float(anomaly_scores[index]),
                "dual_score": float(score),
            }
        )

    return dual_scores, logs


def score_grouped(normal_model, anomaly_model, grouped_data):
    all_scores = []
    all_labels = []
    all_logs = []

    for item in grouped_data:
        scores, logs = score_dual(
            normal_model,
            anomaly_model,
            item["patterns"],
        )

        if len(scores) != len(item["transition_labels"]):
            raise ValueError(
                f"{item['source_file']} skor/etiket sayısı farklı: "
                f"{len(scores)}/{len(item['transition_labels'])}"
            )

        for log in logs:
            log["source_file"] = item["source_file"]

        all_scores.extend(scores.tolist())
        all_labels.extend(item["transition_labels"].tolist())
        all_logs.extend(logs)

    return (
        np.asarray(all_scores, dtype=float),
        np.asarray(all_labels, dtype=int),
        all_logs,
    )


def select_threshold(labels, scores):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)

    if len(labels) != len(scores) or len(scores) == 0:
        raise ValueError("Validation etiket ve skorları uygun değil.")

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


def inject_noise(series, noise_level, seed):
    rng = np.random.default_rng(seed)
    series = np.asarray(series, dtype=float)
    return series + rng.normal(0.0, noise_level, size=len(series))


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


def combine_statistics(normal_model, anomaly_model):
    normal_stats = normal_model.get_model_statistics()
    anomaly_stats = anomaly_model.get_model_statistics()

    return {
        "normal_states_before": normal_stats["original_state_count"],
        "normal_states_after": normal_stats["merged_state_count"],
        "normal_transition_count": normal_stats["transition_count"],
        "normal_transition_density": normal_stats["transition_density"],
        "anomaly_states_before": anomaly_stats["original_state_count"],
        "anomaly_states_after": anomaly_stats["merged_state_count"],
        "anomaly_transition_count": anomaly_stats["transition_count"],
        "anomaly_transition_density": anomaly_stats["transition_density"],
        "total_states_before": (
            normal_stats["original_state_count"]
            + anomaly_stats["original_state_count"]
        ),
        "total_states_after": (
            normal_stats["merged_state_count"]
            + anomaly_stats["merged_state_count"]
        ),
        "total_transition_count": (
            normal_stats["transition_count"]
            + anomaly_stats["transition_count"]
        ),
    }


def create_inner_splitter(n_splits, seed):
    if StratifiedGroupKFold is not None:
        return (
            StratifiedGroupKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=seed,
            ),
            "StratifiedGroupKFold",
        )

    return GroupKFold(n_splits=n_splits), "GroupKFold"


def choose_inner_split(
    X,
    y,
    source_files,
    transformer,
    config,
    fold_id,
    seed,
):
    unique_groups = np.unique(source_files)
    n_splits = min(config["inner_n_splits"], len(unique_groups))

    if n_splits < 2:
        raise ValueError(
            f"Fold {fold_id} içinde inner split için yeterli source_file yok."
        )

    splitter, splitter_name = create_inner_splitter(n_splits, seed)
    dummy = np.zeros(len(X), dtype=int)
    candidates = []
    target_anomaly_ratio = float(np.mean(y))

    for split_id, (train_idx, val_idx) in enumerate(
        splitter.split(dummy, y, groups=source_files),
        start=1,
    ):
        train_groups = sorted(set(source_files[train_idx]))
        val_groups = sorted(set(source_files[val_idx]))

        if set(train_groups).intersection(val_groups):
            raise RuntimeError("Inner train ve validation source_file grupları çakışıyor.")

        train_mean = float(np.mean(X[train_idx]))
        train_std = float(np.std(X[train_idx])) or 1.0

        grouped_train = prepare_grouped_patterns(
            X[train_idx],
            y[train_idx],
            source_files[train_idx],
            transformer,
            config["window_size"],
            train_mean,
            train_std,
        )
        grouped_val = prepare_grouped_patterns(
            X[val_idx],
            y[val_idx],
            source_files[val_idx],
            transformer,
            config["window_size"],
            train_mean,
            train_std,
        )

        normal_sequences, anomaly_sequences = build_grouped_class_sequences(
            grouped_train
        )
        val_labels = (
            np.concatenate(
                [item["transition_labels"] for item in grouped_val]
            )
            if grouped_val
            else np.array([], dtype=int)
        )

        if not normal_sequences or not anomaly_sequences:
            continue
        if len(val_labels) == 0 or len(np.unique(val_labels)) < 2:
            continue

        val_anomaly_ratio = float(np.mean(val_labels))
        ratio_difference = abs(val_anomaly_ratio - target_anomaly_ratio)

        candidates.append(
            {
                "key": (ratio_difference, split_id),
                "splitter": splitter_name,
                "split_id": split_id,
                "train_idx": train_idx,
                "val_idx": val_idx,
                "train_mean": train_mean,
                "train_std": train_std,
                "grouped_train": grouped_train,
                "grouped_val": grouped_val,
                "normal_sequences": normal_sequences,
                "anomaly_sequences": anomaly_sequences,
                "val_labels": val_labels,
                "train_groups": train_groups,
                "val_groups": val_groups,
            }
        )

    if not candidates:
        group_report = (
            pd.DataFrame({"source_file": source_files, "label": y})
            .groupby("source_file")["label"]
            .agg(["count", "sum"])
            .reset_index()
        )
        raise RuntimeError(
            f"Fold {fold_id}, seed={seed} için iki sınıfı içeren uygun "
            "inner validation bulunamadı. Source-file dağılımı:\n"
            f"{group_report.to_string(index=False)}"
        )

    return min(candidates, key=lambda item: item["key"])


def make_result_row(
    fold_id,
    seed,
    scenario,
    metrics,
    best,
    split_info,
    structure_stats,
    sample_count,
    anomaly_count,
    unseen_count,
    test_file_count,
):
    return {
        "dataset": "SKAB",
        "model": "dual_alergia_state_merging",
        "fold": fold_id,
        "seed": seed,
        "scenario": scenario,
        "window_size": best["config"]["window_size"],
        "alphabet_size": best["config"]["alphabet_size"],
        "merge_alpha": best["config"]["merge_alpha"],
        "min_state_count": best["config"]["min_state_count"],
        "smoothing_alpha": best["config"]["smoothing_alpha"],
        "selected_threshold": best["threshold"],
        "validation_f1": best["metrics"]["f1_score"],
        "inner_splitter": split_info["splitter"],
        "inner_split_id": split_info["split_id"],
        "inner_train_file_count": len(split_info["train_groups"]),
        "inner_validation_file_count": len(split_info["val_groups"]),
        "test_file_count": test_file_count,
        "sample_count": sample_count,
        "actual_anomaly_count": anomaly_count,
        "unseen_count": unseen_count,
        **structure_stats,
        **metrics,
    }


def run_fold_seed(fold_id, seed, config, transformer):
    print("\n" + "=" * 72)
    print(f"SKAB FOLD {fold_id} | SEED {seed} | DUAL ALERGIA")
    print("=" * 72)

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
            f"Fold {fold_id} outer train/test source_file çakışması: {overlap}"
        )

    print(
        f"Train X/y/source: {len(X_train)}/{len(y_train)}/{len(train_source)} | "
        f"Test: {len(X_test)}/{len(y_test)}/{len(test_source)}"
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

    validation_rows = []
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

                normal_model, anomaly_model = fit_dual_models(
                    split_info["normal_sequences"],
                    split_info["anomaly_sequences"],
                    kwargs,
                )
                val_scores, val_labels, val_logs = score_grouped(
                    normal_model,
                    anomaly_model,
                    split_info["grouped_val"],
                )
                threshold, metrics = select_threshold(val_labels, val_scores)
                stats = combine_statistics(normal_model, anomaly_model)
                val_unseen_count = sum(
                    log["overall_status"] == "unseen" for log in val_logs
                )

                validation_rows.append(
                    {
                        "dataset": "SKAB",
                        "fold": fold_id,
                        "seed": seed,
                        "inner_splitter": split_info["splitter"],
                        "inner_split_id": split_info["split_id"],
                        "window_size": config["window_size"],
                        "alphabet_size": config["alphabet_size"],
                        **kwargs,
                        "threshold": threshold,
                        "validation_sample_count": len(val_labels),
                        "validation_anomaly_count": int(val_labels.sum()),
                        "validation_unseen_count": val_unseen_count,
                        **stats,
                        **{
                            f"validation_{key}": value
                            for key, value in metrics.items()
                        },
                    }
                )

                print(
                    f"alpha={merge_alpha} | min={min_count} | "
                    f"smooth={smoothing_alpha} | "
                    f"states={stats['total_states_before']}"
                    f"->{stats['total_states_after']} | "
                    f"F1={metrics['f1_score']:.4f}"
                )

                key = (
                    metrics["f1_score"],
                    metrics["precision"],
                    metrics["recall"],
                    -stats["total_states_after"],
                )

                if best is None or key > best["key"]:
                    best = {
                        "key": key,
                        "threshold": threshold,
                        "metrics": metrics,
                        "config": {
                            "window_size": config["window_size"],
                            "alphabet_size": config["alphabet_size"],
                            "merge_alpha": float(merge_alpha),
                            "min_state_count": int(min_count),
                            "smoothing_alpha": float(smoothing_alpha),
                        },
                    }

    print("\n--- FOLD/SEED İÇİN EN İYİ AYAR ---")
    print(
        f"alpha={best['config']['merge_alpha']} | "
        f"min={best['config']['min_state_count']} | "
        f"smooth={best['config']['smoothing_alpha']} | "
        f"threshold={best['threshold']:.6f} | "
        f"val F1={best['metrics']['f1_score']:.4f}"
    )

    # Parametre seçimi bittikten sonra final model outer-train'in tamamıyla eğitilir.
    full_train_mean = float(np.mean(X_train))
    full_train_std = float(np.std(X_train)) or 1.0
    grouped_full_train = prepare_grouped_patterns(
        X_train,
        y_train,
        train_source,
        transformer,
        config["window_size"],
        full_train_mean,
        full_train_std,
    )
    full_normal_sequences, full_anomaly_sequences = build_grouped_class_sequences(
        grouped_full_train
    )

    if not full_normal_sequences or not full_anomaly_sequences:
        raise RuntimeError(
            f"Fold {fold_id} full train içinde iki sınıf için sequence oluşmadı."
        )

    final_kwargs = {
        "merge_alpha": best["config"]["merge_alpha"],
        "min_state_count": best["config"]["min_state_count"],
        "smoothing_alpha": best["config"]["smoothing_alpha"],
        "max_pattern_distance": config["max_pattern_distance"],
    }
    normal_model, anomaly_model = fit_dual_models(
        full_normal_sequences,
        full_anomaly_sequences,
        final_kwargs,
    )
    structure_stats = combine_statistics(normal_model, anomaly_model)

    grouped_test = prepare_grouped_patterns(
        X_test,
        y_test,
        test_source,
        transformer,
        config["window_size"],
        full_train_mean,
        full_train_std,
    )
    test_scores, test_labels, test_logs = score_grouped(
        normal_model,
        anomaly_model,
        grouped_test,
    )
    test_predictions = (test_scores >= best["threshold"]).astype(int)
    original_metrics = calculate_metrics(test_labels, test_predictions)
    original_unseen_count = sum(
        log["overall_status"] == "unseen" for log in test_logs
    )

    noisy_test = inject_noise(X_test, config["noise_level"], seed)
    grouped_noisy_test = prepare_grouped_patterns(
        noisy_test,
        y_test,
        test_source,
        transformer,
        config["window_size"],
        full_train_mean,
        full_train_std,
    )
    noisy_scores, noisy_labels, noisy_logs = score_grouped(
        normal_model,
        anomaly_model,
        grouped_noisy_test,
    )
    noisy_predictions = (noisy_scores >= best["threshold"]).astype(int)
    noise_metrics = calculate_metrics(noisy_labels, noisy_predictions)
    noise_unseen_count = sum(
        log["overall_status"] == "unseen" for log in noisy_logs
    )

    unseen = evaluate_unseen(test_predictions, test_labels, test_logs)

    rows = [
        make_result_row(
            fold_id,
            seed,
            "original",
            original_metrics,
            best,
            split_info,
            structure_stats,
            len(test_labels),
            int(test_labels.sum()),
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
            structure_stats,
            len(noisy_labels),
            int(noisy_labels.sum()),
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
            structure_stats,
            unseen["count"],
            unseen["anomaly_count"],
            unseen["count"],
            len(test_files),
        ),
    ]

    best_config_row = {
        "dataset": "SKAB",
        "fold": fold_id,
        "seed": seed,
        **best["config"],
        "selected_threshold": best["threshold"],
        **{
            f"validation_{key}": value
            for key, value in best["metrics"].items()
        },
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
        "metrics_rows": rows,
        "validation_rows": validation_rows,
        "best_config_row": best_config_row,
        "explainability": {
            "original": test_logs[:100],
            "gaussian_noise": noisy_logs[:100],
        },
    }


def create_summary(metrics_df):
    columns = ["accuracy", "precision", "recall", "f1_score", "unseen_count"]
    summary = metrics_df.groupby("scenario")[columns].agg(["mean", "std"])
    summary.columns = [f"{column}_{stat}" for column, stat in summary.columns]
    return summary.reset_index()


def create_fold_summary(metrics_df):
    columns = ["accuracy", "precision", "recall", "f1_score", "unseen_count"]
    summary = metrics_df.groupby(["fold", "scenario"])[columns].agg(
        ["mean", "std"]
    )
    summary.columns = [f"{column}_{stat}" for column, stat in summary.columns]
    return summary.reset_index()


def main():
    print("\n--- SKAB 5-FOLD × 5-SEED DUAL ALERGIA DENEYİ BAŞLATILIYOR ---")

    config = load_config()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    transformer = SaxPaaTransformer(alphabet_size=config["alphabet_size"])

    all_metric_rows = []
    all_validation_rows = []
    all_best_config_rows = []
    explainability = {}

    for fold_id in range(1, config["n_folds"] + 1):
        for seed in config["seeds"]:
            result = run_fold_seed(fold_id, seed, config, transformer)
            all_metric_rows.extend(result["metrics_rows"])
            all_validation_rows.extend(result["validation_rows"])
            all_best_config_rows.append(result["best_config_row"])
            explainability[f"fold_{fold_id}_seed_{seed}"] = result[
                "explainability"
            ]

    metrics_df = pd.DataFrame(all_metric_rows)
    validation_df = pd.DataFrame(all_validation_rows)
    best_config_df = pd.DataFrame(all_best_config_rows)
    summary_df = create_summary(metrics_df)
    fold_summary_df = create_fold_summary(metrics_df)

    metrics_df.to_csv(
        os.path.join(OUTPUT_DIR, "skab_dual_alergia_fold_seed_metrics.csv"),
        index=False,
    )
    validation_df.to_csv(
        os.path.join(OUTPUT_DIR, "skab_dual_alergia_validation_search.csv"),
        index=False,
    )
    best_config_df.to_csv(
        os.path.join(OUTPUT_DIR, "skab_dual_alergia_best_configs.csv"),
        index=False,
    )
    fold_summary_df.to_csv(
        os.path.join(OUTPUT_DIR, "skab_dual_alergia_fold_summary.csv"),
        index=False,
    )
    summary_df.to_csv(
        os.path.join(OUTPUT_DIR, "skab_dual_alergia_overall_summary.csv"),
        index=False,
    )

    with open(
        os.path.join(OUTPUT_DIR, "skab_dual_alergia_explainability.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(explainability, file, indent=4, ensure_ascii=False)

    print("\n--- SKAB GENEL ÖZET ---")
    print(summary_df.to_string(index=False))
    print("\nSonuç dosyaları results/outputs klasörüne kaydedildi.")


if __name__ == "__main__":
    main()