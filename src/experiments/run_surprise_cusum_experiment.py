import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score
)
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..")
    )
)


from src.data_pipeline.sax_paa import SaxPaaTransformer
from src.experiments.evaluator import calculate_metrics
from src.models.automata_model import ProbabilisticAutomata


def load_automata_config(
    config_path="src/config/settings.json"
):
    """
    Surprise + CUSUM deneyinde kullanılacak parametreleri
    merkezi settings.json dosyasından okur.
    """

    default_config = {
        "window_size": 4,
        "alphabet_size": 3,
        "cusum_k_quantiles": [
            0.50,
            0.60,
            0.70,
            0.80,
            0.90,
            0.95
        ]
    }

    if not os.path.exists(config_path):
        return default_config

    with open(
        config_path,
        "r",
        encoding="utf-8"
    ) as file:
        full_config = json.load(file)

    automata_config = full_config.get(
        "automata",
        {}
    )

    return {
        "window_size": automata_config.get(
            "window_size",
            default_config["window_size"]
        ),
        "alphabet_size": automata_config.get(
            "alphabet_size",
            default_config["alphabet_size"]
        ),
        "cusum_k_quantiles": automata_config.get(
            "cusum_k_quantiles",
            default_config["cusum_k_quantiles"]
        )
    }


def load_vector(file_path):

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Gerekli dosya bulunamadı: {file_path}"
        )

    return pd.read_csv(
        file_path
    ).values.flatten()


def convert_labels_to_binary(labels):
 
    labels = np.asarray(
        labels
    ).flatten()

    labels = np.where(
        labels == -999,
        0,
        labels
    )

    labels = np.where(
        labels > 0,
        1,
        0
    )

    return labels.astype(int)


def transform_with_train_statistics(
    transformer,
    series,
    window_size,
    train_mean,
    train_std
):

    series = np.asarray(
        series,
        dtype=float
    ).flatten()

    normalized_series = (
        series - train_mean
    ) / train_std

    paa_signal = transformer.apply_paa(
        normalized_series,
        window_size
    )

    sax_sequence = transformer.convert_to_sax(
        paa_signal
    )

    patterns = transformer.get_sliding_windows(
        sax_sequence,
        window_size
    )

    return patterns


def align_labels_to_patterns(
    labels,
    window_size
):
    """
    Ham nokta etiketlerini SAX pattern seviyesine hizalar.

    Bir pattern'in kapsadığı bölgede en az bir anomaly varsa
    o pattern'in etiketi anomaly=1 olur.
    """

    labels = convert_labels_to_binary(
        labels
    )

    if window_size <= 0:
        raise ValueError(
            "window_size sıfırdan büyük olmalıdır."
        )

    #paa kalan noktaları kırptığı için etiketler de aynı uzunluğa kırpılır
    usable_length = (
        len(labels) // window_size
    ) * window_size

    if usable_length == 0:
        return np.array(
            [],
            dtype=int
        )

    trimmed_labels = labels[
        :usable_length
    ]

    #herr paa bloğunun etiketi 
    paa_block_labels = trimmed_labels.reshape(
        -1,
        window_size
    ).max(axis=1)

    pattern_count = (
        len(paa_block_labels)
        - window_size
        + 1
    )

    if pattern_count <= 0:
        return np.array(
            [],
            dtype=int
        )

    pattern_labels = []

    for start_index in range(pattern_count):
        end_index = (
            start_index + window_size
        )

        pattern_region = paa_block_labels[
            start_index:end_index
        ]

        pattern_labels.append(
            int(np.max(pattern_region))
        )

    return np.asarray(
        pattern_labels,
        dtype=int
    )


def print_score_distribution(
    score_name,
    scores
):

    scores = np.asarray(
        scores,
        dtype=float
    )

    if len(scores) == 0:
        print(
            f"{score_name}: skor bulunamadı."
        )
        return

    print(
        f"{score_name}: "
        f"n={len(scores)} | "
        f"min={np.min(scores):.4f} | "
        f"ortalama={np.mean(scores):.4f} | "
        f"medyan={np.median(scores):.4f} | "
        f"max={np.max(scores):.4f}"
    )


def calculate_auc_metrics(
    true_labels,
    anomaly_scores
):

    true_labels = np.asarray(
        true_labels,
        dtype=int
    )

    anomaly_scores = np.asarray(
        anomaly_scores,
        dtype=float
    )

    if len(np.unique(true_labels)) != 2:
        return float("nan"), float("nan")

    pr_auc = average_precision_score(
        true_labels,
        anomaly_scores
    )

    roc_auc = roc_auc_score(
        true_labels,
        anomaly_scores
    )

    return float(pr_auc), float(roc_auc)


def select_score_threshold(
    true_labels,
    anomaly_scores
):

    true_labels = np.asarray(
        true_labels,
        dtype=int
    ).flatten()

    anomaly_scores = np.asarray(
        anomaly_scores,
        dtype=float
    ).flatten()

    if len(true_labels) != len(anomaly_scores):
        raise ValueError(
            "Etiket ve skor uzunlukları eşit değil: "
            f"etiket={len(true_labels)}, "
            f"skor={len(anomaly_scores)}"
        )

    if len(anomaly_scores) == 0:
        raise ValueError(
            "Threshold seçimi için skor bulunamadı."
        )

    unique_scores = np.unique(
        anomaly_scores
    )

    epsilon = 1e-12

    threshold_candidates = np.concatenate(
        [
            [
                unique_scores.min()
                - epsilon
            ],
            unique_scores,
            [
                unique_scores.max()
                + epsilon
            ]
        ]
    )

    best_threshold = None
    best_metrics = None
    best_key = None

    for threshold in threshold_candidates:
        predictions = (
            anomaly_scores >= threshold
        ).astype(int)

        metrics = calculate_metrics(
            true_labels,
            predictions
        )

        comparison_key = (
            metrics["f1_score"],
            metrics["precision"],
            metrics["recall"],
            float(threshold)
        )

        if (
            best_key is None
            or comparison_key > best_key
        ):
            best_key = comparison_key
            best_threshold = float(
                threshold
            )
            best_metrics = metrics

    return (
        best_threshold,
        best_metrics
    )


def calculate_cusum(
    surprise_scores,
    reference_k
):
    surprise_scores = np.asarray(
        surprise_scores,
        dtype=float
    ).flatten()

    if reference_k < 0:
        raise ValueError(
            "CUSUM k değeri negatif olamaz."
        )

    cusum_scores = []
    cumulative_score = 0.0

    for surprise_score in surprise_scores:
        cumulative_score = max(
            0.0,
            cumulative_score
            + float(surprise_score)
            - float(reference_k)
        )

        cusum_scores.append(
            cumulative_score
        )

    return np.asarray(
        cusum_scores,
        dtype=float
    )


def select_cusum_parameters(
    true_labels,
    surprise_scores,
    k_quantiles
):
    """
    k: Validation normal surprise skorlarının belirtilen
    quantile değerlerinden üretilir.
    h: Her k değeri için oluşan CUSUM skorlarından otomatik
    threshold taramasıyla seçilir.
    """

    true_labels = np.asarray(
        true_labels,
        dtype=int
    ).flatten()

    surprise_scores = np.asarray(
        surprise_scores,
        dtype=float
    ).flatten()

    normal_surprises = surprise_scores[
        true_labels == 0
    ]

    if len(normal_surprises) == 0:
        raise ValueError(
            "CUSUM k seçimi için normal validation "
            "skoru bulunamadı."
        )

    cleaned_quantiles = []

    for quantile in k_quantiles:
        quantile = float(quantile)

        if quantile < 0 or quantile > 1:
            raise ValueError(
                "CUSUM quantile değerleri 0 ile 1 "
                "arasında olmalıdır."
            )

        cleaned_quantiles.append(
            quantile
        )

    candidates = []

    for quantile in cleaned_quantiles:
        reference_k = float(
            np.quantile(
                normal_surprises,
                quantile
            )
        )

        candidates.append(
            (
                quantile,
                reference_k
            )
        )

    #aynı k değerine karşılık gelen tekrarları kaldır
    unique_candidates = []
    seen_k_values = set()

    for quantile, reference_k in candidates:
        rounded_k = round(
            reference_k,
            12
        )

        if rounded_k not in seen_k_values:
            seen_k_values.add(
                rounded_k
            )

            unique_candidates.append(
                (
                    quantile,
                    reference_k
                )
            )

    best_result = None
    best_key = None

    search_rows = []

    for quantile, reference_k in unique_candidates:
        cusum_scores = calculate_cusum(
            surprise_scores,
            reference_k
        )

        selected_h, metrics = (
            select_score_threshold(
                true_labels,
                cusum_scores
            )
        )

        search_rows.append(
            {
                "k_quantile": quantile,
                "reference_k": reference_k,
                "selected_h": selected_h,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"]
            }
        )

        comparison_key = (
            metrics["f1_score"],
            metrics["precision"],
            metrics["recall"],
            selected_h,
            reference_k
        )

        if (
            best_key is None
            or comparison_key > best_key
        ):
            best_key = comparison_key

            best_result = {
                "k_quantile": quantile,
                "reference_k": reference_k,
                "selected_h": selected_h,
                "metrics": metrics,
                "cusum_scores": cusum_scores
            }

    return (
        best_result,
        pd.DataFrame(search_rows)
    )


def main():
    config = load_automata_config()

    window_size = int(
        config["window_size"]
    )

    alphabet_size = int(
        config["alphabet_size"]
    )

    k_quantiles = config[
        "cusum_k_quantiles"
    ]

    print(
        "\n--- BATADAL TRANSITION SURPRISE + "
        "CUSUM DENEYİ BAŞLATILIYOR ---"
    )

    print(
        f"window_size={window_size} | "
        f"alphabet_size={alphabet_size}"
    )

    print(
        f"CUSUM k quantiles={k_quantiles}"
    )

    # 1. VERİLERİ YÜKLE
    X_train = load_vector(
        "data/processed/"
        "batadal_X_train_adasyn_pc1.csv"
    )

    X_val = load_vector(
        "data/processed/"
        "batadal_X_val_pc1.csv"
    )

    y_val = load_vector(
        "data/processed/"
        "batadal_y_val.csv"
    )

    X_test = load_vector(
        "data/processed/"
        "batadal_X_test_pc1.csv"
    )

    y_test = load_vector(
        "data/processed/"
        "batadal_y_test.csv"
    )

    print("\n--- HAM VERİ KONTROLÜ ---")

    print(
        f"Train X : {len(X_train)}"
    )

    print(
        f"Val X/y : "
        f"{len(X_val)} / {len(y_val)}"
    )

    print(
        f"Test X/y: "
        f"{len(X_test)} / {len(y_test)}"
    )

    if len(X_val) != len(y_val):
        raise ValueError(
            "Validation X/y uzunlukları farklı."
        )

    if len(X_test) != len(y_test):
        raise ValueError(
            "Test X/y uzunlukları farklı."
        )

    # 2. PAA / SAX / SLIDING WINDOW
    transformer = SaxPaaTransformer(
        alphabet_size=alphabet_size
    )

    # Yalnızca train istatistikleri kullanılır.
    train_mean = float(
        np.mean(X_train)
    )

    train_std = float(
        np.std(X_train)
    )

    if np.isclose(
        train_std,
        0.0
    ):
        train_std = 1.0

    print(
        "\n--- ORTAK SAX NORMALİZASYONU ---"
    )

    print(
        f"Train mean={train_mean:.6f} | "
        f"Train std={train_std:.6f}"
    )

    train_patterns = transform_with_train_statistics(
        transformer=transformer,
        series=X_train,
        window_size=window_size,
        train_mean=train_mean,
        train_std=train_std
    )

    validation_patterns = (
        transform_with_train_statistics(
            transformer=transformer,
            series=X_val,
            window_size=window_size,
            train_mean=train_mean,
            train_std=train_std
        )
    )

    test_patterns = transform_with_train_statistics(
        transformer=transformer,
        series=X_test,
        window_size=window_size,
        train_mean=train_mean,
        train_std=train_std
    )

    validation_pattern_labels = (
        align_labels_to_patterns(
            y_val,
            window_size
        )
    )

    test_pattern_labels = (
        align_labels_to_patterns(
            y_test,
            window_size
        )
    )

    print(
        "\n--- PATTERN HİZALAMA KONTROLÜ ---"
    )

    print(
        f"Train pattern sayısı: "
        f"{len(train_patterns)}"
    )

    print(
        f"Val pattern/etiket  : "
        f"{len(validation_patterns)} / "
        f"{len(validation_pattern_labels)}"
    )

    print(
        f"Test pattern/etiket : "
        f"{len(test_patterns)} / "
        f"{len(test_pattern_labels)}"
    )

    if (
        len(validation_patterns)
        != len(validation_pattern_labels)
    ):
        raise ValueError(
            "Validation pattern ve etiket sayıları "
            "eşit değil."
        )

    if (
        len(test_patterns)
        != len(test_pattern_labels)
    ):
        raise ValueError(
            "Test pattern ve etiket sayıları eşit değil."
        )

    validation_transition_labels = (
        validation_pattern_labels[1:]
    )

    test_transition_labels = (
        test_pattern_labels[1:]
    )

    # 3. TEMEL OTOMATAYI EĞİT
    model = ProbabilisticAutomata(
        smoothing=True
    )

    model.fit(
        train_patterns
    )

    num_states = len(
        model.trained_patterns
    )

    num_transitions = sum(
        len(targets)
        for targets
        in model.transitions.values()
    )

    transition_density = (
        num_transitions
        / (num_states * num_states)
        if num_states > 0
        else 0.0
    )

    print("\n--- OTOMATA YAPISI ---")

    print(
        f"State sayısı       : {num_states}"
    )

    print(
        f"Geçiş sayısı       : {num_transitions}"
    )

    print(
        f"Geçiş yoğunluğu    : "
        f"{transition_density:.6f}"
    )

    # 4. VALIDATION SURPRISE SKORLARI
    (
        validation_surprises,
        validation_details
    ) = model.calculate_transition_surprises(
        validation_patterns
    )

    validation_surprises = np.asarray(
        validation_surprises,
        dtype=float
    )

    validation_transition_labels = np.asarray(
        validation_transition_labels,
        dtype=int
    )

    if (
        len(validation_surprises)
        != len(validation_transition_labels)
    ):
        raise ValueError(
            "Validation surprise ve geçiş etiketi "
            "sayıları eşit değil."
        )

    normal_validation_surprises = (
        validation_surprises[
            validation_transition_labels == 0
        ]
    )

    anomaly_validation_surprises = (
        validation_surprises[
            validation_transition_labels == 1
        ]
    )

    print(
        "\n--- VALIDATION SURPRISE DAĞILIMI ---"
    )

    print_score_distribution(
        "Normal surprise",
        normal_validation_surprises
    )

    print_score_distribution(
        "Anomaly surprise",
        anomaly_validation_surprises
    )

    print(
        "Farklı surprise sayısı: "
        f"{len(np.unique(validation_surprises))}"
    )

    validation_anomaly_rate = float(
        np.mean(validation_transition_labels)
    )

    print(
        "Validation anomaly oranı: "
        f"{validation_anomaly_rate:.4f}"
    )

    # 5. RAW SURPRISE THRESHOLD BASELINE
    (
        raw_surprise_threshold,
        raw_validation_metrics
    ) = select_score_threshold(
        validation_transition_labels,
        validation_surprises
    )

    raw_validation_predictions = (
        validation_surprises
        >= raw_surprise_threshold
    ).astype(int)

    (
        raw_validation_pr_auc,
        raw_validation_roc_auc
    ) = calculate_auc_metrics(
        validation_transition_labels,
        validation_surprises
    )

    print(
        "\n--- VALIDATION: RAW SURPRISE ---"
    )

    print(
        f"Threshold="
        f"{raw_surprise_threshold:.6f} | "
        f"Precision="
        f"{raw_validation_metrics['precision']:.4f} | "
        f"Recall="
        f"{raw_validation_metrics['recall']:.4f} | "
        f"F1="
        f"{raw_validation_metrics['f1_score']:.4f} | "
        f"PR-AUC="
        f"{raw_validation_pr_auc:.4f} | "
        f"ROC-AUC="
        f"{raw_validation_roc_auc:.4f}"
    )

    print(
        "Raw surprise anomaly tahmini: "
        f"{int(np.sum(raw_validation_predictions))}/"
        f"{len(raw_validation_predictions)}"
    )

    # 6. CUSUM k VE h SEÇİMİ

    (
        best_cusum,
        cusum_search_df
    ) = select_cusum_parameters(
        true_labels=validation_transition_labels,
        surprise_scores=validation_surprises,
        k_quantiles=k_quantiles
    )

    selected_k_quantile = best_cusum[
        "k_quantile"
    ]

    selected_k = best_cusum[
        "reference_k"
    ]

    selected_h = best_cusum[
        "selected_h"
    ]

    cusum_validation_metrics = best_cusum[
        "metrics"
    ]

    validation_cusum_scores = best_cusum[
        "cusum_scores"
    ]

    validation_cusum_predictions = (
        validation_cusum_scores
        >= selected_h
    ).astype(int)

    (
        cusum_validation_pr_auc,
        cusum_validation_roc_auc
    ) = calculate_auc_metrics(
        validation_transition_labels,
        validation_cusum_scores
    )

    print(
        "\n--- VALIDATION: SURPRISE + CUSUM ---"
    )

    print(
        f"Seçilen k quantile="
        f"{selected_k_quantile:.2f} | "
        f"k={selected_k:.6f} | "
        f"h={selected_h:.6f}"
    )

    print(
        f"Precision="
        f"{cusum_validation_metrics['precision']:.4f} | "
        f"Recall="
        f"{cusum_validation_metrics['recall']:.4f} | "
        f"F1="
        f"{cusum_validation_metrics['f1_score']:.4f} | "
        f"PR-AUC="
        f"{cusum_validation_pr_auc:.4f} | "
        f"ROC-AUC="
        f"{cusum_validation_roc_auc:.4f}"
    )

    print(
        "CUSUM anomaly tahmini: "
        f"{int(np.sum(validation_cusum_predictions))}/"
        f"{len(validation_cusum_predictions)}"
    )

    # ---------------------------------------------------------
    # 7. TEST SKORLARI
    # ---------------------------------------------------------

    (
        test_surprises,
        test_details
    ) = model.calculate_transition_surprises(
        test_patterns
    )

    test_surprises = np.asarray(
        test_surprises,
        dtype=float
    )

    test_transition_labels = np.asarray(
        test_transition_labels,
        dtype=int
    )

    if (
        len(test_surprises)
        != len(test_transition_labels)
    ):
        raise ValueError(
            "Test surprise ve geçiş etiketi sayıları "
            "eşit değil."
        )

    # Validation'dan seçilen raw surprise threshold.
    raw_test_predictions = (
        test_surprises
        >= raw_surprise_threshold
    ).astype(int)

    raw_test_metrics = calculate_metrics(
        test_transition_labels,
        raw_test_predictions
    )

    (
        raw_test_pr_auc,
        raw_test_roc_auc
    ) = calculate_auc_metrics(
        test_transition_labels,
        test_surprises
    )

    # Validation'dan seçilen k ile test CUSUM'u.
    test_cusum_scores = calculate_cusum(
        test_surprises,
        selected_k
    )

    # Validation'dan seçilen h testte aynen kullanılır.
    test_cusum_predictions = (
        test_cusum_scores
        >= selected_h
    ).astype(int)

    cusum_test_metrics = calculate_metrics(
        test_transition_labels,
        test_cusum_predictions
    )

    (
        cusum_test_pr_auc,
        cusum_test_roc_auc
    ) = calculate_auc_metrics(
        test_transition_labels,
        test_cusum_scores
    )

    print(
        "\n--- TEST: RAW SURPRISE ---"
    )

    print(
        f"Accuracy="
        f"{raw_test_metrics['accuracy']:.4f} | "
        f"Precision="
        f"{raw_test_metrics['precision']:.4f} | "
        f"Recall="
        f"{raw_test_metrics['recall']:.4f} | "
        f"F1="
        f"{raw_test_metrics['f1_score']:.4f} | "
        f"PR-AUC={raw_test_pr_auc:.4f} | "
        f"ROC-AUC={raw_test_roc_auc:.4f}"
    )

    print(
        "Raw surprise anomaly tahmini: "
        f"{int(np.sum(raw_test_predictions))}/"
        f"{len(raw_test_predictions)}"
    )

    print(
        "\n--- TEST: SURPRISE + CUSUM ---"
    )

    print(
        f"Accuracy="
        f"{cusum_test_metrics['accuracy']:.4f} | "
        f"Precision="
        f"{cusum_test_metrics['precision']:.4f} | "
        f"Recall="
        f"{cusum_test_metrics['recall']:.4f} | "
        f"F1="
        f"{cusum_test_metrics['f1_score']:.4f} | "
        f"PR-AUC={cusum_test_pr_auc:.4f} | "
        f"ROC-AUC={cusum_test_roc_auc:.4f}"
    )

    print(
        "CUSUM anomaly tahmini: "
        f"{int(np.sum(test_cusum_predictions))}/"
        f"{len(test_cusum_predictions)}"
    )

    # ---------------------------------------------------------
    # 8. SONUÇLARI KAYDET
    # ---------------------------------------------------------

    os.makedirs(
        "results/outputs",
        exist_ok=True
    )

    metrics_rows = [
        {
            "dataset": "BATADAL",
            "method": "raw_transition_surprise",
            "window_size": window_size,
            "alphabet_size": alphabet_size,
            "num_states": num_states,
            "num_transitions": num_transitions,
            "transition_density": transition_density,
            "selected_k_quantile": np.nan,
            "selected_k": np.nan,
            "selected_threshold":
                raw_surprise_threshold,

            "validation_precision":
                raw_validation_metrics["precision"],
            "validation_recall":
                raw_validation_metrics["recall"],
            "validation_f1":
                raw_validation_metrics["f1_score"],
            "validation_pr_auc":
                raw_validation_pr_auc,
            "validation_roc_auc":
                raw_validation_roc_auc,

            "test_accuracy":
                raw_test_metrics["accuracy"],
            "test_precision":
                raw_test_metrics["precision"],
            "test_recall":
                raw_test_metrics["recall"],
            "test_f1":
                raw_test_metrics["f1_score"],
            "test_pr_auc":
                raw_test_pr_auc,
            "test_roc_auc":
                raw_test_roc_auc
        },
        {
            "dataset": "BATADAL",
            "method": "transition_surprise_cusum",
            "window_size": window_size,
            "alphabet_size": alphabet_size,
            "num_states": num_states,
            "num_transitions": num_transitions,
            "transition_density": transition_density,
            "selected_k_quantile":
                selected_k_quantile,
            "selected_k":
                selected_k,
            "selected_threshold":
                selected_h,

            "validation_precision":
                cusum_validation_metrics["precision"],
            "validation_recall":
                cusum_validation_metrics["recall"],
            "validation_f1":
                cusum_validation_metrics["f1_score"],
            "validation_pr_auc":
                cusum_validation_pr_auc,
            "validation_roc_auc":
                cusum_validation_roc_auc,

            "test_accuracy":
                cusum_test_metrics["accuracy"],
            "test_precision":
                cusum_test_metrics["precision"],
            "test_recall":
                cusum_test_metrics["recall"],
            "test_f1":
                cusum_test_metrics["f1_score"],
            "test_pr_auc":
                cusum_test_pr_auc,
            "test_roc_auc":
                cusum_test_roc_auc
        }
    ]

    pd.DataFrame(
        metrics_rows
    ).to_csv(
        "results/outputs/"
        "automata_surprise_cusum_batadal_metrics.csv",
        index=False
    )

    cusum_search_df.to_csv(
        "results/outputs/"
        "automata_surprise_cusum_validation_search.csv",
        index=False
    )

    validation_df = pd.DataFrame(
        validation_details
    )

    validation_df["true_label"] = (
        validation_transition_labels
    )

    validation_df["raw_prediction"] = (
        raw_validation_predictions
    )

    validation_df["cusum_score"] = (
        validation_cusum_scores
    )

    validation_df["cusum_prediction"] = (
        validation_cusum_predictions
    )

    validation_df.to_csv(
        "results/outputs/"
        "automata_surprise_cusum_validation_scores.csv",
        index=False
    )

    test_df = pd.DataFrame(
        test_details
    )

    test_df["true_label"] = (
        test_transition_labels
    )

    test_df["raw_prediction"] = (
        raw_test_predictions
    )

    test_df["cusum_score"] = (
        test_cusum_scores
    )

    test_df["cusum_prediction"] = (
        test_cusum_predictions
    )

    test_df.to_csv(
        "results/outputs/"
        "automata_surprise_cusum_test_scores.csv",
        index=False
    )

    print(
        "\nSurprise + CUSUM sonuçları "
        "results/outputs klasörüne kaydedildi.\n"
    )


if __name__ == "__main__":
    main()