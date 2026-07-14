import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


# Dosya doğrudan çalıştırıldığında src importlarının bulunabilmesi için.
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..")
    )
)


from src.data_pipeline.sax_paa import SaxPaaTransformer
from src.experiments.evaluator import calculate_metrics
from src.models.likelihood_ratio_automata import (
    LikelihoodRatioAutomata
)

#
def load_automata_config(
    config_path="src/config/settings.json"
):

    default_config = {
        "window_size": 4,
        "alphabet_size": 3,
        "likelihood_smoothing_alpha": 1.0
    }

    if not os.path.exists(config_path):
        return default_config

    with open(config_path, "r", encoding="utf-8") as file:
        full_config = json.load(file)

    automata_config = full_config.get("automata", {})

    return {
        "window_size": automata_config.get(
            "window_size",
            default_config["window_size"]
        ),
        "alphabet_size": automata_config.get(
            "alphabet_size",
            default_config["alphabet_size"]
        ),
        "likelihood_smoothing_alpha": automata_config.get(
            "likelihood_smoothing_alpha",
            default_config["likelihood_smoothing_alpha"]
        )
    }

#csv dosyasını tek boyutlu numpy dızısı olarak yukler
def load_vector(file_path):

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Gerekli dosya bulunamadı: {file_path}"
        )

    return pd.read_csv(file_path).values.flatten()

#batadal etıketlerını 0 ve 1 bıcımıne donusturur (-999->0 ve pozitf değerleri->1)
def convert_labels_to_binary(labels):

    labels = np.asarray(labels).flatten()
    labels = np.where(labels == -999, 0, labels)
    labels = np.where(labels > 0, 1, 0)

    return labels.astype(int)

def print_adasyn_order_diagnostics(
    adasyn_labels,
    original_labels
):
    """
    ADASYN sonrasında oluşan eğitim sırasının geçiş modeli
    açısından incelenmesi için temel kontrolleri yazdırır.
    """

    adasyn_labels = convert_labels_to_binary(
        adasyn_labels
    )

    original_labels = convert_labels_to_binary(
        original_labels
    )

    original_length = len(original_labels)
    adasyn_length = len(adasyn_labels)
    synthetic_count = adasyn_length - original_length

    change_indices = np.flatnonzero(
        adasyn_labels[1:] != adasyn_labels[:-1]
    ) + 1

    boundaries = np.concatenate(
        [
            [0],
            change_indices,
            [adasyn_length]
        ]
    )

    run_lengths = np.diff(boundaries)

    print("\n--- ADASYN SIRA KONTROLÜ ---")

    print(
        f"Orijinal train uzunluğu : {original_length}"
    )

    print(
        f"ADASYN train uzunluğu   : {adasyn_length}"
    )

    print(
        f"Eklenen sentetik örnek  : {synthetic_count}"
    )

    print(
        "ADASYN normal/anomaly   : "
        f"{np.sum(adasyn_labels == 0)} / "
        f"{np.sum(adasyn_labels == 1)}"
    )

    print(
        "Ardışık etiket değişimi : "
        f"{len(change_indices)}"
    )

    print(
        "En uzun aynı sınıf bloğu: "
        f"{int(np.max(run_lengths))}"
    )

    print(
        "İlk 30 ADASYN etiketi   : "
        f"{adasyn_labels[:30].tolist()}"
    )

    print(
        "Son 30 ADASYN etiketi   : "
        f"{adasyn_labels[-30:].tolist()}"
    )

    boundary_start = max(
        0,
        original_length - 10
    )

    boundary_end = min(
        adasyn_length,
        original_length + 10
    )

    print(
        "Orijinal/sentetik sınırı: "
        f"{adasyn_labels[boundary_start:boundary_end].tolist()}"
    )

#ham zaman noktası etiketlerini sax pattern seviyesine hizalar
def align_labels_to_patterns(labels, window_size):

    labels = convert_labels_to_binary(labels)

    if window_size <= 0:
        raise ValueError(
            "window_size sıfırdan büyük olmalıdır."
        )

    #paa işlemi kalan noktaları kırptığı için etiketleri de aynı biçimde kırpıyoruz
    usable_length = (
        len(labels) // window_size
    ) * window_size

    if usable_length == 0:
        return np.array([], dtype=int)

    trimmed_labels = labels[:usable_length]

    # Her PAA bloğunda en az bir anomaly varsa o SAX sembolünün etiketi anomaly olur.
    paa_block_labels = trimmed_labels.reshape(
        -1,
        window_size
    ).max(axis=1)

    pattern_count = (
        len(paa_block_labels) - window_size + 1
    )

    if pattern_count <= 0:
        return np.array([], dtype=int)

    pattern_labels = []

    for start_index in range(pattern_count):
        end_index = start_index + window_size

        pattern_region = paa_block_labels[
            start_index:end_index
        ]

        pattern_label = int(
            np.max(pattern_region)
        )

        pattern_labels.append(pattern_label)

    return np.asarray(pattern_labels, dtype=int)

#oncelık f1 yuksekse
def select_validation_threshold(
    true_labels,
    likelihood_scores
):

    true_labels = np.asarray(
        true_labels,
        dtype=int
    ).flatten()

    likelihood_scores = np.asarray(
        likelihood_scores,
        dtype=float
    ).flatten()

    if len(true_labels) != len(likelihood_scores):
        raise ValueError(
            "Validation etiket ve skor uzunlukları eşit değil. "
            f"Etiket={len(true_labels)}, "
            f"skor={len(likelihood_scores)}"
        )

    if len(likelihood_scores) == 0:
        raise ValueError(
            "Validation likelihood-ratio skoru bulunamadı."
        )

    unique_scores = np.unique(likelihood_scores)

    # Bütün örnekleri normal veya bütün örnekleri anomaly yapabilecek sınır değerlerini de adaylara ekliyoruz.
    epsilon = 1e-12

    threshold_candidates = np.concatenate(
        [
            [unique_scores.min() - epsilon],
            unique_scores,
            [unique_scores.max() + epsilon]
        ]
    )

    best_threshold = None
    best_metrics = None
    best_comparison_key = None

    for threshold in threshold_candidates:
        predictions = (
            likelihood_scores >= threshold
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
            best_comparison_key is None
            or comparison_key > best_comparison_key
        ):
            best_comparison_key = comparison_key
            best_threshold = float(threshold)
            best_metrics = metrics

    return best_threshold, best_metrics


def print_score_distribution(
    score_name,
    scores
):
    scores = np.asarray(scores, dtype=float)

    if len(scores) == 0:
        print(f"{score_name}: skor bulunamadı.")
        return

    print(
        f"{score_name}: "
        f"n={len(scores)} | "
        f"min={np.min(scores):.4f} | "
        f"ortalama={np.mean(scores):.4f} | "
        f"medyan={np.median(scores):.4f} | "
        f"max={np.max(scores):.4f}"
    )

def transform_with_train_statistics(
    transformer,
    series,
    window_size,
    train_mean,
    train_std
):
    """
    SAX öncesinde bütün veri bölümlerini yalnızca train
    verisinden öğrenilen ortalama ve standart sapmayla dönüştürür.

    Veri seti, split ve etiketler değişmez.
    """

    series = np.asarray(series, dtype=float).flatten()

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

    return transformer.get_sliding_windows(
        sax_sequence,
        window_size
    )

def main():
    config = load_automata_config()

    window_size = int(config["window_size"])
    alphabet_size = int(config["alphabet_size"])
    smoothing_alpha = float(
        config["likelihood_smoothing_alpha"]
    )

    print(
        "\n--- BATADAL LIKELIHOOD-RATIO "
        "AUTOMATA DENEYİ BAŞLATILIYOR ---"
    )

    print(
        f"window_size={window_size} | "
        f"alphabet_size={alphabet_size} | "
        f"smoothing_alpha={smoothing_alpha}"
    )

    #1.veri yukleme 
    X_train = load_vector(
        "data/processed/"
        "batadal_X_train_adasyn_pc1.csv"
    )

    y_train = load_vector(
        "data/processed/"
        "batadal_y_train_adasyn.csv"
    )
    y_train_original = load_vector(
        "data/processed/"
        "batadal_y_train.csv"
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
        f"Train X/y : {len(X_train)} / {len(y_train)}"
    )
    print(
        f"Val X/y   : {len(X_val)} / {len(y_val)}"
    )
    print(
        f"Test X/y  : {len(X_test)} / {len(y_test)}"
    )

    for split_name, features, labels in [
        ("Train", X_train, y_train),
        ("Validation", X_val, y_val),
        ("Test", X_test, y_test)
    ]:
        if len(features) != len(labels):
            raise ValueError(
                f"{split_name} X/y uzunlukları farklı: "
                f"X={len(features)}, y={len(labels)}"
            )
    print_adasyn_order_diagnostics(
        adasyn_labels=y_train,
        original_labels=y_train_original
    )    

    # 2. SAX/PAA PATTERN'LERİNİ OLUŞTUR
    transformer = SaxPaaTransformer(
        alphabet_size=alphabet_size
    )
    # Normalizasyon değerleri yalnızca train verisinden öğrenilir.
    train_mean = float(np.mean(X_train))
    train_std = float(np.std(X_train))

    if train_std == 0:
        train_std = 1.0

    print("\n--- ORTAK SAX NORMALİZASYONU ---")
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

    validation_patterns = transform_with_train_statistics(
    transformer=transformer,
    series=X_val,
    window_size=window_size,
    train_mean=train_mean,
    train_std=train_std
    )

    test_patterns = transform_with_train_statistics(
    transformer=transformer,
    series=X_test,
    window_size=window_size,
    train_mean=train_mean,
    train_std=train_std
    )

    # 3. ETİKETLERİ PATTERN SEVİYESİNE HİZALA

    train_pattern_labels = align_labels_to_patterns(
        y_train,
        window_size
    )

    validation_pattern_labels = align_labels_to_patterns(
        y_val,
        window_size
    )

    test_pattern_labels = align_labels_to_patterns(
        y_test,
        window_size
    )

    print("\n--- PATTERN HİZALAMA KONTROLÜ ---")

    print(
        f"Train pattern/etiket : "
        f"{len(train_patterns)} / "
        f"{len(train_pattern_labels)}"
    )

    print(
        f"Val pattern/etiket   : "
        f"{len(validation_patterns)} / "
        f"{len(validation_pattern_labels)}"
    )

    print(
        f"Test pattern/etiket  : "
        f"{len(test_patterns)} / "
        f"{len(test_pattern_labels)}"
    )

    for split_name, patterns, labels in [
        (
            "Train",
            train_patterns,
            train_pattern_labels
        ),
        (
            "Validation",
            validation_patterns,
            validation_pattern_labels
        ),
        (
            "Test",
            test_patterns,
            test_pattern_labels
        )
    ]:
        if len(patterns) != len(labels):
            raise ValueError(
                f"{split_name} pattern ve etiket sayıları "
                f"eşit değil: pattern={len(patterns)}, "
                f"etiket={len(labels)}"
            )

    # pattern[i] -> pattern[i+1] geçişinin etiketi olarak
    # hedef pattern'in etiketi kullanılır.
    train_transition_labels = train_pattern_labels[1:]
    validation_transition_labels = (
        validation_pattern_labels[1:]
    )
    test_transition_labels = test_pattern_labels[1:]

    # 4. LIKELIHOOD-RATIO MODELİNİ EĞİT
    model = LikelihoodRatioAutomata(
        smoothing_alpha=smoothing_alpha
    )

    model.fit(
        train_patterns,
        train_transition_labels
    )

    print("\n--- EĞİTİM GEÇİŞLERİ ---")

    print(
        "Normal geçiş sayısı  : "
        f"{np.sum(train_transition_labels == 0)}"
    )

    print(
        "Anomaly geçiş sayısı : "
        f"{np.sum(train_transition_labels == 1)}"
    )

    print(
        "Benzersiz state sayısı: "
        f"{len(model.trained_patterns)}"
    )

    # 5. VALIDATION SKORLARINI ÜRET
    validation_scores, validation_details = model.score(
        validation_patterns
    )

    # Model skorları liste olarak döndürür.
    # Boolean maskeleme ve sayısal işlemler için NumPy dizisine çevrilir.
    validation_scores = np.asarray(
        validation_scores,
        dtype=float
    )

    validation_transition_labels = np.asarray(
        validation_transition_labels,
        dtype=int
    )

    if (
        len(validation_scores)
        != len(validation_transition_labels)
    ):       
        raise ValueError(
            "Validation skor ve geçiş etiketi sayıları "
            "eşit değil."
        )


    normal_validation_scores = validation_scores[
        validation_transition_labels == 0
    ]

    anomaly_validation_scores = validation_scores[
        validation_transition_labels == 1
    ]

    print("\n--- VALIDATION SKOR DAĞILIMI ---")

    print_score_distribution(
        "Normal skorlar",
        normal_validation_scores
    )

    print_score_distribution(
        "Anomaly skorlar",
        anomaly_validation_scores
    )

    print(
        "Farklı skor sayısı: "
        f"{len(np.unique(validation_scores))}"
    )

    validation_anomaly_rate = float(
        np.mean(validation_transition_labels)
    )

    print(
        "Validation anomaly oranı: "
        f"{validation_anomaly_rate:.4f}"
    )

    # PR-AUC'nin karşılaştırılacağı basit referans,
    # validation anomaly oranıdır.
    if (
        len(np.unique(validation_transition_labels))
        == 2
    ):
        validation_pr_auc = average_precision_score(
            validation_transition_labels,
            validation_scores
        )

        validation_roc_auc = roc_auc_score(
            validation_transition_labels,
            validation_scores
        )
    else:
        validation_pr_auc = float("nan")
        validation_roc_auc = float("nan")

    # 6. THRESHOLD'U VALIDATION ÜZERİNDEN SEÇ
    selected_threshold, validation_metrics = (
        select_validation_threshold(
            validation_transition_labels,
            validation_scores
        )
    )

    print("\n--- VALIDATION THRESHOLD SEÇİMİ ---")

    print(
        f"Seçilen threshold: "
        f"{selected_threshold:.6f}"
    )

    print(
        f"Precision="
        f"{validation_metrics['precision']:.4f} | "
        f"Recall="
        f"{validation_metrics['recall']:.4f} | "
        f"F1="
        f"{validation_metrics['f1_score']:.4f} | "
        f"PR-AUC={validation_pr_auc:.4f} | "
        f"ROC-AUC={validation_roc_auc:.4f}"
    )

    # 7. SEÇİLEN THRESHOLD İLE TEST
    (
        test_predictions,
        test_scores,
        test_details
    ) = model.predict(
        test_patterns,
        score_threshold=selected_threshold
    )
    # Metrik, indeksleme ve karşılaştırma işlemleri için
    # sonuçlar NumPy dizisine dönüştürülür.
    test_predictions = np.asarray(
        test_predictions,
        dtype=int
    )

    test_scores = np.asarray(
        test_scores,
        dtype=float
    )

    test_transition_labels = np.asarray(
        test_transition_labels,
        dtype=int
    )
    if (
        len(test_predictions)
        != len(test_transition_labels)
    ):
        raise ValueError(
            "Test tahmin ve geçiş etiketi sayıları "
            "eşit değil."
        )

    test_metrics = calculate_metrics(
        test_transition_labels,
        test_predictions
    )

    if len(np.unique(test_transition_labels)) == 2:
        test_pr_auc = average_precision_score(
            test_transition_labels,
            test_scores
        )

        test_roc_auc = roc_auc_score(
            test_transition_labels,
            test_scores
        )
    else:
        test_pr_auc = float("nan")
        test_roc_auc = float("nan")

    print(
        "\n--- TEST SONUCU: LIKELIHOOD-RATIO ---"
    )

    print(
        f"Accuracy={test_metrics['accuracy']:.4f} | "
        f"Precision={test_metrics['precision']:.4f} | "
        f"Recall={test_metrics['recall']:.4f} | "
        f"F1={test_metrics['f1_score']:.4f} | "
        f"PR-AUC={test_pr_auc:.4f} | "
        f"ROC-AUC={test_roc_auc:.4f}"
    )

    print(
        "Anomaly tahmini: "
        f"{int(np.sum(test_predictions))}/"
        f"{len(test_predictions)}"
    )

    # 8. SONUÇLARI KAYDET
    os.makedirs(
        "results/outputs",
        exist_ok=True
    )

    result_row = {
        "dataset": "BATADAL",
        "model": "likelihood_ratio_automata",
        "window_size": window_size,
        "alphabet_size": alphabet_size,
        "smoothing_alpha": smoothing_alpha,
        "selected_threshold": selected_threshold,

        "validation_anomaly_rate":
            validation_anomaly_rate,
        "validation_precision":
            validation_metrics["precision"],
        "validation_recall":
            validation_metrics["recall"],
        "validation_f1":
            validation_metrics["f1_score"],
        "validation_pr_auc":
            validation_pr_auc,
        "validation_roc_auc":
            validation_roc_auc,

        "test_accuracy":
            test_metrics["accuracy"],
        "test_precision":
            test_metrics["precision"],
        "test_recall":
            test_metrics["recall"],
        "test_f1":
            test_metrics["f1_score"],
        "test_pr_auc":
            test_pr_auc,
        "test_roc_auc":
            test_roc_auc
    }

    pd.DataFrame([result_row]).to_csv(
        "results/outputs/"
        "automata_likelihood_ratio_batadal_metrics.csv",
        index=False
    )

    validation_score_df = pd.DataFrame(
        validation_details
    )

    validation_score_df["true_label"] = (
        validation_transition_labels
    )

    validation_score_df["prediction"] = (
        validation_scores >= selected_threshold
    ).astype(int)

    validation_score_df.to_csv(
        "results/outputs/"
        "automata_likelihood_ratio_validation_scores.csv",
        index=False
    )

    test_score_df = pd.DataFrame(test_details)
    test_score_df["true_label"] = (
        test_transition_labels
    )
    test_score_df["prediction"] = test_predictions

    test_score_df.to_csv(
        "results/outputs/"
        "automata_likelihood_ratio_test_scores.csv",
        index=False
    )

    print(
        "\nLikelihood-ratio sonuçları "
        "results/outputs klasörüne kaydedildi."
    )

    print(
        "CUSUM henüz eklenmedi. Önce likelihood-ratio "
        "skorlarının ayrışması kontrol edilecek.\n"
    )


if __name__ == "__main__":
    main()