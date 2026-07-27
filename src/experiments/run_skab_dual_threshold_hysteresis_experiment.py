import json
import os
import sys
from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError: 
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
        "n_folds": int(automata.get("skab_n_folds", deep_learning.get("n_folds", 5))),
        "inner_n_splits": int(automata.get("alergia_skab_inner_n_splits", 5)),
        "merge_alphas": automata.get("alergia_merge_alpha_values", [0.01, 0.05, 0.1]),
        "min_counts": automata.get("alergia_min_state_count_values", [2, 5, 10]),
        "smoothing_alphas": automata.get("alergia_smoothing_alpha_values", [0.1, 0.5, 1.0]),
        "max_pattern_distance": automata.get("alergia_max_pattern_distance", None),
        "distance_penalties": automata.get(
            "alergia_distance_penalty_values",
            [0.0, 0.25, 0.5, 1.0, 2.0],
        ),
        #hysteresis deneyi için yeni parametreler
        "hysteresis_min_anomaly_runs": [
            int(value)
            for value in automata.get(
                "alergia_skab_hysteresis_min_anomaly_run_values",
                [1, 2, 3, 4, 5, 6, 7, 8, 10, 12],
            )
        ],
        "hysteresis_min_validation_recall": float(
            automata.get(
                "alergia_skab_hysteresis_min_validation_recall",
                0.70,
            )
        ),
        "hysteresis_quantile_count": int(
            automata.get(
                "alergia_skab_hysteresis_quantile_count",
                21,
            )
        ),
    }


def transform_patterns(transformer, series, window_size, train_mean, train_std):
    # Veriyi önce Z-score ile normalize edip, sonra PAA (boyut küçültme) ve SAX (harflere çevirme) işlemlerinden geçirir En son kayan pencere (sliding windows) formatına getirir
    normalized = (np.asarray(series, dtype=float) - train_mean) / train_std
    paa = transformer.apply_paa(normalized, window_size)
    sax = transformer.convert_to_sax(paa)
    return transformer.get_sliding_windows(sax, window_size)


def align_labels(labels, window_size):
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

#csv dosyalarını birbirine karıştırmamak için aynı dosyadan gelen verilerin indekslerini gruplar
def grouped_indices(source_files):
    grouped = OrderedDict()

    for index, source_file in enumerate(source_files):
        grouped.setdefault(str(source_file), []).append(index)

    for source_file, indices in grouped.items():
        yield source_file, np.asarray(indices, dtype=int)

#ham değerleri ve etiketleri dosya bazlı gruplara ayırarak dönüşüm işlemlerini (SAX/PAA) uygulayan ana veri hazırlama fonksiyonu
def prepare_grouped_patterns(values, labels, source_files, transformer, window_size, train_mean, train_std):
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

        # En az iki pattern olmalı ki aralarında geçiş (transition) bulabilelim.
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
    #oluşturduğumuz pattern geçişlerini normal ve anomali olarak iki farklı sequence listesine ayırır
    #sınıf değişimi sırasında (normalden anomaliye geçerken) sahte bir geçiş öğrenmesin diye diziyi oradan keser
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

        #etiket aynıysa sequence e eklemeye devam et
        if label == current_label:
            current_sequence.append(patterns[index + 1])
            continue

        #etiket değiştiyse mevcut sequence'i ilgili listeye kaydet ve yeni bir tane başlat
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

#tüm dosyalardaki normal ve anomali sequence'leri büyük listelerde birleştirir
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
    # Bir pattern dizisi geldiğinde hem normal modele hem anomali modeline puanlatıyoruz.
    # Hangi modelin olasılığı/skoru yüksekse veriyi o modelin daha iyi açıkladığını varsayacağız.
    normal_scores, normal_logs = normal_model.score_patterns(patterns)
    anomaly_scores, anomaly_logs = anomaly_model.score_patterns(patterns)

    if len(normal_scores) != len(anomaly_scores):
        raise ValueError("Normal ve anomaly skor sayıları farklı.")

    # Dual skor: (Normal Model Skoru - Anomali Model Skoru)
    # Skor pozitifse, geçiş anomali modeline daha uygun demektir.
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
                "normal_levenshtein_distance": normal_logs[index][
                    "levenshtein_distance"
                ],
                "normal_mapping_penalty": normal_logs[index][
                    "mapping_penalty"
                ],
                "normal_surprise": float(normal_scores[index]),
                "anomaly_mapped_to": anomaly_logs[index]["mapped_to"],
                "anomaly_probability": anomaly_logs[index]["transition_probability"],
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


def score_grouped(normal_model, anomaly_model, grouped_data):
    #tüm dosyalar için model skorlarını toplayıp tek bir dizi haline getiren bir wrapper
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


def score_grouped_items(normal_model, anomaly_model, grouped_data):
    """
    Her source_file dosyasını ayrı tutarak skor üretir.
    Hysteresis bir durum (state) makinesi mantığıyla çalıştığı için dosya sınırları kopmamalı.
    Bütün skorları tek bir dizi olarak birleştirirsek, bir dosyanın sonundaki anomali durumu 
    tamamen farklı bir dosyanın başındaki veriye taşar.
    Bu fonksiyon bu yüzden dosyaları liste içinde ayrı ayrı korur.
    """
    scored_items = []
    all_scores = []
    all_labels = []
    all_logs = []

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
        all_scores.extend(scores.tolist())
        all_labels.extend(labels.tolist())
        all_logs.extend(logs)

    return (
        scored_items,
        np.asarray(all_scores, dtype=float),
        np.asarray(all_labels, dtype=int),
        all_logs,
    )


def apply_dual_threshold_hysteresis(scores, high_threshold, low_threshold):
    """
    Sistem normalken anomali alarmı vermek için 'high_threshold' geçilmelidir.
    Alarm bir kere çaldığında, anomali durumunun bitmesi için skorun 'low_threshold' altına inmesi beklenir.
    Böylece sınırda dolaşan ufak gürültüler yüzünden sistem sürekli alarm açıp kapatmaz, daha istikrarlı olur.
    """
    scores = np.asarray(scores, dtype=float).flatten()
    high_threshold = float(high_threshold)
    low_threshold = float(low_threshold)

    if low_threshold > high_threshold:
        raise ValueError(
            "low_threshold, high_threshold değerinden büyük olamaz."
        )

    predictions = np.zeros(len(scores), dtype=int)
    anomaly_active = False

    for index, score in enumerate(scores):
        if not anomaly_active:
            if score >= high_threshold:
                anomaly_active = True
                predictions[index] = 1
        else:
            if score < low_threshold:
                anomaly_active = False
            else:
                predictions[index] = 1

    return predictions


def apply_temporal_persistence(predictions, min_anomaly_run):
    """
    Eğer sistem çok kısa süreli (örn: sadece 1 birim süren) bir anomali bulursa bunu yok say (filtrele).
    Gerçek anomaliler genelde arka arkaya belli bir süre devam eder. 'min_anomaly_run' parametresi bu süreyi belirler.
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

        #bulunan anomali bloğu yeterince uzunsa kabul et
        if end - start >= min_anomaly_run:
            filtered[start:end] = 1

        start = end

    return filtered


def evaluate_grouped_hysteresis_candidate(scored_items, high_threshold, low_threshold, min_anomaly_run):
    """
    Verilen eşik ve uzunluk parametrelerini her bir dosya üzerinde test eder.
    Ayrıca her dosyanın F1 skorunu hesaplayarak stabiliteyi (tüm dosyalarda tutarlı çalışıp çalışmadığını) ölçeer
    """
    all_raw_predictions = []
    all_predictions = []
    file_rows = []

    for item in scored_items:
        #ilk Hysteresis uygular
        raw_predictions = apply_dual_threshold_hysteresis(
            item["scores"],
            high_threshold,
            low_threshold,
        )
        # Sonra çok kısa anomalileri temizler
        predictions = apply_temporal_persistence(
            raw_predictions,
            min_anomaly_run,
        )
        file_metrics = calculate_metrics(item["labels"], predictions)

        all_raw_predictions.extend(raw_predictions.tolist())
        all_predictions.extend(predictions.tolist())
        file_rows.append(
            {
                "source_file": item["source_file"],
                "sample_count": int(len(item["labels"])),
                "anomaly_count": int(item["labels"].sum()),
                "raw_predicted_anomaly_count": int(raw_predictions.sum()),
                "predicted_anomaly_count": int(predictions.sum()),
                "precision": float(file_metrics["precision"]),
                "recall": float(file_metrics["recall"]),
                "f1_score": float(file_metrics["f1_score"]),
            }
        )

    f1_values = np.asarray(
        [row["f1_score"] for row in file_rows],
        dtype=float,
    )

    stability = {
        "file_rows": file_rows,
        "file_count": int(len(file_rows)),
        "mean_file_f1": float(np.mean(f1_values)) if len(f1_values) else 0.0,
        "min_file_f1": float(np.min(f1_values)) if len(f1_values) else 0.0,
        "std_file_f1": float(np.std(f1_values)) if len(f1_values) else 0.0,
    }

    return (
        np.asarray(all_raw_predictions, dtype=int),
        np.asarray(all_predictions, dtype=int),
        stability,
    )


def build_hysteresis_threshold_candidates(scores, baseline_threshold, quantile_count):
    """
    Yüksek ve düşük eşikler için mantıklı deneme noktaları oluşturur.
    Tüm olası skorları denemek çok uzun sürer, bu yüzden çeyrekliklere (quantiles) bölerek hız kazandırır
    """
    scores = np.asarray(scores, dtype=float).flatten()

    if len(scores) == 0:
        raise ValueError("Validation skorları boş olamaz.")

    quantile_count = max(int(quantile_count), 3)
    quantiles = np.linspace(0.0, 1.0, quantile_count)
    quantile_values = np.quantile(scores, quantiles)

    candidates = np.concatenate(
        (
            [scores.min() - 1e-12],
            quantile_values,
            [float(baseline_threshold)],
            [scores.max() + 1e-12],
        )
    )

    return np.asarray(
        sorted(set(float(value) for value in candidates)),
        dtype=float,
    )


def select_grouped_hysteresis_parameters(scored_items, validation_labels, baseline_threshold, min_anomaly_runs, min_validation_recall, quantile_count):
    """
    Inner-validation (iç doğrulama) veri seti üzerinde high_threshold, low_threshold ve min_anomaly_run 
    için grid search yapar ve en iyi F1 skorunu vereni seçer
    """
    validation_labels = np.asarray(validation_labels, dtype=int).flatten()
    all_scores = np.concatenate(
        [item["scores"] for item in scored_items]
    )

    if len(validation_labels) != len(all_scores) or len(all_scores) == 0:
        raise ValueError("Validation skor ve etiket sayıları uygun değil.")

    run_values = sorted(set(int(value) for value in min_anomaly_runs))
    if not run_values or run_values[0] < 1:
        raise ValueError("min_anomaly_run değerleri en az 1 olmalıdır.")

    threshold_candidates = build_hysteresis_threshold_candidates(
        all_scores,
        baseline_threshold,
        quantile_count,
    )

    search_rows = []
    best_eligible = None
    best_fallback = None

    for high_threshold in threshold_candidates:
        for low_threshold in threshold_candidates:
            if low_threshold > high_threshold:
                continue

            width = float(high_threshold - low_threshold)

            for min_anomaly_run in run_values:
                #modeli aday parametrelerle test edip metriklerini alır
                raw_predictions, predictions, stability = (
                    evaluate_grouped_hysteresis_candidate(
                        scored_items,
                        high_threshold,
                        low_threshold,
                        min_anomaly_run,
                    )
                )
                metrics = calculate_metrics(
                    validation_labels,
                    predictions,
                )
                
                #recall çok düşerse (çok fazla anomali kaçırırsak) o adayı elemek için 
                eligible = bool(
                    metrics["recall"] >= min_validation_recall
                )
                #yeni hysteresis hiçbir iyileştirme sağlamıyorsa sistemin eski baseline ayarına dönebilmesi için bir kontrol
                is_baseline = bool(
                    np.isclose(
                        high_threshold,
                        baseline_threshold,
                        rtol=0.0,
                        atol=1e-12,
                    )
                    and np.isclose(
                        low_threshold,
                        baseline_threshold,
                        rtol=0.0,
                        atol=1e-12,
                    )
                    and min_anomaly_run == 1
                )

                row = {
                    "high_threshold": float(high_threshold),
                    "low_threshold": float(low_threshold),
                    "hysteresis_width": width,
                    "min_anomaly_run": int(min_anomaly_run),
                    "is_baseline": is_baseline,
                    "recall_constraint_met": eligible,
                    "raw_predicted_anomaly_count": int(
                        raw_predictions.sum()
                    ),
                    "predicted_anomaly_count": int(predictions.sum()),
                    "validation_accuracy": float(metrics["accuracy"]),
                    "validation_precision": float(metrics["precision"]),
                    "validation_recall": float(metrics["recall"]),
                    "validation_f1_score": float(metrics["f1_score"]),
                    "validation_mean_file_f1": stability["mean_file_f1"],
                    "validation_min_file_f1": stability["min_file_f1"],
                    "validation_std_file_f1": stability["std_file_f1"],
                    "validation_file_details": json.dumps(
                        stability["file_rows"],
                        ensure_ascii=False,
                    ),
                }
                search_rows.append(row)

                candidate = {
                    "high_threshold": float(high_threshold),
                    "low_threshold": float(low_threshold),
                    "hysteresis_width": width,
                    "min_anomaly_run": int(min_anomaly_run),
                    "raw_predictions": raw_predictions.copy(),
                    "predictions": predictions.copy(),
                    "metrics": metrics,
                    "stability": stability,
                    "recall_constraint_met": eligible,
                    "selection_mode": (
                        "recall_constrained"
                        if eligible
                        else "fallback_highest_recall"
                    ),
                    "is_baseline": is_baseline,
                }

                #coklu sıralama koşulu: Aynı F1 skoruna sahip iki aday varsa sırasıyla
                #precision, recall, stabilite, hysteresis genişliği gibi detaylara bakarak daha sade olanı seçer
                eligible_key = (
                    metrics["f1_score"],
                    metrics["precision"],
                    metrics["recall"],
                    stability["mean_file_f1"],
                    stability["min_file_f1"],
                    -stability["std_file_f1"],
                    -width,
                    -int(predictions.sum()),
                    -int(min_anomaly_run),
                )
                fallback_key = (
                    metrics["recall"],
                    metrics["f1_score"],
                    metrics["precision"],
                    stability["mean_file_f1"],
                    stability["min_file_f1"],
                    -stability["std_file_f1"],
                    -width,
                    -int(predictions.sum()),
                    -int(min_anomaly_run),
                )

                if eligible and (
                    best_eligible is None
                    or eligible_key > best_eligible["key"]
                ):
                    best_eligible = {**candidate, "key": eligible_key}

                if (
                    best_fallback is None
                    or fallback_key > best_fallback["key"]
                ):
                    best_fallback = {**candidate, "key": fallback_key}

    best = best_eligible if best_eligible is not None else best_fallback

    if best is None:
        raise RuntimeError("Hysteresis için uygun validation adayı bulunamadı.")

    return best, search_rows, threshold_candidates


def predict_grouped_with_hysteresis(normal_model, anomaly_model, grouped_data, high_threshold, low_threshold, min_anomaly_run):
    """
    Test setine tahminde bulunurken logları detaylandıran ve seçilen hysteresis ayarlarını uygulayan fonksiyon.
    Hangi dosyanın neresinde eşik aşıldı, anomali ne zaman bitirildi gibi tüm kararları şeffafça loglar
    """
    all_predictions = []
    all_raw_predictions = []
    all_labels = []
    all_logs = []

    for item in grouped_data:
        scores, logs = score_dual(
            normal_model,
            anomaly_model,
            item["patterns"],
        )
        raw_predictions = apply_dual_threshold_hysteresis(
            scores,
            high_threshold,
            low_threshold,
        )
        predictions = apply_temporal_persistence(
            raw_predictions,
            min_anomaly_run,
        )

        if len(predictions) != len(item["transition_labels"]):
            raise ValueError(
                f"{item['source_file']} tahmin/etiket sayısı farklı."
            )

        anomaly_active = False
        for index, (score, raw_prediction, prediction, log) in enumerate(
            zip(scores, raw_predictions, predictions, logs)
        ):
            state_before = anomaly_active

            if not anomaly_active:
                if score >= high_threshold:
                    anomaly_active = True
            elif score < low_threshold:
                anomaly_active = False

            log["source_file"] = item["source_file"]
            log["score_index_in_file"] = int(index)
            log["high_threshold"] = float(high_threshold)
            log["low_threshold"] = float(low_threshold)
            log["hysteresis_width"] = float(
                high_threshold - low_threshold
            )
            log["min_anomaly_run"] = int(min_anomaly_run)
            log["hysteresis_state_before"] = (
                "anomaly" if state_before else "normal"
            )
            log["hysteresis_state_after"] = (
                "anomaly" if anomaly_active else "normal"
            )
            log["raw_decision"] = (
                "anomaly" if raw_prediction == 1 else "normal"
            )
            log["decision"] = (
                "anomaly" if prediction == 1 else "normal"
            )

        all_raw_predictions.extend(raw_predictions.tolist())
        all_predictions.extend(predictions.tolist())
        all_labels.extend(item["transition_labels"].tolist())
        all_logs.extend(logs)

    return (
        np.asarray(all_predictions, dtype=int),
        np.asarray(all_raw_predictions, dtype=int),
        np.asarray(all_labels, dtype=int),
        all_logs,
    )

#tahminlerin ortalamalarını hesaplar
def summarize_scores(labels, scores, logs, threshold):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    predictions = (scores >= threshold).astype(int)
    normal_scores = scores[labels == 0]
    anomaly_scores = scores[labels == 1]

    return {
        "validation_unique_score_count": int(len(np.unique(scores))),
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

#hysteresis olmayan baseline yaklaşımı için skor dağılımı üzerinden en iyi standart (tekil) eşiği bulur
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

#modelin dayanıklılığını (robustness) ölçmek için test verisine ufak Gaussian rastgele gürültü ekler
def inject_noise(series, noise_level, seed):
    rng = np.random.default_rng(seed)
    series = np.asarray(series, dtype=float)
    return series + rng.normal(0.0, noise_level, size=len(series))

#modelin unseen patternlere nası tepkı verdıgını olcer
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
    #otomata state merging (durum birleştirme) sonrasında modellerin ne kadar küçüldüğünü izler
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

#cross validation yaparken sınıflar dengeli dağılsın splittler olusturur
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


def choose_inner_split(X, y, source_files, transformer, config, fold_id, seed):
    #modeli doğrulamak (validation) için Fold içindeki eğitim verisini kendi içinde ikiye böler (Inner Split)
    #bu bölünmeyi yaparken anomali oranının genel eğitim setiyle tutarlı olmasına özen gösteri4
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

    #anomali oranı hedefe en yakın bölünmeyi seç
    return min(candidates, key=lambda item: item["key"])

#csv dosyalarını kaydetmek uzere experiment sonuçlarını içeren dictionary hazırlar
def make_result_row(fold_id, seed, scenario, method, metrics, selection, baseline_threshold, best, split_info, structure_stats, sample_count, anomaly_count, unseen_count, test_file_count, raw_predicted_anomaly_count, predicted_anomaly_count):
    return {
        "dataset": "SKAB",
        "model": "dual_alergia_hysteresis",
        "method": method,
        "fold": fold_id,
        "seed": seed,
        "scenario": scenario,
        "window_size": best["config"]["window_size"],
        "alphabet_size": best["config"]["alphabet_size"],
        "merge_alpha": best["config"]["merge_alpha"],
        "min_state_count": best["config"]["min_state_count"],
        "smoothing_alpha": best["config"]["smoothing_alpha"],
        "distance_penalty": best["config"]["distance_penalty"],
        "baseline_threshold": float(baseline_threshold),
        "high_threshold": selection["high_threshold"],
        "low_threshold": selection["low_threshold"],
        "hysteresis_width": selection["hysteresis_width"],
        "min_anomaly_run": selection["min_anomaly_run"],
        "baseline_selected": bool(selection.get("is_baseline", False)),
        "validation_selection_mode": selection["selection_mode"],
        "validation_recall_constraint_met": selection[
            "recall_constraint_met"
        ],
        "validation_accuracy": selection["metrics"]["accuracy"],
        "validation_precision": selection["metrics"]["precision"],
        "validation_recall": selection["metrics"]["recall"],
        "validation_f1_score": selection["metrics"]["f1_score"],
        "validation_mean_file_f1": selection["stability"][
            "mean_file_f1"
        ],
        "validation_min_file_f1": selection["stability"][
            "min_file_f1"
        ],
        "validation_std_file_f1": selection["stability"][
            "std_file_f1"
        ],
        "final_training_scope": "inner_train_calibrated",
        "inner_splitter": split_info["splitter"],
        "inner_split_id": split_info["split_id"],
        "inner_train_file_count": len(split_info["train_groups"]),
        "inner_validation_file_count": len(split_info["val_groups"]),
        "test_file_count": test_file_count,
        "sample_count": sample_count,
        "actual_anomaly_count": anomaly_count,
        "raw_predicted_anomaly_count": raw_predicted_anomaly_count,
        "predicted_anomaly_count": predicted_anomaly_count,
        "unseen_count": unseen_count,
        **structure_stats,
        **metrics,
    }

#ana deney akisi
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

    #train ve test verilerinde aynı sensör (source file) kaydının olup olmadığını test eder data leakage olmaması için
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
                    "distance_penalty": 0.0,
                }

                #modelleri eğitir
                normal_model, anomaly_model = fit_dual_models(
                    split_info["normal_sequences"],
                    split_info["anomaly_sequences"],
                    kwargs,
                )
                
                #validation seti üzerinde puanlama
                val_scores, val_labels, val_logs = score_grouped(
                    normal_model,
                    anomaly_model,
                    split_info["grouped_val"],
                )
                threshold, metrics = select_threshold(val_labels, val_scores)
                diagnostics = summarize_scores(
                    val_labels,
                    val_scores,
                    val_logs,
                    threshold,
                )
                stats = combine_statistics(normal_model, anomaly_model)
                val_unseen_count = sum(
                    log["overall_status"] == "unseen" for log in val_logs
                )

                validation_rows.append(
                    {
                        "dataset": "SKAB",
                        "search_stage": "state_merging",
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
                        **diagnostics,
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
                        "normal_model": normal_model,
                        "anomaly_model": anomaly_model,
                        "structure_stats": stats,
                        "threshold": threshold,
                        "metrics": metrics,
                        "config": {
                            "window_size": config["window_size"],
                            "alphabet_size": config["alphabet_size"],
                            "merge_alpha": float(merge_alpha),
                            "min_state_count": int(min_count),
                            "smoothing_alpha": float(smoothing_alpha),
                            "distance_penalty": 0.0,
                        },
                    }

    print("\n--- LEVENSHTEIN UZAKLIK CEZASI TARAMASI ---")

    penalty_best = None
    for distance_penalty in config["distance_penalties"]:
        distance_penalty = float(distance_penalty)
        best["normal_model"].distance_penalty = distance_penalty
        best["anomaly_model"].distance_penalty = distance_penalty

        val_scores, val_labels, val_logs = score_grouped(
            best["normal_model"],
            best["anomaly_model"],
            split_info["grouped_val"],
        )
        threshold, metrics = select_threshold(val_labels, val_scores)
        diagnostics = summarize_scores(
            val_labels,
            val_scores,
            val_logs,
            threshold,
        )
        val_unseen_count = sum(
            log["overall_status"] == "unseen" for log in val_logs
        )

        validation_rows.append(
            {
                "dataset": "SKAB",
                "search_stage": "distance_penalty",
                "fold": fold_id,
                "seed": seed,
                "inner_splitter": split_info["splitter"],
                "inner_split_id": split_info["split_id"],
                "window_size": config["window_size"],
                "alphabet_size": config["alphabet_size"],
                "merge_alpha": best["config"]["merge_alpha"],
                "min_state_count": best["config"]["min_state_count"],
                "smoothing_alpha": best["config"]["smoothing_alpha"],
                "max_pattern_distance": config["max_pattern_distance"],
                "distance_penalty": distance_penalty,
                "threshold": threshold,
                "validation_sample_count": len(val_labels),
                "validation_anomaly_count": int(val_labels.sum()),
                "validation_unseen_count": val_unseen_count,
                **best["structure_stats"],
                **{
                    f"validation_{key}": value
                    for key, value in metrics.items()
                },
                **diagnostics,
            }
        )

        print(
            f"distance_penalty={distance_penalty} | "
            f"threshold={threshold:.6f} | "
            f"F1={metrics['f1_score']:.4f} | "
            f"score_gap={diagnostics['validation_score_mean_gap']:.4f}"
        )

        key = (
            metrics["f1_score"],
            metrics["precision"],
            metrics["recall"],
            -int((val_scores >= threshold).sum()),
        )
        if penalty_best is None or key > penalty_best["key"]:
            penalty_best = {
                "key": key,
                "distance_penalty": distance_penalty,
                "threshold": threshold,
                "metrics": metrics,
            }

    best["normal_model"].distance_penalty = penalty_best[
        "distance_penalty"
    ]
    best["anomaly_model"].distance_penalty = penalty_best[
        "distance_penalty"
    ]
    best["threshold"] = penalty_best["threshold"]
    best["metrics"] = penalty_best["metrics"]
    best["config"]["distance_penalty"] = penalty_best[
        "distance_penalty"
    ]

    print("\n--- FOLD/SEED İÇİN EN İYİ ALERGIA AYARI ---")
    print(
        f"alpha={best['config']['merge_alpha']} | "
        f"min={best['config']['min_state_count']} | "
        f"smooth={best['config']['smoothing_alpha']} | "
        f"distance_penalty={best['config']['distance_penalty']} | "
        f"baseline_threshold={best['threshold']:.6f} | "
        f"val F1={best['metrics']['f1_score']:.4f}"
    )

    normal_model = best["normal_model"]
    anomaly_model = best["anomaly_model"]
    structure_stats = best["structure_stats"]
    final_train_mean = split_info["train_mean"]
    final_train_std = split_info["train_std"]

    scored_val_items, val_scores, val_labels, val_logs = score_grouped_items(
        normal_model,
        anomaly_model,
        split_info["grouped_val"],
    )

    print("\n--- INNER VALIDATION HYSTERESIS TARAMASI ---")
    hysteresis_best, hysteresis_rows, threshold_candidates = (
        select_grouped_hysteresis_parameters(
            scored_val_items,
            val_labels,
            best["threshold"],
            config["hysteresis_min_anomaly_runs"],
            config["hysteresis_min_validation_recall"],
            config["hysteresis_quantile_count"],
        )
    )

    for row in hysteresis_rows:
        validation_rows.append(
            {
                "dataset": "SKAB",
                "search_stage": "dual_threshold_hysteresis",
                "fold": fold_id,
                "seed": seed,
                "inner_splitter": split_info["splitter"],
                "inner_split_id": split_info["split_id"],
                "window_size": config["window_size"],
                "alphabet_size": config["alphabet_size"],
                "merge_alpha": best["config"]["merge_alpha"],
                "min_state_count": best["config"]["min_state_count"],
                "smoothing_alpha": best["config"]["smoothing_alpha"],
                "distance_penalty": best["config"]["distance_penalty"],
                "baseline_threshold": best["threshold"],
                "validation_sample_count": len(val_labels),
                "validation_anomaly_count": int(val_labels.sum()),
                **structure_stats,
                **row,
            }
        )

    #single threshold baselineı hesaplıyor
    baseline_raw, baseline_predictions, baseline_stability = (
        evaluate_grouped_hysteresis_candidate(
            scored_val_items,
            best["threshold"],
            best["threshold"],
            1,
        )
    )
    baseline_metrics = calculate_metrics(val_labels, baseline_predictions)
    baseline_selection = {
        "high_threshold": float(best["threshold"]),
        "low_threshold": float(best["threshold"]),
        "hysteresis_width": 0.0,
        "min_anomaly_run": 1,
        "raw_predictions": baseline_raw,
        "predictions": baseline_predictions,
        "metrics": baseline_metrics,
        "stability": baseline_stability,
        "recall_constraint_met": bool(
            baseline_metrics["recall"]
            >= config["hysteresis_min_validation_recall"]
        ),
        "selection_mode": "existing_single_threshold_baseline",
        "is_baseline": True,
    }

    print(
        f"Threshold adayı={len(threshold_candidates)} | "
        f"min_run adayı={config['hysteresis_min_anomaly_runs']}"
    )
    print(
        "Baseline -> "
        f"threshold={best['threshold']:.6f} | min_run=1 | "
        f"P={baseline_metrics['precision']:.4f} | "
        f"R={baseline_metrics['recall']:.4f} | "
        f"F1={baseline_metrics['f1_score']:.4f}"
    )
    print(
        "Hysteresis -> "
        f"high={hysteresis_best['high_threshold']:.6f} | "
        f"low={hysteresis_best['low_threshold']:.6f} | "
        f"min_run={hysteresis_best['min_anomaly_run']} | "
        f"P={hysteresis_best['metrics']['precision']:.4f} | "
        f"R={hysteresis_best['metrics']['recall']:.4f} | "
        f"F1={hysteresis_best['metrics']['f1_score']:.4f} | "
        f"baseline_selected={hysteresis_best['is_baseline']}"
    )

    #X_test üzerinde tahminde bulunuyor
    grouped_test = prepare_grouped_patterns(
        X_test,
        y_test,
        test_source,
        transformer,
        config["window_size"],
        final_train_mean,
        final_train_std,
    )

    (
        baseline_test_predictions,
        baseline_test_raw,
        test_labels,
        baseline_test_logs,
    ) = predict_grouped_with_hysteresis(
        normal_model,
        anomaly_model,
        grouped_test,
        baseline_selection["high_threshold"],
        baseline_selection["low_threshold"],
        baseline_selection["min_anomaly_run"],
    )
    (
        hysteresis_test_predictions,
        hysteresis_test_raw,
        hysteresis_test_labels,
        hysteresis_test_logs,
    ) = predict_grouped_with_hysteresis(
        normal_model,
        anomaly_model,
        grouped_test,
        hysteresis_best["high_threshold"],
        hysteresis_best["low_threshold"],
        hysteresis_best["min_anomaly_run"],
    )

    if not np.array_equal(test_labels, hysteresis_test_labels):
        raise RuntimeError("Baseline ve hysteresis test etiketleri farklı.")

    baseline_original_metrics = calculate_metrics(
        test_labels,
        baseline_test_predictions,
    )
    hysteresis_original_metrics = calculate_metrics(
        test_labels,
        hysteresis_test_predictions,
    )

    #sistemin dayanıklılığını görmek için test setine Gaussian gürültü ekleme
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

    (
        baseline_noisy_predictions,
        baseline_noisy_raw,
        noisy_labels,
        baseline_noisy_logs,
    ) = predict_grouped_with_hysteresis(
        normal_model,
        anomaly_model,
        grouped_noisy_test,
        baseline_selection["high_threshold"],
        baseline_selection["low_threshold"],
        baseline_selection["min_anomaly_run"],
    )
    (
        hysteresis_noisy_predictions,
        hysteresis_noisy_raw,
        hysteresis_noisy_labels,
        hysteresis_noisy_logs,
    ) = predict_grouped_with_hysteresis(
        normal_model,
        anomaly_model,
        grouped_noisy_test,
        hysteresis_best["high_threshold"],
        hysteresis_best["low_threshold"],
        hysteresis_best["min_anomaly_run"],
    )

    if not np.array_equal(noisy_labels, hysteresis_noisy_labels):
        raise RuntimeError("Noise baseline ve hysteresis etiketleri farklı.")

    baseline_noise_metrics = calculate_metrics(
        noisy_labels,
        baseline_noisy_predictions,
    )
    hysteresis_noise_metrics = calculate_metrics(
        noisy_labels,
        hysteresis_noisy_predictions,
    )

    baseline_unseen = evaluate_unseen(
        baseline_test_predictions,
        test_labels,
        baseline_test_logs,
    )
    hysteresis_unseen = evaluate_unseen(
        hysteresis_test_predictions,
        test_labels,
        hysteresis_test_logs,
    )

    baseline_unseen_count = sum(
        log["overall_status"] == "unseen"
        for log in baseline_test_logs
    )
    hysteresis_unseen_count = sum(
        log["overall_status"] == "unseen"
        for log in hysteresis_test_logs
    )
    baseline_noise_unseen_count = sum(
        log["overall_status"] == "unseen"
        for log in baseline_noisy_logs
    )
    hysteresis_noise_unseen_count = sum(
        log["overall_status"] == "unseen"
        for log in hysteresis_noisy_logs
    )

    rows = [
        make_result_row(
            fold_id, seed, "original", "baseline_single_threshold",
            baseline_original_metrics, baseline_selection,
            best["threshold"], best, split_info, structure_stats,
            len(test_labels), int(test_labels.sum()), baseline_unseen_count,
            len(test_files), int(baseline_test_raw.sum()),
            int(baseline_test_predictions.sum()),
        ),
        make_result_row(
            fold_id, seed, "original", "dual_threshold_hysteresis",
            hysteresis_original_metrics, hysteresis_best,
            best["threshold"], best, split_info, structure_stats,
            len(test_labels), int(test_labels.sum()), hysteresis_unseen_count,
            len(test_files), int(hysteresis_test_raw.sum()),
            int(hysteresis_test_predictions.sum()),
        ),
        make_result_row(
            fold_id, seed, "gaussian_noise", "baseline_single_threshold",
            baseline_noise_metrics, baseline_selection,
            best["threshold"], best, split_info, structure_stats,
            len(noisy_labels), int(noisy_labels.sum()),
            baseline_noise_unseen_count, len(test_files),
            int(baseline_noisy_raw.sum()),
            int(baseline_noisy_predictions.sum()),
        ),
        make_result_row(
            fold_id, seed, "gaussian_noise", "dual_threshold_hysteresis",
            hysteresis_noise_metrics, hysteresis_best,
            best["threshold"], best, split_info, structure_stats,
            len(noisy_labels), int(noisy_labels.sum()),
            hysteresis_noise_unseen_count, len(test_files),
            int(hysteresis_noisy_raw.sum()),
            int(hysteresis_noisy_predictions.sum()),
        ),
        make_result_row(
            fold_id, seed, "unseen_data", "baseline_single_threshold",
            baseline_unseen["metrics"], baseline_selection,
            best["threshold"], best, split_info, structure_stats,
            baseline_unseen["count"], baseline_unseen["anomaly_count"],
            baseline_unseen["count"], len(test_files),
            baseline_unseen["count"], baseline_unseen["count"],
        ),
        make_result_row(
            fold_id, seed, "unseen_data", "dual_threshold_hysteresis",
            hysteresis_unseen["metrics"], hysteresis_best,
            best["threshold"], best, split_info, structure_stats,
            hysteresis_unseen["count"], hysteresis_unseen["anomaly_count"],
            hysteresis_unseen["count"], len(test_files),
            hysteresis_unseen["count"], hysteresis_unseen["count"],
        ),
    ]

    best_config_row = {
        "dataset": "SKAB",
        "fold": fold_id,
        "seed": seed,
        **best["config"],
        "baseline_threshold": best["threshold"],
        "high_threshold": hysteresis_best["high_threshold"],
        "low_threshold": hysteresis_best["low_threshold"],
        "hysteresis_width": hysteresis_best["hysteresis_width"],
        "min_anomaly_run": hysteresis_best["min_anomaly_run"],
        "baseline_selected": hysteresis_best["is_baseline"],
        "validation_selection_mode": hysteresis_best["selection_mode"],
        "validation_recall_constraint_met": hysteresis_best[
            "recall_constraint_met"
        ],
        **{
            f"validation_{key}": value
            for key, value in hysteresis_best["metrics"].items()
        },
        "validation_mean_file_f1": hysteresis_best["stability"][
            "mean_file_f1"
        ],
        "validation_min_file_f1": hysteresis_best["stability"][
            "min_file_f1"
        ],
        "validation_std_file_f1": hysteresis_best["stability"][
            "std_file_f1"
        ],
        "inner_splitter": split_info["splitter"],
        "inner_split_id": split_info["split_id"],
        "inner_train_files": "|".join(split_info["train_groups"]),
        "inner_validation_files": "|".join(split_info["val_groups"]),
        **structure_stats,
    }

    print("\n--- FOLD TEST KARŞILAŞTIRMASI ---")
    print(
        "Baseline Original -> "
        f"P={baseline_original_metrics['precision']:.4f} | "
        f"R={baseline_original_metrics['recall']:.4f} | "
        f"F1={baseline_original_metrics['f1_score']:.4f}"
    )
    print(
        "Hysteresis Original -> "
        f"P={hysteresis_original_metrics['precision']:.4f} | "
        f"R={hysteresis_original_metrics['recall']:.4f} | "
        f"F1={hysteresis_original_metrics['f1_score']:.4f}"
    )
    print(
        "Baseline Noise -> "
        f"F1={baseline_noise_metrics['f1_score']:.4f} | "
        "Hysteresis Noise -> "
        f"F1={hysteresis_noise_metrics['f1_score']:.4f}"
    )
    print(
        f"Unseen baseline/hysteresis: "
        f"{baseline_unseen['count']}/{hysteresis_unseen['count']}"
    )
    return {
        "metrics_rows": rows,
        "validation_rows": validation_rows,
        "best_config_row": best_config_row,
        "explainability": {
            "validation": {
                "baseline": {
                    "threshold": best["threshold"],
                    "metrics": baseline_metrics,
                },
                "hysteresis": {
                    "high_threshold": hysteresis_best[
                        "high_threshold"
                    ],
                    "low_threshold": hysteresis_best[
                        "low_threshold"
                    ],
                    "min_anomaly_run": hysteresis_best[
                        "min_anomaly_run"
                    ],
                    "metrics": hysteresis_best["metrics"],
                    "file_details": hysteresis_best["stability"][
                        "file_rows"
                    ],
                },
                "logs": val_logs[:100],
            },
            "baseline_original": baseline_test_logs[:100],
            "hysteresis_original": hysteresis_test_logs[:100],
            "baseline_gaussian_noise": baseline_noisy_logs[:100],
            "hysteresis_gaussian_noise": hysteresis_noisy_logs[:100],
        },
    }

#cıkan bütün test metriklerini DataFrame üzerinden gruplayıp genel (overall) ortalama ve standart sapmalarını çıkarır
def create_summary(metrics_df):
    columns = ["accuracy", "precision", "recall", "f1_score", "unseen_count"]
    summary = metrics_df.groupby(["method", "scenario"])[columns].agg(["mean", "std"])
    summary.columns = [f"{column}_{stat}" for column, stat in summary.columns]
    return summary.reset_index()

#herbir fold için sonucları özetler
def create_fold_summary(metrics_df):
    columns = ["accuracy", "precision", "recall", "f1_score", "unseen_count"]
    summary = metrics_df.groupby(["fold", "method", "scenario"])[columns].agg(["mean", "std"])
    summary.columns = [f"{column}_{stat}" for column, stat in summary.columns]
    return summary.reset_index()


def main():    
    print("\n--- SKAB 5-FOLD × 5-SEED DUAL-THRESHOLD HYSTERESIS DENEYİ BAŞLATILIYOR ---")

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

    metrics_df.to_csv(os.path.join(OUTPUT_DIR, "skab_hysteresis_fold_seed_metrics.csv"),index=False,)
    validation_df.to_csv(os.path.join(OUTPUT_DIR, "skab_hysteresis_validation_search.csv"),index=False,)
    best_config_df.to_csv(os.path.join(OUTPUT_DIR, "skab_hysteresis_best_configs.csv"),index=False,)
    fold_summary_df.to_csv(os.path.join(OUTPUT_DIR, "skab_hysteresis_fold_summary.csv"),index=False,)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "skab_hysteresis_overall_summary.csv"),index=False,)

    with open(
        os.path.join(OUTPUT_DIR, "skab_hysteresis_explainability.json"),
        "w",
        "utf-8",
    ) as file:
        json.dump(explainability, file, indent=4, ensure_ascii=False)

    print("\n--- SKAB GENEL ÖZET ---")
    print(summary_df.to_string(index=False))
    print("\nSonuç dosyaları results/outputs klasörüne kaydedildi.")


if __name__ == "__main__":
    main()