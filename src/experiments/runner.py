import os
import json
import sys
import pandas as pd
import numpy as np
 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
 
from src.data_pipeline.sax_paa import SaxPaaTransformer
#from src.models.automata_model import ProbabilisticAutomata
from src.models.vomm_model import VariableOrderMarkovModel
from src.models.dual_vomm_model import DualVariableOrderMarkovModel
from src.experiments.evaluator import calculate_metrics
 
def load_json_config(config_path="src/config/settings.json"):
    default_config = {
        "window_size": 4,
        "alphabet_size": 3,
        "window_sizes": [3, 4, 5, 6],
        "alphabet_sizes": [3, 4, 5, 6], 
        "anomaly_threshold": 0.05,
        "skab_anomaly_threshold": 0.90,
        "batadal_anomaly_threshold": 0.05,
        "noise_level": 0.1,
        "seeds": [42, 123, 2026, 7, 999],
        "batadal_train_ratio": 0.60,
        "batadal_val_ratio": 0.20,
         #vomm-pst için ek parametreler
        "min_count": 2,
        "smoothing_alpha": 1.0,
        "min_count_values": [1, 2, 3, 5],
        "smoothing_alpha_values": [0.1, 0.5, 1.0],
        "vomm_threshold_values": [0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5],
        "min_context_depth": 0,
        "min_context_depth_values": [0, 1, 2, 3],
        "min_context_count": 0,
        "min_context_count_values": [0, 1, 2, 3, 5, 10, 20, 30],
        "smooth_window": 1,
        "smooth_window_values": [1, 2, 3, 5, 7],
    }
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                user_config = json.load(f)
            automata_config = user_config.get("automata", {})
            return {
                "window_size": automata_config.get("window_size", default_config["window_size"]),
                "alphabet_size": automata_config.get("alphabet_size", default_config["alphabet_size"]),
                "window_sizes": automata_config.get("window_sizes", default_config["window_sizes"]),
                "alphabet_sizes": automata_config.get("alphabet_sizes", default_config["alphabet_sizes"]),
                "anomaly_threshold": automata_config.get("anomaly_threshold", default_config["anomaly_threshold"]),
                "skab_anomaly_threshold": automata_config.get("skab_anomaly_threshold", default_config["skab_anomaly_threshold"]),
                "batadal_anomaly_threshold": automata_config.get("batadal_anomaly_threshold", default_config["batadal_anomaly_threshold"]),
                "noise_level": automata_config.get("noise_level", default_config["noise_level"]),
                "seeds": automata_config.get("seeds", default_config["seeds"]),
                "batadal_train_ratio": automata_config.get("batadal_train_ratio", default_config["batadal_train_ratio"]),
                "batadal_val_ratio": automata_config.get("batadal_val_ratio", default_config["batadal_val_ratio"]),
                "min_count": automata_config.get("min_count", default_config["min_count"]),
                "smoothing_alpha":  automata_config.get("smoothing_alpha", default_config["smoothing_alpha"]),
                "min_count_values":  automata_config.get("min_count_values", default_config["min_count_values"]),
                "smoothing_alpha_values": automata_config.get("smoothing_alpha_values", default_config["smoothing_alpha_values"]),
                "vomm_threshold_values": automata_config.get("vomm_threshold_values", default_config["vomm_threshold_values"]),
                "min_context_depth": automata_config.get("min_context_depth",default_config["min_context_depth"]),
                "min_context_depth_values": automata_config.get("min_context_depth_values",default_config["min_context_depth_values"]),
                "min_context_count": automata_config.get("min_context_count", default_config["min_context_count"]),
                "min_context_count_values": automata_config.get("min_context_count_values",default_config["min_context_count_values"]),
                "smooth_window": automata_config.get("smooth_window", default_config["smooth_window"]),
                "smooth_window_values": automata_config.get("smooth_window_values", default_config["smooth_window_values"])
            }
        except Exception:
            return default_config
    return default_config
 
def inject_gaussian_noise(series, noise_level=0.1, seed=42):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, noise_level, series.shape)
    return series + noise
#zaman sırasını bozmadan val ayırma 
def split_train_validation(X_train, y_train, val_ratio=0.2):
    split_idx = int(len(X_train) * (1 - val_ratio))

    X_model_train = X_train[:split_idx]
    X_val = X_train[split_idx:]

    y_model_train = y_train[:split_idx]
    y_val = y_train[split_idx:]

    return X_model_train, X_val, y_model_train, y_val

#etiketlerı pattern hizalama
def align_labels_to_patterns(y, num_patterns, paa_window_size, pattern_length=None):
    aligned_labels = []

    #eğer pattern_length ayrıca verilmezse window_size hem paa penceresi hem de pattern uzunluğu olarak kabul et
    if pattern_length is None:
        pattern_length = paa_window_size

    for pattern_idx in range(1, num_patterns):
        start_idx = pattern_idx * paa_window_size
        end_idx = min(
            (pattern_idx + pattern_length) * paa_window_size,
            len(y)
        )

        if end_idx <= start_idx:
            break

        label = 1 if np.any(y[start_idx:end_idx] == 1) else 0
        aligned_labels.append(label)

    return np.array(aligned_labels)
#threshold secme
def select_best_threshold_on_validation(
    model,
    transformer,
    X_val,
    y_val,
    threshold_values,
    window_size,
    min_context_depth_values=None,
    min_context_count_values=None,
    smooth_window_values=None
):
    best_threshold = threshold_values[0]
    best_context_depth = 0
    best_context_count = 0
    best_smooth_window = 1
    best_f1 = -1.0
    best_metrics = None

    if min_context_depth_values is None:
        min_context_depth_values = [0]
    if min_context_count_values is None:
        min_context_count_values = [0]
    if smooth_window_values is None:
        smooth_window_values = [1]

    val_patterns = transformer.transform(X_val, window_size=window_size)
    y_val_aligned = align_labels_to_patterns(y_val, len(val_patterns), window_size)

    for threshold in threshold_values:
        for min_context_depth in min_context_depth_values:
            for min_context_count in min_context_count_values:
                for smooth_window in smooth_window_values:
                    preds_val, _ = model.predict_smoothed(
                        val_patterns,
                        anomaly_threshold=threshold,
                        smooth_window=smooth_window,
                        min_context_depth=min_context_depth,
                        min_context_count=min_context_count
                    )

                    preds_val = preds_val[:len(y_val_aligned)]
                    metrics = calculate_metrics(y_val_aligned, preds_val)

                    if metrics["f1_score"] > best_f1:
                        best_f1 = metrics["f1_score"]
                        best_threshold = threshold
                        best_context_depth = min_context_depth
                        best_context_count = min_context_count
                        best_smooth_window = smooth_window
                        best_metrics = metrics

    return best_threshold, best_context_depth, best_context_count, best_smooth_window, best_metrics

#Pattern-level etiketlerden normal/anomaly sınıf oranlarını hesapla
#Dual-PST skoruna bu prior değerleri eklenerek ADASYN kaynaklı yapay sınıf dengesi etkisi azaltılmaya çalışılır.
def calculate_pattern_priors(pattern_labels):
    eps = 1e-12

    normal_count = np.sum(pattern_labels == 0)
    anomaly_count = np.sum(pattern_labels == 1)
    total_count = normal_count + anomaly_count

    if total_count == 0:
        return 0.5, 0.5

    prior_normal = normal_count / total_count
    prior_anomaly = anomaly_count / total_count

    #0 olasılık oluşmasını engelle
    prior_normal = max(prior_normal, eps)
    prior_anomaly = max(prior_anomaly, eps)

    return prior_normal, prior_anomaly

#skor yüksekse pattern anomalili modele daha yakın kabul edilir.
def select_best_dual_score_threshold_on_validation(
    model,
    val_patterns,
    y_val,
    score_threshold_values,
    window_size
):
    best_threshold = score_threshold_values[0]
    best_f1 = -1.0
    best_metrics = None

    # Validation etiketlerini pattern seviyesine hizalıyoruz.
    y_val_aligned = align_labels_to_patterns(
        y_val,
        len(val_patterns),
        window_size
    )
    #validation setindeki gerçek pattern-level sınıf dağılımını hesaplayıp dual-pst karar skoruna eklenecek.
    prior_normal, prior_anomaly = calculate_pattern_priors(y_val_aligned)

    for score_threshold in score_threshold_values:
        preds_val, _ = model.predict(
            val_patterns,
            score_threshold=score_threshold,
            prior_normal=prior_normal,
            prior_anomaly=prior_anomaly          
        )

        preds_val = preds_val[:len(y_val_aligned)]
        metrics = calculate_metrics(y_val_aligned, preds_val)

        # En iyi F1 veren skor threshold'unu seçiyoruz.
        if metrics["f1_score"] > best_f1:
            best_f1 = metrics["f1_score"]
            best_threshold = score_threshold
            best_metrics = metrics

    return best_threshold, best_metrics, prior_normal, prior_anomaly
#hiperparametre optimizasyonu
def select_best_vomm_config_on_validation(
    X_train,
    X_val,
    y_val,
    config,
    dataset_name
):
    best_config = None
    best_f1 = -1.0
    best_metrics = None

    window_sizes = config.get("window_sizes", [3, 4, 5, 6])
    alphabet_sizes = config.get("alphabet_sizes", [3, 4, 5, 6])

    #skab için min_count ve smoothing değerlerini de validation üzerinde tarıyoruz
    min_count_values = config.get("min_count_values", [1, 2, 3, 5])
    smoothing_alpha_values = config.get("smoothing_alpha_values", [0.1, 0.5, 1.0])

    threshold_values = config.get(
        "vomm_threshold_values",
        [0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5]
    )

    print(f"\n>> {dataset_name} validation tabanli VOMM/PST config taramasi...")

    for window_size in window_sizes:
        for alphabet_size in alphabet_sizes:
            # Her window/alphabet kombinasyonu için veriyi sembolik patternlere çeviriyoruz.
            transformer = SaxPaaTransformer(alphabet_size=alphabet_size)

            train_patterns = transformer.transform(
                X_train,
                window_size=window_size
            )

            val_patterns = transformer.transform(
                X_val,
                window_size=window_size
            )

            # Validation etiketlerini pattern seviyesine hizalıyoruz.
            # Böylece model tahminleri ile gerçek etiketler aynı seviyede karşılaştırılır.
            y_val_aligned = align_labels_to_patterns(
                y_val,
                len(val_patterns),
                window_size
            )

            for min_count in min_count_values:
                for smoothing_alpha in smoothing_alpha_values:
                    # Bu kombinasyon için VOMM/PST modelini kurup eğitiyoruz.
                    model = VariableOrderMarkovModel(
                        max_depth=window_size,
                        min_count=min_count,
                        smoothing=True,
                        smoothing_alpha=smoothing_alpha
                    )

                    model.fit(train_patterns)

                    for threshold in threshold_values:
                        preds_val, _ = model.predict(
                            val_patterns,
                            anomaly_threshold=threshold
                        )

                        preds_val = preds_val[:len(y_val_aligned)]

                        metrics = calculate_metrics(y_val_aligned, preds_val)

                        # En iyi validation F1 veren tüm parametreleri saklıyoruz.
                        if metrics["f1_score"] > best_f1:
                            best_f1 = metrics["f1_score"]
                            best_metrics = metrics
                            best_config = {
                                "window_size": window_size,
                                "alphabet_size": alphabet_size,
                                "min_count": min_count,
                                "smoothing_alpha": smoothing_alpha,
                                "anomaly_threshold": threshold
                            }


    print(
        f"{dataset_name} best validation config -> "
        f"window_size={best_config['window_size']}, "
        f"alphabet_size={best_config['alphabet_size']}, "
        f"min_count={best_config['min_count']}, "
        f"smoothing_alpha={best_config['smoothing_alpha']}, "
        f"threshold={best_config['anomaly_threshold']} | "
        f"val F1={best_f1:.4f}"
    )

    return best_config, best_metrics

def run_experiment_pipeline(X_train, X_test, y_test, config, dataset_name, fold_name="", seed=42,X_val=None,y_val=None,use_validation_threshold=True):
    results = []
    #SAX-PAA Dönüşümü ve Model Eğitimi
    transformer = SaxPaaTransformer(alphabet_size=config["alphabet_size"])
    train_patterns = transformer.transform(X_train, window_size=config["window_size"])
 
    model = VariableOrderMarkovModel(
        max_depth=config["window_size"],
        min_count=config.get("min_count", 2),
        smoothing=True,
        smoothing_alpha=config.get("smoothing_alpha", 1.0)
    )
    model.fit(train_patterns)
    validation_threshold_f1 = None
    selected_threshold = config["anomaly_threshold"]

    #context reliability filter için seçilen minimum context derinliği 0 olursa eski davranış koru
    selected_min_context_depth = config.get("min_context_depth", 0)
    selected_min_context_count = config.get("min_context_count", 0)

    validation_threshold_f1 = None
    selected_smooth_window = config.get("smooth_window", 1)
    #eğer aktifse dinamik eşik bulma mekanizmasını çalıştırır
    if use_validation_threshold and X_val is not None and y_val is not None:
       threshold_values = config.get(
           "vomm_threshold_values",
           [0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5]
        )
       min_context_depth_values = config.get(
            "min_context_depth_values",
            [0, 1, 2, 3]
        )
       min_context_count_values = config.get(
            "min_context_count_values",
            [0, 1, 2, 3, 5, 10, 20, 30]
        )
       smooth_window_values = config.get("smooth_window_values", [1, 2, 3, 5, 7])
       selected_threshold, selected_min_context_depth, selected_min_context_count, selected_smooth_window, val_metrics = select_best_threshold_on_validation(
            model,
            transformer,
            X_val,
            y_val,
            threshold_values,
            config["window_size"],
            min_context_depth_values,
            min_context_count_values,
            smooth_window_values
        )
       validation_threshold_f1 = val_metrics["f1_score"]
    if use_validation_threshold and X_val is not None and y_val is not None:
        print(
            f"{dataset_name} {fold_name} validation selected threshold: "
            f"{selected_threshold} | min_context_depth: {selected_min_context_depth} | "
            f"min_context_count: {selected_min_context_count} | "
            f"smooth_window: {selected_smooth_window} | "
            f"val F1: {validation_threshold_f1:.4f}"
        )
    #modelin yapısal karmaşıklığını hesaplama(gerçek durum/geçiş sayılarını pst ağacının içinden hesaplar)
    model_stats = model.model_stats

    num_states = model_stats["num_nodes"]#pstiçindeki toplam context/node sayısı

    num_transitions = model_stats["num_transitions"]#pst nodelarında gözlenen toplam farklı hedef sembol/geçiş sayısı.

    #geçiş yoğunluğu: teorik olarak mümkün durum çiftlerine göre
    #öğrenilen geçişlerin ne kadar yoğun olduğunu gösterir
    transition_density = (
        num_transitions / (num_states * num_states)
        if num_states > 0
        else 0.0
    )
 
    common_fields = {
        "dataset": dataset_name, "fold": fold_name, "seed": seed,
        "window_size": config["window_size"],
        "alphabet_size": config["alphabet_size"],
        "min_count": config.get("min_count", 2),
        "smoothing_alpha": config.get("smoothing_alpha", 1.0),
        "num_states": num_states, "num_transitions": num_transitions,
        "transition_density": transition_density,
        "selected_threshold": selected_threshold,
        "selected_min_context_depth": selected_min_context_depth,
        "selected_min_context_count": selected_min_context_count,
        "validation_threshold_f1": validation_threshold_f1,
        "selected_smooth_window": selected_smooth_window,
    }
 
    # --- SENARYO 1: Orijinal Veri ---
    test_patterns_orig = transformer.transform(X_test, window_size=config["window_size"])
    preds_orig, logs_orig = model.predict_smoothed(test_patterns_orig, anomaly_threshold=selected_threshold,smooth_window=selected_smooth_window,min_context_depth=selected_min_context_depth,min_context_count=selected_min_context_count)
    y_test_aligned_orig = align_labels_to_patterns(
        y_test,
        len(test_patterns_orig),
        config["window_size"]
    )
    preds_orig = preds_orig[:len(y_test_aligned_orig)]
    metrics_orig = calculate_metrics(y_test_aligned_orig, preds_orig)
    metrics_orig.update({"scenario": "original", **common_fields})
    results.append(metrics_orig)
 
    # --- SENARYO 2: Gaussian Noise ---
    X_test_noisy = inject_gaussian_noise(X_test, noise_level=config["noise_level"], seed=seed)
    test_patterns_noisy = transformer.transform(X_test_noisy, window_size=config["window_size"])
    preds_noisy, _ = model.predict_smoothed(test_patterns_noisy, anomaly_threshold=selected_threshold,smooth_window=selected_smooth_window,min_context_depth=selected_min_context_depth,min_context_count=selected_min_context_count)
    y_test_aligned_noisy = align_labels_to_patterns(
        y_test,
        len(test_patterns_noisy),
        config["window_size"]
    )
    preds_noisy = preds_noisy[:len(y_test_aligned_noisy)]
    metrics_noisy = calculate_metrics(y_test_aligned_noisy, preds_noisy)
    metrics_noisy.update({"scenario": "gaussian_noise", **common_fields})
    results.append(metrics_noisy)
 
    # --- SENARYO 3: Unseen Veri ---
    unseen_preds = []
    unseen_labels = []

    for pred_idx, pred in enumerate(preds_orig):
        pattern_idx = pred_idx + 1

        if pattern_idx < len(test_patterns_orig):
            current_pattern = test_patterns_orig[pattern_idx]

            if current_pattern not in model.trained_patterns:
                unseen_preds.append(pred)
                unseen_labels.append(y_test_aligned_orig[pred_idx])

    if len(unseen_preds) > 0:
        metrics_unseen = calculate_metrics(
            np.array(unseen_labels),
            np.array(unseen_preds)
        )
    else:
        metrics_unseen = {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0
        }

    metrics_unseen.update({"scenario": "unseen_data", **common_fields})
    results.append(metrics_unseen)

    return results, logs_orig
 
#parametre duyarlık analizi
def run_parameter_sensitivity_analysis(config):
    print("\n--- PARAMETRE DUYARLILIK ANALİZİ (GRID SEARCH) BAŞLATILIYOR ---")
    
    window_sizes = config.get("window_sizes", [3, 4, 5, 6])
    alphabet_sizes = config.get("alphabet_sizes", [3, 4, 5, 6])
    sensitivity_results = []
    
    # 1. batada (adasyn ile dengelenmiş veri) için parametre taraması
    if os.path.exists("data/processed/batadal_X_train_adasyn_pc1.csv"):
        X_train_b = pd.read_csv("data/processed/batadal_X_train_adasyn_pc1.csv").values.flatten()
        X_test_b = pd.read_csv("data/processed/batadal_X_test_pc1.csv").values.flatten()
        y_test_b = pd.read_csv("data/processed/batadal_y_test.csv").values.flatten()
        y_test_b = np.where(y_test_b == -999, 0, y_test_b)
 
        print("\n>> BATADAL (ADASYN) Parametre Taraması...")
        for w in window_sizes:
            for a in alphabet_sizes:
                cc = {
                    "window_size": w,
                    "alphabet_size": a,
                    "anomaly_threshold": config.get("batadal_anomaly_threshold", 0.05),
                    "noise_level": config["noise_level"]
                }
                res, _ = run_experiment_pipeline(X_train_b, X_test_b, y_test_b, cc, "BATADAL", "param_search", seed=config["seeds"][0])
                orig_res = [r for r in res if r["scenario"] == "original"][0]
                sensitivity_results.append(orig_res)
                print(f"BATADAL -> Window Size: {w}, Alphabet Size: {a} | Density: {orig_res['transition_density']:.4f}, F1: {orig_res['f1_score']:.4f}")
    #skab ıcın parametre taraması
    skab_train_path = "data/processed/skab_fold1_X_train_pc1.csv"
    skab_test_path = "data/processed/skab_fold1_X_test_pc1.csv"
    skab_y_path = "data/processed/skab_fold1_y_test.csv"
    
    if os.path.exists(skab_train_path) and os.path.exists(skab_test_path) and os.path.exists(skab_y_path):
        X_train_s = pd.read_csv(skab_train_path).values.flatten()
        X_test_s = pd.read_csv(skab_test_path).values.flatten()
        y_test_s = pd.read_csv(skab_y_path).values.flatten()
        
        print("\n>> SKAB Parametre Taraması...")
        for w in window_sizes:
            for a in alphabet_sizes:
                cc = {
                    "window_size": w,
                    "alphabet_size": a,
                    "anomaly_threshold": config.get("skab_anomaly_threshold", 0.90),
                    "noise_level": config["noise_level"]
                }
                res, _ = run_experiment_pipeline(X_train_s, X_test_s, y_test_s, cc, "SKAB", "param_search", seed=config["seeds"][0])
                orig_res = [r for r in res if r["scenario"] == "original"][0]
                sensitivity_results.append(orig_res)
                print(f"SKAB -> Window Size: {w}, Alphabet Size: {a} | Density: {orig_res['transition_density']:.4f}, F1: {orig_res['f1_score']:.4f}")
 
    if sensitivity_results:
        pd.DataFrame(sensitivity_results).to_csv("results/outputs/automata_param_sensitivity_metrics.csv", index=False)

#threshold analızı
def run_vomm_threshold_sensitivity_analysis(config):
    print("\n--- VOMM/PST THRESHOLD DUYARLILIK ANALİZİ BAŞLATILIYOR ---")

    threshold_results = []

    batadal_thresholds = [0.001, 0.005, 0.01, 0.02, 0.03, 0.05]
    skab_thresholds = [0.01, 0.03, 0.05, 0.10, 0.20, 0.30, 0.50]

    if os.path.exists("data/processed/batadal_X_train_adasyn_pc1.csv"):
        X_train_b = pd.read_csv("data/processed/batadal_X_train_adasyn_pc1.csv").values.flatten()
        X_test_b = pd.read_csv("data/processed/batadal_X_test_pc1.csv").values.flatten()
        y_test_b = pd.read_csv("data/processed/batadal_y_test.csv").values.flatten()
        y_test_b = np.where(y_test_b == -999, 0, y_test_b)

        print("\n>> BATADAL VOMM/PST Threshold Taraması...")

        for threshold in batadal_thresholds:
            cc = {
                **config,
                "window_size": 4,
                "alphabet_size": 3,
                "anomaly_threshold": threshold
            }

            res, _ = run_experiment_pipeline(
                X_train_b, X_test_b, y_test_b,
                cc, "BATADAL", "vomm_threshold_search",
                seed=config["seeds"][0]
            )

            orig_res = [r for r in res if r["scenario"] == "original"][0]
            threshold_results.append(orig_res)

            print(
                f"BATADAL -> Threshold: {threshold} | "
                f"Precision: {orig_res['precision']:.4f} | "
                f"Recall: {orig_res['recall']:.4f} | "
                f"F1: {orig_res['f1_score']:.4f}"
            )

    skab_train_path = "data/processed/skab_fold1_X_train_pc1.csv"
    skab_test_path = "data/processed/skab_fold1_X_test_pc1.csv"
    skab_y_path = "data/processed/skab_fold1_y_test.csv"

    if os.path.exists(skab_train_path) and os.path.exists(skab_test_path) and os.path.exists(skab_y_path):
        X_train_s = pd.read_csv(skab_train_path).values.flatten()
        X_test_s = pd.read_csv(skab_test_path).values.flatten()
        y_test_s = pd.read_csv(skab_y_path).values.flatten()

        print("\n>> SKAB VOMM/PST Threshold Taraması...")

        for threshold in skab_thresholds:
            cc = {
                **config,
                "window_size": 4,
                "alphabet_size": 4,
                "anomaly_threshold": threshold
            }

            res, _ = run_experiment_pipeline(
                X_train_s, X_test_s, y_test_s,
                cc, "SKAB", "vomm_threshold_search",
                seed=config["seeds"][0]
            )

            orig_res = [r for r in res if r["scenario"] == "original"][0]
            threshold_results.append(orig_res)

            print(
                f"SKAB -> Threshold: {threshold} | "
                f"Precision: {orig_res['precision']:.4f} | "
                f"Recall: {orig_res['recall']:.4f} | "
                f"F1: {orig_res['f1_score']:.4f}"
            )

    if threshold_results:
        pd.DataFrame(threshold_results).to_csv(
            "results/outputs/vomm_pst_threshold_sensitivity.csv",
            index=False
        ) 

#batadal ıyılestırma
def run_batadal_vomm_regularization_analysis(config):
    print("\n--- BATADAL VOMM/PST MIN_COUNT & SMOOTHING ANALİZİ BAŞLATILIYOR ---")

    results = []

    min_count_values = [1, 2, 3, 5, 10]
    smoothing_alpha_values = [0.01, 0.05, 0.1, 0.5, 1.0]

    if not os.path.exists("data/processed/batadal_X_train_adasyn_pc1.csv"):
        print("BATADAL verisi bulunamadı.")
        return

    X_train = pd.read_csv("data/processed/batadal_X_train_adasyn_pc1.csv").values.flatten()
    X_val = pd.read_csv("data/processed/batadal_X_val_pc1.csv").values.flatten()
    y_val = pd.read_csv("data/processed/batadal_y_val.csv").values.flatten()
    X_test = pd.read_csv("data/processed/batadal_X_test_pc1.csv").values.flatten()
    y_test = pd.read_csv("data/processed/batadal_y_test.csv").values.flatten()

    y_val = np.where(y_val == -999, 0, y_val)
    y_test = np.where(y_test == -999, 0, y_test)

    for min_count in min_count_values:
        for smoothing_alpha in smoothing_alpha_values:
            cc = {
                **config,
                "window_size": 6,
                "alphabet_size": 4,
                "anomaly_threshold": 0.05,
                "min_count": min_count,
                "smoothing_alpha": smoothing_alpha
            }

            res, _ = run_experiment_pipeline(
                X_train,
                X_test,
                y_test,
                cc,
                "BATADAL",
                "regularization_search",
                seed=config["seeds"][0],
                X_val=X_val,
                y_val=y_val,
                use_validation_threshold=True
            )

            original_result = [r for r in res if r["scenario"] == "original"][0]
            results.append(original_result)

            print(
                f"BATADAL -> min_count={min_count}, "
                f"smoothing_alpha={smoothing_alpha} | "
                f"threshold={original_result['selected_threshold']} | "
                f"Precision={original_result['precision']:.4f} | "
                f"Recall={original_result['recall']:.4f} | "
                f"F1={original_result['f1_score']:.4f}"
            )

    if results:
        df_results = pd.DataFrame(results)
        df_results.to_csv(
            "results/outputs/batadal_vomm_regularization_sensitivity.csv",
            index=False
        )

        best_row = df_results.loc[df_results["f1_score"].idxmax()]
        print("\nEn iyi BATADAL min_count & smoothing ayarı:")
        print(
            f"min_count={best_row['min_count']}, "
            f"smoothing_alpha={best_row['smoothing_alpha']}, "
            f"threshold={best_row['selected_threshold']} | "
            f"F1={best_row['f1_score']:.4f}"
        )    


#batadal için dual vomm-pst deneyi
def run_batadal_dual_vomm_experiment(config):
    print("\n--- BATADAL DUAL VOMM/PST ANALİZİ BAŞLATILIYOR ---")

    results = []

    required_files = [
        "data/processed/batadal_X_train_adasyn_pc1.csv",
        "data/processed/batadal_y_train_adasyn.csv",
        "data/processed/batadal_X_val_pc1.csv",
        "data/processed/batadal_y_val.csv",
        "data/processed/batadal_X_test_pc1.csv",
        "data/processed/batadal_y_test.csv"
    ]

    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"Eksik dosya bulundu, Dual VOMM/PST deneyi atlandı: {file_path}")
            return

    X_train = pd.read_csv("data/processed/batadal_X_train_adasyn_pc1.csv").values.flatten()
    y_train = pd.read_csv("data/processed/batadal_y_train_adasyn.csv").values.flatten()

    X_val = pd.read_csv("data/processed/batadal_X_val_pc1.csv").values.flatten()
    y_val = pd.read_csv("data/processed/batadal_y_val.csv").values.flatten()

    X_test = pd.read_csv("data/processed/batadal_X_test_pc1.csv").values.flatten()
    y_test = pd.read_csv("data/processed/batadal_y_test.csv").values.flatten()

    #batadalda -999 normal sınıfı temsil ettiği için 0 a çeviriyoruz.
    y_train = np.where(y_train == -999, 0, y_train)
    y_val = np.where(y_val == -999, 0, y_val)
    y_test = np.where(y_test == -999, 0, y_test)

    #onceki batadal analizlerinde en iyi çalışan sembolik temsil.
    window_size = 6
    alphabet_size = 4
    min_count = 3
    smoothing_alpha = 1.0

    transformer = SaxPaaTransformer(alphabet_size=alphabet_size)

    train_patterns = transformer.transform(
        X_train,
        window_size=window_size
    )

    val_patterns = transformer.transform(
        X_val,
        window_size=window_size
    )

    test_patterns = transformer.transform(
        X_test,
        window_size=window_size
    )

    #train labellarini pattern seviyesine hizalama
    y_train_aligned = align_labels_to_patterns(
        y_train,
        len(train_patterns),
        window_size
    )

    model = DualVariableOrderMarkovModel(
        max_depth=window_size,
        min_count=min_count,
        smoothing=True,
        smoothing_alpha=smoothing_alpha
    )

    model.fit(train_patterns, y_train_aligned)

    print(
        f"Dual-PST eğitim geçişleri -> "
        f"normal={model.normal_transition_count}, "
        f"anomaly={model.anomaly_transition_count}"
    )

    score_threshold_values = [-20, -15, -10, -7, -5, -3, -2, -1, -0.5,0, 0.5, 1, 2, 3, 5, 7, 10, 15, 20]

    selected_score_threshold, val_metrics, prior_normal, prior_anomaly = select_best_dual_score_threshold_on_validation(
        model,
        val_patterns,
        y_val,
        score_threshold_values,
        window_size
    )

    print(
        f"BATADAL Dual VOMM/PST validation selected score_threshold: "
        f"{selected_score_threshold} | val F1: {val_metrics['f1_score']:.4f} | "
        f"prior_normal={prior_normal:.4f}, prior_anomaly={prior_anomaly:.4f}"

    )

    # --- Orijinal test verisi ---
    preds_test, logs_test = model.predict(
        test_patterns,
        score_threshold=selected_score_threshold,
        prior_normal=prior_normal,
        prior_anomaly=prior_anomaly
    )
    y_test_aligned = align_labels_to_patterns(
        y_test,
        len(test_patterns),
        window_size
    )

    preds_test = preds_test[:len(y_test_aligned)]
    metrics_original = calculate_metrics(y_test_aligned, preds_test)

    metrics_original.update({
        "dataset": "BATADAL",
        "scenario": "dual_original",
        "window_size": window_size,
        "alphabet_size": alphabet_size,
        "min_count": min_count,
        "smoothing_alpha": smoothing_alpha,
        "selected_score_threshold": selected_score_threshold,
        "validation_f1": val_metrics["f1_score"],
        "normal_transition_count": model.normal_transition_count,
        "anomaly_transition_count": model.anomaly_transition_count
    })

    results.append(metrics_original)

    print(
        f"BATADAL Dual-PST Original -> "
        f"Precision={metrics_original['precision']:.4f} | "
        f"Recall={metrics_original['recall']:.4f} | "
        f"F1={metrics_original['f1_score']:.4f}"
    )

    # --- Gaussian noise testi ---
    for seed in config["seeds"]:
        X_test_noisy = inject_gaussian_noise(
            X_test,
            noise_level=config["noise_level"],
            seed=seed
        )

        noisy_patterns = transformer.transform(
            X_test_noisy,
            window_size=window_size
        )

        preds_noisy, _ = model.predict(
            noisy_patterns,
            score_threshold=selected_score_threshold,
            prior_normal=prior_normal,
            prior_anomaly=prior_anomaly
        )

        y_noisy_aligned = align_labels_to_patterns(
            y_test,
            len(noisy_patterns),
            window_size
        )

        preds_noisy = preds_noisy[:len(y_noisy_aligned)]
        metrics_noisy = calculate_metrics(y_noisy_aligned, preds_noisy)

        metrics_noisy.update({
            "dataset": "BATADAL",
            "scenario": "dual_gaussian_noise",
            "seed": seed,
            "window_size": window_size,
            "alphabet_size": alphabet_size,
            "min_count": min_count,
            "smoothing_alpha": smoothing_alpha,
            "selected_score_threshold": selected_score_threshold,
            "validation_f1": val_metrics["f1_score"],
            "normal_transition_count": model.normal_transition_count,
            "anomaly_transition_count": model.anomaly_transition_count
        })

        results.append(metrics_noisy)

    if results:
        pd.DataFrame(results).to_csv(
            "results/outputs/batadal_dual_vomm_pst_metrics.csv",
            index=False
        )

    with open("results/outputs/batadal_dual_vomm_pst_explainability.json", "w") as f:
        json.dump(logs_test[:100], f, indent=4)

    print("BATADAL Dual VOMM/PST sonuçları kaydedildi.")       
def main():
    config = load_json_config()
    seeds = config["seeds"]
    os.makedirs("results/outputs", exist_ok=True)
    all_res = []
 
    print("\n[INFO] SKAB Veri Seti Bölme Stratejisi:")
    print("-> Fiziksel düzenek bütünlüğü ve zamansal bağımlılıkları korumak adına GroupKFold mimarisi aktiftir.")
 
    # ANA EĞİTİM İÇİN ADASYN VERİSİ BAĞLANDI
    if os.path.exists("data/processed/batadal_X_train_adasyn_pc1.csv"):
        print("\n" + "="*60)
        print("[ADASYN ENTEGRASYONU] Otomata modeli ADASYN_PC1 verisiyle eğitiliyor...")
        print("="*60 + "\n")
        
        X_train = pd.read_csv("data/processed/batadal_X_train_adasyn_pc1.csv").values.flatten()
        X_test = pd.read_csv("data/processed/batadal_X_test_pc1.csv").values.flatten()
        y_test = pd.read_csv("data/processed/batadal_y_test.csv").values.flatten()
        y_test = np.where(y_test == -999, 0, y_test)
        X_val = pd.read_csv("data/processed/batadal_X_val_pc1.csv").values.flatten()
        y_val = pd.read_csv("data/processed/batadal_y_val.csv").values.flatten()
        y_val = np.where(y_val == -999, 0, y_val)
 
        config_batadal = {
            **config,
            "window_size": 6,
            "alphabet_size": 4,
            "anomaly_threshold": 0.005,
            "min_count": 3,
            "smoothing_alpha": 1.0,
            "min_context_depth_values": [0, 1, 2, 3],
            "min_context_count_values": [0, 1, 2, 3, 5, 10, 20, 30]
        }
 
        batadal_logs = None
        for seed in seeds:
            print(f"BATADAL automata seed={seed} çalışıyor...")
            batadal_res, logs = run_experiment_pipeline(
                X_train,
                X_test,
                y_test,
                config_batadal,
                "BATADAL",
                "single",
                seed=seed,
                X_val=X_val,
                y_val=y_val,
                use_validation_threshold=True
           )
            all_res.extend(batadal_res)
            if batadal_logs is None:
                batadal_logs = logs
 
        with open("results/outputs/automata_batadal_advanced_explainability.json", "w") as f:
            json.dump(batadal_logs[:100], f, indent=4)
 
    config_skab = {
        **config,
        "window_size": 4,
        "alphabet_size": 4,
        "anomaly_threshold": 0.5 
    }
 
    for fold in range(1, 6):
        train_file = f"data/processed/skab_fold{fold}_X_train_pc1.csv"
        test_file = f"data/processed/skab_fold{fold}_X_test_pc1.csv"
        y_test_file = f"data/processed/skab_fold{fold}_y_test.csv"
        if os.path.exists(train_file) and os.path.exists(test_file) and os.path.exists(y_test_file):
            X_train = pd.read_csv(train_file).values.flatten()
            X_test = pd.read_csv(test_file).values.flatten()
            y_test = pd.read_csv(y_test_file).values.flatten()
            y_train_file = f"data/processed/skab_fold{fold}_y_train.csv"
            if not os.path.exists(y_train_file):
                print(f"SKAB fold={fold} için y_train dosyası bulunamadı, bu fold atlandı.")
                continue

            y_train = pd.read_csv(y_train_file).values.flatten()

            X_model_train, X_val, _, y_val = split_train_validation(
                X_train,
                y_train,
                val_ratio=0.2
            )
            best_skab_config, best_skab_val_metrics = select_best_vomm_config_on_validation(
                X_model_train,
                X_val,
                y_val,
                config,
                f"SKAB fold_{fold}"
            )
            config_skab_fold = { **config,**best_skab_config}
                
            for seed in seeds:
                print(f"SKAB automata fold={fold}, seed={seed} çalışıyor...")
                fold_res, _ = run_experiment_pipeline(
                    X_model_train,
                    X_test,
                    y_test,
                    config_skab_fold,
                    "SKAB",
                    f"fold_{fold}",
                    seed=seed,
                    X_val=X_val,
                    y_val=y_val,
                    use_validation_threshold=False
                )
                all_res.extend(fold_res)
 
    df_all = pd.DataFrame(all_res)
    df_all.to_csv("results/outputs/automata_advanced_all_scenarios_metrics.csv", index=False)
 
    # SKAB özet
    df_skab = df_all[df_all["dataset"] == "SKAB"]
    if not df_skab.empty:
        summary_df = df_skab.groupby("scenario").agg(
            accuracy_mean=('accuracy', 'mean'),
            accuracy_std=('accuracy', 'std'),
            precision_mean=('precision', 'mean'),
            precision_std=('precision', 'std'),
            recall_mean=('recall', 'mean'),
            recall_std=('recall', 'std'),
            f1_score_mean=('f1_score', 'mean'),
            f1_score_std=('f1_score', 'std')
        ).reset_index()
        summary_df.to_csv("results/outputs/automata_skab_fold_summary.csv", index=False)
        print("\nSKAB VOMM/PST (5 fold x 5 seed):")
        print(summary_df)
 
    # BATADAL özet
    df_batadal = df_all[df_all["dataset"] == "BATADAL"]
    if not df_batadal.empty:
        batadal_summary = df_batadal.groupby("scenario").agg(
            accuracy_mean=('accuracy', 'mean'),
            accuracy_std=('accuracy', 'std'),
            precision_mean=('precision', 'mean'),
            precision_std=('precision', 'std'),
            recall_mean=('recall', 'mean'),
            recall_std=('recall', 'std'),
            f1_score_mean=('f1_score', 'mean'),
            f1_score_std=('f1_score', 'std')
        ).reset_index()
        batadal_summary.to_csv("results/outputs/automata_batadal_seed_summary.csv", index=False)
        print("\nBATADAL VOMM/PST (5 seed):")
        print(batadal_summary)
 
    #run_parameter_sensitivity_analysis(config)
    #run_vomm_threshold_sensitivity_analysis(config)
    #run_batadal_vomm_regularization_analysis(config)
    #run_batadal_dual_vomm_experiment(config)


    try:
        from src.experiments.statistical_tests import main as run_statistical_main
        run_statistical_main()
    except Exception as e:
        print(f"İstatistiksel testler tetiklenirken bir hata meydana geldi: {str(e)}")
    print("=" * 60 + "\n")
 
if __name__ == "__main__":
    main()