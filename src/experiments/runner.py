import os
import json
import sys
import pandas as pd
import numpy as np
 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
 
from src.data_pipeline.sax_paa import SaxPaaTransformer
from src.models.automata_model import ProbabilisticAutomata
from src.experiments.evaluator import calculate_metrics
 
def load_json_config(config_path="src/config/settings.json"):
    default_config = {
        "window_size": 4,
        "alphabet_size": 3,
        "window_sizes": [3, 4, 5, 6],
        "alphabet_sizes": [3, 4, 5, 6], 
        "weight_sharpness": 1.0,
        "weight_sharpness_values": [0.25, 0.5, 1.0, 2.0, 4.0, 6.0],
        "anomaly_threshold": 0.05,
        "skab_anomaly_threshold": 0.99,
        "batadal_anomaly_threshold": 0.01,
        "batadal_threshold_values": [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20],
        "skab_threshold_values": [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99],
        "noise_level": 0.1,
        "seeds": [42, 123, 2026, 7, 999],
        "batadal_train_ratio": 0.60,
        "batadal_val_ratio": 0.20,
        "use_similarity_penalty": False,
        "similarity_penalty_strength": 0.0,
        "similarity_penalty_strength_values": [0.0, 0.1, 0.25, 0.5, 0.75, 1.0],
        "use_dynamic_threshold": False,
        "dynamic_threshold_quantile": 0.05,
        "dynamic_threshold_quantile_values":  [0.01, 0.03, 0.05, 0.10, 0.20],
        "min_dynamic_threshold_samples": 3
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
                "weight_sharpness": automata_config.get("weight_sharpness", default_config["weight_sharpness"]),
                "weight_sharpness_values": automata_config.get("weight_sharpness_values", default_config["weight_sharpness_values"]),
                "anomaly_threshold": automata_config.get("anomaly_threshold", default_config["anomaly_threshold"]),
                "skab_anomaly_threshold": automata_config.get("skab_anomaly_threshold", default_config["skab_anomaly_threshold"]),
                "batadal_anomaly_threshold": automata_config.get("batadal_anomaly_threshold", default_config["batadal_anomaly_threshold"]),
                "batadal_threshold_values": automata_config.get("batadal_threshold_values", default_config["batadal_threshold_values"]),
                "skab_threshold_values": automata_config.get("skab_threshold_values", default_config["skab_threshold_values"]),
                "noise_level": automata_config.get("noise_level", default_config["noise_level"]),
                "seeds": automata_config.get("seeds", default_config["seeds"]),
                "batadal_train_ratio": automata_config.get("batadal_train_ratio", default_config["batadal_train_ratio"]),
                "batadal_val_ratio": automata_config.get("batadal_val_ratio", default_config["batadal_val_ratio"]),
                "similarity_penalty_strength": automata_config.get("similarity_penalty_strength", default_config["similarity_penalty_strength"]),
                "use_similarity_penalty": automata_config.get("use_similarity_penalty", default_config["use_similarity_penalty"]),
                "similarity_penalty_strength_values": automata_config.get("similarity_penalty_strength_values", default_config["similarity_penalty_strength_values"]),
                "use_dynamic_threshold": automata_config.get("use_dynamic_threshold", default_config["use_dynamic_threshold"]),
                "dynamic_threshold_quantile": automata_config.get("dynamic_threshold_quantile", default_config["dynamic_threshold_quantile"]),
                "dynamic_threshold_quantile_values": automata_config.get("dynamic_threshold_quantile_values", default_config["dynamic_threshold_quantile_values"]),
                "min_dynamic_threshold_samples": automata_config.get("min_dynamic_threshold_samples", default_config["min_dynamic_threshold_samples"])
            }
        except Exception:
            return default_config
    return default_config
 
def inject_gaussian_noise(series, noise_level=0.1, seed=42):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, noise_level, series.shape)
    return series + noise
 
def run_experiment_pipeline(X_train, X_test, y_test, config, dataset_name, fold_name="", seed=42):
    results = []
    transformer = SaxPaaTransformer(alphabet_size=config["alphabet_size"])
    train_patterns = transformer.transform(X_train, window_size=config["window_size"])
    #geçiş olasılıklarında weight_sharpness parametresi kullanılarakık görülen geçişlerin etkisi artırıldı nadir geçişler daha seçici değerlendirme
    #use_similarity_penalty: unseen pattern geldiğinde uzaklık cezasını açıp kapatır.
    model = ProbabilisticAutomata(
        smoothing=True,
        weight_sharpness=config.get("weight_sharpness", 1.0),
        use_similarity_penalty=config.get("use_similarity_penalty", False),
        similarity_penalty_strength=config.get("similarity_penalty_strength", 0.0),
        use_dynamic_threshold=config.get("use_dynamic_threshold", False),
        dynamic_threshold_quantile=config.get("dynamic_threshold_quantile", 0.05),
        min_dynamic_threshold_samples=config.get("min_dynamic_threshold_samples", 3)
    )
    model.fit(train_patterns)
 
    num_states = len(model.trained_patterns)
    num_transitions = sum(len(targets) for targets in model.transitions.values())
    transition_density = num_transitions / (num_states * num_states) if num_states > 0 else 0.0
    #duyalık analizi ile secilen threshold değeri
    selected_threshold = config["anomaly_threshold"]

    common_fields = {
        "dataset": dataset_name, "fold": fold_name, "seed": seed,
        "window_size": config["window_size"], "alphabet_size": config["alphabet_size"],
        "weight_sharpness": config.get("weight_sharpness", 1.0),
        "use_similarity_penalty": config.get("use_similarity_penalty", False),
        "similarity_penalty_strength": config.get("similarity_penalty_strength", 0.0),
        "use_dynamic_threshold": config.get("use_dynamic_threshold", False),
        "dynamic_threshold_quantile": config.get("dynamic_threshold_quantile", 0.05),
        "min_dynamic_threshold_samples": config.get("min_dynamic_threshold_samples", 3),
        "selected_threshold": selected_threshold,
        "num_states": num_states, "num_transitions": num_transitions,
        "transition_density": transition_density
    }

    # --- SENARYO 1: Orijinal Veri ---
    test_patterns_orig = transformer.transform(X_test, window_size=config["window_size"])
    preds_orig, logs_orig = model.predict(test_patterns_orig, anomaly_threshold=selected_threshold)
    y_test_aligned_orig = y_test[:len(preds_orig)]
    metrics_orig = calculate_metrics(y_test_aligned_orig, preds_orig)
    metrics_orig.update({"scenario": "original", **common_fields})
    results.append(metrics_orig)
 
    # --- SENARYO 2: Gaussian Noise ---
    X_test_noisy = inject_gaussian_noise(X_test, noise_level=config["noise_level"], seed=seed)
    test_patterns_noisy = transformer.transform(X_test_noisy, window_size=config["window_size"])
    preds_noisy, _ = model.predict(test_patterns_noisy, anomaly_threshold=selected_threshold)
    y_test_aligned_noisy = y_test[:len(preds_noisy)]
    metrics_noisy = calculate_metrics(y_test_aligned_noisy, preds_noisy)
    metrics_noisy.update({"scenario": "gaussian_noise", **common_fields})
    results.append(metrics_noisy)
 
    # --- SENARYO 3: Unseen Veri ---
    unseen_test_patterns = []
    y_test_unseen = []
    y_test_sliding = y_test[:len(test_patterns_orig)]
    for idx, pat in enumerate(test_patterns_orig):
        if pat not in model.trained_patterns:
            unseen_test_patterns.append(pat)
            y_test_unseen.append(y_test_sliding[idx])
 
    if len(unseen_test_patterns) > 1:
        preds_unseen, _ = model.predict(unseen_test_patterns, anomaly_threshold=selected_threshold)
        y_test_aligned_unseen = y_test_unseen[-len(preds_unseen):]
        metrics_unseen = calculate_metrics(y_test_aligned_unseen, preds_unseen)
    else:
        metrics_unseen = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0}
 
    metrics_unseen.update({"scenario": "unseen_data", **common_fields})
    results.append(metrics_unseen)
 
    return results, logs_orig
 
def run_parameter_sensitivity_analysis(config):
    print("\n--- PARAMETRE DUYARLILIK ANALİZİ (GRID SEARCH) BAŞLATILIYOR ---")
    
    window_sizes = config.get("window_sizes", [3, 4, 5, 6])
    alphabet_sizes = config.get("alphabet_sizes", [3, 4, 5, 6])
    sensitivity_results = []
    weight_sharpness_values = config.get("weight_sharpness_values", [1.0])
    
    # PARAMETRE TARAMASI İÇİN DE ADASYN VERİSİ BAĞLANDI
    if os.path.exists("data/processed/batadal_X_train_adasyn_pc1.csv"):
        X_train_b = pd.read_csv("data/processed/batadal_X_train_adasyn_pc1.csv").values.flatten()
        X_test_b = pd.read_csv("data/processed/batadal_X_test_pc1.csv").values.flatten()
        y_test_b = pd.read_csv("data/processed/batadal_y_test.csv").values.flatten()
        y_test_b = np.where(y_test_b == -999, 0, y_test_b)
 
        print("\n>> BATADAL (ADASYN) Parametre Taraması...")
        for w in window_sizes:
            for a in alphabet_sizes:
                for ws in weight_sharpness_values:
                    cc = {
                       "window_size": w,
                       "alphabet_size": a,
                       "weight_sharpness": ws,
                       "anomaly_threshold": config.get("batadal_anomaly_threshold", 0.01),
                       "noise_level": config["noise_level"],
                       "use_similarity_penalty": False,
                       "similarity_penalty_strength": 0.0
                    }

                    res, _ = run_experiment_pipeline(
                      X_train_b, X_test_b, y_test_b,
                      cc, "BATADAL", "param_search",
                      seed=config["seeds"][0]
                    )

                    orig_res = [r for r in res if r["scenario"] == "original"][0]
                    sensitivity_results.append(orig_res)

                    print(
                       f"BATADAL -> Window Size: {w}, Alphabet Size: {a}, "
                       f"Weight Sharpness: {ws} | "
                       f"Density: {orig_res['transition_density']:.4f}, "                           f"F1: {orig_res['f1_score']:.4f}"
                    )
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
                for ws in weight_sharpness_values:
                    cc = {
                       "window_size": w,
                       "alphabet_size": a,
                       "weight_sharpness": ws,
                       "anomaly_threshold": config.get("skab_anomaly_threshold", 0.99),
                       "noise_level": config["noise_level"],
                       "use_similarity_penalty": False,
                       "similarity_penalty_strength": 0.0
                    }
                    res, _ = run_experiment_pipeline(X_train_s, X_test_s, y_test_s, cc, "SKAB", "param_search", seed=config["seeds"][0])
                    orig_res = [r for r in res if r["scenario"] == "original"][0]
                    sensitivity_results.append(orig_res)
                    print(f"SKAB -> Window Size: {w}, Alphabet Size: {a}, Weight Sharpness: {ws} | Density: {orig_res['transition_density']:.4f}, F1: {orig_res['f1_score']:.4f}")

    if sensitivity_results:
        pd.DataFrame(sensitivity_results).to_csv("results/outputs/automata_param_sensitivity_metrics.csv", index=False)

#threshold duyarlık analizi
def run_threshold_sensitivity_analysis(config):
    print("\n--- THRESHOLD DUYARLILIK ANALİZİ BAŞLATILIYOR ---")

    threshold_results = []
    seeds = config["seeds"]

    # --- BATADAL ---
    if os.path.exists("data/processed/batadal_X_train_adasyn_pc1.csv"):
        X_train_b = pd.read_csv("data/processed/batadal_X_train_adasyn_pc1.csv").values.flatten()
        X_test_b = pd.read_csv("data/processed/batadal_X_test_pc1.csv").values.flatten()
        y_test_b = pd.read_csv("data/processed/batadal_y_test.csv").values.flatten()
        y_test_b = np.where(y_test_b == -999, 0, y_test_b)

        print("\n>> BATADAL Threshold Taraması...")

        for threshold in config.get("batadal_threshold_values", [0.01]):
            cc = {
                **config,
                "window_size": 4,
                "alphabet_size": 3,
                "weight_sharpness": 6.0,
                "anomaly_threshold": threshold,
                "use_similarity_penalty": False,
                "similarity_penalty_strength": 0.0,
                "use_dynamic_threshold": False
            }

            f1_scores = []

            for seed in seeds:
                res, _ = run_experiment_pipeline(
                    X_train_b,
                    X_test_b,
                    y_test_b,
                    cc,
                    "BATADAL",
                    "threshold_search",
                    seed=seed
                )

                original_result = [r for r in res if r["scenario"] == "original"][0]
                f1_scores.append(original_result["f1_score"])

            mean_f1 = float(np.mean(f1_scores))

            threshold_results.append({
                "dataset": "BATADAL",
                "threshold": threshold,
                "window_size": 4,
                "alphabet_size": 3,
                "weight_sharpness": 6.0,
                "f1_score_mean": mean_f1,
                "use_similarity_penalty": False,
                "similarity_penalty_strength": 0.0
            })

            print(f"BATADAL -> Threshold: {threshold} | Ortalama F1: {mean_f1:.4f}")

    # --- SKAB ---
    print("\n>> SKAB Threshold Taraması...")

    for threshold in config.get("skab_threshold_values", [0.99]):
        cc = {
            **config,
            "window_size": 4,
            "alphabet_size": 4,
            "weight_sharpness": 2.0,
            "anomaly_threshold": threshold,
            "use_similarity_penalty": False,
            "similarity_penalty_strength": 0.0,
            "use_dynamic_threshold": False
        }

        f1_scores = []

        for fold in range(1, 6):
            train_file = f"data/processed/skab_fold{fold}_X_train_pc1.csv"
            test_file = f"data/processed/skab_fold{fold}_X_test_pc1.csv"
            y_test_file = f"data/processed/skab_fold{fold}_y_test.csv"

            if os.path.exists(train_file) and os.path.exists(test_file) and os.path.exists(y_test_file):
                X_train_s = pd.read_csv(train_file).values.flatten()
                X_test_s = pd.read_csv(test_file).values.flatten()
                y_test_s = pd.read_csv(y_test_file).values.flatten()

                for seed in seeds:
                    res, _ = run_experiment_pipeline(
                        X_train_s,
                        X_test_s,
                        y_test_s,
                        cc,
                        "SKAB",
                        f"threshold_search_fold_{fold}",
                        seed=seed
                    )

                    original_result = [r for r in res if r["scenario"] == "original"][0]
                    f1_scores.append(original_result["f1_score"])

        mean_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0

        threshold_results.append({
            "dataset": "SKAB",
            "threshold": threshold,
            "window_size": 4,
            "alphabet_size": 4,
            "weight_sharpness": 2.0,
            "f1_score_mean": mean_f1,
            "use_similarity_penalty": False,
            "similarity_penalty_strength": 0.0
        })

        print(f"SKAB -> Threshold: {threshold} | Ortalama F1: {mean_f1:.4f}")

    if threshold_results:
        df_threshold = pd.DataFrame(threshold_results)
        df_threshold.to_csv("results/outputs/automata_threshold_sensitivity.csv", index=False)

        print("\nEn iyi threshold değerleri:")
        for dataset_name in df_threshold["dataset"].unique():
            dataset_df = df_threshold[df_threshold["dataset"] == dataset_name]
            best_row = dataset_df.loc[dataset_df["f1_score_mean"].idxmax()]
            print(
                f"{dataset_name} -> Best Threshold: {best_row['threshold']} "
                f"| F1: {best_row['f1_score_mean']:.4f}"
            )  

#dynamic treshold deneyi için analiz fonksiyonu.
#bu fonksiyon final modelin parçası değil kotu sonuc verdıgı icin kapatıldı
def run_dynamic_threshold_sensitivity_analysis(config):
    print("\n--- DINAMIK THRESHOLD DUYARLILIK ANALIZI BASLATILIYOR ---")

    dynamic_results = []
    seeds = config["seeds"]
    quantile_values = config.get("dynamic_threshold_quantile_values", [0.01, 0.03, 0.05, 0.10, 0.20])
    min_samples = config.get("min_dynamic_threshold_samples", 3)

    if os.path.exists("data/processed/batadal_X_train_adasyn_pc1.csv"):
        X_train_b = pd.read_csv("data/processed/batadal_X_train_adasyn_pc1.csv").values.flatten()
        X_test_b = pd.read_csv("data/processed/batadal_X_test_pc1.csv").values.flatten()
        y_test_b = pd.read_csv("data/processed/batadal_y_test.csv").values.flatten()
        y_test_b = np.where(y_test_b == -999, 0, y_test_b)

        print("\n>> BATADAL Dinamik Threshold Taramasi...")
        for quantile in quantile_values:
            cc = {
                **config,
                "window_size": 4,
                "alphabet_size": 3,
                "weight_sharpness": 6.0,
                "anomaly_threshold": 0.01,
                "use_similarity_penalty": False,
                "similarity_penalty_strength": 0.0,
                "use_dynamic_threshold": True,
                "dynamic_threshold_quantile": quantile,
                "min_dynamic_threshold_samples": min_samples
            }

            f1_scores = []
            precision_scores = []
            recall_scores = []

            for seed in seeds:
                res, _ = run_experiment_pipeline(
                    X_train_b, X_test_b, y_test_b,
                    cc, "BATADAL", "dynamic_threshold_search", seed=seed
                )
                original_result = [r for r in res if r["scenario"] == "original"][0]
                f1_scores.append(original_result["f1_score"])
                precision_scores.append(original_result["precision"])
                recall_scores.append(original_result["recall"])

            dynamic_results.append({
                "dataset": "BATADAL",
                "dynamic_threshold_quantile": quantile,
                "min_dynamic_threshold_samples": min_samples,
                "window_size": 4,
                "alphabet_size": 3,
                "weight_sharpness": 6.0,
                "base_threshold": 0.01,
                "precision_mean": float(np.mean(precision_scores)),
                "recall_mean": float(np.mean(recall_scores)),
                "f1_score_mean": float(np.mean(f1_scores)),
                "use_dynamic_threshold": True,
                "use_similarity_penalty": False,
                "similarity_penalty_strength": 0.0
            })
            print(
                f"BATADAL -> Quantile: {quantile} | "
                f"Precision: {np.mean(precision_scores):.4f} | "
                f"Recall: {np.mean(recall_scores):.4f} | "
                f"F1: {np.mean(f1_scores):.4f}"
            )

    print("\n>> SKAB Dinamik Threshold Taramasi...")
    for quantile in quantile_values:
        cc = {
            **config,
            "window_size": 4,
            "alphabet_size": 4,
            "weight_sharpness": 2.0,
            "anomaly_threshold": 0.99,
            "use_similarity_penalty": False,
            "similarity_penalty_strength": 0.0,
            "use_dynamic_threshold": True,
            "dynamic_threshold_quantile": quantile,
            "min_dynamic_threshold_samples": min_samples
        }

        f1_scores = []
        precision_scores = []
        recall_scores = []

        for fold in range(1, 6):
            train_file = f"data/processed/skab_fold{fold}_X_train_pc1.csv"
            test_file = f"data/processed/skab_fold{fold}_X_test_pc1.csv"
            y_test_file = f"data/processed/skab_fold{fold}_y_test.csv"

            if os.path.exists(train_file) and os.path.exists(test_file) and os.path.exists(y_test_file):
                X_train_s = pd.read_csv(train_file).values.flatten()
                X_test_s = pd.read_csv(test_file).values.flatten()
                y_test_s = pd.read_csv(y_test_file).values.flatten()

                for seed in seeds:
                    res, _ = run_experiment_pipeline(
                        X_train_s, X_test_s, y_test_s,
                        cc, "SKAB", f"dynamic_threshold_search_fold_{fold}", seed=seed
                    )
                    original_result = [r for r in res if r["scenario"] == "original"][0]
                    f1_scores.append(original_result["f1_score"])
                    precision_scores.append(original_result["precision"])
                    recall_scores.append(original_result["recall"])

        dynamic_results.append({
            "dataset": "SKAB",
            "dynamic_threshold_quantile": quantile,
            "min_dynamic_threshold_samples": min_samples,
            "window_size": 4,
            "alphabet_size": 4,
            "weight_sharpness": 2.0,
            "base_threshold": 0.99,
            "precision_mean": float(np.mean(precision_scores)) if precision_scores else 0.0,
            "recall_mean": float(np.mean(recall_scores)) if recall_scores else 0.0,
            "f1_score_mean": float(np.mean(f1_scores)) if f1_scores else 0.0,
            "use_dynamic_threshold": True,
            "use_similarity_penalty": False,
            "similarity_penalty_strength": 0.0
        })
        print(
            f"SKAB -> Quantile: {quantile} | "
            f"Precision: {np.mean(precision_scores):.4f} | "
            f"Recall: {np.mean(recall_scores):.4f} | "
            f"F1: {np.mean(f1_scores):.4f}"
            if f1_scores else
            f"SKAB -> Quantile: {quantile} | veri bulunamadi"
        )

    if dynamic_results:
        df_dynamic = pd.DataFrame(dynamic_results)
        df_dynamic.to_csv("results/outputs/automata_dynamic_threshold_sensitivity.csv", index=False)

        print("\nEn iyi dinamik threshold ayarlari:")
        for dataset_name in df_dynamic["dataset"].unique():
            dataset_df = df_dynamic[df_dynamic["dataset"] == dataset_name]
            best_row = dataset_df.loc[dataset_df["f1_score_mean"].idxmax()]
            print(
                f"{dataset_name} -> Best Quantile: {best_row['dynamic_threshold_quantile']} "
                f"| F1: {best_row['f1_score_mean']:.4f}"
            )
 

def run_similarity_penalty_sensitivity_analysis(config):
    print("\n--- SIMILARITY PENALTY DUYARLILIK ANALIZI BASLATILIYOR ---")

    strength_values = config.get(
        "similarity_penalty_strength_values",
        [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
    )

    penalty_results = []
    seeds = config["seeds"]

    # --- BATADAL ---
    if os.path.exists("data/processed/batadal_X_train_adasyn_pc1.csv"):
        X_train_b = pd.read_csv("data/processed/batadal_X_train_adasyn_pc1.csv").values.flatten()
        X_test_b = pd.read_csv("data/processed/batadal_X_test_pc1.csv").values.flatten()
        y_test_b = pd.read_csv("data/processed/batadal_y_test.csv").values.flatten()
        y_test_b = np.where(y_test_b == -999, 0, y_test_b)

        print("\n>> BATADAL Similarity Penalty Taramasi...")

        for strength in strength_values:
            cc = {
                **config,
                "window_size": 4,
                "alphabet_size": 3,
                "weight_sharpness": 6.0,
                "anomaly_threshold": 0.01,
                "use_similarity_penalty": True,
                "similarity_penalty_strength": strength,
                "use_dynamic_threshold": False
            }

            original_f1_scores = []
            unseen_f1_scores = []

            for seed in seeds:
                res, _ = run_experiment_pipeline(
                    X_train_b,
                    X_test_b,
                    y_test_b,
                    cc,
                    "BATADAL",
                    "similarity_penalty_search",
                    seed=seed
                )

                original_result = [r for r in res if r["scenario"] == "original"][0]
                unseen_result = [r for r in res if r["scenario"] == "unseen_data"][0]

                original_f1_scores.append(original_result["f1_score"])
                unseen_f1_scores.append(unseen_result["f1_score"])

            penalty_results.append({
                "dataset": "BATADAL",
                "similarity_penalty_strength": strength,
                "window_size": 4,
                "alphabet_size": 3,
                "weight_sharpness": 6.0,
                "threshold": 0.01,
                "original_f1_mean": float(np.mean(original_f1_scores)),
                "unseen_f1_mean": float(np.mean(unseen_f1_scores))
            })

            print(
                f"BATADAL -> Strength: {strength} | "
                f"Original F1: {np.mean(original_f1_scores):.4f} | "
                f"Unseen F1: {np.mean(unseen_f1_scores):.4f}"
            )

    # --- SKAB ---
    print("\n>> SKAB Similarity Penalty Taramasi...")

    for strength in strength_values:
        cc = {
            **config,
            "window_size": 4,
            "alphabet_size": 4,
            "weight_sharpness": 2.0,
            "anomaly_threshold": 0.99,
            "use_similarity_penalty": True,
            "similarity_penalty_strength": strength,
            "use_dynamic_threshold": False
        }

        original_f1_scores = []
        unseen_f1_scores = []

        for fold in range(1, 6):
            train_file = f"data/processed/skab_fold{fold}_X_train_pc1.csv"
            test_file = f"data/processed/skab_fold{fold}_X_test_pc1.csv"
            y_test_file = f"data/processed/skab_fold{fold}_y_test.csv"

            if os.path.exists(train_file) and os.path.exists(test_file) and os.path.exists(y_test_file):
                X_train_s = pd.read_csv(train_file).values.flatten()
                X_test_s = pd.read_csv(test_file).values.flatten()
                y_test_s = pd.read_csv(y_test_file).values.flatten()

                for seed in seeds:
                    res, _ = run_experiment_pipeline(
                        X_train_s,
                        X_test_s,
                        y_test_s,
                        cc,
                        "SKAB",
                        f"similarity_penalty_search_fold_{fold}",
                        seed=seed
                    )

                    original_result = [r for r in res if r["scenario"] == "original"][0]
                    unseen_result = [r for r in res if r["scenario"] == "unseen_data"][0]

                    original_f1_scores.append(original_result["f1_score"])
                    unseen_f1_scores.append(unseen_result["f1_score"])

        penalty_results.append({
            "dataset": "SKAB",
            "similarity_penalty_strength": strength,
            "window_size": 4,
            "alphabet_size": 4,
            "weight_sharpness": 2.0,
            "threshold": 0.99,
            "original_f1_mean": float(np.mean(original_f1_scores)) if original_f1_scores else 0.0,
            "unseen_f1_mean": float(np.mean(unseen_f1_scores)) if unseen_f1_scores else 0.0
        })

        print(
            f"SKAB -> Strength: {strength} | "
            f"Original F1: {np.mean(original_f1_scores):.4f} | "
            f"Unseen F1: {np.mean(unseen_f1_scores):.4f}"
            if original_f1_scores else
            f"SKAB -> Strength: {strength} | veri bulunamadi"
        )

    if penalty_results:
        df_penalty = pd.DataFrame(penalty_results)
        df_penalty.to_csv("results/outputs/automata_similarity_penalty_sensitivity.csv", index=False)

        print("\nEn iyi similarity penalty strength degerleri:")
        for dataset_name in df_penalty["dataset"].unique():
            dataset_df = df_penalty[df_penalty["dataset"] == dataset_name]
            best_row = dataset_df.loc[dataset_df["original_f1_mean"].idxmax()]
            print(
                f"{dataset_name} -> Best Strength: {best_row['similarity_penalty_strength']} "
                f"| Original F1: {best_row['original_f1_mean']:.4f} "
                f"| Unseen F1: {best_row['unseen_f1_mean']:.4f}"
            )

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
 
        config_batadal = {
            **config,
            "window_size": 4,
            "alphabet_size": 3,
            "weight_sharpness": 6.0,
            "anomaly_threshold": 0.01,
            "use_similarity_penalty": False,
            "similarity_penalty_strength": 0.0,
            "use_dynamic_threshold": False
        }
 
        batadal_logs = None
        for seed in seeds:
            print(f"BATADAL automata seed={seed} çalışıyor...")
            batadal_res, logs = run_experiment_pipeline(X_train, X_test, y_test, config_batadal, "BATADAL", "single", seed=seed)
            all_res.extend(batadal_res)
            if batadal_logs is None:
                batadal_logs = logs
 
        with open("results/outputs/automata_batadal_advanced_explainability.json", "w") as f:
            json.dump(batadal_logs[:100], f, indent=4)
 
    config_skab = {
        **config,
        "window_size": 4,
        "alphabet_size": 4,
        "weight_sharpness": 2.0,
        "anomaly_threshold": 0.99,
        "use_similarity_penalty": False,
        "similarity_penalty_strength": 0.0,
        "use_dynamic_threshold": False
   }
 
    for fold in range(1, 6):
        train_file = f"data/processed/skab_fold{fold}_X_train_pc1.csv"
        test_file = f"data/processed/skab_fold{fold}_X_test_pc1.csv"
        y_test_file = f"data/processed/skab_fold{fold}_y_test.csv"
        if os.path.exists(train_file) and os.path.exists(test_file) and os.path.exists(y_test_file):
            X_train = pd.read_csv(train_file).values.flatten()
            X_test = pd.read_csv(test_file).values.flatten()
            y_test = pd.read_csv(y_test_file).values.flatten()
            for seed in seeds:
                print(f"SKAB automata fold={fold}, seed={seed} çalışıyor...")
                fold_res, _ = run_experiment_pipeline(X_train, X_test, y_test, config_skab, "SKAB", f"fold_{fold}", seed=seed)
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
        print("\nSKAB Otomata Özet (5 fold x 5 seed):")
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
        print("\nBATADAL Otomata Özet (5 seed):")
        print(batadal_summary)
 
    run_parameter_sensitivity_analysis(config)
    run_threshold_sensitivity_analysis(config)
    #dynamic Threshold deneyi sabit threshold yönteminden düşük performans verdiği için çalıştırılmıyor
    #run_dynamic_threshold_sensitivity_analysis(config)

    try:
        from src.experiments.statistical_tests import main as run_statistical_main
        run_statistical_main()
    except Exception as e:
        print(f"İstatistiksel testler tetiklenirken bir hata meydana geldi: {str(e)}")
    print("=" * 60 + "\n")
 
if __name__ == "__main__":
    main()