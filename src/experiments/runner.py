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
        "anomaly_threshold": 0.05,
        "skab_anomaly_threshold": 0.90,
        "batadal_anomaly_threshold": 0.05,
        "weight_sharpness": 1.0,
        "weight_sharpness_values": [0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0],
        "batadal_window_size": 4,
        "batadal_alphabet_size": 3,
        "batadal_weight_sharpness": 2.0,
        "batadal_threshold_values": [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3],
        "skab_window_size": 4,
        "skab_alphabet_size": 4,
        "skab_weight_sharpness": 2.0,
        "skab_threshold_values": [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99],
        "noise_level": 0.1,
        "seeds": [42, 123, 2026, 7, 999],
        "batadal_train_ratio": 0.60,
        "batadal_val_ratio": 0.20
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
                "weight_sharpness": automata_config.get("weight_sharpness", default_config["weight_sharpness"]),
                "weight_sharpness_values": automata_config.get("weight_sharpness_values", default_config["weight_sharpness_values"]),
                "batadal_window_size": automata_config.get("batadal_window_size", default_config["batadal_window_size"]),
                "batadal_alphabet_size": automata_config.get("batadal_alphabet_size", default_config["batadal_alphabet_size"]),
                "batadal_weight_sharpness": automata_config.get("batadal_weight_sharpness", default_config["batadal_weight_sharpness"]),
                "batadal_threshold_values": automata_config.get("batadal_threshold_values", default_config["batadal_threshold_values"]),
                "skab_window_size": automata_config.get("skab_window_size", default_config["skab_window_size"]),
                "skab_alphabet_size": automata_config.get("skab_alphabet_size", default_config["skab_alphabet_size"]),
                "skab_weight_sharpness": automata_config.get("skab_weight_sharpness", default_config["skab_weight_sharpness"]),
                "skab_threshold_values": automata_config.get("skab_threshold_values", default_config["skab_threshold_values"]),
                "noise_level": automata_config.get("noise_level", default_config["noise_level"]),
                "seeds": automata_config.get("seeds", default_config["seeds"]),
                "batadal_train_ratio": automata_config.get("batadal_train_ratio", default_config["batadal_train_ratio"]),
                "batadal_val_ratio": automata_config.get("batadal_val_ratio", default_config["batadal_val_ratio"])
            }
        except Exception:
            return default_config
    return default_config
 
def inject_gaussian_noise(series, noise_level=0.1, seed=42):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, noise_level, series.shape)
    return series + noise

def find_optimal_threshold(model, val_patterns, y_val, thresholds=None):
    """
    Test setine BAKMADAN, ayrı bir validation seti üzerinde F1'i
    maksimize eden anomaly_threshold degerini bulur. Bunu test seti
    üzerinde çalıştırmak veri sızıntısı olur - sadece val icin kullanın.
    """
    from src.experiments.evaluator import calculate_metrics
    if thresholds is None:
        thresholds = np.arange(0.01, 0.99, 0.02)
    best_threshold, best_f1 = thresholds[0], -1.0
    for t in thresholds:
        preds, _ = model.predict(val_patterns, anomaly_threshold=t)
        y_aligned = y_val[:len(preds)]
        f1 = calculate_metrics(y_aligned, preds).get("f1_score", 0.0)
        if f1 > best_f1:
            best_f1, best_threshold = f1, t
    return float(best_threshold), float(best_f1)
 
def run_experiment_pipeline(X_train, X_test, y_test, config, dataset_name, fold_name="", seed=42):
    results = []
    transformer = SaxPaaTransformer(alphabet_size=config["alphabet_size"])
    train_patterns = transformer.transform(X_train, window_size=config["window_size"])
 
    model = ProbabilisticAutomata(smoothing=True, weight_sharpness=config.get("weight_sharpness", 1.0))
    model.fit(train_patterns)
 
    num_states = len(model.trained_patterns)
    num_transitions = sum(len(targets) for targets in model.transitions.values())
    transition_density = num_transitions / (num_states * num_states) if num_states > 0 else 0.0
 
    common_fields = {
        "dataset": dataset_name, "fold": fold_name, "seed": seed,
        "window_size": config["window_size"], "alphabet_size": config["alphabet_size"],
        "num_states": num_states, "num_transitions": num_transitions,
        "transition_density": transition_density
    }
 
    # --- SENARYO 1: Orijinal Veri ---
    test_patterns_orig = transformer.transform(X_test, window_size=config["window_size"])
    preds_orig, logs_orig = model.predict(test_patterns_orig, anomaly_threshold=config["anomaly_threshold"])
    y_test_aligned_orig = y_test[:len(preds_orig)]
    metrics_orig = calculate_metrics(y_test_aligned_orig, preds_orig)
    metrics_orig.update({"scenario": "original", **common_fields})
    results.append(metrics_orig)
 
    # --- SENARYO 2: Gaussian Noise ---
    X_test_noisy = inject_gaussian_noise(X_test, noise_level=config["noise_level"], seed=seed)
    test_patterns_noisy = transformer.transform(X_test_noisy, window_size=config["window_size"])
    preds_noisy, _ = model.predict(test_patterns_noisy, anomaly_threshold=config["anomaly_threshold"])
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
        preds_unseen, _ = model.predict(unseen_test_patterns, anomaly_threshold=config["anomaly_threshold"])
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
    weight_sharpness_values = config.get("weight_sharpness_values", [1.0])
    sensitivity_results = []
    
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
                        "anomaly_threshold": config.get("batadal_anomaly_threshold", 0.05),
                        "weight_sharpness": ws,
                        "noise_level": config["noise_level"]
                    }
                    res, _ = run_experiment_pipeline(X_train_b, X_test_b, y_test_b, cc, "BATADAL", "param_search", seed=config["seeds"][0])
                    orig_res = [r for r in res if r["scenario"] == "original"][0]
                    orig_res["weight_sharpness"] = ws
                    sensitivity_results.append(orig_res)
                    print(f"BATADAL -> Window Size: {w}, Alphabet Size: {a}, Sharpness: {ws} | Density: {orig_res['transition_density']:.4f}, F1: {orig_res['f1_score']:.4f}")

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
                        "anomaly_threshold": config.get("skab_anomaly_threshold", 0.90),
                        "weight_sharpness": ws,
                        "noise_level": config["noise_level"]
                    }
                    res, _ = run_experiment_pipeline(X_train_s, X_test_s, y_test_s, cc, "SKAB", "param_search", seed=config["seeds"][0])
                    orig_res = [r for r in res if r["scenario"] == "original"][0]
                    orig_res["weight_sharpness"] = ws
                    sensitivity_results.append(orig_res)
                    print(f"SKAB -> Window Size: {w}, Alphabet Size: {a}, Sharpness: {ws} | Density: {orig_res['transition_density']:.4f}, F1: {orig_res['f1_score']:.4f}")
 
    if sensitivity_results:
        pd.DataFrame(sensitivity_results).to_csv("results/outputs/automata_param_sensitivity_metrics.csv", index=False)

def run_threshold_sensitivity_analysis(config):
    """
    Zaten secilmis en iyi window/alphabet/weight_sharpness sabit tutulup
    SADECE anomaly_threshold taranir. SKAB icin tum fold+seed, BATADAL icin
    tum seed ortalamasi alinir - boylece ana ozet tablosuyla dogrudan
    kiyaslanabilir, tek fold'a ozgu yanilgiya dusulmez.
    """
    print("\n--- THRESHOLD DUYARLILIK ANALİZİ BAŞLATILIYOR ---")
    threshold_results = []
    seeds = config["seeds"]

    # --- BATADAL ---
    if os.path.exists("data/processed/batadal_X_train_adasyn_pc1.csv"):
        X_train_b = pd.read_csv("data/processed/batadal_X_train_adasyn_pc1.csv").values.flatten()
        X_test_b = pd.read_csv("data/processed/batadal_X_test_pc1.csv").values.flatten()
        y_test_b = pd.read_csv("data/processed/batadal_y_test.csv").values.flatten()
        y_test_b = np.where(y_test_b == -999, 0, y_test_b)

        batadal_thresholds = config.get("batadal_threshold_values", [0.05])
        print("\n>> BATADAL Threshold Taraması...")
        for t in batadal_thresholds:
            cc = {
                "window_size": config.get("batadal_window_size", config["window_size"]),
                "alphabet_size": config.get("batadal_alphabet_size", config["alphabet_size"]),
                "weight_sharpness": config.get("batadal_weight_sharpness", config["weight_sharpness"]),
                "anomaly_threshold": t,
                "noise_level": config["noise_level"]
            }
            f1s = []
            for seed in seeds:
                res, _ = run_experiment_pipeline(X_train_b, X_test_b, y_test_b, cc, "BATADAL", "thresh_search", seed=seed)
                orig_res = [r for r in res if r["scenario"] == "original"][0]
                f1s.append(orig_res["f1_score"])
            mean_f1 = float(np.mean(f1s))
            threshold_results.append({"dataset": "BATADAL", "threshold": t, "f1_score_mean": mean_f1})
            print(f"BATADAL -> Threshold: {t} | F1 (ortalama, {len(seeds)} seed): {mean_f1:.4f}")

    # --- SKAB (tum fold + seed ortalamasi) ---
    skab_thresholds = config.get("skab_threshold_values", [0.90])
    print("\n>> SKAB Threshold Taraması...")
    for t in skab_thresholds:
        cc = {
            "window_size": config.get("skab_window_size", config["window_size"]),
            "alphabet_size": config.get("skab_alphabet_size", config["alphabet_size"]),
            "weight_sharpness": config.get("skab_weight_sharpness", config["weight_sharpness"]),
            "anomaly_threshold": t,
            "noise_level": config["noise_level"]
        }
        f1s = []
        for fold in range(1, 6):
            train_file = f"data/processed/skab_fold{fold}_X_train_pc1.csv"
            test_file = f"data/processed/skab_fold{fold}_X_test_pc1.csv"
            y_test_file = f"data/processed/skab_fold{fold}_y_test.csv"
            if not (os.path.exists(train_file) and os.path.exists(test_file) and os.path.exists(y_test_file)):
                continue
            X_train_s = pd.read_csv(train_file).values.flatten()
            X_test_s = pd.read_csv(test_file).values.flatten()
            y_test_s = pd.read_csv(y_test_file).values.flatten()
            for seed in seeds:
                res, _ = run_experiment_pipeline(X_train_s, X_test_s, y_test_s, cc, "SKAB", f"thresh_search_fold{fold}", seed=seed)
                orig_res = [r for r in res if r["scenario"] == "original"][0]
                f1s.append(orig_res["f1_score"])
        mean_f1 = float(np.mean(f1s)) if f1s else 0.0
        threshold_results.append({"dataset": "SKAB", "threshold": t, "f1_score_mean": mean_f1})
        print(f"SKAB -> Threshold: {t} | F1 (ortalama, {len(f1s)} çalıştırma): {mean_f1:.4f}")

    if threshold_results:
        df_thresh = pd.DataFrame(threshold_results)
        df_thresh.to_csv("results/outputs/automata_threshold_sensitivity.csv", index=False)
        for ds in df_thresh["dataset"].unique():
            sub = df_thresh[df_thresh["dataset"] == ds]
            best_row = sub.loc[sub["f1_score_mean"].idxmax()]
            print(f"\n>>> {ds} EN İYİ THRESHOLD: {best_row['threshold']} (F1={best_row['f1_score_mean']:.4f})")
 
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
            "anomaly_threshold": config.get("batadal_anomaly_threshold", 0.05),
            "window_size": config.get("batadal_window_size", config["window_size"]),
            "alphabet_size": config.get("batadal_alphabet_size", config["alphabet_size"]),
            "weight_sharpness": config.get("batadal_weight_sharpness", config["weight_sharpness"])
        }
        print(f"[BATADAL] window_size={config_batadal['window_size']}, alphabet_size={config_batadal['alphabet_size']}, "
              f"weight_sharpness={config_batadal['weight_sharpness']}, threshold={config_batadal['anomaly_threshold']}")
 
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
        "anomaly_threshold": config.get("skab_anomaly_threshold", 0.90),
        "window_size": config.get("skab_window_size", config["window_size"]),
        "alphabet_size": config.get("skab_alphabet_size", config["alphabet_size"]),
        "weight_sharpness": config.get("skab_weight_sharpness", config["weight_sharpness"])
    }
    print(f"[SKAB] window_size={config_skab['window_size']}, alphabet_size={config_skab['alphabet_size']}, "
          f"weight_sharpness={config_skab['weight_sharpness']}, threshold={config_skab['anomaly_threshold']}")
 
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
 
    try:
        from src.experiments.statistical_tests import main as run_statistical_main
        run_statistical_main()
    except Exception as e:
        print(f"İstatistiksel testler tetiklenirken bir hata meydana geldi: {str(e)}")
    print("=" * 60 + "\n")
 
if __name__ == "__main__":
    main()