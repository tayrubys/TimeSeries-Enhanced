import os
import json
import sys
import pandas as pd
import numpy as np
 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
 
from src.data_pipeline.sax_paa import SaxPaaTransformer
#from src.models.automata_model import ProbabilisticAutomata
from src.models.vomm_model import VariableOrderMarkovModel
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
# 
def split_train_validation(X_train, y_train, val_ratio=0.2):
    split_idx = int(len(X_train) * (1 - val_ratio))

    X_model_train = X_train[:split_idx]
    X_val = X_train[split_idx:]

    y_model_train = y_train[:split_idx]
    y_val = y_train[split_idx:]

    return X_model_train, X_val, y_model_train, y_val

#
def align_labels_to_patterns(y, num_patterns, window_size):
    aligned_labels = []

    for i in range(1, num_patterns):
        start_idx = i * window_size
        end_idx = min(start_idx + window_size, len(y))

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
    window_size
):
    best_threshold = threshold_values[0]
    best_f1 = -1.0
    best_metrics = None

    val_patterns = transformer.transform(X_val, window_size=window_size)

    for threshold in threshold_values:
        preds_val, _ = model.predict(val_patterns, anomaly_threshold=threshold)
        label_start = window_size
        y_val_aligned = align_labels_to_patterns(
            y_val,
            len(val_patterns),
            window_size
        )
        preds_val = preds_val[:len(y_val_aligned)]

        metrics = calculate_metrics(y_val_aligned, preds_val)

        if metrics["f1_score"] > best_f1:
            best_f1 = metrics["f1_score"]
            best_threshold = threshold
            best_metrics = metrics

    return best_threshold, best_metrics
#
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
    threshold_values = config.get(
        "vomm_threshold_values",
        [0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5]
    )

    print(f"\n>> {dataset_name} validation tabanli VOMM/PST config taramasi...")

    for window_size in window_sizes:
        for alphabet_size in alphabet_sizes:
            transformer = SaxPaaTransformer(alphabet_size=alphabet_size)

            train_patterns = transformer.transform(
                X_train,
                window_size=window_size
            )

            model = VariableOrderMarkovModel(
                max_depth=window_size,
                min_count=2,
                smoothing=True,
                smoothing_alpha=1.0
            )
            model.fit(train_patterns)

            val_patterns = transformer.transform(
                X_val,
                window_size=window_size
            )

            y_val_aligned = align_labels_to_patterns(
                y_val,
                len(val_patterns),
                window_size
            )

            for threshold in threshold_values:
                preds_val, _ = model.predict(
                    val_patterns,
                    anomaly_threshold=threshold
                )
                preds_val = preds_val[:len(y_val_aligned)]

                metrics = calculate_metrics(y_val_aligned, preds_val)

                if metrics["f1_score"] > best_f1:
                    best_f1 = metrics["f1_score"]
                    best_metrics = metrics
                    best_config = {
                        "window_size": window_size,
                        "alphabet_size": alphabet_size,
                        "anomaly_threshold": threshold
                    }

    print(
        f"{dataset_name} best validation config -> "
        f"window_size={best_config['window_size']}, "
        f"alphabet_size={best_config['alphabet_size']}, "
        f"threshold={best_config['anomaly_threshold']} | "
        f"val F1={best_f1:.4f}"
    )

    return best_config, best_metrics

def run_experiment_pipeline(X_train, X_test, y_test, config, dataset_name, fold_name="", seed=42,X_val=None,y_val=None,use_validation_threshold=True):
    results = []
    transformer = SaxPaaTransformer(alphabet_size=config["alphabet_size"])
    train_patterns = transformer.transform(X_train, window_size=config["window_size"])
 
    model = VariableOrderMarkovModel(
    max_depth=config["window_size"],
    min_count=2,
    smoothing=True,
    smoothing_alpha=1.0
    )
    model.fit(train_patterns)
    selected_threshold = config["anomaly_threshold"]
    validation_threshold_f1 = None
    if use_validation_threshold and X_val is not None and y_val is not None:
       threshold_values = config.get(
           "vomm_threshold_values",
           [0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5]
        )
       selected_threshold, val_metrics = select_best_threshold_on_validation(
            model,
            transformer,
            X_val,
            y_val,
            threshold_values,
            config["window_size"]
        )
       validation_threshold_f1 = val_metrics["f1_score"]
    if use_validation_threshold and X_val is not None and y_val is not None:
        print(
            f"{dataset_name} {fold_name} validation selected threshold: "
            f"{selected_threshold} | val F1: {validation_threshold_f1:.4f}"
         )

    num_states = len(model.trained_patterns)
    num_transitions = sum(len(targets) for targets in model.transitions.values())
    transition_density = num_transitions / (num_states * num_states) if num_states > 0 else 0.0
 
    common_fields = {
        "dataset": dataset_name, "fold": fold_name, "seed": seed,
        "window_size": config["window_size"], "alphabet_size": config["alphabet_size"],
        "num_states": num_states, "num_transitions": num_transitions,
        "transition_density": transition_density,
        "selected_threshold": selected_threshold,
        "validation_threshold_f1": validation_threshold_f1
    }
 
    # --- SENARYO 1: Orijinal Veri ---
    test_patterns_orig = transformer.transform(X_test, window_size=config["window_size"])
    preds_orig, logs_orig = model.predict(test_patterns_orig, anomaly_threshold=selected_threshold)
    label_start = config["window_size"]
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
    preds_noisy, _ = model.predict(test_patterns_noisy, anomaly_threshold=selected_threshold)
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
    
    # PARAMETRE TARAMASI İÇİN DE ADASYN VERİSİ BAĞLANDI
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
            "anomaly_threshold": 0.05
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
            if os.path.exists(train_file) and os.path.exists(test_file) and os.path.exists(y_test_file) and os.path.exists(y_train_file):
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

                config_skab_fold = {
                    **config,
                    **best_skab_config
           }
                
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
 
    run_parameter_sensitivity_analysis(config)
    run_vomm_threshold_sensitivity_analysis(config)
 
    try:
        from src.experiments.statistical_tests import main as run_statistical_main
        run_statistical_main()
    except Exception as e:
        print(f"İstatistiksel testler tetiklenirken bir hata meydana geldi: {str(e)}")
    print("=" * 60 + "\n")
 
if __name__ == "__main__":
    main()