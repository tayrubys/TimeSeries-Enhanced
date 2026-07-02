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
    """Automata ayarlarını okur. High-order Markov sweep için order/threshold listeleri eklenmiştir."""
    default_config = {
        "window_size": 4,
        "alphabet_size": 3,

        # High-order Markov fine-tuning parametreleri
        "order": 2,
        "orders": [2, 3, 4],
        "batadal_thresholds": [0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02],
        "skab_thresholds": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05],
        # Geriye uyumluluk: eski ortak liste config'te varsa sadece fallback olarak kullanılır.
        "thresholds": [0.05, 0.01, 0.005, 0.001],

        "window_sizes": [3, 4, 5, 6],
        "alphabet_sizes": [3, 4, 5, 6],

        "anomaly_threshold": 0.05,
        "skab_anomaly_threshold": 0.90,
        "batadal_anomaly_threshold": 0.05,
        "noise_level": 0.1,
        "seeds": [42, 123, 2026, 7, 999],
        "batadal_train_ratio": 0.60,
        "batadal_val_ratio": 0.20,
    }

    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                user_config = json.load(f)
            automata_config = user_config.get("automata", {})
            merged = default_config.copy()
            for key in merged.keys():
                merged[key] = automata_config.get(key, merged[key])
            return merged
        except Exception as exc:
            print(f"[UYARI] Config okunamadı, varsayılan ayarlar kullanılacak: {exc}")
            return default_config

    return default_config


def inject_gaussian_noise(series, noise_level=0.1, seed=42):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, noise_level, series.shape)
    return series + noise


def run_experiment_pipeline(X_train, X_test, y_test, config, dataset_name, fold_name="", seed=42):
    """Tek bir dataset/fold/seed/parametre kombinasyonu için otomata deneyini çalıştırır."""
    results = []

    transformer = SaxPaaTransformer(alphabet_size=config["alphabet_size"])
    train_patterns = transformer.transform(X_train, window_size=config["window_size"])

    model = ProbabilisticAutomata(
        smoothing=True,
        order=config.get("order", 2),
        learning_rate=config.get("learning_rate", 0.0),
    )
    model.fit(train_patterns)

    num_states = len(model.total_exits)
    vocab_size = len(model.trained_patterns)
    num_transitions = sum(len(targets) for targets in model.transitions.values())
    transition_density = num_transitions / (num_states * vocab_size) if num_states > 0 and vocab_size > 0 else 0.0

    common_fields = {
        "dataset": dataset_name,
        "fold": fold_name,
        "seed": seed,
        "window_size": config["window_size"],
        "alphabet_size": config["alphabet_size"],
        "order": config.get("order", 2),
        "anomaly_threshold": config["anomaly_threshold"],
        "num_states": num_states,
        "vocab_size": vocab_size,
        "num_transitions": num_transitions,
        "transition_density": transition_density,
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


def run_markov_order_threshold_sweep(config):
    """Order ve threshold kombinasyonlarını tarar; eski window/alphabet grid search yerine bunu kullanır."""
    print("\n--- HIGH-ORDER MARKOV FINE-TUNING GRID SEARCH BAŞLATILIYOR ---")

    orders = config.get("orders", [2, 3, 4])
    fallback_thresholds = config.get("thresholds", [0.05, 0.01, 0.005, 0.001])
    batadal_thresholds = config.get("batadal_thresholds", fallback_thresholds)
    skab_thresholds = config.get("skab_thresholds", fallback_thresholds)
    seeds = config.get("seeds", [42, 123, 2026, 7, 999])

    sweep_results = []

    # BATADAL sweep
    if os.path.exists("data/processed/batadal_X_train_adasyn_pc1.csv"):
        X_train_b = pd.read_csv("data/processed/batadal_X_train_adasyn_pc1.csv").values.flatten()
        X_test_b = pd.read_csv("data/processed/batadal_X_test_pc1.csv").values.flatten()
        y_test_b = pd.read_csv("data/processed/batadal_y_test.csv").values.flatten()
        y_test_b = np.where(y_test_b == -999, 0, y_test_b)

        print("\n>> BATADAL Markov Order/Threshold Taraması...")
        print(f"   Threshold aralığı: {batadal_thresholds}")
        for order in orders:
            for threshold in batadal_thresholds:
                for seed in seeds:
                    cc = {
                        **config,
                        "order": order,
                        "anomaly_threshold": threshold,
                    }
                    res, _ = run_experiment_pipeline(
                        X_train_b,
                        X_test_b,
                        y_test_b,
                        cc,
                        "BATADAL",
                        "markov_sweep",
                        seed=seed,
                    )
                    sweep_results.extend(res)

                temp_df = pd.DataFrame([r for r in sweep_results if r["dataset"] == "BATADAL"
                                        and r["order"] == order
                                        and r["anomaly_threshold"] == threshold
                                        and r["scenario"] == "original"])
                if not temp_df.empty:
                    print(
                        f"BATADAL -> order={order}, threshold={threshold} | "
                        f"Precision={temp_df['precision'].mean():.4f}, "
                        f"Recall={temp_df['recall'].mean():.4f}, "
                        f"F1={temp_df['f1_score'].mean():.4f}"
                    )

    # SKAB sweep: bütün fold ve seed kombinasyonları
    print("\n>> SKAB Markov Order/Threshold Taraması...")
    print(f"   Threshold aralığı: {skab_thresholds}")
    for order in orders:
        for threshold in skab_thresholds:
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
                    cc = {
                        **config,
                        "order": order,
                        "anomaly_threshold": threshold,
                    }
                    res, _ = run_experiment_pipeline(
                        X_train_s,
                        X_test_s,
                        y_test_s,
                        cc,
                        "SKAB",
                        f"fold_{fold}",
                        seed=seed,
                    )
                    sweep_results.extend(res)

            temp_df = pd.DataFrame([r for r in sweep_results if r["dataset"] == "SKAB"
                                    and r["order"] == order
                                    and r["anomaly_threshold"] == threshold
                                    and r["scenario"] == "original"])
            if not temp_df.empty:
                print(
                    f"SKAB -> order={order}, threshold={threshold} | "
                    f"Precision={temp_df['precision'].mean():.4f}, "
                    f"Recall={temp_df['recall'].mean():.4f}, "
                    f"F1={temp_df['f1_score'].mean():.4f}"
                )

    if not sweep_results:
        print("[UYARI] Sweep için uygun veri dosyası bulunamadı.")
        return pd.DataFrame()

    df_sweep = pd.DataFrame(sweep_results)
    sweep_path = "results/outputs/automata_markov_sweep_metrics.csv"
    dataset_threshold_sweep_path = "results/outputs/automata_markov_dataset_threshold_sweep_metrics.csv"
    df_sweep.to_csv(sweep_path, index=False)
    df_sweep.to_csv(dataset_threshold_sweep_path, index=False)
    print(f"\n[OK] Markov sweep sonuçları kaydedildi: {sweep_path}")
    print(f"[OK] Dataset bazlı threshold sweep sonuçları kaydedildi: {dataset_threshold_sweep_path}")

    # Optimizasyon için original senaryonun özetini ayrıca yazıyoruz.
    df_original = df_sweep[df_sweep["scenario"] == "original"].copy()
    if not df_original.empty:
        summary_df = df_original.groupby(["dataset", "order", "anomaly_threshold"]).agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            precision_mean=("precision", "mean"),
            precision_std=("precision", "std"),
            recall_mean=("recall", "mean"),
            recall_std=("recall", "std"),
            f1_score_mean=("f1_score", "mean"),
            f1_score_std=("f1_score", "std"),
            transition_density_mean=("transition_density", "mean"),
            num_states_mean=("num_states", "mean"),
            num_transitions_mean=("num_transitions", "mean"),
        ).reset_index()

        summary_path = "results/outputs/automata_markov_sweep_summary.csv"
        dataset_threshold_summary_path = "results/outputs/automata_markov_dataset_threshold_sweep_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        summary_df.to_csv(dataset_threshold_summary_path, index=False)
        print(f"[OK] Markov sweep özeti kaydedildi: {summary_path}")
        print(f"[OK] Dataset bazlı threshold sweep özeti kaydedildi: {dataset_threshold_summary_path}")

        # Recall düşmeden en yüksek precision/F1 adaylarını görmeyi kolaylaştırır.
        best_df = summary_df.sort_values(
            by=["dataset", "recall_mean", "precision_mean", "f1_score_mean"],
            ascending=[True, False, False, False],
        )
        best_path = "results/outputs/automata_markov_sweep_best_candidates.csv"
        dataset_threshold_best_path = "results/outputs/automata_markov_dataset_threshold_best_candidates.csv"
        best_df.to_csv(best_path, index=False)
        best_df.to_csv(dataset_threshold_best_path, index=False)
        print(f"[OK] En iyi adaylar kaydedildi: {best_path}")
        print(f"[OK] Dataset bazlı en iyi adaylar kaydedildi: {dataset_threshold_best_path}")

    return df_sweep


def write_summary_files(df_all):
    """Ana metrik CSV'sinden SKAB/BATADAL özet dosyalarını üretir."""
    if df_all.empty:
        return

    df_skab = df_all[df_all["dataset"] == "SKAB"]
    if not df_skab.empty:
        summary_df = df_skab.groupby(["scenario", "order", "anomaly_threshold"]).agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            precision_mean=("precision", "mean"),
            precision_std=("precision", "std"),
            recall_mean=("recall", "mean"),
            recall_std=("recall", "std"),
            f1_score_mean=("f1_score", "mean"),
            f1_score_std=("f1_score", "std"),
        ).reset_index()
        summary_df.to_csv("results/outputs/automata_skab_fold_summary.csv", index=False)
        print("\nSKAB Otomata Özet:")
        print(summary_df)

    df_batadal = df_all[df_all["dataset"] == "BATADAL"]
    if not df_batadal.empty:
        batadal_summary = df_batadal.groupby(["scenario", "order", "anomaly_threshold"]).agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            precision_mean=("precision", "mean"),
            precision_std=("precision", "std"),
            recall_mean=("recall", "mean"),
            recall_std=("recall", "std"),
            f1_score_mean=("f1_score", "mean"),
            f1_score_std=("f1_score", "std"),
        ).reset_index()
        batadal_summary.to_csv("results/outputs/automata_batadal_seed_summary.csv", index=False)
        print("\nBATADAL Otomata Özet:")
        print(batadal_summary)


def main():
    config = load_json_config()
    seeds = config["seeds"]
    os.makedirs("results/outputs", exist_ok=True)
    all_res = []

    print("\n[INFO] SKAB Veri Seti Bölme Stratejisi:")
    print("-> Fiziksel düzenek bütünlüğü ve zamansal bağımlılıkları korumak adına GroupKFold mimarisi aktiftir.")

    # ANA EĞİTİM: config'teki tekil order/threshold ile çalışır.
    # Fine-tuning sonuçları ayrıca automata_markov_sweep_metrics.csv içine yazılır.
    if os.path.exists("data/processed/batadal_X_train_adasyn_pc1.csv"):
        print("\n" + "=" * 60)
        print("[ADASYN ENTEGRASYONU] Otomata modeli ADASYN_PC1 verisiyle eğitiliyor...")
        print("=" * 60 + "\n")

        X_train = pd.read_csv("data/processed/batadal_X_train_adasyn_pc1.csv").values.flatten()
        X_test = pd.read_csv("data/processed/batadal_X_test_pc1.csv").values.flatten()
        y_test = pd.read_csv("data/processed/batadal_y_test.csv").values.flatten()
        y_test = np.where(y_test == -999, 0, y_test)

        config_batadal = {
            **config,
            "order": config.get("order", 2),
            "anomaly_threshold": config.get("batadal_anomaly_threshold", config.get("anomaly_threshold", 0.05)),
        }

        batadal_logs = None
        for seed in seeds:
            print(f"BATADAL automata seed={seed}, order={config_batadal['order']}, threshold={config_batadal['anomaly_threshold']} çalışıyor...")
            batadal_res, logs = run_experiment_pipeline(
                X_train,
                X_test,
                y_test,
                config_batadal,
                "BATADAL",
                "single",
                seed=seed,
            )
            all_res.extend(batadal_res)
            if batadal_logs is None:
                batadal_logs = logs

        if batadal_logs:
            with open("results/outputs/automata_batadal_advanced_explainability.json", "w") as f:
                json.dump(batadal_logs[:100], f, indent=4)

    config_skab = {
        **config,
        "order": config.get("order", 2),
        "anomaly_threshold": config.get("skab_anomaly_threshold", config.get("anomaly_threshold", 0.90)),
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
                print(f"SKAB automata fold={fold}, seed={seed}, order={config_skab['order']}, threshold={config_skab['anomaly_threshold']} çalışıyor...")
                fold_res, _ = run_experiment_pipeline(
                    X_train,
                    X_test,
                    y_test,
                    config_skab,
                    "SKAB",
                    f"fold_{fold}",
                    seed=seed,
                )
                all_res.extend(fold_res)

    df_all = pd.DataFrame(all_res)
    df_all.to_csv("results/outputs/automata_advanced_all_scenarios_metrics.csv", index=False)
    write_summary_files(df_all)

    # Eski window/alphabet sensitivity yerine yeni Markov order/threshold sweep.
    run_markov_order_threshold_sweep(config)

    try:
        from src.experiments.statistical_tests import main as run_statistical_main
        run_statistical_main()
    except Exception as e:
        print(f"İstatistiksel testler tetiklenirken bir hata meydana geldi: {str(e)}")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
