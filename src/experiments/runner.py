import os
import json
import sys
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

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
        "noise_level": 0.1,
        "seeds": [42, 123, 2026, 7, 999],
        "batadal_train_ratio": 0.60,
        "batadal_val_ratio": 0.20,

        # Genel PST ayarları
        "suffix_max_order": 3,
        "suffix_min_context_count": 2,
        "suffix_smoothing_alpha": 1.0,

        # Grid search için PST listeleri
        "suffix_max_orders": [2, 3, 4],
        "suffix_min_context_counts": [1, 2, 3],
        "suffix_smoothing_alphas": [0.1, 0.5, 1.0],

        # BATADAL özel PST ayarları
        "batadal_window_size": 4,
        "batadal_alphabet_size": 3,
        "batadal_suffix_max_order": 3,
        "batadal_suffix_min_context_count": 2,
        "batadal_suffix_smoothing_alpha": 1.0,

        # SKAB özel PST ayarları
        "skab_window_size": 4,
        "skab_alphabet_size": 4,
        "skab_suffix_max_order": 3,
        "skab_suffix_min_context_count": 2,
        "skab_suffix_smoothing_alpha": 1.0,
    }

    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                user_config = json.load(f)

            automata_config = user_config.get("automata", {})

            config = default_config.copy()

            for key, value in automata_config.items():
                config[key] = value

            return config

        except Exception as e:
            print(f"[WARN] Config okunamadı, varsayılan ayarlar kullanılacak: {str(e)}")
            return default_config

    return default_config


def inject_gaussian_noise(series, noise_level=0.1, seed=42):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, noise_level, series.shape)
    return series + noise


def build_suffix_automata(config):
    
    return ProbabilisticAutomata(
        smoothing=True,
        max_order=config.get("suffix_max_order", 3),
        min_context_count=config.get("suffix_min_context_count", 2),
        smoothing_alpha=config.get("suffix_smoothing_alpha", 1.0),
    )

def get_dataset_specific_config(config, dataset_name):
    """
    Her veri seti için ayrı PST/SAX/threshold ayarlarını üretir.

    dataset_name:
        "BATADAL" veya "SKAB"
    """

    dataset_key = dataset_name.lower()

    dataset_config = config.copy()

    dataset_config["window_size"] = config.get(
        f"{dataset_key}_window_size",
        config.get("window_size", 4),
    )

    dataset_config["alphabet_size"] = config.get(
        f"{dataset_key}_alphabet_size",
        config.get("alphabet_size", 3),
    )

    dataset_config["suffix_max_order"] = config.get(
        f"{dataset_key}_suffix_max_order",
        config.get("suffix_max_order", 3),
    )

    dataset_config["suffix_min_context_count"] = config.get(
        f"{dataset_key}_suffix_min_context_count",
        config.get("suffix_min_context_count", 2),
    )

    dataset_config["suffix_smoothing_alpha"] = config.get(
        f"{dataset_key}_suffix_smoothing_alpha",
        config.get("suffix_smoothing_alpha", 1.0),
    )

    dataset_config["anomaly_threshold"] = config.get(
        f"{dataset_key}_anomaly_threshold",
        config.get("anomaly_threshold", 0.05),
    )

    return dataset_config


def get_model_complexity_fields(model):

    num_states = len(model.trained_patterns)
    num_contexts = len(model.transitions)
    num_transitions = sum(len(targets) for targets in model.transitions.values())

    if num_contexts > 0 and num_states > 0:
        transition_density = num_transitions / (num_contexts * num_states)
    else:
        transition_density = 0.0

    return {
        "num_states": num_states,
        "num_contexts": num_contexts,
        "num_transitions": num_transitions,
        "transition_density": transition_density,
    }


def run_experiment_pipeline(
    X_train,
    X_test,
    y_test,
    config,
    dataset_name,
    fold_name="",
    seed=42,
):
    results = []

    transformer = SaxPaaTransformer(
        alphabet_size=config["alphabet_size"],
    )

    train_patterns = transformer.transform(
        X_train,
        window_size=config["window_size"],
    )

    model = build_suffix_automata(config)
    model.fit(train_patterns)

    complexity_fields = get_model_complexity_fields(model)

    common_fields = {
        "dataset": dataset_name,
        "fold": fold_name,
        "seed": seed,
        "window_size": config["window_size"],
        "alphabet_size": config["alphabet_size"],
        "model_variant": "probabilistic_suffix_tree",
        "suffix_max_order": config.get("suffix_max_order", 3),
        "suffix_min_context_count": config.get("suffix_min_context_count", 2),
        "suffix_smoothing_alpha": config.get("suffix_smoothing_alpha", 1.0),
        **complexity_fields,
    }

    # --- SENARYO 1: Orijinal Veri ---
    test_patterns_orig = transformer.transform(
        X_test,
        window_size=config["window_size"],
    )

    preds_orig, logs_orig = model.predict(
        test_patterns_orig,
        anomaly_threshold=config["anomaly_threshold"],
    )

    y_test_aligned_orig = y_test[:len(preds_orig)]
    metrics_orig = calculate_metrics(y_test_aligned_orig, preds_orig)
    metrics_orig.update(
        {
            "scenario": "original",
            **common_fields,
        }
    )
    results.append(metrics_orig)

    # --- SENARYO 2: Gaussian Noise ---
    X_test_noisy = inject_gaussian_noise(
        X_test,
        noise_level=config["noise_level"],
        seed=seed,
    )

    test_patterns_noisy = transformer.transform(
        X_test_noisy,
        window_size=config["window_size"],
    )

    preds_noisy, _ = model.predict(
        test_patterns_noisy,
        anomaly_threshold=config["anomaly_threshold"],
    )

    y_test_aligned_noisy = y_test[:len(preds_noisy)]
    metrics_noisy = calculate_metrics(y_test_aligned_noisy, preds_noisy)
    metrics_noisy.update(
        {
            "scenario": "gaussian_noise",
            **common_fields,
        }
    )
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
        preds_unseen, _ = model.predict(
            unseen_test_patterns,
            anomaly_threshold=config["anomaly_threshold"],
        )
        y_test_aligned_unseen = y_test_unseen[-len(preds_unseen):]
        metrics_unseen = calculate_metrics(y_test_aligned_unseen, preds_unseen)
    else:
        metrics_unseen = {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
        }

    metrics_unseen.update(
        {
            "scenario": "unseen_data",
            **common_fields,
        }
    )
    results.append(metrics_unseen)

    return results, logs_orig


def run_parameter_sensitivity_analysis(config):
    print("\n--- PST PARAMETRE + THRESHOLD TARAMASI BAŞLATILIYOR ---")

    window_sizes = config.get("window_sizes", [3, 4, 5, 6])
    alphabet_sizes = config.get("alphabet_sizes", [3, 4, 5, 6])

    suffix_max_orders = config.get(
        "suffix_max_orders",
        [config.get("suffix_max_order", 3)],
    )

    suffix_min_context_counts = config.get(
        "suffix_min_context_counts",
        [config.get("suffix_min_context_count", 2)],
    )

    suffix_smoothing_alphas = config.get(
        "suffix_smoothing_alphas",
        [config.get("suffix_smoothing_alpha", 1.0)],
    )

    batadal_thresholds = config.get(
        "batadal_thresholds",
        [0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20],
    )

    skab_thresholds = config.get(
        "skab_thresholds",
        [0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 0.90],
    )

    sensitivity_results = []

    def run_grid_for_dataset(
        dataset_name,
        X_train,
        X_test,
        y_test,
        thresholds,
        fold_name="param_search",
    ):
        dataset_results = []

        print(f"\n>> {dataset_name} PST Parametre + Threshold Taraması...")

        for w in window_sizes:
            for a in alphabet_sizes:
                for max_order in suffix_max_orders:
                    for min_context_count in suffix_min_context_counts:
                        for smoothing_alpha in suffix_smoothing_alphas:
                            for threshold in thresholds:
                                cc = {
                                    **config,
                                    "window_size": w,
                                    "alphabet_size": a,
                                    "suffix_max_order": max_order,
                                    "suffix_min_context_count": min_context_count,
                                    "suffix_smoothing_alpha": smoothing_alpha,
                                    "anomaly_threshold": threshold,
                                    "noise_level": config["noise_level"],
                                }

                                res, _ = run_experiment_pipeline(
                                    X_train,
                                    X_test,
                                    y_test,
                                    cc,
                                    dataset_name,
                                    fold_name,
                                    seed=config["seeds"][0],
                                )

                                for row in res:
                                    row["tuning_type"] = "pst_threshold_grid"
                                    row["threshold"] = threshold
                                    row["suffix_smoothing_alpha"] = smoothing_alpha
                                    dataset_results.append(row)
                                    sensitivity_results.append(row)

                                orig_res = [
                                    r for r in res if r["scenario"] == "original"
                                ][0]

                                print(
                                    f"{dataset_name} -> "
                                    f"W={w}, A={a}, "
                                    f"Order={max_order}, "
                                    f"MinCtx={min_context_count}, "
                                    f"Alpha={smoothing_alpha}, "
                                    f"Thr={threshold} | "
                                    f"F1={orig_res['f1_score']:.4f}, "
                                    f"P={orig_res['precision']:.4f}, "
                                    f"R={orig_res['recall']:.4f}, "
                                    f"Density={orig_res['transition_density']:.4f}"
                                )

        return dataset_results

    # BATADAL
    if os.path.exists("data/processed/batadal_X_train_adasyn_pc1.csv"):
        X_train_b = pd.read_csv(
            "data/processed/batadal_X_train_adasyn_pc1.csv"
        ).values.flatten()
        X_test_b = pd.read_csv(
            "data/processed/batadal_X_test_pc1.csv"
        ).values.flatten()
        y_test_b = pd.read_csv(
            "data/processed/batadal_y_test.csv"
        ).values.flatten()
        y_test_b = np.where(y_test_b == -999, 0, y_test_b)

        run_grid_for_dataset(
            dataset_name="BATADAL",
            X_train=X_train_b,
            X_test=X_test_b,
            y_test=y_test_b,
            thresholds=batadal_thresholds,
            fold_name="param_search",
        )

    # SKAB için fold1 hızlı tarama
    skab_train_path = "data/processed/skab_fold1_X_train_pc1.csv"
    skab_test_path = "data/processed/skab_fold1_X_test_pc1.csv"
    skab_y_path = "data/processed/skab_fold1_y_test.csv"

    if (
        os.path.exists(skab_train_path)
        and os.path.exists(skab_test_path)
        and os.path.exists(skab_y_path)
    ):
        X_train_s = pd.read_csv(skab_train_path).values.flatten()
        X_test_s = pd.read_csv(skab_test_path).values.flatten()
        y_test_s = pd.read_csv(skab_y_path).values.flatten()

        run_grid_for_dataset(
            dataset_name="SKAB",
            X_train=X_train_s,
            X_test=X_test_s,
            y_test=y_test_s,
            thresholds=skab_thresholds,
            fold_name="fold1_param_search",
        )

    if not sensitivity_results:
        print("[WARN] Parametre taraması için uygun veri bulunamadı.")
        return

    df_sensitivity = pd.DataFrame(sensitivity_results)

    os.makedirs("results/outputs/pst_ablation", exist_ok=True)

    all_candidates_path = "results/outputs/pst_ablation/pst_all_candidates.csv"
    best_original_path = "results/outputs/pst_ablation/pst_best_original_by_dataset.csv"
    best_stable_path = "results/outputs/pst_ablation/pst_best_stable_by_dataset.csv"

    df_sensitivity.to_csv(all_candidates_path, index=False)

    # Eski dosya adıyla da kaydedelim ki mevcut akış bozulmasın.
    df_sensitivity.to_csv(
        "results/outputs/automata_param_sensitivity_metrics.csv",
        index=False,
    )

    df_original = df_sensitivity[df_sensitivity["scenario"] == "original"].copy()

    if not df_original.empty:
        best_original = (
            df_original.sort_values(
                ["dataset", "f1_score", "recall", "precision"],
                ascending=[True, False, False, False],
            )
            .groupby("dataset")
            .head(10)
            .reset_index(drop=True)
        )

        best_original.to_csv(best_original_path, index=False)

        print("\n--- DATASET BAZLI EN İYİ ORIGINAL F1 ADAYLARI ---")

        for dataset_name in best_original["dataset"].unique():
            print(f"\n{dataset_name} top candidates:")

            cols = [
                "dataset",
                "scenario",
                "window_size",
                "alphabet_size",
                "suffix_max_order",
                "suffix_min_context_count",
                "suffix_smoothing_alpha",
                "threshold",
                "accuracy",
                "precision",
                "recall",
                "f1_score",
                "num_contexts",
                "transition_density",
            ]

            print(best_original[best_original["dataset"] == dataset_name][cols].head(5))

    # Stabil aday seçimi:
    # original + gaussian_noise birlikte iyi olsun diye ortalama F1 hesaplıyoruz.
    df_stability = (
        df_sensitivity[df_sensitivity["scenario"].isin(["original", "gaussian_noise"])]
        .groupby(
            [
                "dataset",
                "window_size",
                "alphabet_size",
                "suffix_max_order",
                "suffix_min_context_count",
                "suffix_smoothing_alpha",
                "threshold",
            ]
        )
        .agg(
            stability_f1_mean=("f1_score", "mean"),
            stability_f1_min=("f1_score", "min"),
            accuracy_mean=("accuracy", "mean"),
            precision_mean=("precision", "mean"),
            recall_mean=("recall", "mean"),
            num_contexts_mean=("num_contexts", "mean"),
            transition_density_mean=("transition_density", "mean"),
        )
        .reset_index()
    )

    if not df_stability.empty:
        best_stable = (
            df_stability.sort_values(
                ["dataset", "stability_f1_mean", "stability_f1_min", "recall_mean"],
                ascending=[True, False, False, False],
            )
            .groupby("dataset")
            .head(10)
            .reset_index(drop=True)
        )

        best_stable.to_csv(best_stable_path, index=False)

        print("\n--- DATASET BAZLI EN STABİL PST ADAYLARI ---")

        for dataset_name in best_stable["dataset"].unique():
            print(f"\n{dataset_name} stable candidates:")
            print(best_stable[best_stable["dataset"] == dataset_name].head(5))

    print("\nPST tarama dosyaları kaydedildi:")
    print(f"- {all_candidates_path}")
    print(f"- {best_original_path}")
    print(f"- {best_stable_path}")


def main():
    config = load_json_config()
    seeds = config["seeds"]

    os.makedirs("results/outputs", exist_ok=True)

    all_res = []

    print("\n[INFO] SKAB Veri Seti Bölme Stratejisi:")
    print(
        "-> Fiziksel düzenek bütünlüğü ve zamansal bağımlılıkları "
        "korumak adına GroupKFold mimarisi aktiftir."
    )

    config_batadal_preview = get_dataset_specific_config(config, "BATADAL")
    config_skab_preview = get_dataset_specific_config(config, "SKAB")

    print("\n[INFO] Probabilistic Suffix Tree ayarları:")

    print(
        "-> BATADAL: "
        f"window={config_batadal_preview['window_size']}, "
        f"alphabet={config_batadal_preview['alphabet_size']}, "
        f"suffix_max_order={config_batadal_preview['suffix_max_order']}, "
        f"min_context_count={config_batadal_preview['suffix_min_context_count']}, "
        f"smoothing_alpha={config_batadal_preview['suffix_smoothing_alpha']}, "
        f"threshold={config_batadal_preview['anomaly_threshold']}"
    )

    print(
        "-> SKAB: "
        f"window={config_skab_preview['window_size']}, "
        f"alphabet={config_skab_preview['alphabet_size']}, "
        f"suffix_max_order={config_skab_preview['suffix_max_order']}, "
        f"min_context_count={config_skab_preview['suffix_min_context_count']}, "
        f"smoothing_alpha={config_skab_preview['suffix_smoothing_alpha']}, "
        f"threshold={config_skab_preview['anomaly_threshold']}"
    )

    if os.path.exists("data/processed/batadal_X_train_adasyn_pc1.csv"):
        print("\n" + "=" * 60)
        print("[ADASYN ENTEGRASYONU] PST otomata modeli ADASYN_PC1 verisiyle eğitiliyor...")
        print("=" * 60 + "\n")

        X_train = pd.read_csv(
            "data/processed/batadal_X_train_adasyn_pc1.csv"
        ).values.flatten()
        X_test = pd.read_csv(
            "data/processed/batadal_X_test_pc1.csv"
        ).values.flatten()
        y_test = pd.read_csv(
            "data/processed/batadal_y_test.csv"
        ).values.flatten()
        y_test = np.where(y_test == -999, 0, y_test)

        config_batadal = get_dataset_specific_config(config, "BATADAL")

        batadal_logs = None

        for seed in seeds:
            print(
                "BATADAL PST automata "
                f"seed={seed}, "
                f"window={config_batadal['window_size']}, "
                f"alphabet={config_batadal['alphabet_size']}, "
                f"suffix_max_order={config_batadal.get('suffix_max_order', 3)}, "
                f"min_context_count={config_batadal.get('suffix_min_context_count', 2)}, "
                f"threshold={config_batadal['anomaly_threshold']} çalışıyor..."
            )

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

        if batadal_logs is not None:
            with open(
                "results/outputs/automata_batadal_advanced_explainability.json",
                "w",
            ) as f:
                json.dump(batadal_logs[:100], f, indent=4)

    config_skab = get_dataset_specific_config(config, "SKAB")

    for fold in range(1, 6):
        train_file = f"data/processed/skab_fold{fold}_X_train_pc1.csv"
        test_file = f"data/processed/skab_fold{fold}_X_test_pc1.csv"
        y_test_file = f"data/processed/skab_fold{fold}_y_test.csv"

        if (
            os.path.exists(train_file)
            and os.path.exists(test_file)
            and os.path.exists(y_test_file)
        ):
            X_train = pd.read_csv(train_file).values.flatten()
            X_test = pd.read_csv(test_file).values.flatten()
            y_test = pd.read_csv(y_test_file).values.flatten()

            for seed in seeds:
                print(
                    "SKAB PST automata "
                    f"fold={fold}, "
                    f"seed={seed}, "
                    f"window={config_skab['window_size']}, "
                    f"alphabet={config_skab['alphabet_size']}, "
                    f"suffix_max_order={config_skab.get('suffix_max_order', 3)}, "
                    f"min_context_count={config_skab.get('suffix_min_context_count', 2)}, "
                    f"threshold={config_skab['anomaly_threshold']} çalışıyor..."
                )

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
    df_all.to_csv(
        "results/outputs/automata_advanced_all_scenarios_metrics.csv",
        index=False,
    )

    # SKAB özet
    df_skab = df_all[df_all["dataset"] == "SKAB"]

    if not df_skab.empty:
        summary_df = df_skab.groupby("scenario").agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            precision_mean=("precision", "mean"),
            precision_std=("precision", "std"),
            recall_mean=("recall", "mean"),
            recall_std=("recall", "std"),
            f1_score_mean=("f1_score", "mean"),
            f1_score_std=("f1_score", "std"),
            num_contexts_mean=("num_contexts", "mean"),
            transition_density_mean=("transition_density", "mean"),
        ).reset_index()

        summary_df.to_csv(
            "results/outputs/automata_skab_fold_summary.csv",
            index=False,
        )

        print("\nSKAB PST Otomata Özet (5 fold x 5 seed):")
        print(summary_df)

    # BATADAL özet
    df_batadal = df_all[df_all["dataset"] == "BATADAL"]

    if not df_batadal.empty:
        batadal_summary = df_batadal.groupby("scenario").agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            precision_mean=("precision", "mean"),
            precision_std=("precision", "std"),
            recall_mean=("recall", "mean"),
            recall_std=("recall", "std"),
            f1_score_mean=("f1_score", "mean"),
            f1_score_std=("f1_score", "std"),
            num_contexts_mean=("num_contexts", "mean"),
            transition_density_mean=("transition_density", "mean"),
        ).reset_index()

        batadal_summary.to_csv(
            "results/outputs/automata_batadal_seed_summary.csv",
            index=False,
        )

        print("\nBATADAL PST Otomata Özet (5 seed):")
        print(batadal_summary)

    run_parameter_sensitivity_analysis(config)

    try:
        from src.experiments.statistical_tests import main as run_statistical_main

        run_statistical_main()
    except Exception as e:
        print(f"İstatistiksel testler tetiklenirken bir hata meydana geldi: {str(e)}")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()