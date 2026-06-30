"""
İstatistiksel Anlamlılık Testleri
- SKAB: LSTM vs GRU vs Automata (fold bazlı F1 skorları) → Wilcoxon Signed-Rank Test
- BATADAL: LSTM vs GRU (seed bazlı F1 skorları) → Wilcoxon Signed-Rank Test
"""

import os
import sys
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.config import load_config

config = load_config()
DL_CONFIG = config["deep_learning"]

OUTPUT_PATH = "results/outputs/statistical_test_results.csv"


def run_wilcoxon(scores_a, scores_b, label_a, label_b, dataset):
    scores_a = np.array(scores_a)
    scores_b = np.array(scores_b)
    diff = scores_a - scores_b

    if np.all(diff == 0):
        return {
            "dataset": dataset,
            "model_a": label_a,
            "model_b": label_b,
            "mean_a": float(np.mean(scores_a)),
            "mean_b": float(np.mean(scores_b)),
            "statistic": None,
            "p_value": None,
            "significant (p<0.05)": "N/A (identical scores)"
        }

    stat, p = wilcoxon(scores_a, scores_b)
    return {
        "dataset": dataset,
        "model_a": label_a,
        "model_b": label_b,
        "mean_a": float(np.mean(scores_a)),
        "mean_b": float(np.mean(scores_b)),
        "statistic": float(stat),
        "p_value": float(p),
        "significant (p<0.05)": "Evet" if p < 0.05 else "Hayir"
    }


def run_skab_tests():
    """SKAB: LSTM vs GRU vs Automata — fold bazlı F1 skorları üzerinden Wilcoxon testi."""
    results = []

    lstm_path = "results/outputs/skab_lstm_results.csv"
    gru_path = "results/outputs/skab_gru_results.csv"
    automata_path = "results/outputs/automata_advanced_all_scenarios_metrics.csv"

    if not (os.path.exists(lstm_path) and os.path.exists(gru_path)):
        print("SKAB DL sonuç dosyaları bulunamadı, SKAB testi atlanıyor.")
        return results

    lstm_df = pd.read_csv(lstm_path).sort_values("fold")
    gru_df = pd.read_csv(gru_path).sort_values("fold")

    lstm_f1 = lstm_df["f1"].values
    gru_f1 = gru_df["f1"].values

    # LSTM vs GRU
    results.append(run_wilcoxon(lstm_f1, gru_f1, "LSTM", "GRU", "SKAB"))

    # Automata fold F1 skorları (original senaryo)
    if os.path.exists(automata_path):
        auto_df = pd.read_csv(automata_path)
        auto_skab = auto_df[
            (auto_df["dataset"] == "SKAB") & (auto_df["scenario"] == "original")
        ]

        # 5 seed x 5 fold durumunda fold başına seed ortalaması alınır → 5 değer
        auto_skab_fold = auto_skab.groupby("fold")["f1_score"].mean().sort_index()
        auto_f1 = auto_skab_fold.values

        if len(auto_f1) == len(lstm_f1):
            results.append(run_wilcoxon(lstm_f1, auto_f1, "LSTM", "Automata", "SKAB"))
            results.append(run_wilcoxon(gru_f1, auto_f1, "GRU", "Automata", "SKAB"))
        else:
            print(
                f"SKAB automata fold sayısı uyuşmuyor "
                f"({len(auto_f1)} vs {len(lstm_f1)}), Automata karşılaştırması atlanıyor."
            )

    return results


def run_batadal_tests():
    """BATADAL: LSTM vs GRU — seed bazlı F1 skorları üzerinden Wilcoxon testi."""
    results = []

    seed_path = "results/outputs/batadal_deep_learning_seed_results.csv"
    if not os.path.exists(seed_path):
        print("BATADAL seed sonuç dosyası bulunamadı, BATADAL testi atlanıyor.")
        return results

    df = pd.read_csv(seed_path)
    seeds = DL_CONFIG["seeds"]

    lstm_f1 = []
    gru_f1 = []

    for seed in seeds:
        lstm_row = df[(df["model"] == "LSTM") & (df["seed"] == seed)]
        gru_row = df[(df["model"] == "GRU") & (df["seed"] == seed)]

        if not lstm_row.empty and not gru_row.empty:
            lstm_f1.append(lstm_row["f1"].values[0])
            gru_f1.append(gru_row["f1"].values[0])

    if len(lstm_f1) >= 2:
        results.append(run_wilcoxon(lstm_f1, gru_f1, "LSTM", "GRU", "BATADAL"))
    else:
        print(f"BATADAL için yeterli eşleşen seed verisi bulunamadı ({len(lstm_f1)} çift).")

    return results


def main():
    os.makedirs("results/outputs", exist_ok=True)
    all_results = []

    print("--- SKAB Wilcoxon Testleri ---")
    skab_results = run_skab_tests()
    all_results.extend(skab_results)

    for r in skab_results:
        print(
            f"  {r['model_a']} vs {r['model_b']}: "
            f"p={r['p_value']}, anlamlı={r['significant (p<0.05)']}"
        )

    print("\n--- BATADAL Wilcoxon Testleri ---")
    batadal_results = run_batadal_tests()
    all_results.extend(batadal_results)

    for r in batadal_results:
        print(
            f"  {r['model_a']} vs {r['model_b']}: "
            f"p={r['p_value']}, anlamlı={r['significant (p<0.05)']}"
        )

    if all_results:
        df_out = pd.DataFrame(all_results)
        df_out.to_csv(OUTPUT_PATH, index=False)
        print(f"\nSonuçlar kaydedildi: {OUTPUT_PATH}")
        print(df_out.to_string(index=False))


if __name__ == "__main__":
    main()