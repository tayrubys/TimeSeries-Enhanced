from pathlib import Path

import pandas as pd


OUTPUT_DIR = Path("results/outputs")


def add_deep_learning_results(rows):
    """
    Deep learning seed summary dosyalarını okuyup
    ortak karşılaştırma tablosuna LSTM ve GRU sonuçlarını ekler.
    """

    dl_files = [
        OUTPUT_DIR / "batadal_deep_learning_seed_summary.csv",
        OUTPUT_DIR / "skab_deep_learning_seed_summary.csv",
    ]

    for file_path in dl_files:
        if not file_path.exists():
            print(f"Dosya bulunamadı, atlandı: {file_path}")
            continue

        df = pd.read_csv(file_path)

        for _, row in df.iterrows():
            rows.append({
                "dataset": row["dataset"],
                "model": row["model"],
                "scenario": "original",
                "accuracy": row["accuracy_mean"],
                "precision": row["precision_mean"],
                "recall": row["recall_mean"],
                "f1_score": row["f1_mean"],
                "source_file": file_path.name,
            })


def add_batadal_automata_results(rows):
    """
    BATADAL automata sonuçlarını ortak tabloya ekler.
    Öncelik: automata_advanced_all_scenarios_metrics.csv içindeki original senaryo.
    """

    file_path = OUTPUT_DIR / "automata_advanced_all_scenarios_metrics.csv"

    if not file_path.exists():
        print(f"Dosya bulunamadı, atlandı: {file_path}")
        return

    df = pd.read_csv(file_path)

    # Kolon isimlerini küçük harfe çekmeden kontrollü kullanıyoruz.
    # Beklenen kolonlar: dataset, scenario, accuracy, precision, recall, f1_score
    if "dataset" in df.columns:
        df_batadal = df[df["dataset"].astype(str).str.upper() == "BATADAL"]
    else:
        df_batadal = df.copy()

    if "scenario" in df_batadal.columns:
        df_batadal = df_batadal[df_batadal["scenario"].astype(str) == "original"]

    if df_batadal.empty:
        print("BATADAL automata original sonucu bulunamadı.")
        return

    row = df_batadal.iloc[0]

    rows.append({
        "dataset": "BATADAL",
        "model": "Automata",
        "scenario": "original",
        "accuracy": row["accuracy"],
        "precision": row["precision"],
        "recall": row["recall"],
        "f1_score": row["f1_score"],
        "source_file": file_path.name,
    })


def add_skab_automata_results(rows):
    """
    SKAB automata fold summary dosyasından original senaryo mean sonuçlarını ekler.
    """

    file_path = OUTPUT_DIR / "automata_skab_fold_summary.csv"

    if not file_path.exists():
        print(f"Dosya bulunamadı, atlandı: {file_path}")
        return

    df = pd.read_csv(file_path)

    if "scenario" in df.columns:
        df_original = df[df["scenario"].astype(str) == "original"]
    else:
        df_original = df.copy()

    if df_original.empty:
        print("SKAB automata original sonucu bulunamadı.")
        return

    row = df_original.iloc[0]

    # Mean/std formatı varsa mean kolonlarını kullanıyoruz.
    if "accuracy_mean" in df_original.columns:
        accuracy = row["accuracy_mean"]
        precision = row["precision_mean"]
        recall = row["recall_mean"]
        f1_score = row["f1_score_mean"]
    else:
        # Eski format olursa fallback
        accuracy = row["accuracy"]
        precision = row["precision"]
        recall = row["recall"]
        f1_score = row["f1_score"]

    rows.append({
        "dataset": "SKAB",
        "model": "Automata",
        "scenario": "original",
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "source_file": file_path.name,
    })


def main():
    rows = []

    add_deep_learning_results(rows)
    add_batadal_automata_results(rows)
    add_skab_automata_results(rows)

    comparison_df = pd.DataFrame(rows)

    if comparison_df.empty:
        print("Karşılaştırma tablosu oluşturulamadı. Girdi dosyalarını kontrol edin.")
        return

    comparison_df = comparison_df[
        [
            "dataset",
            "model",
            "scenario",
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "source_file",
        ]
    ]

    output_path = OUTPUT_DIR / "model_comparison_results.csv"
    comparison_df.to_csv(output_path, index=False)

    print("\nModel karşılaştırma tablosu:")
    print(comparison_df)

    print(f"\nKaydedildi: {output_path}")


if __name__ == "__main__":
    main()