import pandas as pd
from pathlib import Path


def merge_deep_learning_results():
    output_dir = Path("results/outputs")

    batadal_path = output_dir / "batadal_deep_learning_results.csv"
    skab_lstm_path = output_dir / "skab_lstm_results.csv"
    skab_gru_path = output_dir / "skab_gru_results.csv"

    batadal_df = pd.read_csv(batadal_path)
    skab_lstm_df = pd.read_csv(skab_lstm_path)
    skab_gru_df = pd.read_csv(skab_gru_path)

    all_results_df = pd.concat(
        [batadal_df, skab_lstm_df, skab_gru_df],
        ignore_index=True
    )

    output_path = output_dir / "deep_learning_results.csv"
    all_results_df.to_csv(output_path, index=False)

    print(f"Tüm deep learning sonuçları kaydedildi: {output_path}")
    print(all_results_df)


if __name__ == "__main__":
    merge_deep_learning_results()