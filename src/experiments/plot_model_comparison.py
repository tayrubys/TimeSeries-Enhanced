from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def plot_f1_comparison(df, output_dir):
    """
    Dataset bazında LSTM, GRU ve Automata F1-score karşılaştırması çizer.
    """

    df = df.copy()
    df["label"] = df["dataset"] + " - " + df["model"]

    plt.figure(figsize=(10, 6))
    plt.bar(df["label"], df["f1_score"])

    plt.ylim(0, 1)
    plt.xlabel("Dataset - Model")
    plt.ylabel("F1-score")
    plt.title("LSTM vs GRU vs Automata F1-score Comparison")
    plt.xticks(rotation=25)
    plt.tight_layout()

    output_path = output_dir / "model_comparison_f1_score.png"
    plt.savefig(output_path)
    plt.close()

    print(f"Kaydedildi: {output_path}")


def plot_metric_comparison(df, output_dir):
    """
    Tüm modeller için accuracy, precision, recall ve F1-score karşılaştırması çizer.
    """

    df = df.copy()
    df["label"] = df["dataset"] + " - " + df["model"]

    metrics = ["accuracy", "precision", "recall", "f1_score"]

    x = range(len(df))
    width = 0.2

    plt.figure(figsize=(12, 6))

    for i, metric in enumerate(metrics):
        values = df[metric].tolist()
        positions = [pos + (i - 1.5) * width for pos in x]
        plt.bar(positions, values, width=width, label=metric)

    plt.xticks(list(x), df["label"], rotation=25)
    plt.ylim(0, 1)
    plt.xlabel("Dataset - Model")
    plt.ylabel("Score")
    plt.title("Model Performance Comparison")
    plt.legend()
    plt.tight_layout()

    output_path = output_dir / "model_comparison_all_metrics.png"
    plt.savefig(output_path)
    plt.close()

    print(f"Kaydedildi: {output_path}")


def main():
    input_path = Path("results/outputs/model_comparison_results.csv")
    output_dir = Path("results/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)

    print("Model comparison results:")
    print(df)

    plot_f1_comparison(df, output_dir)
    plot_metric_comparison(df, output_dir)


if __name__ == "__main__":
    main()