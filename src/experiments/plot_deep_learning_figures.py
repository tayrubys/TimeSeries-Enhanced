from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

#BATADAL ve SKAB seed summary dosyalarını okur
def load_summary_files():

    batadal_df = pd.read_csv("results/outputs/batadal_deep_learning_seed_summary.csv")
    skab_df = pd.read_csv("results/outputs/skab_deep_learning_seed_summary.csv")

    return batadal_df, skab_df

#Bir veri seti için LSTM ve GRU modellerinin accuracy, precision, recall ve f1 mean değerlerini çizer.
def plot_dataset_metric_comparison(df, dataset_name, output_dir):

    models = df["model"].tolist()

    accuracy_values = df["accuracy_mean"].tolist()
    precision_values = df["precision_mean"].tolist()
    recall_values = df["recall_mean"].tolist()
    f1_values = df["f1_mean"].tolist()

    x = range(len(models))
    width = 0.2

    plt.figure(figsize=(10, 6))

    plt.bar([i - 1.5 * width for i in x], accuracy_values, width=width, label="Accuracy")
    plt.bar([i - 0.5 * width for i in x], precision_values, width=width, label="Precision")
    plt.bar([i + 0.5 * width for i in x], recall_values, width=width, label="Recall")
    plt.bar([i + 1.5 * width for i in x], f1_values, width=width, label="F1-score")

    plt.xticks(list(x), models)
    plt.ylim(0, 1)
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.title(f"{dataset_name} Deep Learning Metric Comparison")
    plt.legend()
    plt.tight_layout()

    output_path = output_dir / f"{dataset_name.lower()}_metric_comparison.png"
    plt.savefig(output_path)
    plt.close()

    print(f"Kaydedildi: {output_path}")

#BATADAL ve SKAB için LSTM / GRU F1-score karşılaştırması çizer
def plot_f1_comparison(batadal_df, skab_df, output_dir):

    labels = []
    values = []

    for _, row in batadal_df.iterrows():
        labels.append(f"BATADAL-{row['model']}")
        values.append(row["f1_mean"])

    for _, row in skab_df.iterrows():
        labels.append(f"SKAB-{row['model']}")
        values.append(row["f1_mean"])

    plt.figure(figsize=(10, 6))
    plt.bar(labels, values)

    plt.ylim(0, 1)
    plt.xlabel("Dataset - Model")
    plt.ylabel("F1-score")
    plt.title("Deep Learning F1-score Comparison")
    plt.xticks(rotation=20)
    plt.tight_layout()

    output_path = output_dir / "deep_learning_f1_comparison.png"
    plt.savefig(output_path)
    plt.close()

    print(f"Kaydedildi: {output_path}")


def main():
    output_dir = Path("results/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    batadal_df, skab_df = load_summary_files()

    print("BATADAL summary:")
    print(batadal_df)

    print("\nSKAB summary:")
    print(skab_df)

    plot_dataset_metric_comparison(
        df=batadal_df,
        dataset_name="BATADAL",
        output_dir=output_dir
    )

    plot_dataset_metric_comparison(
        df=skab_df,
        dataset_name="SKAB",
        output_dir=output_dir
    )

    plot_f1_comparison(
        batadal_df=batadal_df,
        skab_df=skab_df,
        output_dir=output_dir
    )


if __name__ == "__main__":
    main()