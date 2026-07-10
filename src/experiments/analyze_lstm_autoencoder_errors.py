import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


PROCESSED_DIR = Path("data/processed")
OUTPUT_DIR = Path("results/outputs/autoencoder_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    X_val = np.load(PROCESSED_DIR / "batadal_X_val_seq.npy").astype("float32")
    y_val = np.load(PROCESSED_DIR / "batadal_y_val_seq.npy").astype("int32")

    y_val = np.where(y_val == -999, 0, y_val)

    return X_val, y_val


def load_errors():
    """
    Bu dosyaların AutoEncoder deneyinden sonra kaydedilmiş olması gerekir.
    Eğer henüz kaydetmediysen, run_batadal_lstm_autoencoder.py içinde
    val_errors hesaplandıktan sonra np.save ile kaydetmelisin.
    """
    val_errors_path = OUTPUT_DIR / "val_errors_seed_42.npy"

    if not val_errors_path.exists():
        raise FileNotFoundError(
            f"{val_errors_path} bulunamadı. "
            "Önce AutoEncoder deneyinde val_errors değerlerini kaydetmelisin."
        )

    return np.load(val_errors_path)


def save_error_summary(y_val, val_errors):
    normal_errors = val_errors[y_val == 0]
    attack_errors = val_errors[y_val == 1]

    summary = {
        "normal_count": len(normal_errors),
        "attack_count": len(attack_errors),
        "normal_mean": normal_errors.mean(),
        "attack_mean": attack_errors.mean(),
        "normal_min": normal_errors.min(),
        "normal_max": normal_errors.max(),
        "attack_min": attack_errors.min(),
        "attack_max": attack_errors.max(),
        "normal_std": normal_errors.std(),
        "attack_std": attack_errors.std(),
    }

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(OUTPUT_DIR / "reconstruction_error_summary.csv", index=False)

    print("\nReconstruction Error Özeti")
    print(summary_df)


def find_extreme_normal_sequences(X_val, y_val, val_errors, top_k=10):
    normal_indices_global = np.where(y_val == 0)[0]
    normal_errors = val_errors[y_val == 0]

    top_local_indices = np.argsort(normal_errors)[-top_k:][::-1]

    rows = []

    for rank, local_idx in enumerate(top_local_indices, start=1):
        global_idx = normal_indices_global[local_idx]

        rows.append({
            "rank": rank,
            "local_normal_index": int(local_idx),
            "global_val_index": int(global_idx),
            "reconstruction_error": float(normal_errors[local_idx]),
        })

    extreme_df = pd.DataFrame(rows)
    extreme_df.to_csv(OUTPUT_DIR / "top_extreme_normal_sequences.csv", index=False)

    print("\nEn yüksek reconstruction error'a sahip normal sequence'ler")
    print(extreme_df)

    return extreme_df


def plot_error_histogram(y_val, val_errors):
    normal_errors = val_errors[y_val == 0]
    attack_errors = val_errors[y_val == 1]

    plt.figure(figsize=(8, 5))
    plt.hist(normal_errors, bins=50, alpha=0.6, label="Normal")
    plt.hist(attack_errors, bins=50, alpha=0.6, label="Attack")
    plt.xlabel("Reconstruction Error")
    plt.ylabel("Frekans")
    plt.title("Validation Reconstruction Error Dağılımı")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "validation_error_histogram.png")
    plt.close()


def plot_error_boxplot(y_val, val_errors):
    normal_errors = val_errors[y_val == 0]
    attack_errors = val_errors[y_val == 1]

    plt.figure(figsize=(6, 5))
    plt.boxplot(
        [normal_errors, attack_errors],
        tick_labels=["Normal", "Attack"],
        showfliers=True
    )
    plt.ylabel("Reconstruction Error")
    plt.title("Validation Reconstruction Error Boxplot")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "validation_error_boxplot.png")
    plt.close()


def main():
    X_val, y_val = load_data()
    val_errors = load_errors()

    print("X_val shape:", X_val.shape)
    print("y_val shape:", y_val.shape)
    print("val_errors shape:", val_errors.shape)

    save_error_summary(y_val, val_errors)

    extreme_df = find_extreme_normal_sequences(
        X_val=X_val,
        y_val=y_val,
        val_errors=val_errors,
        top_k=10
    )

    plot_error_histogram(y_val, val_errors)
    plot_error_boxplot(y_val, val_errors)

    print("\nAnaliz dosyaları kaydedildi:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()