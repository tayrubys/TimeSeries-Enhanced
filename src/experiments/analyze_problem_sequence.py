import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path

from src.experiments.run_batadal_lstm_autoencoder import build_lstm_autoencoder

PROCESSED_DIR = Path("data/processed")
OUTPUT_DIR = Path("results/outputs/autoencoder_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEQUENCE_INDEX = 671


def main():
    X_val = np.load(PROCESSED_DIR / "batadal_X_val_seq.npy").astype("float32")
    y_val = np.load(PROCESSED_DIR / "batadal_y_val_seq.npy").astype("int32")
    y_val = np.where(y_val == -999, 0, y_val)

    # Burada model ağırlığını kaydetmediğimiz için modeli tekrar eğitmeden tahmin alamayız.
    # Bu yüzden önce AutoEncoder kodunda modeli kaydetmen gerekiyor.
    model_path = OUTPUT_DIR / "lstm_autoencoder_seed_42.keras"

    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_path} bulunamadı. Önce AutoEncoder eğitim kodunda modeli kaydetmelisin."
        )

    model = tf.keras.models.load_model(model_path)

    sequence = X_val[SEQUENCE_INDEX]
    pred_sequence = model.predict(sequence[np.newaxis, :, :], verbose=0)[0]

    diff = np.square(sequence - pred_sequence)

    # Her sensör için ortalama reconstruction error
    sensor_errors = diff.mean(axis=0)

    sensor_df = pd.DataFrame({
        "sensor_index": range(len(sensor_errors)),
        "sensor_error": sensor_errors
    }).sort_values("sensor_error", ascending=False)

    sensor_df.to_csv(OUTPUT_DIR / f"sequence_{SEQUENCE_INDEX}_sensor_errors.csv", index=False)

    print("Sequence label:", y_val[SEQUENCE_INDEX])
    print(sensor_df.head(10))

    # En problemli sensörü çiz
    top_sensor = int(sensor_df.iloc[0]["sensor_index"])

    plt.figure(figsize=(10, 4))
    plt.plot(sequence[:, top_sensor], label="Gerçek")
    plt.plot(pred_sequence[:, top_sensor], label="Tahmin")
    plt.title(f"Sequence {SEQUENCE_INDEX} - Sensor {top_sensor}")
    plt.xlabel("Zaman adımı")
    plt.ylabel("Scaled değer")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"sequence_{SEQUENCE_INDEX}_top_sensor_plot.png")
    plt.show()


if __name__ == "__main__":
    main()