import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from tensorflow.keras.callbacks import ReduceLROnPlateau

# Eğer validation loss 3 epoch boyunca iyileşmezse learning rate'i yarıya düşürür
lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5)



def load_batadal_sequence_data(processed_dir="data/processed"):
    X_train = np.load(f"{processed_dir}/batadal_X_train_seq.npy").astype("float32")
    y_train = np.load(f"{processed_dir}/batadal_y_train_seq.npy").astype("int32")

    X_val = np.load(f"{processed_dir}/batadal_X_val_seq.npy").astype("float32")
    y_val = np.load(f"{processed_dir}/batadal_y_val_seq.npy").astype("int32")

    X_test = np.load(f"{processed_dir}/batadal_X_test_seq.npy").astype("float32")
    y_test = np.load(f"{processed_dir}/batadal_y_test_seq.npy").astype("int32")

    y_train = np.where(y_train == -999, 0, y_train)
    y_val = np.where(y_val == -999, 0, y_val)
    y_test = np.where(y_test == -999, 0, y_test)

    return X_train, y_train, X_val, y_val, X_test, y_test


def build_lstm_autoencoder(input_shape, units=128, dropout_rate=0.2):
    timesteps = input_shape[0]
    features = input_shape[1]

    model = Sequential([
        # Encoder Katmanı
        LSTM(units, activation="tanh", input_shape=input_shape, return_sequences=False),
        Dropout(dropout_rate),
        
        # Sıkıştırma (Darboğaz) Katmanı - Özellikleri 64'e indiriyoruz
        Dense(64, activation="relu"), 
        
        # Zaman boyutunu tekrar kopyalama
        RepeatVector(timesteps),
        
        # Decoder Katmanı
        LSTM(units, activation="tanh", return_sequences=True),
        Dropout(dropout_rate),
        
        # Çıktıyı orijinal sensör sayısına (features) eşitleme
        TimeDistributed(Dense(features))
    ])

    model.compile(
        optimizer="adam",
        loss="mse"
    )

    return model


def reconstruction_errors(model, X):
    X_pred = model.predict(X, verbose=0)
    errors = np.mean(np.square(X - X_pred), axis=(1, 2))
    return errors


def evaluate_binary(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def find_best_threshold_from_validation(y_val, val_errors):
    percentiles = np.arange(90, 100, 0.1)
    thresholds = np.percentile(val_errors, percentiles)

    best_threshold = None
    best_percentile = None
    best_metrics = None
    best_f1 = -1

    for percentile, threshold in zip(percentiles, thresholds):
        y_pred = (val_errors >= threshold).astype(int)

        metrics = evaluate_binary(y_val, y_pred)

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_threshold = threshold
            best_percentile = percentile
            best_metrics = metrics

    return best_threshold, best_percentile, best_metrics


def train_one_seed(seed):
    print("\n==============================")
    print(f"BATADAL LSTM AutoEncoder seed={seed}")
    print("==============================")

    np.random.seed(seed)
    tf.random.set_seed(seed)

    X_train, y_train, X_val, y_val, X_test, y_test = load_batadal_sequence_data()

    # AutoEncoder sadece normal sequence'lerle eğitilir
    X_train_normal = X_train[y_train == 0]

    print(f"Train shape: {X_train.shape}")
    print(f"Normal train shape: {X_train_normal.shape}")
    print(f"Val shape: {X_val.shape}")
    print(f"Test shape: {X_test.shape}")

    model = build_lstm_autoencoder(
        input_shape=X_train.shape[1:],
        units=128,
        dropout_rate=0.2
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )

    model.fit(
        X_train_normal,
        X_train_normal,
        validation_data=(X_val[y_val == 0], X_val[y_val == 0]),
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping, lr_reducer],
        verbose=1
    )

    val_errors = reconstruction_errors(model, X_val)

    best_threshold, best_percentile, val_metrics = find_best_threshold_from_validation(
        y_val=y_val,
        val_errors=val_errors
    )

    test_errors = reconstruction_errors(model, X_test)
    y_test_pred = (test_errors >= best_threshold).astype(int)

    test_metrics = evaluate_binary(y_test, y_test_pred)

    result = {
        "dataset": "BATADAL",
        "model": "LSTM_AutoEncoder",
        "seed": seed,
        "threshold_percentile": best_percentile,
        "threshold": best_threshold,
        "accuracy": test_metrics["accuracy"],
        "precision": test_metrics["precision"],
        "recall": test_metrics["recall"],
        "f1": test_metrics["f1"],
        "val_accuracy": val_metrics["accuracy"],
        "val_precision": val_metrics["precision"],
        "val_recall": val_metrics["recall"],
        "val_f1": val_metrics["f1"],
    }

    print("Validation sonucu:", val_metrics)
    print("Test sonucu:", result)

    return result


def summarize_results(results_df):
    return (
        results_df
        .groupby(["dataset", "model"])
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            precision_mean=("precision", "mean"),
            precision_std=("precision", "std"),
            recall_mean=("recall", "mean"),
            recall_std=("recall", "std"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
        )
        .reset_index()
    )


def main():
    seeds = [42, 123, 2026, 7, 999]

    all_results = []

    for seed in seeds:
        result = train_one_seed(seed)
        all_results.append(result)

    output_dir = Path("results/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(all_results)
    summary_df = summarize_results(results_df)

    results_path = output_dir / "batadal_lstm_autoencoder_seed_results.csv"
    summary_path = output_dir / "batadal_lstm_autoencoder_seed_summary.csv"

    results_df.to_csv(results_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print("\nBATADAL LSTM AutoEncoder seed bazlı sonuçlar:")
    print(results_df)

    print("\nBATADAL LSTM AutoEncoder mean/std özet:")
    print(summary_df)

    print(f"\nSonuç dosyası kaydedildi: {results_path}")
    print(f"Özet dosyası kaydedildi: {summary_path}")


if __name__ == "__main__":
    main()