import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, RepeatVector, TimeDistributed, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.config import get_dl_config


def load_batadal_sequence_data(processed_dir="data/processed", clip_value=5.0):
    X_train = np.load(f"{processed_dir}/batadal_X_train_seq.npy").astype("float32")
    y_train = np.load(f"{processed_dir}/batadal_y_train_seq.npy").astype("int32")

    X_val = np.load(f"{processed_dir}/batadal_X_val_seq.npy").astype("float32")
    y_val = np.load(f"{processed_dir}/batadal_y_val_seq.npy").astype("int32")

    X_test = np.load(f"{processed_dir}/batadal_X_test_seq.npy").astype("float32")
    y_test = np.load(f"{processed_dir}/batadal_y_test_seq.npy").astype("int32")

    y_train = np.where(y_train == -999, 0, y_train).ravel()
    y_val = np.where(y_val == -999, 0, y_val).ravel()
    y_test = np.where(y_test == -999, 0, y_test).ravel()

    if clip_value is not None:
        X_train = np.clip(X_train, -clip_value, clip_value)
        X_val = np.clip(X_val, -clip_value, clip_value)
        X_test = np.clip(X_test, -clip_value, clip_value)

    return X_train, y_train, X_val, y_val, X_test, y_test


def build_cnn_lstm_autoencoder(
    input_shape,
    conv_filters=32,
    kernel_size=3,
    lstm_units=64,
    latent_dim=32,
    dropout_rate=0.3,
    learning_rate=0.001,
):
    timesteps = input_shape[0]
    features = input_shape[1]

    model = Sequential([
        tf.keras.Input(shape=input_shape),

        Conv1D(
            filters=conv_filters,
            kernel_size=kernel_size,
            activation="relu",
            padding="same",
            name="conv1d_feature_extractor",
        ),

        LSTM(
            lstm_units,
            activation="tanh",
            return_sequences=False,
            name="lstm_encoder",
        ),
        Dropout(dropout_rate, name="encoder_dropout"),

        Dense(
            latent_dim,
            activation="relu",
            name="latent_bottleneck",
        ),

        RepeatVector(timesteps, name="repeat_latent_vector"),

        LSTM(
            lstm_units,
            activation="tanh",
            return_sequences=True,
            name="lstm_decoder",
        ),
        Dropout(dropout_rate, name="decoder_dropout"),

        TimeDistributed(
            Dense(features),
            name="reconstruction_output",
        ),
    ])

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="mse",
    )

    return model


def reconstruction_errors(model, X):
    X_pred = model.predict(X, verbose=0)
    return np.mean(np.square(X - X_pred), axis=(1, 2))


def evaluate_binary(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def find_best_threshold_from_validation(
    y_val,
    val_errors,
    percentile_start=90.0,
    percentile_end=100.0,
    percentile_step=0.1,
):
    percentiles = np.arange(percentile_start, percentile_end, percentile_step)
    thresholds = np.percentile(val_errors, percentiles)

    best_threshold = None
    best_percentile = None
    best_metrics = None
    best_f1 = -1.0

    for percentile, threshold in zip(percentiles, thresholds):
        y_pred = (val_errors >= threshold).astype(int)
        metrics = evaluate_binary(y_val, y_pred)

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_threshold = float(threshold)
            best_percentile = float(percentile)
            best_metrics = metrics

    return best_threshold, best_percentile, best_metrics


def train_one_seed(seed):
    cfg = get_dl_config()

    print("\n" + "=" * 60)
    print(f"BATADAL CNN-LSTM AutoEncoder | seed={seed}")
    print("=" * 60)

    np.random.seed(seed)
    tf.random.set_seed(seed)

    X_train, y_train, X_val, y_val, X_test, y_test = load_batadal_sequence_data(
        processed_dir="data/processed",
        clip_value=5.0,
    )

    X_train_normal = X_train[y_train == 0]
    X_val_normal = X_val[y_val == 0]

    if len(X_train_normal) == 0:
        raise ValueError("Eğitim kümesinde normal sequence bulunamadı.")

    if len(X_val_normal) == 0:
        raise ValueError("Validation kümesinde normal sequence bulunamadı.")

    print("X_train shape       :", X_train.shape)
    print("X_train_normal shape:", X_train_normal.shape)
    print("X_val shape         :", X_val.shape)
    print("X_test shape        :", X_test.shape)
    print("Train min/max       :", X_train.min(), X_train.max())
    print("Val min/max         :", X_val.min(), X_val.max())
    print("Test min/max        :", X_test.min(), X_test.max())

    model = build_cnn_lstm_autoencoder(
        input_shape=X_train.shape[1:],
        conv_filters=32,
        kernel_size=3,
        lstm_units=64,
        latent_dim=32,
        dropout_rate=0.3,
        learning_rate=cfg.get("learning_rate", 0.001),
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        min_delta=1e-4,
    )

    lr_reducer = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1,
    )

    history = model.fit(
        X_train_normal,
        X_train_normal,
        validation_data=(X_val_normal, X_val_normal),
        epochs=cfg.get("epochs", 100),
        batch_size=cfg.get("batch_size", 32),
        callbacks=[early_stopping, lr_reducer],
        verbose=1,
        shuffle=False,
    )

    val_errors = reconstruction_errors(model, X_val)

    best_threshold, best_percentile, val_metrics = find_best_threshold_from_validation(
        y_val=y_val,
        val_errors=val_errors,
        percentile_start=90.0,
        percentile_end=100.0,
        percentile_step=0.1,
    )

    test_errors = reconstruction_errors(model, X_test)
    y_test_pred = (test_errors >= best_threshold).astype(int)

    test_metrics = evaluate_binary(y_test, y_test_pred)

    output_dir = Path("results/outputs/cnn_lstm_autoencoder")
    output_dir.mkdir(parents=True, exist_ok=True)

    model.save(output_dir / f"cnn_lstm_autoencoder_seed_{seed}.keras")
    np.save(output_dir / f"val_errors_seed_{seed}.npy", val_errors)
    np.save(output_dir / f"test_errors_seed_{seed}.npy", test_errors)

    pd.DataFrame(history.history).to_csv(
        output_dir / f"training_history_seed_{seed}.csv",
        index=False,
    )

    result = {
        "dataset": "BATADAL",
        "model": "CNN_LSTM_AutoEncoder",
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
        "epochs_trained": len(history.history["loss"]),
    }

    print("\nValidation sonucu:", val_metrics)
    print("Test sonucu:", result)

    tf.keras.backend.clear_session()
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
            threshold_mean=("threshold", "mean"),
            threshold_std=("threshold", "std"),
        )
        .reset_index()
    )


def main():
    cfg = get_dl_config()
    seeds = cfg.get("seeds", [42, 123, 2026, 7, 999])

    all_results = []

    for seed in seeds:
        result = train_one_seed(seed)
        all_results.append(result)

    output_dir = Path("results/outputs/cnn_lstm_autoencoder")
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(all_results)
    summary_df = summarize_results(results_df)

    results_path = output_dir / "batadal_cnn_lstm_autoencoder_seed_results.csv"
    summary_path = output_dir / "batadal_cnn_lstm_autoencoder_seed_summary.csv"

    results_df.to_csv(results_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print("\nCNN-LSTM AutoEncoder seed bazlı sonuçlar:")
    print(results_df)

    print("\nCNN-LSTM AutoEncoder mean/std özeti:")
    print(summary_df)

    print(f"\nSonuç dosyası: {results_path}")
    print(f"Özet dosyası : {summary_path}")


if __name__ == "__main__":
    main()