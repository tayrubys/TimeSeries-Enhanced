import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight

from src.models.deep_model import build_lstm_model, build_gru_model
from src.experiments.evaluator import evaluate_binary_classification
from src.config import get_dl_config


def load_batadal_sequence_data(processed_dir="data/processed"):
    X_train = np.load(f"{processed_dir}/batadal_X_train_seq.npy").astype("float32")
    y_train = np.load(f"{processed_dir}/batadal_y_train_seq.npy").astype("float32")

    X_val = np.load(f"{processed_dir}/batadal_X_val_seq.npy").astype("float32")
    y_val = np.load(f"{processed_dir}/batadal_y_val_seq.npy").astype("float32")

    X_test = np.load(f"{processed_dir}/batadal_X_test_seq.npy").astype("float32")
    y_test = np.load(f"{processed_dir}/batadal_y_test_seq.npy").astype("float32")

    return X_train, y_train, X_val, y_val, X_test, y_test


def build_model(model_type, input_shape):
    if model_type == "LSTM":
        return build_lstm_model(input_shape=input_shape)

    if model_type == "GRU":
        return build_gru_model(input_shape=input_shape)

    raise ValueError(f"Desteklenmeyen model tipi: {model_type}")


def apply_moving_average(y_pred_prob, window_size=3):
    """
    Modelin ürettiği probability çıktısına Moving Average uygular.
    Threshold'tan önce kullanılır.
    """
    y_pred_prob = np.asarray(y_pred_prob).ravel()

    y_pred_prob_ma = (
        pd.Series(y_pred_prob)
        .rolling(window=window_size, min_periods=1)
        .mean()
        .values
    )

    return y_pred_prob_ma


def find_best_threshold(y_true, y_pred_prob, thresholds):
    best_threshold = None
    best_metrics = None
    best_f1 = -1

    for threshold in thresholds:
        y_pred = (y_pred_prob >= threshold).astype(int).ravel()

        metrics = evaluate_binary_classification(
            y_true=y_true,
            y_pred=y_pred
        )

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_threshold = threshold
            best_metrics = metrics

    return best_threshold, best_metrics


def train_one_batadal_experiment_ma(model_type, seed, ma_window):
    cfg = get_dl_config()

    print("\n==============================")
    print(f"BATADAL {model_type} seed={seed} MA window={ma_window} eğitimi başlıyor")
    print("==============================")

    np.random.seed(seed)
    tf.random.set_seed(seed)

    X_train, y_train, X_val, y_val, X_test, y_test = load_batadal_sequence_data()

    model = build_model(
        model_type=model_type,
        input_shape=X_train.shape[1:]
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=cfg["early_stopping_patience"],
        restore_best_weights=True
    )

    class_weights_array = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=y_train.astype(int)
    )

    class_weights = {
        0: class_weights_array[0],
        1: class_weights_array[1]
    }

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        callbacks=[early_stopping],
        class_weight=class_weights,
        verbose=1
    )

    # Validation probability
    y_val_pred_prob = model.predict(X_val).ravel()

    # Moving Average validation probability
    y_val_pred_prob_ma = apply_moving_average(
        y_pred_prob=y_val_pred_prob,
        window_size=ma_window
    )

    # Threshold validation üzerinde seçilir
    best_threshold, val_metrics = find_best_threshold(
        y_true=y_val,
        y_pred_prob=y_val_pred_prob_ma,
        thresholds=cfg["thresholds"]
    )

    # Test probability
    y_test_pred_prob = model.predict(X_test).ravel()

    # Moving Average test probability
    y_test_pred_prob_ma = apply_moving_average(
        y_pred_prob=y_test_pred_prob,
        window_size=ma_window
    )

    # Validation'da seçilen threshold test'e uygulanır
    y_test_pred = (y_test_pred_prob_ma >= best_threshold).astype(int).ravel()

    test_metrics = evaluate_binary_classification(
        y_true=y_test,
        y_pred=y_test_pred
    )

    result = {
        "dataset": "BATADAL",
        "model": model_type,
        "seed": seed,
        "fold": "-",
        "ma_window": ma_window,
        "threshold": best_threshold,
        "accuracy": test_metrics["accuracy"],
        "precision": test_metrics["precision"],
        "recall": test_metrics["recall"],
        "f1": test_metrics["f1"],
    }

    print("Validation sonucu:", val_metrics)
    print("Test sonucu:", result)

    return result


def summarize_results(results_df):
    return (
        results_df
        .groupby(["dataset", "model", "ma_window"])
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
    cfg = get_dl_config()

    all_results = []

    # İstersen burada sadece [3] bırakabilirsin.
    ma_windows = [3, 5, 7]

    for ma_window in ma_windows:
        for model_type in ["LSTM", "GRU"]:
            for seed in cfg["seeds"]:
                result = train_one_batadal_experiment_ma(
                    model_type=model_type,
                    seed=seed,
                    ma_window=ma_window
                )
                all_results.append(result)

    output_dir = Path("results/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(all_results)

    results_path = output_dir / "batadal_deep_learning_seed_results_ma.csv"
    summary_path = output_dir / "batadal_deep_learning_seed_summary_ma.csv"

    results_df.to_csv(results_path, index=False)

    summary_df = summarize_results(results_df)
    summary_df.to_csv(summary_path, index=False)

    print("\nBATADAL Moving Average seed bazlı sonuçlar:")
    print(results_df)

    print("\nBATADAL Moving Average mean/std özet:")
    print(summary_df)

    print(f"\nSonuç dosyası kaydedildi: {results_path}")
    print(f"Özet dosyası kaydedildi: {summary_path}")


if __name__ == "__main__":
    main()