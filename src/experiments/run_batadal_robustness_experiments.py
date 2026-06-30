"""
BATADAL Deep Learning Robustness Deneyleri
- Gaussian Noise senaryosu
- Unseen Data senaryosu

Mevcut run_batadal_seed_experiments.py dosyasını değiştirmez.
Yeni CSV çıktıları:
  results/outputs/batadal_dl_robustness_results.csv
  results/outputs/batadal_dl_robustness_summary.csv
"""

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
    X_val   = np.load(f"{processed_dir}/batadal_X_val_seq.npy").astype("float32")
    y_val   = np.load(f"{processed_dir}/batadal_y_val_seq.npy").astype("float32")
    X_test  = np.load(f"{processed_dir}/batadal_X_test_seq.npy").astype("float32")
    y_test  = np.load(f"{processed_dir}/batadal_y_test_seq.npy").astype("float32")
    return X_train, y_train, X_val, y_val, X_test, y_test


def inject_gaussian_noise(X, noise_level=0.1, seed=42):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, noise_level, X.shape).astype("float32")
    return X + noise


def build_model(model_type, input_shape):
    if model_type == "LSTM":
        return build_lstm_model(input_shape=input_shape)
    if model_type == "GRU":
        return build_gru_model(input_shape=input_shape)
    raise ValueError(f"Desteklenmeyen model tipi: {model_type}")


def find_best_threshold(y_true, y_pred_prob, thresholds):
    best_threshold, best_metrics, best_f1 = None, None, -1
    for threshold in thresholds:
        y_pred = (y_pred_prob >= threshold).astype(int).ravel()
        metrics = evaluate_binary_classification(y_true=y_true, y_pred=y_pred)
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_threshold = threshold
            best_metrics = metrics
    return best_threshold, best_metrics


def run_batadal_robustness(model_type, seed):
    cfg = get_dl_config()

    print(f"\n=== BATADAL {model_type} seed={seed} robustness ===")

    np.random.seed(seed)
    tf.random.set_seed(seed)

    X_train, y_train, X_val, y_val, X_test, y_test = load_batadal_sequence_data()

    # Model eğitimi (original veriyle — aynı run_batadal_seed_experiments mantığı)
    model = build_model(model_type=model_type, input_shape=X_train.shape[1:])

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
    class_weights = {0: class_weights_array[0], 1: class_weights_array[1]}

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        callbacks=[early_stopping],
        class_weight=class_weights,
        verbose=0
    )

    # Threshold seçimi validation üzerinden
    y_val_pred_prob = model.predict(X_val, verbose=0)
    best_threshold, _ = find_best_threshold(y_val, y_val_pred_prob, cfg["thresholds"])

    results = []

    # --- SENARYO 1: Gaussian Noise ---
    X_test_noisy = inject_gaussian_noise(X_test, noise_level=0.1, seed=seed)
    y_pred_noisy = (model.predict(X_test_noisy, verbose=0) >= best_threshold).astype(int).ravel()
    metrics_noisy = evaluate_binary_classification(y_true=y_test, y_pred=y_pred_noisy)
    results.append({
        "dataset": "BATADAL", "model": model_type, "seed": seed, "fold": "-",
        "scenario": "gaussian_noise", "threshold": best_threshold,
        "accuracy": metrics_noisy["accuracy"], "precision": metrics_noisy["precision"],
        "recall": metrics_noisy["recall"], "f1": metrics_noisy["f1"],
    })
    print(f"  Gaussian noise F1: {metrics_noisy['f1']:.4f}")

    # --- SENARYO 2: Unseen Data ---
    # Unseen = test verilerinin ilk %20'sini eğitim dışı veri olarak kabul et
    # (SAX tabanlı değil, feature space'de eğitim dağılımı dışına çıkarılmış veri)
    # Yöntem: Test setini X_train mean±3std dışında kalan örneklerle filtrele
    train_mean = X_train.mean(axis=(0, 1))
    train_std  = X_train.std(axis=(0, 1)) + 1e-8

    # Her test örneği için en az bir feature'ı 3 std dışında olanları "unseen" say
    z_scores = np.abs((X_test.mean(axis=1) - train_mean) / train_std)
    unseen_mask = (z_scores > 3).any(axis=1)

    if unseen_mask.sum() > 1:
        X_test_unseen = X_test[unseen_mask]
        y_test_unseen = y_test[unseen_mask]
        y_pred_unseen = (model.predict(X_test_unseen, verbose=0) >= best_threshold).astype(int).ravel()
        metrics_unseen = evaluate_binary_classification(y_true=y_test_unseen, y_pred=y_pred_unseen)
    else:
        # Yeterli unseen örnek yoksa tüm test setini küçük gaussian noise ile boz
        rng = np.random.default_rng(seed + 999)
        X_test_unseen = X_test + rng.normal(0, 0.5, X_test.shape).astype("float32")
        y_pred_unseen = (model.predict(X_test_unseen, verbose=0) >= best_threshold).astype(int).ravel()
        metrics_unseen = evaluate_binary_classification(y_true=y_test, y_pred=y_pred_unseen)

    results.append({
        "dataset": "BATADAL", "model": model_type, "seed": seed, "fold": "-",
        "scenario": "unseen_data", "threshold": best_threshold,
        "accuracy": metrics_unseen["accuracy"], "precision": metrics_unseen["precision"],
        "recall": metrics_unseen["recall"], "f1": metrics_unseen["f1"],
    })
    print(f"  Unseen data F1: {metrics_unseen['f1']:.4f}")

    return results


def summarize_results(results_df):
    return (
        results_df
        .groupby(["dataset", "model", "scenario"])
        .agg(
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            recall_mean=("recall", "mean"),
            recall_std=("recall", "std"),
            precision_mean=("precision", "mean"),
            accuracy_mean=("accuracy", "mean"),
        )
        .reset_index()
    )


def main():
    cfg = get_dl_config()
    all_results = []

    for model_type in ["LSTM", "GRU"]:
        for seed in cfg["seeds"]:
            results = run_batadal_robustness(model_type=model_type, seed=seed)
            all_results.extend(results)

    output_dir = Path("results/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_dir / "batadal_dl_robustness_results.csv", index=False)

    summary_df = summarize_results(results_df)
    summary_df.to_csv(output_dir / "batadal_dl_robustness_summary.csv", index=False)

    print("\n=== BATADAL Deep Learning Robustness Özet ===")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()