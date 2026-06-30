import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight

from src.models.deep_model import build_lstm_model, build_gru_model
from src.experiments.evaluator import evaluate_binary_classification
from src.config import get_dl_config


def load_skab_fold_data(fold_id, processed_dir="data/processed"):
    X_train = np.load(f"{processed_dir}/skab_fold{fold_id}_X_train_seq.npy").astype("float32")
    y_train = np.load(f"{processed_dir}/skab_fold{fold_id}_y_train_seq.npy").astype("float32")
    X_test  = np.load(f"{processed_dir}/skab_fold{fold_id}_X_test_seq.npy").astype("float32")
    y_test  = np.load(f"{processed_dir}/skab_fold{fold_id}_y_test_seq.npy").astype("float32")
    return X_train, y_train, X_test, y_test


def split_train_validation(X_train, y_train, val_ratio):
    val_start = int(len(X_train) * (1 - val_ratio))
    return X_train[:val_start], y_train[:val_start], X_train[val_start:], y_train[val_start:]


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


def train_one_skab_experiment(model_type, fold_id, seed):
    cfg = get_dl_config()

    print("\n==============================")
    print(f"SKAB {model_type} fold={fold_id}, seed={seed} eğitimi başlıyor")
    print("==============================")

    np.random.seed(seed)
    tf.random.set_seed(seed)

    X_train, y_train, X_test, y_test = load_skab_fold_data(fold_id)
    X_tr, y_tr, X_val, y_val = split_train_validation(X_train, y_train, cfg["val_ratio"])

    model = build_model(model_type=model_type, input_shape=X_tr.shape[1:])

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=cfg["early_stopping_patience"],
        restore_best_weights=True
    )

    class_weights_array = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=y_tr.astype(int)
    )
    class_weights = {0: class_weights_array[0], 1: class_weights_array[1]}

    model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        callbacks=[early_stopping],
        class_weight=class_weights,
        verbose=1
    )

    y_val_pred_prob = model.predict(X_val)
    best_threshold, val_metrics = find_best_threshold(
        y_val,
        y_val_pred_prob,
        cfg["thresholds"]
    )

    y_test_pred_prob = model.predict(X_test)
    y_test_pred = (y_test_pred_prob >= best_threshold).astype(int).ravel()

    best_metrics = evaluate_binary_classification(
        y_true=y_test,
        y_pred=y_test_pred
    )
    result = {
        "dataset": "SKAB",
        "model": model_type,
        "seed": seed,
        "fold": fold_id,
        "threshold": best_threshold,
        "accuracy": best_metrics["accuracy"],
        "precision": best_metrics["precision"],
        "recall": best_metrics["recall"],
        "f1": best_metrics["f1"],
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
    cfg = get_dl_config()
    all_results = []

    for model_type in ["LSTM", "GRU"]:
        for seed in cfg["seeds"]:
            for fold_id in range(1, 6):
                result = train_one_skab_experiment(
                    model_type=model_type,
                    fold_id=fold_id,
                    seed=seed
                )
                all_results.append(result)

    output_dir = Path("results/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_dir / "skab_deep_learning_seed_results.csv", index=False)

    summary_df = summarize_results(results_df)
    summary_df.to_csv(output_dir / "skab_deep_learning_seed_summary.csv", index=False)

    print("\nSKAB LSTM + GRU seed + fold bazlı sonuçlar:")
    print(results_df)
    print("\nSKAB mean/std özet:")
    print(summary_df)


if __name__ == "__main__":
    main()