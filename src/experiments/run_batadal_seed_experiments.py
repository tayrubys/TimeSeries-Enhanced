import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight

from src.models.deep_model import build_lstm_model, build_gru_model
from src.experiments.evaluator import evaluate_binary_classification
from src.config import get_dl_config


def load_batadal_sequence_data(processed_dir="data/processed", balancing_method="class_weight"):
    """
    balancing_method:
      "class_weight" -> ham (dengesiz) train + class_weight kullanılacak
      "smote"        -> SMOTE ile dengelenmiş train (class_weight KULLANILMAZ)
      "adasyn"       -> ADASYN ile dengelenmiş train (class_weight KULLANILMAZ)
    """
    if balancing_method == "class_weight":
        X_train = np.load(f"{processed_dir}/batadal_X_train_seq.npy").astype("float32")
        y_train = np.load(f"{processed_dir}/batadal_y_train_seq.npy").astype("float32")
    elif balancing_method == "smote":
        X_train = np.load(f"{processed_dir}/batadal_X_train_seq_balanced.npy").astype("float32")
        y_train = np.load(f"{processed_dir}/batadal_y_train_seq_balanced.npy").astype("float32")
    elif balancing_method == "adasyn":
        X_train = np.load(f"{processed_dir}/batadal_X_train_seq_adasyn.npy").astype("float32")
        y_train = np.load(f"{processed_dir}/batadal_y_train_seq_adasyn.npy").astype("float32")
    else:
        raise ValueError(f"Bilinmeyen balancing_method: {balancing_method}")

    # val/test HER ZAMAN dengesiz, orijinal hâliyle yüklenir
    X_val   = np.load(f"{processed_dir}/batadal_X_val_seq.npy").astype("float32")
    y_val   = np.load(f"{processed_dir}/batadal_y_val_seq.npy").astype("float32")
    X_test  = np.load(f"{processed_dir}/batadal_X_test_seq.npy").astype("float32")
    y_test  = np.load(f"{processed_dir}/batadal_y_test_seq.npy").astype("float32")
    return X_train, y_train, X_val, y_val, X_test, y_test


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


def train_one_batadal_experiment(model_type, seed, balancing_method="class_weight"):
    cfg = get_dl_config()

    print("\n==============================")
    print(f"BATADAL {model_type} seed={seed} balancing={balancing_method} eğitimi başlıyor")
    print("==============================")

    np.random.seed(seed)
    tf.random.set_seed(seed)

    X_train, y_train, X_val, y_val, X_test, y_test = load_batadal_sequence_data(
        balancing_method=balancing_method
    )

    model = build_model(model_type=model_type, input_shape=X_train.shape[1:])

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=cfg["early_stopping_patience"],
        restore_best_weights=True
    )

    if balancing_method == "class_weight":
        class_weights_array = compute_class_weight(
            class_weight="balanced",
            classes=np.array([0, 1]),
            y=y_train.astype(int)
        )
        class_weights = {0: class_weights_array[0], 1: class_weights_array[1]}
    else:
        class_weights = None

    model.fit(
        X_train, y_train,
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
        "dataset": "BATADAL",
        "model": model_type,
        "balancing_method": balancing_method,   
        "seed": seed,
        "fold": "-",
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
        .groupby(["dataset", "model", "balancing_method"])  
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

    balancing_methods = ["adasyn"]

    for model_type in ["LSTM", "GRU"]:
        for balancing_method in balancing_methods:
            for seed in cfg["seeds"]:
                result = train_one_batadal_experiment(
                    model_type=model_type,
                    seed=seed,
                    balancing_method=balancing_method
                )
                all_results.append(result)

    output_dir = Path("results/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_dir / "batadal_deep_learning_seed_results.csv", index=False)

    summary_df = summarize_results(results_df)
    summary_df.to_csv(output_dir / "batadal_deep_learning_seed_summary.csv", index=False)

    print("\nBATADAL seed bazlı sonuçlar:")
    print(results_df)
    print("\nBATADAL mean/std özet:")
    print(summary_df)


if __name__ == "__main__":
    main()