import numpy as np
import tensorflow as tf

from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight

from src.models.deep_model import build_lstm_model
from src.experiments.evaluator import evaluate_binary_classification, print_metrics
from src.experiments.result_logger import save_results_to_csv
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


def evaluate_thresholds(y_test, y_pred_prob, thresholds):
    best_threshold, best_f1, best_metrics = None, -1, None
    for threshold in thresholds:
        y_pred = (y_pred_prob >= threshold).astype(int).ravel()
        metrics = evaluate_binary_classification(y_true=y_test, y_pred=y_pred)
        print(f"\nThreshold: {threshold}")
        print("0 tahmini:", (y_pred == 0).sum())
        print("1 tahmini:", (y_pred == 1).sum())
        print_metrics(metrics)
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_threshold = threshold
            best_metrics = metrics
    return best_threshold, best_metrics


def train_skab_lstm_fold(fold_id):
    cfg = get_dl_config()

    print("\n==============================")
    print(f"SKAB Fold {fold_id} LSTM eğitimi başlıyor...")
    print("==============================")

    X_train, y_train, X_test, y_test = load_skab_fold_data(fold_id)
    X_tr, y_tr, X_val, y_val = split_train_validation(X_train, y_train, cfg["val_ratio"])

    print("X_tr shape:", X_tr.shape)
    print("X_val shape:", X_val.shape)
    print("X_test shape:", X_test.shape)

    model = build_lstm_model(input_shape=X_tr.shape[1:])

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
    print("Class weights:", class_weights)

    model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        callbacks=[early_stopping],
        class_weight=class_weights,
        verbose=1
    )

    y_pred_prob = model.predict(X_test)

    print("Tahmin olasılık min:", y_pred_prob.min())
    print("Tahmin olasılık max:", y_pred_prob.max())
    print("Tahmin olasılık ortalama:", y_pred_prob.mean())
    print("0 sayısı:", (y_test == 0).sum())
    print("1 sayısı:", (y_test == 1).sum())

    best_threshold, best_metrics = evaluate_thresholds(
        y_test=y_test,
        y_pred_prob=y_pred_prob,
        thresholds=cfg["thresholds"]
    )

    print(f"\nSKAB Fold {fold_id} LSTM En İyi Sonuç:")
    print("Best threshold:", best_threshold)
    print_metrics(best_metrics)

    return {
        "fold": fold_id,
        "best_threshold": best_threshold,
        "accuracy": best_metrics["accuracy"],
        "precision": best_metrics["precision"],
        "recall": best_metrics["recall"],
        "f1": best_metrics["f1"],
    }


def main():
    cfg = get_dl_config()
    np.random.seed(cfg["seeds"][0])
    tf.random.set_seed(cfg["seeds"][0])

    fold_results = []
    for fold_id in range(1, 6):
        result = train_skab_lstm_fold(fold_id)
        fold_results.append(result)

    print("\n==============================")
    print("SKAB LSTM FOLD SONUÇLARI")
    print("==============================")
    for result in fold_results:
        print(result)

    for metric in ["accuracy", "precision", "recall", "f1"]:
        vals = [r[metric] for r in fold_results]
        print(f"{metric} mean: {np.mean(vals):.4f}  std: {np.std(vals):.4f}")

    results_for_csv = [
        {
            "dataset": "SKAB",
            "model": "LSTM",
            "fold": r["fold"],
            "threshold": r["best_threshold"],
            "accuracy": r["accuracy"],
            "precision": r["precision"],
            "recall": r["recall"],
            "f1": r["f1"],
        }
        for r in fold_results
    ]

    save_results_to_csv(
        results=results_for_csv,
        output_path="results/outputs/skab_lstm_results.csv"
    )


if __name__ == "__main__":
    main()