import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from pathlib import Path
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_curve, average_precision_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.config import load_config
from src.models.deep_model import build_lstm_model, build_gru_model
from src.experiments.evaluator import evaluate_binary_classification


config = load_config()
DL_CONFIG = config["deep_learning"]

SEED = DL_CONFIG["seeds"][0]
THRESHOLDS = DL_CONFIG["thresholds"]
EPOCHS = DL_CONFIG["epochs"]
BATCH_SIZE = DL_CONFIG["batch_size"]
EARLY_STOPPING_PATIENCE = DL_CONFIG["early_stopping_patience"]
LEARNING_RATE = DL_CONFIG["learning_rate"]
VAL_RATIO = DL_CONFIG["val_ratio"]
N_FOLDS = DL_CONFIG["n_folds"]


def load_skab_fold_data(fold_id, processed_dir="data/processed"):
    X_train = np.load(f"{processed_dir}/skab_fold{fold_id}_X_train_seq.npy")
    y_train = np.load(f"{processed_dir}/skab_fold{fold_id}_y_train_seq.npy")

    X_test = np.load(f"{processed_dir}/skab_fold{fold_id}_X_test_seq.npy")
    y_test = np.load(f"{processed_dir}/skab_fold{fold_id}_y_test_seq.npy")

    X_train = X_train.astype("float32")
    y_train = y_train.astype("float32")

    X_test = X_test.astype("float32")
    y_test = y_test.astype("float32")

    return X_train, y_train, X_test, y_test


def split_train_validation(X_train, y_train, val_ratio=VAL_RATIO):
    val_start = int(len(X_train) * (1 - val_ratio))

    X_tr = X_train[:val_start]
    y_tr = y_train[:val_start]

    X_val = X_train[val_start:]
    y_val = y_train[val_start:]

    return X_tr, y_tr, X_val, y_val


def build_model(model_type, input_shape):
    if model_type == "LSTM":
        return build_lstm_model(
            input_shape=input_shape,
            learning_rate=LEARNING_RATE
        )

    if model_type == "GRU":
        return build_gru_model(
            input_shape=input_shape,
            learning_rate=LEARNING_RATE
        )

    raise ValueError(f"Desteklenmeyen model tipi: {model_type}")


def find_best_threshold(y_true, y_pred_prob):
    best_threshold = None
    best_metrics = None
    best_f1 = -1

    for threshold in THRESHOLDS:
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


def train_and_predict_one_fold(model_type, fold_id):
    print("\n==============================")
    print(f"SKAB {model_type} fold={fold_id} eğitimi başlıyor")
    print("==============================")

    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    X_train, y_train, X_test, y_test = load_skab_fold_data(fold_id)

    X_tr, y_tr, X_val, y_val = split_train_validation(
        X_train=X_train,
        y_train=y_train,
        val_ratio=VAL_RATIO
    )

    input_shape = X_tr.shape[1:]

    model = build_model(
        model_type=model_type,
        input_shape=input_shape
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True
    )

    class_weights_array = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=y_tr.astype(int)
    )

    class_weights = {
        0: class_weights_array[0],
        1: class_weights_array[1],
    }

    model.fit(
        X_tr,
        y_tr,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping],
        class_weight=class_weights,
        verbose=1
    )

    y_pred_prob = model.predict(X_test).ravel()

    best_threshold, best_metrics = find_best_threshold(
        y_true=y_test,
        y_pred_prob=y_pred_prob
    )

    y_pred = (y_pred_prob >= best_threshold).astype(int)

    print(f"{model_type} fold={fold_id} best threshold:", best_threshold)
    print(f"{model_type} fold={fold_id} metrics:", best_metrics)

    return y_test, y_pred_prob, y_pred


def collect_skab_predictions(model_type):
    all_y_true = []
    all_y_pred_prob = []
    all_y_pred = []

    for fold_id in range(1, N_FOLDS + 1):
        y_true, y_pred_prob, y_pred = train_and_predict_one_fold(
            model_type=model_type,
            fold_id=fold_id
        )

        all_y_true.append(y_true)
        all_y_pred_prob.append(y_pred_prob)
        all_y_pred.append(y_pred)

    all_y_true = np.concatenate(all_y_true)
    all_y_pred_prob = np.concatenate(all_y_pred_prob)
    all_y_pred = np.concatenate(all_y_pred)

    return all_y_true, all_y_pred_prob, all_y_pred


def plot_confusion_matrix_figure(y_true, y_pred, model_type, output_dir):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"SKAB {model_type} Confusion Matrix")
    plt.tight_layout()

    output_path = output_dir / f"skab_{model_type.lower()}_confusion_matrix.png"
    plt.savefig(output_path)
    plt.close()

    print(f"Kaydedildi: {output_path}")


def plot_precision_recall_curve(y_true, y_pred_prob, model_type, output_dir):
    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
    ap_score = average_precision_score(y_true, y_pred_prob)

    plt.figure(figsize=(7, 6))
    plt.plot(recall, precision, label=f"AP = {ap_score:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"SKAB {model_type} Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()

    output_path = output_dir / f"skab_{model_type.lower()}_pr_curve.png"
    plt.savefig(output_path)
    plt.close()

    print(f"Kaydedildi: {output_path}")


def main():
    output_dir = Path("results/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_type in ["LSTM", "GRU"]:
        y_true, y_pred_prob, y_pred = collect_skab_predictions(model_type)

        plot_confusion_matrix_figure(
            y_true=y_true,
            y_pred=y_pred,
            model_type=model_type,
            output_dir=output_dir
        )

        plot_precision_recall_curve(
            y_true=y_true,
            y_pred_prob=y_pred_prob,
            model_type=model_type,
            output_dir=output_dir
        )


if __name__ == "__main__":
    main()