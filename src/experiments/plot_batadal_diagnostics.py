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

SEED = DL_CONFIG["seeds"][3]
THRESHOLDS = DL_CONFIG["thresholds"]
EPOCHS = DL_CONFIG["epochs"]
BATCH_SIZE = DL_CONFIG["batch_size"]
EARLY_STOPPING_PATIENCE = DL_CONFIG["early_stopping_patience"]
LEARNING_RATE = DL_CONFIG["learning_rate"]

# BATADAL sequence dosyalarını yükler
def load_batadal_sequence_data(processed_dir="data/processed"):

    X_train = np.load(f"{processed_dir}/batadal_X_train_seq_adasyn.npy")
    y_train = np.load(f"{processed_dir}/batadal_y_train_seq_adasyn.npy")

    X_val = np.load(f"{processed_dir}/batadal_X_val_seq.npy")
    y_val = np.load(f"{processed_dir}/batadal_y_val_seq.npy")

    X_test = np.load(f"{processed_dir}/batadal_X_test_seq.npy")
    y_test = np.load(f"{processed_dir}/batadal_y_test_seq.npy")

    X_train = X_train.astype("float32")
    y_train = y_train.astype("float32")

    X_val = X_val.astype("float32")
    y_val = y_val.astype("float32")

    X_test = X_test.astype("float32")
    y_test = y_test.astype("float32")

    return X_train, y_train, X_val, y_val, X_test, y_test


# Model tipine göre model oluşturur
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


# En iyi threshold'u bulur
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


# Eğitim ve test tahmini
def train_and_predict(model_type):

    print(f"\n{model_type} eğitimi başlıyor...")

    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    X_train, y_train, X_val, y_val, X_test, y_test = load_batadal_sequence_data()

    input_shape = X_train.shape[1:]

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
        y=y_train.astype(int)
    )

    class_weights = {
        0: class_weights_array[0],
        1: class_weights_array[1],
    }

    print("Class weights:", class_weights)

    model.fit(
        X_train,
        y_train,
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

    print(f"{model_type} best threshold:", best_threshold)
    print(f"{model_type} best metrics:", best_metrics)

    return y_test, y_pred_prob, y_pred, best_threshold, best_metrics


# Confusion matrix çizer
def plot_confusion_matrix_figure(y_true, y_pred, model_type, output_dir):

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"BATADAL {model_type} Confusion Matrix")
    plt.tight_layout()

    output_path = output_dir / f"batadal_{model_type.lower()}_confusion_matrix.png"
    plt.savefig(output_path)
    plt.close()

    print(f"Kaydedildi: {output_path}")


# Precision-Recall eğrisi çizer
def plot_precision_recall_curve(y_true, y_pred_prob, model_type, output_dir):

    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
    ap_score = average_precision_score(y_true, y_pred_prob)

    plt.figure(figsize=(7, 6))
    plt.plot(recall, precision, label=f"AP = {ap_score:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"BATADAL {model_type} Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()

    output_path = output_dir / f"batadal_{model_type.lower()}_pr_curve.png"
    plt.savefig(output_path)
    plt.close()

    print(f"Kaydedildi: {output_path}")


def main():
    output_dir = Path("results/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_type in ["LSTM", "GRU"]:
        y_true, y_pred_prob, y_pred, best_threshold, best_metrics = train_and_predict(model_type)

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