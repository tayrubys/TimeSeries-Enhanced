import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# Binary classification sonuçlarını değerlendirme
# Deep learning tarafı bu fonksiyonu kullanıyor.
def evaluate_binary_classification(y_true, y_pred):
    """
    Binary classification sonuçlarını değerlendirir.

    0 = normal
    1 = anomaly
    """

    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm,
    }


# Otomata tarafı için metrik hesaplama
def calculate_metrics(y_true, y_pred):
    """
    Otomata modeli için binary classification metriklerini hesaplar.

    0 = normal
    1 = anomaly

    Deep learning modelleri ile tutarlı karşılaştırma için
    precision, recall ve F1-score anomaly=1 sınıfı üzerinden hesaplanır.
    """

    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
    }


# Metrikleri ekrana yazdır
def print_metrics(metrics):

    print("Accuracy:", metrics["accuracy"])
    print("Precision:", metrics["precision"])
    print("Recall:", metrics["recall"])

    if "f1" in metrics:
        print("F1-score:", metrics["f1"])
    elif "f1_score" in metrics:
        print("F1-score:", metrics["f1_score"])

    if "confusion_matrix" in metrics:
        print("Confusion Matrix:")
        print(metrics["confusion_matrix"])
