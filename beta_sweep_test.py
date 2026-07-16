"""
Interpolated back-off (beta sweep) ile mevcut predict_smoothed sonuclarini
validation setinde karsilastirir ayrı dosyada yapma sebebi:eğer kotu etkilerse dahil etmemek
"""

import os
import sys
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from src.data_pipeline.sax_paa import SaxPaaTransformer
from src.models.vomm_model import VariableOrderMarkovModel
from src.experiments.evaluator import calculate_metrics


def align_labels_to_patterns(y, num_patterns, paa_window_size, pattern_length=None):
    aligned_labels = []
    if pattern_length is None:
        pattern_length = paa_window_size
    for pattern_idx in range(1, num_patterns):
        start_idx = pattern_idx * paa_window_size
        end_idx = min((pattern_idx + pattern_length) * paa_window_size, len(y))
        if end_idx <= start_idx:
            break
        label = 1 if np.any(y[start_idx:end_idx] == 1) else 0
        aligned_labels.append(label)
    return np.array(aligned_labels)


def run_beta_sweep(X_train, X_val, y_val, X_test, y_test,
                    window_size, alphabet_size, min_count, smoothing_alpha,
                    threshold_values, beta_values):

    transformer = SaxPaaTransformer(alphabet_size=alphabet_size)
    train_patterns = transformer.transform(X_train, window_size=window_size)
    val_patterns = transformer.transform(X_val, window_size=window_size)
    test_patterns = transformer.transform(X_test, window_size=window_size)

    y_val_aligned = align_labels_to_patterns(y_val, len(val_patterns), window_size)
    y_test_aligned = align_labels_to_patterns(y_test, len(test_patterns), window_size)

    model = VariableOrderMarkovModel(
        max_depth=window_size,
        min_count=min_count,
        smoothing=True,
        smoothing_alpha=smoothing_alpha
    )
    model.fit(train_patterns)

    #eski yontem:predict_smoothed, smooth_window=1
    best_old_f1 = -1
    best_old_threshold = None
    for threshold in threshold_values:
        preds_val, _ = model.predict_smoothed(val_patterns, anomaly_threshold=threshold, smooth_window=1)
        preds_val = preds_val[:len(y_val_aligned)]
        metrics = calculate_metrics(y_val_aligned, preds_val)
        if metrics["f1_score"] > best_old_f1:
            best_old_f1 = metrics["f1_score"]
            best_old_threshold = threshold

    preds_test_old, _ = model.predict_smoothed(test_patterns, anomaly_threshold=best_old_threshold, smooth_window=1)
    preds_test_old = preds_test_old[:len(y_test_aligned)]
    old_test_metrics = calculate_metrics(y_test_aligned, preds_test_old)

    print(f"\n[ESKI YONTEM - hard back-off] val_threshold={best_old_threshold} | val_F1={best_old_f1:.4f}")
    print(f"  TEST -> Precision={old_test_metrics['precision']:.4f} "
          f"Recall={old_test_metrics['recall']:.4f} F1={old_test_metrics['f1_score']:.4f}")

    #yeni yontem:beta + threshold sweep
    best_new_f1 = -1
    best_new_threshold = None
    best_beta = None

    for beta in beta_values:
        for threshold in threshold_values:
            preds_val, _ = model.predict_interpolated(val_patterns, anomaly_threshold=threshold, beta=beta)
            preds_val = preds_val[:len(y_val_aligned)]
            metrics = calculate_metrics(y_val_aligned, preds_val)
            if metrics["f1_score"] > best_new_f1:
                best_new_f1 = metrics["f1_score"]
                best_new_threshold = threshold
                best_beta = beta

    preds_test_new, _ = model.predict_interpolated(test_patterns, anomaly_threshold=best_new_threshold, beta=best_beta)
    preds_test_new = preds_test_new[:len(y_test_aligned)]
    new_test_metrics = calculate_metrics(y_test_aligned, preds_test_new)

    print(f"\nYENI YONTEM - interpolated back-off beta={best_beta} "
          f"val_threshold={best_new_threshold} | val_F1={best_new_f1:.4f}")
    print(f"  TEST -> Precision={new_test_metrics['precision']:.4f} "
          f"Recall={new_test_metrics['recall']:.4f} F1={new_test_metrics['f1_score']:.4f}")

    print(f"\nFARK Test F1 degisimi: {old_test_metrics['f1_score']:.4f} -> "
          f"{new_test_metrics['f1_score']:.4f} "
          f"({new_test_metrics['f1_score'] - old_test_metrics['f1_score']:+.4f})")

    return old_test_metrics, new_test_metrics, best_beta, best_new_threshold


if __name__ == "__main__":
    threshold_values = [0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5]
    beta_values = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]

    #batadal
    if os.path.exists("data/processed/batadal_X_train_adasyn_pc1.csv"):
        print("=" * 60)
        print("BATADAL BETA SWEEP")
        print("=" * 60)

        X_train = pd.read_csv("data/processed/batadal_X_train_adasyn_pc1.csv").values.flatten()
        X_val = pd.read_csv("data/processed/batadal_X_val_pc1.csv").values.flatten()
        y_val = pd.read_csv("data/processed/batadal_y_val.csv").values.flatten()
        y_val = np.where(y_val == -999, 0, y_val)
        X_test = pd.read_csv("data/processed/batadal_X_test_pc1.csv").values.flatten()
        y_test = pd.read_csv("data/processed/batadal_y_test.csv").values.flatten()
        y_test = np.where(y_test == -999, 0, y_test)

        run_beta_sweep(
            X_train, X_val, y_val, X_test, y_test,
            window_size=6, alphabet_size=4, min_count=3, smoothing_alpha=1.0,
            threshold_values=threshold_values, beta_values=beta_values
        )

    #skab
    if os.path.exists("data/processed/skab_fold1_X_train_pc1.csv"):
        print("\n" + "=" * 60)
        print("SKAB (fold 1) BETA SWEEP")
        print("=" * 60)

        X_train_full = pd.read_csv("data/processed/skab_fold1_X_train_pc1.csv").values.flatten()
        y_train_full = pd.read_csv("data/processed/skab_fold1_y_train.csv").values.flatten()
        split_idx = int(len(X_train_full) * 0.8)
        X_train = X_train_full[:split_idx]
        X_val = X_train_full[split_idx:]
        y_val = y_train_full[split_idx:]

        X_test = pd.read_csv("data/processed/skab_fold1_X_test_pc1.csv").values.flatten()
        y_test = pd.read_csv("data/processed/skab_fold1_y_test.csv").values.flatten()

        run_beta_sweep(
            X_train, X_val, y_val, X_test, y_test,
            window_size=12, alphabet_size=4, min_count=2, smoothing_alpha=1.0,
            threshold_values=threshold_values, beta_values=beta_values
        )