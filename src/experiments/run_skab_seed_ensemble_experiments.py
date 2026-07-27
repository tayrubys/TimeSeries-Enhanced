import os

# TensorFlow yüklenmeden önce deterministik GPU işlemlerini etkinleştir.
os.environ["TF_DETERMINISTIC_OPS"] = "1"

import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path
from tensorflow.keras.callbacks import EarlyStopping

from src.models.deep_model import build_lstm_model, build_gru_model
from src.experiments.evaluator import evaluate_binary_classification
from src.config import get_dl_config


tf.config.experimental.enable_op_determinism()


def set_deterministic_seed(seed):
    """Keras durumunu temizler ve Python/NumPy/TensorFlow seed'lerini ayarlar."""
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)


def load_skab_fold_data(fold_id, processed_dir="data/processed"):
    X_train = np.load(
        f"{processed_dir}/skab_fold{fold_id}_X_train_seq.npy"
    ).astype("float32")
    y_train = np.load(
        f"{processed_dir}/skab_fold{fold_id}_y_train_seq.npy"
    ).astype("float32")
    X_test = np.load(
        f"{processed_dir}/skab_fold{fold_id}_X_test_seq.npy"
    ).astype("float32")
    y_test = np.load(
        f"{processed_dir}/skab_fold{fold_id}_y_test_seq.npy"
    ).astype("float32")
    return X_train, y_train, X_test, y_test


def split_train_validation(X_train, y_train, val_ratio):
    """Zaman sırasını bozmadan train'in son bölümünü validation olarak ayırır."""
    val_start = int(len(X_train) * (1 - val_ratio))
    return (
        X_train[:val_start],
        y_train[:val_start],
        X_train[val_start:],
        y_train[val_start:],
    )


def build_model(model_type, input_shape):
    if model_type == "LSTM":
        return build_lstm_model(input_shape=input_shape)
    if model_type == "GRU":
        return build_gru_model(input_shape=input_shape)
    raise ValueError(f"Desteklenmeyen model tipi: {model_type}")


def find_best_threshold(y_true, y_pred_prob):
    thresholds = np.arange(0.01, 0.51, 0.01)

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

    set_deterministic_seed(seed)

    X_train, y_train, X_test, y_test = load_skab_fold_data(fold_id)
    X_tr, y_tr, X_val, y_val = split_train_validation(
        X_train,
        y_train,
        cfg["val_ratio"],
    )

    print(
        f"Fold {fold_id} | window={X_tr.shape[1]} | "
        f"features={X_tr.shape[2]}"
    )

    model = build_model(model_type=model_type, input_shape=X_tr.shape[1:])

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=cfg["early_stopping_patience"],
        restore_best_weights=True,
    )

    # BATADAL betiğiyle tutarlılık için class_weight kullanılmıyor.
    model.fit(
        X_tr,
        y_tr,
        validation_data=(X_val, y_val),
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        callbacks=[early_stopping],
        class_weight=None,
        shuffle=False,
        verbose=1,
    )

    y_val_pred_prob = model.predict(X_val, verbose=0).ravel()
    best_threshold, val_metrics = find_best_threshold(
        y_val,
        y_val_pred_prob,
    )

    y_test_pred_prob = model.predict(X_test, verbose=0).ravel()
    y_test_pred = (y_test_pred_prob >= best_threshold).astype(int)

    test_metrics = evaluate_binary_classification(
        y_true=y_test,
        y_pred=y_test_pred,
    )

    result = {
        "dataset": "SKAB",
        "model": model_type,
        "seed": seed,
        "fold": fold_id,
        "threshold": best_threshold,
        "accuracy": test_metrics["accuracy"],
        "precision": test_metrics["precision"],
        "recall": test_metrics["recall"],
        "f1": test_metrics["f1"],
    }

    print("Validation sonucu:", val_metrics)
    print("Test sonucu:", result)

    # Fold içindeki seed ensemble için olasılıklar ve etiketler de döndürülür.
    return (
        result,
        y_val_pred_prob,
        y_test_pred_prob,
        y_val.astype(int).ravel(),
        y_test.astype(int).ravel(),
    )


def evaluate_fold_seed_ensemble(
    model_type,
    fold_id,
    seeds,
    val_probabilities,
    test_probabilities,
    y_val,
    y_test,
):
    """Aynı fold'da eğitilen seed modellerinin olasılıklarını ortalar."""
    val_probability_matrix = np.stack(val_probabilities, axis=0)
    test_probability_matrix = np.stack(test_probabilities, axis=0)

    ensemble_val_prob = val_probability_matrix.mean(axis=0)
    ensemble_test_prob = test_probability_matrix.mean(axis=0)

    # Ensemble threshold'u yalnızca ilgili fold'un validation verisinden seçilir.
    best_threshold, val_metrics = find_best_threshold(
        y_val,
        ensemble_val_prob,
    )

    ensemble_test_pred = (ensemble_test_prob >= best_threshold).astype(int)
    test_metrics = evaluate_binary_classification(
        y_true=y_test,
        y_pred=ensemble_test_pred,
    )

    result = {
        "dataset": "SKAB",
        "model": model_type,
        "fold": fold_id,
        "ensemble_size": len(seeds),
        "seeds": ",".join(map(str, seeds)),
        "threshold": best_threshold,
        "val_f1": val_metrics["f1"],
        "accuracy": test_metrics["accuracy"],
        "precision": test_metrics["precision"],
        "recall": test_metrics["recall"],
        "f1": test_metrics["f1"],
    }

    probability_df = pd.DataFrame({
        "dataset": "SKAB",
        "model": model_type,
        "fold": fold_id,
        "sample_index": np.arange(len(y_test)),
        "y_true": y_test,
        **{
            f"seed_{seed}_prob": test_probability_matrix[index]
            for index, seed in enumerate(seeds)
        },
        "ensemble_prob": ensemble_test_prob,
        "ensemble_pred": ensemble_test_pred,
    })

    print("\n==============================")
    print(f"SKAB {model_type} Fold {fold_id} - {len(seeds)} Seed Ensemble")
    print("==============================")
    print("Ensemble validation sonucu:", val_metrics)
    print("Ensemble test sonucu:", result)

    return result, probability_df


def summarize_seed_results(results_df):
    """Tüm seed ve fold deneylerinin genel ortalama/std değerlerini hesaplar."""
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


def summarize_ensemble_results(ensemble_results_df):
    """Beş fold'un ensemble sonuçlarını ortalama ± std olarak özetler."""
    return (
        ensemble_results_df
        .groupby(["dataset", "model"])
        .agg(
            fold_count=("fold", "count"),
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
    seeds = list(cfg["seeds"])

    all_results = []
    ensemble_results = []
    ensemble_probability_outputs = []

    # Ensemble aynı fold'un aynı örnekleri üzerinde kurulacağı için
    # döngü sırası model -> fold -> seed şeklindedir.
    for model_type in ["LSTM", "GRU"]:
        for fold_id in range(1, 2):
            val_probabilities = []
            test_probabilities = []
            ensemble_y_val = None
            ensemble_y_test = None

            for seed in seeds:
                result, val_prob, test_prob, y_val, y_test = (
                    train_one_skab_experiment(
                        model_type=model_type,
                        fold_id=fold_id,
                        seed=seed,
                    )
                )

                all_results.append(result)
                val_probabilities.append(val_prob)
                test_probabilities.append(test_prob)

                if ensemble_y_val is None:
                    ensemble_y_val = y_val
                    ensemble_y_test = y_test
                else:
                    if not np.array_equal(ensemble_y_val, y_val):
                        raise ValueError(
                            f"{model_type} fold {fold_id}: seed'ler arasında "
                            "validation etiketleri değişti."
                        )
                    if not np.array_equal(ensemble_y_test, y_test):
                        raise ValueError(
                            f"{model_type} fold {fold_id}: seed'ler arasında "
                            "test etiketleri değişti."
                        )

            ensemble_result, probability_df = evaluate_fold_seed_ensemble(
                model_type=model_type,
                fold_id=fold_id,
                seeds=seeds,
                val_probabilities=val_probabilities,
                test_probabilities=test_probabilities,
                y_val=ensemble_y_val,
                y_test=ensemble_y_test,
            )
            ensemble_results.append(ensemble_result)
            ensemble_probability_outputs.append(probability_df)

    output_dir = Path("results/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(
        output_dir / "skab_seed_ensemble_individual_results.csv",
        index=False,
    )

    seed_summary_df = summarize_seed_results(results_df)
    seed_summary_df.to_csv(
        output_dir / "skab_seed_ensemble_individual_summary.csv",
        index=False,
    )

    ensemble_results_df = pd.DataFrame(ensemble_results)
    ensemble_results_df.to_csv(
        output_dir / "skab_seed_ensemble_fold_results.csv",
        index=False,
    )

    ensemble_summary_df = summarize_ensemble_results(ensemble_results_df)
    ensemble_summary_df.to_csv(
        output_dir / "skab_seed_ensemble_fold_summary.csv",
        index=False,
    )

    ensemble_probabilities_df = pd.concat(
        ensemble_probability_outputs,
        ignore_index=True,
    )
    ensemble_probabilities_df.to_csv(
        output_dir / "skab_seed_ensemble_probabilities.csv",
        index=False,
    )

    print("\nSKAB seed + fold bazlı tekil sonuçlar:")
    print(results_df)
    print("\nSKAB tekil modellerin genel mean/std özeti:")
    print(seed_summary_df)
    print("\nSKAB fold bazlı ensemble sonuçları:")
    print(ensemble_results_df)
    print("\nSKAB 5-fold ensemble mean/std özeti:")
    print(ensemble_summary_df)


if __name__ == "__main__":
    main()