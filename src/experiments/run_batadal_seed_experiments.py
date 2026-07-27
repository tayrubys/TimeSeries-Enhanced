import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight

from src.models.deep_model import build_lstm_model, build_gru_model
from src.experiments.evaluator import evaluate_binary_classification
from src.config import get_dl_config


def load_batadal_sequence_data(processed_dir="data/processed/",balancing_method="class_weight", model_type="GRU"):
    X_train = np.load(f"{processed_dir}/batadal_X_train_seq.npy").astype("float32")
    y_train = np.load(f"{processed_dir}/batadal_y_train_seq.npy").astype("float32")
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


def find_best_threshold(y_true, y_pred_prob):
    thresholds = np.arange(0.01, 0.51, 0.01)

    best_threshold = None
    best_metrics = None
    best_f1 = -1

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
        balancing_method=balancing_method,
        model_type=model_type
    )
    actual_window_size = X_train.shape[1]  
    cfg["sequence_window_size"] = actual_window_size

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
        verbose=1,

    )

    y_val_pred_prob = model.predict(X_val)
    best_threshold, val_metrics = find_best_threshold(
        y_val,
        y_val_pred_prob
    )

    y_test_pred_prob = model.predict(X_test)

    print("\nProbability Analysis")
    print("Validation")
    print(" Min :", y_val_pred_prob.min())
    print(" Max :", y_val_pred_prob.max())
    print(" Mean:", y_val_pred_prob.mean())

    print("\nTest")
    print(" Min :", y_test_pred_prob.min())
    print(" Max :", y_test_pred_prob.max())
    print(" Mean:", y_test_pred_prob.mean())

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
    return (
        result,
        y_val_pred_prob.ravel(),
        y_test_pred_prob.ravel(),
        y_val.astype(int).ravel(),
        y_test.astype(int).ravel(),
    )


def evaluate_seed_ensemble(
    model_type,
    balancing_method,
    seeds,
    val_probabilities,
    test_probabilities,
    y_val,
    y_test,
):
    """Seed modellerinin olasılıklarını ortalayıp tek final tahmin üretir."""
    val_probability_matrix = np.stack(val_probabilities, axis=0)
    test_probability_matrix = np.stack(test_probabilities, axis=0)

    ensemble_val_prob = val_probability_matrix.mean(axis=0)
    ensemble_test_prob = test_probability_matrix.mean(axis=0)

    # Threshold yalnızca validation verisinden seçilir.
    best_threshold, val_metrics = find_best_threshold(y_val, ensemble_val_prob)
    ensemble_test_pred = (ensemble_test_prob >= best_threshold).astype(int)
    test_metrics = evaluate_binary_classification(
        y_true=y_test,
        y_pred=ensemble_test_pred,
    )

    result = {
        "dataset": "BATADAL",
        "model": model_type,
        "balancing_method": balancing_method,
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
        "y_true": y_test,
        **{
            f"seed_{seed}_prob": test_probability_matrix[index]
            for index, seed in enumerate(seeds)
        },
        "ensemble_prob": ensemble_test_prob,
        "ensemble_pred": ensemble_test_pred,
    })

    print("\n==============================")
    print(f"{len(seeds)}-Seed {model_type} Ensemble Sonucu")
    print("==============================")
    print("Validation sonucu:", val_metrics)
    print("Test sonucu:", result)

    return result, probability_df


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
    ensemble_results = []
    ensemble_probability_outputs = []

    balancing_methods = ["adasyn"]

    for model_type in [ "LSTM","GRU"]:
        for balancing_method in balancing_methods:
            val_probabilities = []
            test_probabilities = []
            ensemble_y_val = None
            ensemble_y_test = None

            for seed in cfg["seeds"]:
                result, val_prob, test_prob, y_val, y_test = train_one_batadal_experiment(
                    model_type=model_type,
                    seed=seed,
                    balancing_method=balancing_method
                )
                all_results.append(result)
                val_probabilities.append(val_prob)
                test_probabilities.append(test_prob)

                if ensemble_y_val is None:
                    ensemble_y_val = y_val
                    ensemble_y_test = y_test
                else:
                    if not np.array_equal(ensemble_y_val, y_val):
                        raise ValueError("Seed'ler arasında validation etiketleri değişti.")
                    if not np.array_equal(ensemble_y_test, y_test):
                        raise ValueError("Seed'ler arasında test etiketleri değişti.")

            ensemble_result, probability_df = evaluate_seed_ensemble(
                model_type=model_type,
                balancing_method=balancing_method,
                seeds=cfg["seeds"],
                val_probabilities=val_probabilities,
                test_probabilities=test_probabilities,
                y_val=ensemble_y_val,
                y_test=ensemble_y_test,
            )
            ensemble_results.append(ensemble_result)
            probability_df.insert(0, "balancing_method", balancing_method)
            probability_df.insert(0, "model", model_type)
            ensemble_probability_outputs.append(probability_df)

    output_dir = Path("results/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_dir / "batadal_ensemble_deep_learning_seed_results.csv", index=False)

    summary_df = summarize_results(results_df)
    summary_df.to_csv(output_dir / "batadal_ensemble_deep_learning_seed_summary.csv", index=False)

    ensemble_results_df = pd.DataFrame(ensemble_results)
    ensemble_results_df.to_csv(
        output_dir / "batadal_ensemble_deep_learning_seed_ensemble_results.csv",
        index=False,
    )

    ensemble_probabilities_df = pd.concat(
        ensemble_probability_outputs,
        ignore_index=True,
    )
    ensemble_probabilities_df.to_csv(
        output_dir / "batadal_ensemble_deep_learning_seed_ensemble_probabilities.csv",
        index=False,
    )

    print("\nBATADAL seed bazlı sonuçlar:")
    print(results_df)
    print("\nBATADAL mean/std özet:")
    print(summary_df)
    print("\nBATADAL seed ensemble sonucu:")
    print(ensemble_results_df)


if __name__ == "__main__":
    main()