import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, GRU, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from src.config import get_dl_config


PROCESSED_DIR = Path("data/processed")
OUTPUT_DIR = Path("results/outputs/batadal_gru_predictor")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_batadal_scaled_data(
    processed_dir: Path = PROCESSED_DIR,
):
    """
    ADASYN uygulanmamış ve zaman sırası korunmuş
    BATADAL train/validation/test CSV dosyalarını yükler.
    """

    X_train = pd.read_csv(
        processed_dir / "batadal_X_train_scaled.csv"
    ).values.astype("float32")

    y_train = pd.read_csv(
        processed_dir / "batadal_y_train.csv"
    ).values.ravel().astype("int32")

    X_val = pd.read_csv(
        processed_dir / "batadal_X_val_scaled.csv"
    ).values.astype("float32")

    y_val = pd.read_csv(
        processed_dir / "batadal_y_val.csv"
    ).values.ravel().astype("int32")

    X_test = pd.read_csv(
        processed_dir / "batadal_X_test_scaled.csv"
    ).values.astype("float32")

    y_test = pd.read_csv(
        processed_dir / "batadal_y_test.csv"
    ).values.ravel().astype("int32")

    # BATADAL etiket düzenlemesi
    y_train = np.where(y_train == -999, 0, y_train)
    y_val = np.where(y_val == -999, 0, y_val)
    y_test = np.where(y_test == -999, 0, y_test)

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    )


def create_prediction_sequences(
    X: np.ndarray,
    y: np.ndarray,
    window_size: int,
):
    """
    Geçmiş window_size satırı kullanarak
    bir sonraki sensör vektörünü tahmin etmek için veri üretir.

    Girdi:
        X[t-window_size:t]

    Hedef:
        X[t]

    Etiket:
        y[t]
    """

    X_inputs = []
    X_targets = []
    y_targets = []

    for target_index in range(
        window_size,
        len(X),
    ):
        start_index = (
            target_index
            - window_size
        )

        X_window = X[
            start_index:target_index
        ]

        next_sensor_values = X[
            target_index
        ]

        next_label = y[
            target_index
        ]

        X_inputs.append(
            X_window
        )

        X_targets.append(
            next_sensor_values
        )

        y_targets.append(
            next_label
        )

    return (
        np.asarray(
            X_inputs,
            dtype="float32",
        ),
        np.asarray(
            X_targets,
            dtype="float32",
        ),
        np.asarray(
            y_targets,
            dtype="int32",
        ),
    )


def build_gru_predictor(
    input_shape: tuple,
    output_feature_count: int,
    gru_units: int,
    dense_units: int,
    dropout_rate_1: float,
    dropout_rate_2: float,
    learning_rate: float,
):
    """
    Geçmiş zaman penceresine bakarak
    bir sonraki sensör vektörünü tahmin eden GRU modeli.
    """

    model = Sequential([
        Input(
            shape=input_shape
        ),

        GRU(
            units=gru_units,
            activation="tanh",
            return_sequences=False,
            name="gru_predictor",
        ),

        Dropout(
            rate=dropout_rate_1,
            name="gru_dropout",
        ),

        Dense(
            units=dense_units,
            activation="relu",
            name="dense_hidden",
        ),

        Dropout(
            rate=dropout_rate_2,
            name="dense_dropout",
        ),

        Dense(
            units=output_feature_count,
            activation="linear",
            name="next_step_output",
        ),
    ])

    model.compile(
        optimizer=Adam(
            learning_rate=learning_rate,
            clipnorm=1.0,
        ),
        loss="mse",
    )

    return model


def calculate_prediction_errors(
    model,
    X_inputs: np.ndarray,
    X_targets: np.ndarray,
) -> np.ndarray:
    """
    Gerçek sonraki sensör değerleri ile model tahminleri arasındaki
    ortalama kare hatayı hesaplar.
    """

    X_predictions = model.predict(
        X_inputs,
        verbose=0,
    )

    errors = np.max(
        np.square(
            X_targets
            - X_predictions
        ),
        axis=1,
    )

    return errors


def evaluate_binary(
    y_true,
    y_pred,
):
    y_true = np.asarray(
        y_true
    ).ravel()

    y_pred = np.asarray(
        y_pred
    ).ravel()

    return {
        "accuracy": accuracy_score(
            y_true,
            y_pred,
        ),
        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
    }


def find_best_threshold_from_validation(
    y_val: np.ndarray,
    val_errors: np.ndarray,
    percentile_start: float = 80.0,
    percentile_end: float = 100.0,
    percentile_step: float = 0.1,
):
    """
    Validation prediction error değerlerinde
    en yüksek F1-score'u veren threshold'u seçer.
    """

    y_val = np.asarray(
        y_val
    ).ravel()

    val_errors = np.asarray(
        val_errors
    ).ravel()

    percentiles = np.arange(
        percentile_start,
        percentile_end,
        percentile_step,
    )

    thresholds = np.percentile(
        val_errors,
        percentiles,
    )

    best_threshold = None
    best_percentile = None
    best_metrics = None
    best_f1 = -1.0

    for percentile, threshold in zip(
        percentiles,
        thresholds,
    ):
        y_pred = (
            val_errors >= threshold
        ).astype("int32")

        metrics = evaluate_binary(
            y_true=y_val,
            y_pred=y_pred,
        )

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_threshold = float(
                threshold
            )
            best_percentile = float(
                percentile
            )
            best_metrics = metrics

    return (
        best_threshold,
        best_percentile,
        best_metrics,
    )


def print_error_statistics(
    split_name: str,
    errors: np.ndarray,
    labels: np.ndarray,
) -> None:
    normal_errors = errors[
        labels == 0
    ]

    anomaly_errors = errors[
        labels == 1
    ]

    print(
        f"\n{split_name} prediction error:"
    )

    if len(normal_errors) > 0:
        print(
            "Normal mean/std:",
            float(
                normal_errors.mean()
            ),
            float(
                normal_errors.std()
            ),
        )

    if len(anomaly_errors) > 0:
        print(
            "Anomaly mean/std:",
            float(
                anomaly_errors.mean()
            ),
            float(
                anomaly_errors.std()
            ),
        )


def train_one_seed(
    seed: int,
):
    cfg = get_dl_config()

    print("\n" + "=" * 70)
    print(
        "BATADAL GRU Prediction-Based "
        f"Anomaly Detection | seed={seed}"
    )
    print("=" * 70)

    tf.keras.backend.clear_session()
    set_seed(seed)

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    ) = load_batadal_scaled_data()

    window_size = cfg[
        "sequence_window_size"
    ]

    (
        X_train_inputs,
        X_train_targets,
        y_train_targets,
    ) = create_prediction_sequences(
        X=X_train,
        y=y_train,
        window_size=window_size,
    )

    (
        X_val_inputs,
        X_val_targets,
        y_val_targets,
    ) = create_prediction_sequences(
        X=X_val,
        y=y_val,
        window_size=window_size,
    )

    (
        X_test_inputs,
        X_test_targets,
        y_test_targets,
    ) = create_prediction_sequences(
        X=X_test,
        y=y_test,
        window_size=window_size,
    )

    # Model sadece normal bir sonraki adıma sahip
    # sequence'lerle eğitilir.
    normal_train_mask = (
        y_train_targets == 0
    )

    normal_val_mask = (
        y_val_targets == 0
    )

    X_train_normal_inputs = (
        X_train_inputs[
            normal_train_mask
        ]
    )

    X_train_normal_targets = (
        X_train_targets[
            normal_train_mask
        ]
    )

    X_val_normal_inputs = (
        X_val_inputs[
            normal_val_mask
        ]
    )

    X_val_normal_targets = (
        X_val_targets[
            normal_val_mask
        ]
    )

    if len(
        X_train_normal_inputs
    ) == 0:
        raise ValueError(
            "Normal train prediction sequence bulunamadı."
        )

    if len(
        X_val_normal_inputs
    ) == 0:
        raise ValueError(
            "Normal validation prediction sequence bulunamadı."
        )

    print(
        "X_train input shape:",
        X_train_inputs.shape,
    )

    print(
        "Normal train input shape:",
        X_train_normal_inputs.shape,
    )

    print(
        "X_val input shape:",
        X_val_inputs.shape,
    )

    print(
        "X_test input shape:",
        X_test_inputs.shape,
    )

    print(
        "Train hedef sınıfları:",
        np.unique(
            y_train_targets,
            return_counts=True,
        ),
    )

    print(
        "Validation hedef sınıfları:",
        np.unique(
            y_val_targets,
            return_counts=True,
        ),
    )

    print(
        "Test hedef sınıfları:",
        np.unique(
            y_test_targets,
            return_counts=True,
        ),
    )
    print(
    "\nTrain attack ratio:",
    y_train_targets.mean()
    )

    print(
        "Val attack ratio:",
        y_val_targets.mean()
    )

    print(
        "Test attack ratio:",
        y_test_targets.mean()
    )

    model = build_gru_predictor(
        input_shape=(
            X_train_inputs.shape[1:]
        ),
        output_feature_count=(
            X_train_targets.shape[1]
        ),
        gru_units=cfg[
            "gru_units"
        ],
        dense_units=cfg[
            "dense_units"
        ],
        dropout_rate_1=cfg[
            "dropout_rate_1"
        ],
        dropout_rate_2=cfg[
            "dropout_rate_2"
        ],
        learning_rate=cfg[
            "learning_rate"
        ],
    )

    model.summary()

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=cfg[
            "early_stopping_patience"
        ],
        restore_best_weights=True,
        min_delta=1e-4,
        verbose=1,
    )

    history = model.fit(
        X_train_normal_inputs,
        X_train_normal_targets,
        validation_data=(
            X_val_normal_inputs,
            X_val_normal_targets,
        ),
        epochs=cfg[
            "epochs"
        ],
        batch_size=cfg[
            "batch_size"
        ],
        callbacks=[
            early_stopping
        ],
        shuffle=False,
        verbose=1,
    )

    val_errors = (
        calculate_prediction_errors(
            model=model,
            X_inputs=X_val_inputs,
            X_targets=X_val_targets,
        )
    )

    (
        best_threshold,
        best_percentile,
        val_metrics,
    ) = find_best_threshold_from_validation(
        y_val=y_val_targets,
        val_errors=val_errors,
        percentile_start=80.0,
        percentile_end=100.0,
        percentile_step=1.0,
    )

    test_errors = (
        calculate_prediction_errors(
            model=model,
            X_inputs=X_test_inputs,
            X_targets=X_test_targets,
        )
    )

    y_test_pred = (
        test_errors >= best_threshold
    ).astype("int32")

    test_metrics = evaluate_binary(
        y_true=y_test_targets,
        y_pred=y_test_pred,
    )

    print_error_statistics(
        split_name="Validation",
        errors=val_errors,
        labels=y_val_targets,
    )

    print_error_statistics(
        split_name="Test",
        errors=test_errors,
        labels=y_test_targets,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save(
        OUTPUT_DIR
        / f"gru_predictor_seed_{seed}.keras"
    )

    np.save(
        OUTPUT_DIR
        / f"val_prediction_errors_seed_{seed}.npy",
        val_errors,
    )

    np.save(
        OUTPUT_DIR
        / f"test_prediction_errors_seed_{seed}.npy",
        test_errors,
    )

    pd.DataFrame(
        history.history
    ).to_csv(
        OUTPUT_DIR
        / f"training_history_seed_{seed}.csv",
        index=False,
    )

    result = {
        "dataset": "BATADAL",
        "model": "GRU_Predictor",
        "seed": seed,
        "window_size": window_size,
        "feature_count": (
            X_train.shape[1]
        ),
        "threshold_percentile":
            best_percentile,
        "threshold":
            best_threshold,
        "epochs_trained":
            len(
                history.history[
                    "loss"
                ]
            ),

        "val_accuracy":
            val_metrics["accuracy"],
        "val_precision":
            val_metrics["precision"],
        "val_recall":
            val_metrics["recall"],
        "val_f1":
            val_metrics["f1"],

        "accuracy":
            test_metrics["accuracy"],
        "precision":
            test_metrics["precision"],
        "recall":
            test_metrics["recall"],
        "f1":
            test_metrics["f1"],
    }

    print("\nValidation sonucu:")
    print(val_metrics)

    print("\nTest sonucu:")
    print(result)

    return result


def summarize_results(
    results_df: pd.DataFrame,
):
    return (
        results_df
        .groupby([
            "dataset",
            "model",
            "window_size",
            "feature_count",
        ])
        .agg(
            experiment_count=(
                "seed",
                "count",
            ),

            accuracy_mean=(
                "accuracy",
                "mean",
            ),
            accuracy_std=(
                "accuracy",
                "std",
            ),

            precision_mean=(
                "precision",
                "mean",
            ),
            precision_std=(
                "precision",
                "std",
            ),

            recall_mean=(
                "recall",
                "mean",
            ),
            recall_std=(
                "recall",
                "std",
            ),

            f1_mean=(
                "f1",
                "mean",
            ),
            f1_std=(
                "f1",
                "std",
            ),

            threshold_mean=(
                "threshold",
                "mean",
            ),
            threshold_std=(
                "threshold",
                "std",
            ),
        )
        .reset_index()
    )


def main():
    cfg = get_dl_config()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_results = []

    for seed in cfg["seeds"]:
        result = train_one_seed(
            seed=seed
        )

        all_results.append(
            result
        )

        # Her seed sonrasında ara kayıt
        pd.DataFrame(
            all_results
        ).to_csv(
            OUTPUT_DIR
            / "batadal_gru_predictor_results.csv",
            index=False,
        )

    results_df = pd.DataFrame(
        all_results
    )

    summary_df = summarize_results(
        results_df
    )

    results_path = (
        OUTPUT_DIR
        / "batadal_gru_predictor_results.csv"
    )

    summary_path = (
        OUTPUT_DIR
        / "batadal_gru_predictor_summary.csv"
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    print(
        "\nBATADAL GRU Predictor seed sonuçları:"
    )

    print(
        results_df
    )

    print(
        "\nBATADAL GRU Predictor mean/std özeti:"
    )

    print(
        summary_df
    )

    print(
        f"\nSonuç dosyası: {results_path}"
    )

    print(
        f"Özet dosyası: {summary_path}"
    )


if __name__ == "__main__":
    main()