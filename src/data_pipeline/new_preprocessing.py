from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from imblearn.over_sampling import ADASYN
from sklearn.preprocessing import RobustScaler


DATA2_PROCESSED_DIR = Path("data2/processed")


def normalize_batadal_labels(
    y: pd.Series,
) -> pd.Series:
    """
    BATADAL etiketlerini binary biçime dönüştürür.

    -999 -> 0 normal
    diğer değerler -> 1 saldırı
    """
    y_numeric = pd.to_numeric(
        y,
        errors="coerce",
    ).fillna(-999)

    y_binary = np.where(
        y_numeric == -999,
        0,
        1,
    )

    return pd.Series(
        y_binary,
        index=y.index,
        name="anomaly",
        dtype="int32",
    )


def clean_missing_values(
    X: pd.DataFrame,
) -> pd.DataFrame:
    """
    Sonsuz ve eksik değerleri zaman sırasını koruyarak doldurur.
    """
    X_clean = X.copy()

    X_clean = X_clean.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    X_clean = X_clean.ffill().bfill()

    if X_clean.isna().any().any():
        X_clean = X_clean.fillna(
            X_clean.median()
        )

    return X_clean


def split_batadal_time_ordered(
    batadal_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    train_ratio: float = 0.60,
    val_ratio: float = 0.20,
) -> Dict[str, pd.DataFrame | pd.Series]:
    """
    BATADAL verisini zaman sırasını bozmadan böler.
    """
    X = clean_missing_values(
        batadal_df[feature_cols].copy()
    )

    y = normalize_batadal_labels(
        batadal_df[target_col]
    )

    row_count = len(batadal_df)

    train_end = int(
        row_count * train_ratio
    )

    val_end = int(
        row_count * (
            train_ratio + val_ratio
        )
    )

    return {
        "X_train": X.iloc[:train_end].copy(),
        "y_train": y.iloc[:train_end].copy(),

        "X_val": X.iloc[train_end:val_end].copy(),
        "y_val": y.iloc[train_end:val_end].copy(),

        "X_test": X.iloc[val_end:].copy(),
        "y_test": y.iloc[val_end:].copy(),
    }


def scale_train_val_test_robust(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
):
    """
    RobustScaler yalnızca train üzerinde fit edilir.
    Validation ve test aynı scaler ile dönüştürülür.
    """
    scaler = RobustScaler()

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )

    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val),
        columns=X_val.columns,
        index=X_val.index,
    )

    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )

    return {
        "X_train_scaled": X_train_scaled,
        "X_val_scaled": X_val_scaled,
        "X_test_scaled": X_test_scaled,
        "scaler": scaler,
    }


def apply_adasyn_train_only(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
):
    """
    ADASYN yalnızca eğitim verisine uygulanır.
    """
    print(
        "ADASYN öncesi sınıf dağılımı:",
        y_train.value_counts().to_dict(),
    )

    adasyn = ADASYN(
        random_state=random_state
    )

    X_train_adasyn, y_train_adasyn = (
        adasyn.fit_resample(
            X_train,
            y_train,
        )
    )

    X_train_adasyn = pd.DataFrame(
        X_train_adasyn,
        columns=X_train.columns,
    )

    y_train_adasyn = pd.Series(
        y_train_adasyn,
        name=y_train.name,
        dtype="int32",
    )

    print(
        "ADASYN sonrası sınıf dağılımı:",
        y_train_adasyn.value_counts().to_dict(),
    )

    return (
        X_train_adasyn,
        y_train_adasyn,
    )


def save_batadal_data2(
    experiment_name: str,
    data: dict,
):
    """
    Çıktıları data2/processed altında saklar.
    """
    output_dir = (
        DATA2_PROCESSED_DIR
        / experiment_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for file_name, value in data.items():
        output_path = (
            output_dir / file_name
        )

        if isinstance(value, pd.Series):
            value.to_frame().to_csv(
                output_path,
                index=False,
            )
        else:
            value.to_csv(
                output_path,
                index=False,
            )

        print(f"Kaydedildi: {output_path}")


def process_batadal_data2(
    batadal_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    experiment_name: str = "robust_adasyn",
):
    """
    Akış:

    Zaman sıralı split
    -> RobustScaler
    -> ADASYN yalnızca Train
    -> data2 altında kaydetme
    """

    split_data = split_batadal_time_ordered(
        batadal_df=batadal_df,
        feature_cols=feature_cols,
        target_col=target_col,
    )

    scaled_data = scale_train_val_test_robust(
        X_train=split_data["X_train"],
        X_val=split_data["X_val"],
        X_test=split_data["X_test"],
    )

    (
        X_train_adasyn,
        y_train_adasyn,
    ) = apply_adasyn_train_only(
        X_train=scaled_data[
            "X_train_scaled"
        ],
        y_train=split_data[
            "y_train"
        ],
    )

    save_batadal_data2(
        experiment_name=experiment_name,
        data={
            "X_train_adasyn.csv":
                X_train_adasyn,

            "y_train_adasyn.csv":
                y_train_adasyn,

            "X_val_scaled.csv":
                scaled_data["X_val_scaled"],

            "y_val.csv":
                split_data["y_val"],

            "X_test_scaled.csv":
                scaled_data["X_test_scaled"],

            "y_test.csv":
                split_data["y_test"],
        },
    )

    print(
        "\nRobustScaler + ADASYN "
        "ön işleme tamamlandı."
    )

    print(
        f"Çıktı klasörü: "
        f"data2/processed/{experiment_name}"
    )