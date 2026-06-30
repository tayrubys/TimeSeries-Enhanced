import numpy as np
import pandas as pd
from pathlib import Path

# LSTM / GRU modelleri için zaman pencereleri oluşturur
def create_sequences(X, y, window_size=20):
    X_sequences = []
    y_sequences = []

    for i in range(window_size - 1, len(X)):
        start_index = i - window_size + 1
        end_index = i + 1

        X_window = X[start_index:end_index]
        y_label = y[i]

        X_sequences.append(X_window)
        y_sequences.append(y_label)

    X_seq = np.array(X_sequences)
    y_seq = np.array(y_sequences)

    return X_seq, y_seq


# Scaled feature CSV ve label CSV dosyalarını okur
def load_scaled_data(X_path, y_path):
    X_df = pd.read_csv(X_path)
    y_df = pd.read_csv(y_path)

    X = X_df.values
    y = y_df.values.ravel()

    # BATADAL için -999 normal sınıfı temsil ediyor.
    # Binary classification için bunu 0'a çeviriyoruz.
    y = np.where(y == -999, 0, y)

    return X, y


# CSV dosyalarından LSTM / GRU için sequence verisi üretir
def build_sequences_from_csv(X_path, y_path, window_size=20):
    X, y = load_scaled_data(X_path, y_path)

    X_seq, y_seq = create_sequences(
        X=X,
        y=y,
        window_size=window_size
    )

    return X_seq, y_seq

#batadal train/validation/test scaled csv dosylarından lstm ve gru ıcın sequence verileri uretıp .npy olarak kaydetme
def build_and_save_batadal_sequences(
    processed_dir="data/processed",
    window_size=20
):

    processed_dir = Path(processed_dir)

    datasets = {
        "train": (
            processed_dir / "batadal_X_train_scaled.csv",
            processed_dir / "batadal_y_train.csv"
        ),
        "val": (
            processed_dir / "batadal_X_val_scaled.csv",
            processed_dir / "batadal_y_val.csv"
        ),
        "test": (
            processed_dir / "batadal_X_test_scaled.csv",
            processed_dir / "batadal_y_test.csv"
        )
    }

    for split_name, (X_path, y_path) in datasets.items():
        X_seq, y_seq = build_sequences_from_csv(
            X_path=X_path,
            y_path=y_path,
            window_size=window_size
        )

        X_output_path = processed_dir / f"batadal_X_{split_name}_seq.npy"
        y_output_path = processed_dir / f"batadal_y_{split_name}_seq.npy"

        np.save(X_output_path, X_seq)
        np.save(y_output_path, y_seq)

        print(f"{split_name} kaydedildi:")
        print(f"  X shape: {X_seq.shape}")
        print(f"  y shape: {y_seq.shape}")
        print(f"  X dosyası: {X_output_path}")
        print(f"  y dosyası: {y_output_path}")

#skab ıcın sequence
def create_sequences_by_group(X, y, groups, window_size=20):

    X_sequences = []
    y_sequences = []

    unique_groups = pd.Series(groups).unique()

    for group_name in unique_groups:
        group_mask = groups == group_name

        X_group = X[group_mask]
        y_group = y[group_mask]

        if len(X_group) < window_size:
            continue

        for i in range(window_size - 1, len(X_group)):
            start_index = i - window_size + 1
            end_index = i + 1

            X_window = X_group[start_index:end_index]
            y_label = y_group[i]

            X_sequences.append(X_window)
            y_sequences.append(y_label)

    X_seq = np.array(X_sequences)
    y_seq = np.array(y_sequences)

    return X_seq, y_seq

#skab ıcın .npy kaydetme
def build_and_save_skab_sequences(
    processed_dir="data/processed",
    window_size=20,
    n_folds=5
):

    processed_dir = Path(processed_dir)

    for fold_id in range(1, n_folds + 1):
        for split_name in ["train", "test"]:
            X_path = processed_dir / f"skab_fold{fold_id}_X_{split_name}_scaled.csv"
            y_path = processed_dir / f"skab_fold{fold_id}_y_{split_name}.csv"
            source_file_path = processed_dir / f"skab_fold{fold_id}_{split_name}_source_file.csv"

            X_df = pd.read_csv(X_path)
            y_df = pd.read_csv(y_path)
            source_file_df = pd.read_csv(source_file_path)

            X = X_df.values
            y = y_df.values.ravel()
            groups = source_file_df.iloc[:, 0].values

            X_seq, y_seq = create_sequences_by_group(
                X=X,
                y=y,
                groups=groups,
                window_size=window_size
            )

            X_output_path = processed_dir / f"skab_fold{fold_id}_X_{split_name}_seq.npy"
            y_output_path = processed_dir / f"skab_fold{fold_id}_y_{split_name}_seq.npy"

            np.save(X_output_path, X_seq)
            np.save(y_output_path, y_seq)

            print(f"SKAB fold{fold_id} {split_name} kaydedildi:")
            print(f"  X shape: {X_seq.shape}")
            print(f"  y shape: {y_seq.shape}")
            print(f"  X dosyası: {X_output_path}")
            print(f"  y dosyası: {y_output_path}")
