import numpy as np

from src.data_pipeline.sequence_builder import build_sequences_from_csv, load_scaled_data


WINDOW_SIZE = 10


def check_batadal_sequences():
    """BATADAL train verisi icin sequence olusumunu kontrol eder."""
    x_train_path = "data/processed/batadal_X_train_scaled.csv"
    y_train_path = "data/processed/batadal_y_train.csv"

    X, y = load_scaled_data(
        X_path=x_train_path,
        y_path=y_train_path,
    )

    print("Dönüştürülmüş unique label değerleri:", sorted(set(y)))
    print("0 sayısı:", (y == 0).sum())
    print("1 sayısı:", (y == 1).sum())

    X_seq, y_seq = build_sequences_from_csv(
        X_path=x_train_path,
        y_path=y_train_path,
        window_size=WINDOW_SIZE,
    )

    print("X_seq shape:", X_seq.shape)
    print("y_seq shape:", y_seq.shape)
    print("İlk sequence shape:", X_seq[0].shape)
    print("İlk label:", y_seq[0])
    print("Sequence label değerleri:", sorted(set(y_seq)))
    print("Sequence 0 sayısı:", (y_seq == 0).sum())
    print("Sequence 1 sayısı:", (y_seq == 1).sum())


def check_skab_fold_sequences():
    """SKAB fold sequence dosyalarini kontrol eder."""
    print("\n--- SKAB tüm fold kontrolleri ---")

    for fold_id in range(1, 6):
        for split_name in ["train", "test"]:
            x_path = f"data/processed/skab_fold{fold_id}_X_{split_name}_seq.npy"
            y_path = f"data/processed/skab_fold{fold_id}_y_{split_name}_seq.npy"

            X_seq = np.load(x_path)
            y_seq = np.load(y_path)

            print(f"Fold {fold_id} {split_name}:")
            print("  X shape:", X_seq.shape)
            print("  y shape:", y_seq.shape)
            print("  İlk sequence shape:", X_seq[0].shape)
            print("  Label değerleri:", sorted(set(y_seq)))
            print("  0 sayısı:", (y_seq == 0).sum())
            print("  1 sayısı:", (y_seq == 1).sum())


def main():
    check_batadal_sequences()
    check_skab_fold_sequences()


if __name__ == "__main__":
    main()