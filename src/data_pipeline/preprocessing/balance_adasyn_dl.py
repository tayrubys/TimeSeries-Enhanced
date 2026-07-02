import numpy as np
from pathlib import Path
from imblearn.over_sampling import ADASYN


def balance_batadal_train_sequences_adasyn(
    processed_dir="data/processed",
    random_state=42
):
    """
    ADASYN'i SADECE derin öğrenme (LSTM/GRU/CNN) için train sequence'larına,
    pencere (window) seviyesinde uygular.

    Neden satır bazlı değil pencere bazlı?
    BATADAL zaman sıralı bir veri seti olduğu için, ADASYN'i pencelemeden
    önceki ham satırlara uygularsak, sentetik satırlar gerçek zaman akışıyla
    ilişkisiz konumlara eklenir ve pencereleme sırasında zamansal tutarlılığı
    bozan pencereler oluşur. Bu yüzden önce gerçek pencereler oluşturulur
    (sequence_builder.py -> create_sequences), her pencere TEK BİR birim
    olarak ADASYN'e verilir.
    """
    processed_dir = Path(processed_dir)

    X_seq_path = processed_dir / "batadal_X_train_seq.npy"
    y_seq_path = processed_dir / "batadal_y_train_seq.npy"

    if not X_seq_path.exists() or not y_seq_path.exists():
        raise FileNotFoundError(
            f"Hata: '{X_seq_path}' veya '{y_seq_path}' bulunamadı! "
            f"Lütfen önce run_sequence_building.py scriptini çalıştırın."
        )

    X_seq = np.load(X_seq_path)
    y_seq = np.load(y_seq_path)

    n_pencere, window, n_features = X_seq.shape

    print(f"ADASYN öncesi dağılım: {np.bincount(y_seq.astype(int))}")

    # Her pencereyi tek bir vektör olarak düzleştir (3D -> 2D)
    X_flat = X_seq.reshape(n_pencere, window * n_features)

    adasyn = ADASYN(random_state=random_state)
    X_flat_resampled, y_resampled = adasyn.fit_resample(X_flat, y_seq)

    # Tekrar (n_pencere, window, n_features) şekline döndür (2D -> 3D)
    X_resampled = X_flat_resampled.reshape(-1, window, n_features)

    print(f"ADASYN sonrası dağılım: {np.bincount(y_resampled.astype(int))}")

    X_output_path = processed_dir / "batadal_X_train_seq_adasyn.npy"
    y_output_path = processed_dir / "batadal_y_train_seq_adasyn.npy"

    np.save(X_output_path, X_resampled)
    np.save(y_output_path, y_resampled)

    print(f"ADASYN ile dengelenmiş train kaydedildi:")
    print(f"  X shape: {X_resampled.shape}")
    print(f"  y shape: {y_resampled.shape}")
    print(f"  X dosyası: {X_output_path}")
    print(f"  y dosyası: {y_output_path}")


if __name__ == "__main__":
    print("=" * 50)
    print("BATADAL Sequence Verileri ADASYN ile Dengeleniyor...")
    print("=" * 50)

    balance_batadal_train_sequences_adasyn(
        processed_dir="data/processed",
        random_state=42
    )