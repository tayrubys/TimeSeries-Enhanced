import os
import sys
import numpy as np
import pandas as pd  
from pathlib import Path
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA  

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

def balance_batadal_train_sequences(
    processed_dir="data/processed",
    sampling_strategy=0.3,
    random_state=42
):
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

    print(f"SMOTE öncesi dağılım: {np.bincount(y_seq.astype(int))}")

    # Her pencereyi tek bir vektör olarak düzleştirme (3D -> 2D)
    X_flat = X_seq.reshape(n_pencere, window * n_features)

    # SMOTE uygulaması
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
    X_flat_resampled, y_resampled = smote.fit_resample(X_flat, y_seq)

    # Tekrar (n_pencere, window, n_features) şekline döndür (2D -> 3D)
    X_resampled = X_flat_resampled.reshape(-1, window, n_features)

    print(f"SMOTE sonrası dağılım: {np.bincount(y_resampled.astype(int))}")

    # Derin Öğrenme İçin
    X_output_path = processed_dir / "batadal_X_train_seq_balanced.npy"
    y_output_path = processed_dir / "batadal_y_train_seq_balanced.npy"

    np.save(X_output_path, X_resampled)
    np.save(y_output_path, y_resampled)

    print(f"Dengelenmiş train kaydedildi:")
    print(f"  X shape: {X_resampled.shape}")
    print(f"  y shape: {y_resampled.shape}")
    print(f"  X dosyası: {X_output_path}")
    print(f"  y dosyası: {y_output_path}")

    #SMOTE SONRASI OTOMATA İÇİN PCA (PC1) UYGULAMASI
    print("\n" + "-"*40)
    print("Otomata için SMOTE sonrası PCA (PC1) hesaplanıyor...")
    print("-"*40)

    pca = PCA(n_components=1, random_state=random_state)
    X_resampled_pc1 = pca.fit_transform(X_flat_resampled)
 
    X_resampled_pc1_df = pd.DataFrame(X_resampled_pc1, columns=['PC1'])
    
    automata_input_path = processed_dir / "batadal_X_train_pc1.csv"
    X_resampled_pc1_df.to_csv(automata_input_path, index=False)
    
    y_train_output_path = processed_dir / "batadal_y_train.csv"
    pd.DataFrame(y_resampled, columns=['target']).to_csv(y_train_output_path, index=False)
    
    print(f"Otomata Girdisi Hazır yol: {automata_input_path}")
    print(f"Otomata Girdisi Boyutu: {X_resampled_pc1_df.shape}")
    print("=" * 50 + "\n")

if __name__ == "__main__":
    print("=" * 50)
    print("BATADAL Sequence Verileri SMOTE ile Dengeleniyor...")
    print("=" * 50)
    
    balance_batadal_train_sequences(
        processed_dir="data/processed",
        sampling_strategy=0.3,
        random_state=42
    )