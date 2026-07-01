import pandas as pd
from imblearn.under_sampling import RandomUnderSampler

# 1. Dosya yollarını doğrudan proje ana dizinine (root) göre ayarlıyoruz
x_path = "data/processed/batadal_X_train_scaled.csv"
y_path = "data/processed/batadal_y_train.csv"

# 2. Eğitim (Train) verilerini okutuyoruz
X_train = pd.read_csv(x_path)
y_train = pd.read_csv(y_path)

# --- İŞLEM ÖNCESİ ÇIKTISI ---
print("==========")
print("İŞLEM ÖNCESİ DURUM")
print("==========")
print(f"Orijinal Veri Seti Boyutu (Satır, Sütun): {X_train.shape}")
print("Orijinal Sınıf Dağılımı:")
print(y_train.value_counts())
print("\n") # Araya boşluk bırakıyoruz

# 3. Random Undersampler'ı tanımlayıp uyguluyoruz
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X_train, y_train)

# --- İŞLEM SONRASI ÇIKTISI ---
print("==========")
print("RANDOM UNDERSAMPLING İŞLEMİ SONRASI DURUM")
print("==========")
print(f"Dengelenmiş Veri Seti Boyutu (Satır, Sütun): {X_resampled.shape}")
print("Yeni Sınıf Dağılımı:")
print(y_resampled.value_counts())
print("==========\n")

# 4. Yeni dosyaları yine 'processed' klasörüne kaydediyoruz
X_resampled.to_csv("data/processed/batadal_X_train_random_scaled.csv", index=False)
y_resampled.to_csv("data/processed/batadal_y_train_random.csv", index=False)

print("Dengelenmiş veriler 'data/processed/' klasörüne başarıyla kaydedildi.")