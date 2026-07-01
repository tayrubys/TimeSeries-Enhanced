import pandas as pd
from imblearn.over_sampling import ADASYN
from sklearn.decomposition import PCA

X_train = pd.read_csv('data/processed/batadal_X_train_scaled.csv')
y_train = pd.read_csv('data/processed/batadal_y_train.csv')

print("=" * 10)
print("İŞLEM ÖNCESİ DURUM")
print("=" * 10)
print(f"Orijinal Veri Seti Boyutu (Satır, Sütun): {X_train.shape}")
print("Orijinal Sınıf Dağılımı:")
print(y_train.value_counts())
print("\n" + "=" * 10)

ada = ADASYN(random_state=42)
X_train_adasyn, y_train_adasyn = ada.fit_resample(X_train, y_train.values.ravel())

print("ADASYN İŞLEMİ SONRASI DURUM")
print("=" * 10)
print(f"Dengelenmiş Veri Seti Boyutu (Satır, Sütun): {X_train_adasyn.shape}")
print("Yeni Sınıf Dağılımı:")
print(pd.Series(y_train_adasyn).value_counts())
print("=" * 10)

X_train_adasyn_df = pd.DataFrame(X_train_adasyn, columns=X_train.columns)
y_train_adasyn_df = pd.DataFrame(y_train_adasyn, columns=y_train.columns)

X_train_adasyn_df.to_csv('data/processed/batadal_X_train_adasyn.csv', index=False)
y_train_adasyn_df.to_csv('data/processed/batadal_y_train_adasyn.csv', index=False)

print("Dengelenmiş veriler 'data/processed/' klasörüne kaydedildi.\n")

print("Otomata için PCA:")
pca = PCA(n_components=1, random_state=42)
X_train_adasyn_pc1 = pca.fit_transform(X_train_adasyn)

X_train_adasyn_df = pd.DataFrame(X_train_adasyn, columns=X_train.columns)
y_train_adasyn_df = pd.DataFrame(y_train_adasyn, columns=y_train.columns)

X_train_adasyn_pc1_df = pd.DataFrame(X_train_adasyn_pc1, columns=['PC1'])

X_train_adasyn_df.to_csv('data/processed/batadal_X_train_adasyn.csv', index=False)
y_train_adasyn_df.to_csv('data/processed/batadal_y_train_adasyn.csv', index=False)
X_train_adasyn_pc1_df.to_csv('data/processed/batadal_X_train_adasyn_pc1.csv', index=False)

print("=" * 50)
print("Standart ADASYN ve Otomata için ADASYN_PC1 dosyaları kaydedildi")
print("=" * 50 + "\n")