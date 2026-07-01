import pandas as pd
from sklearn.decomposition import PCA
import os

def main():
    print("\n====== RUS VERİSİ İÇİN PCA İŞLEME BAŞLADI ======")
    
    #dengeli veri
    rus_path = "data/processed/batadal_X_train_random_scaled.csv"
    
    if not os.path.exists(rus_path):
        print(f"Hata: {rus_path} dosyası bulunamadı! Lütfen önce balance_rus.py çalıştır.")
        return
        
    X_train_rus = pd.read_csv(rus_path)
    print(f"Giriş Veri Boyutu: {X_train_rus.shape} (43 Sütun)")

    # 43 sütunu TEK bir ana bileşene (PC1) indirgiyoruz
    pca = PCA(n_components=1, random_state=42)
    X_train_pc1 = pca.fit_transform(X_train_rus)
    
    #yeni dosyaları dataframe olarak kaydetme
    df_pc1 = pd.DataFrame(X_train_pc1, columns=['PC1'])
    output_path = "data/processed/batadal_X_train_random_pc1.csv"
    df_pc1.to_csv(output_path, index=False)
    
    print(f"Sıkıştırma Bitti Yeni Boyut: {df_pc1.shape} (Tek Sütun)")
    print(f"Dosya şuraya yazıldı: {output_path}")
    print("===============================================\n")

if __name__ == "__main__":
    main()