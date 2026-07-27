# BATADAL LSTM/GRU Seed Ensemble Deneyi

**Amaç:** LSTM ve GRU modellerinin rastgele başlangıç ağırlıklarına karşı kararlılığını incelemek. Her model 5 farklı seed (`42, 123, 2026, 7, 999`) ile eğitilip, validation/test olasılıkları örnek bazında ortalanarak seed ensemble oluşturuldu.

**Yöntem:** Her seed için model eğitilir → validation olasılıklarında `0.01–0.51` aralığında F1'i maksimize eden threshold seçilir → bu threshold test setine uygulanır → sonuçlar kaydedilir. Ensemble için: 5 modelin validation olasılıkları ortalanır, ortak threshold validation üzerinden seçilir (test'e sızıntı yok), sonra 5 modelin test olasılıkları ortalanıp bu threshold uygulanır (`ensemble_prob = probability_matrix.mean(axis=0)`).

Deney iki ön işleme varyantıyla tekrarlandı; mimari, seed listesi ve sequence boyutu (10) aynı, sadece scaler farklı.

## (a) Robust-Scaler + ADASYN

**Tekil seed ortalaması:**

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| LSTM | 0.9108±0.0260 | 0.2889±0.3104 | 0.3775±0.5057 | 0.3072±0.4010 |
| GRU | 0.9345±0.0289 | 0.6044±0.1549 | 0.7625±0.3398 | 0.6590±0.2440 |

**Ensemble sonucu:**

| Model | Threshold | Val F1 | Accuracy | Precision | Recall | Test F1 |
|---|---:|---:|---:|---:|---:|---:|
| LSTM | 0.36 | 0.5000 | 0.9274 | 0.7083 | 0.4250 | 0.5313 |
| GRU | 0.37 | 0.5714 | 0.9601 | 0.7582 | 0.8625 | **0.8070** |

→ GRU ensemble, tekil GRU ortalamasını (0.6590) belirgin şekilde aşarak bu varyantın en iyi sonucu oldu; LSTM'de recall zayıf kaldı (0.4250).

## (b) StandardScaler + ADASYN

**Tekil seed ortalaması:**

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| LSTM | 0.9216±0.0265 | 0.4492±0.2987 | 0.5375±0.4844 | 0.4479±0.3913 |
| GRU | 0.9202±0.0219 | 0.5476±0.2123 | 0.4025±0.3032 | 0.4390±0.2911 |

**Ensemble sonucu:**

| Model | Threshold | Val F1 | Accuracy | Precision | Recall | Test F1 |
|---|---:|---:|---:|---:|---:|---:|
| LSTM | 0.12 | 0.4444 | 0.9262 | 0.5736 | **0.9250** | **0.7081** |
| GRU | 0.41 | 0.5135 | 0.9057 | 0.5313 | 0.2125 | 0.3036 |

## Tekrarlanabilirlik

`run_batadal_seed_experiments.py` içinde `np.random.seed(seed)` ve `tf.random.set_seed(seed)` ayarlanır; bu, rastgeleliği azaltır ama özellikle GPU'da tam determinizm garantilemez.

## Sonuç

Sınıf etiketleri değil olasılık çıktıları birleştirildi. GRU seed ensemble (Robust-Scaler varyantı) 0.8070 F1 ile tekil ortalamayı geçti. Seed ensemble F1 artışını garanti etmez, ancak kararı tek bir rastgele başlangıca bağlı kalmaktan kurtarır. Gerçek kullanımda 5 modelin ağırlıkları saklanmalı, yeni veri bu 5 model üzerinden ortalanmalıdır.

---

# SKAB LSTM/GRU Seed Ensemble Deneyi

**Amaç:** Aynı yöntem SKAB veri setine uygulandı; eğitim süresini kısaltmak için yalnızca **Fold 1** kullanıldı (5-fold ortalaması değildir — geçici sonuçtur, final değerlendirmede 5 fold'a genişletilecek). Seed'ler aynı (`42, 123, 2026, 7, 999`), sequence pencere boyutu = 20.

**Akış:** Fold 1 verisi yüklenir → LSTM/GRU 5 seed ile eğitilir → val/test olasılıkları kaydedilir → val olasılıkları ortalanır → ortak threshold (0.01–0.50 aralığında F1 maksimize edilerek) seçilir → test olasılıkları ortalanıp threshold uygulanır.

## Tekil Seed Sonuçları

**LSTM:**

| Seed | Threshold | Accuracy | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|---:|
| 42 | 0.24 | 0.8877 | 0.9677 | 0.6995 | 0.8120 |
| 123 | 0.19 | 0.8866 | 0.9529 | 0.7079 | 0.8123 |
| 2026 | 0.18 | 0.8929 | 0.9611 | 0.7202 | 0.8234 |
| 7 | 0.37 | 0.8709 | 0.9515 | 0.6613 | 0.7803 |
| 999 | 0.26 | 0.8875 | 0.9408 | 0.7209 | 0.8163 |

**GRU:**

| Seed | Threshold | Accuracy | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|---:|
| 42 | 0.32 | 0.8716 | 0.9774 | 0.6444 | 0.7767 |
| 123 | 0.45 | 0.8709 | 0.9792 | 0.6412 | 0.7750 |
| 2026 | 0.30 | 0.8882 | 0.9789 | 0.6924 | 0.8111 |
| 7 | 0.30 | 0.8922 | 0.9767 | 0.7060 | 0.8195 |
| 999 | 0.43 | 0.8635 | 0.9795 | 0.6192 | 0.7587 |

## 5 Seed Ortalaması (Ensemble Değil)

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| LSTM | 0.8851±0.0083 | 0.9548±0.0102 | 0.7019±0.0244 | 0.8089±0.0166 |
| GRU | 0.8773±0.0123 | 0.9784±0.0012 | 0.6606±0.0368 | 0.7882±0.0259 |

## Fold 1 Ensemble Sonucu

| Model | Threshold | Val F1 | Accuracy | Precision | Recall | Test F1 |
|---|---:|---:|---:|---:|---:|---:|
| LSTM | 0.28 | 0.8729 | 0.8929 | 0.9734 | 0.7105 | **0.8214** |
| GRU | 0.23 | 0.8767 | 0.8920 | 0.9758 | 0.7060 | 0.8192 |

## Tekil Ortalama vs. Ensemble

| Model | Tekil F1 Ort. | Ensemble F1 | Değişim |
|---|---:|---:|---:|
| LSTM | 0.8089 | 0.8214 | +0.0125 |
| GRU | 0.7882 | 0.8192 | +0.0310 |

## Değerlendirme

İki model birbirine çok yakın sonuç verdi (LSTM 0.8214 vs GRU 0.8192); LSTM Accuracy/Recall/F1'de, GRU Precision/Val-F1'de hafifçe önde. Ensemble her iki modelde de tekil ortalamayı geçti.
