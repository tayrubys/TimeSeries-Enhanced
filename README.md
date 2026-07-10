# BATADAL CNN-LSTM AutoEncoder Deneyi
Literatürde kullanılan hibrit mimarilerden esinlenerek,buçalışmada BATADAL veri seti üzerinde **CNN-LSTM AutoEncoder tabanlı anomali tespiti** gerçekleştirilmiştir. Amaç, Conv1D katmanı kullanarak sensör verilerindeki yerel örüntüleri çıkarmak ve bu özellikleri LSTM ile zamansal olarak modelleyerek anomali tespit performansını artırmaktır.

---

# Deney Akışı

```text
BATADAL Verisi
        ↓
Sequence Oluşturma
(window = 20)
        ↓
Uç Değer Kırpma
(np.clip [-5, 5])
        ↓
Sadece Normal Sequence'lerle Eğitim
        ↓
CNN-LSTM AutoEncoder
        ↓
Reconstruction Error Hesabı
        ↓
Validation Threshold Optimizasyonu
        ↓
Test Değerlendirmesi
```

---

# Model Mimarisi

```text
Sequence
    ↓
Conv1D
    ↓
LSTM Encoder
    ↓
Dense Darboğaz Katmanı
    ↓
RepeatVector
    ↓
LSTM Decoder
    ↓
TimeDistributed Dense
    ↓
Reconstruction Error
```

---

# Model Parametreleri

| Parametre | Değer |
|------------|--------|
| Window Size | 20 |
| Conv1D Filters | 32 |
| Kernel Size | 3 |
| LSTM Units | 64 |
| Latent Dimension | 32 |
| Dropout | 0.30 |
| Batch Size | 32 |
| Epoch | 50 |
| Optimizer | Adam |
| Loss Function | MSE |

---

# Reconstruction Error

Model yalnızca normal sequence'leri öğrenmektedir. Test sırasında bir sequence'in yeniden üretim hatası aşağıdaki şekilde hesaplanmaktadır:

```math
Reconstruction\ Error =
\frac{1}{N}
\sum_{i=1}^{N}
(X_i-\hat{X}_i)^2
```

Kod karşılığı:

```python
errors = np.mean(
    np.square(X - X_pred),
    axis=(1,2)
)
```

Threshold üzerindeki sequence'ler anomali olarak işaretlenmektedir.

---

# Uç Değer İşlemi

Bazı sensörlerde oldukça yüksek değerler gözlemlenmiştir.

Bu nedenle veri aşağıdaki aralıkta sınırlandırılmıştır:

```python
X_train = np.clip(X_train, -5, 5)
X_val   = np.clip(X_val, -5, 5)
X_test  = np.clip(X_test, -5, 5)
```

Bu işlem reconstruction error dağılımını daha kararlı hale getirmiştir.

---

# Seed Bazlı Sonuçlar

| Seed | Validation F1 | Threshold |
|------|---------------|------------|
| 42 | 0.6596 | 0.4887 |
| 123 | **0.7021** | 0.5012 |
| 2026 | 0.6465 | 0.5092 |
| 7 | 0.6667 | 0.5167 |
| 999 | 0.6882 | 0.5386 |

---

# Ortalama Sonuçlar

| Metrik | Ortalama ± Std |
|---------|----------------|
| Accuracy | 0.8632 ± 0.0132 |
| Precision | 0.4170 ± 0.0230 |
| Recall | **0.9825 ± 0.0190** |
| F1-Score | **0.5851 ± 0.0223** |
| Threshold | 0.5109 ± 0.0186 |

---

# Değerlendirme

CNN katmanı eklenmesi ile model saldırı örneklerini oldukça yüksek oranda yakalayabilmiştir (**Recall ≈ 0.98**). Ancak precision değerinde düşüş meydana gelmiş ve model daha fazla false positive üretmiştir.

Önceki LSTM AutoEncoder deneyinde:

```text
F1 ≈ 0.77
```

elde edilirken, CNN-LSTM AutoEncoder modelinde:

```text
F1 ≈ 0.59
```

elde edilmiştir.

Bu durum, mevcut veri boyutu ve pencere uzunluğu için Conv1D katmanının ek karmaşıklık oluşturduğu ve reconstruction error ayrımını zayıflattığını göstermektedir.

---

# Sonraki Çalışmalar

Planlanan deneyler:

### 1) Daha Hafif CNN-LSTM AutoEncoder

```text
Conv Filters : 16
LSTM Units   : 32
Latent Dim   : 16
Dropout      : 0.20
```

Amaç:

- model karmaşıklığını azaltmak
- overfitting riskini düşürmek

---

### 2) Farklı Window Boyutları

```text
window = 10
window = 30
window = 40
```

farklı zaman ufuklarının etkisi incelenecektir.

---

### 3) Threshold Aralığının Genişletilmesi

Mevcut:

```text
93-99 percentile
```

Yeni:

```text
90-99.9 percentile
```
