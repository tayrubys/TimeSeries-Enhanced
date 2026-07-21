# BATADAL LSTM/GRU Seed Ensemble Deneyi

## 1. Çalışmanın Amacı

Bu deneyde BATADAL veri seti üzerinde LSTM ve GRU modellerinin rastgele başlangıç ağırlıklarına karşı kararlılığı incelenmiştir. Aynı model beş farklı seed ile eğitilmiş, her modelin validation ve test olasılıkları kaydedilmiş ve bu olasılıklar ortalanarak bir seed ensemble tahmini oluşturulmuştur.

Kullanılan seed değerleri:

```text
42, 123, 2026, 7, 999
```

Seed ensemble yaklaşımının amacı, tek bir modelin rastgele başlangıç ağırlıklarına bağlı olarak çok iyi veya çok kötü sonuç üretmesinin etkisini azaltmaktır.

## 2. Kullanılan Veri Dosyaları

Kod, önceden hazırlanmış boyutu 20 olan sequence verilerini aşağıdaki klasörden yüklemektedir:

```text
data2/processed/robust_adasyn/
├── batadal_X_train_seq.npy
├── batadal_y_train_seq.npy
├── batadal_X_val_seq.npy
├── batadal_y_val_seq.npy
├── batadal_X_test_seq.npy
└── batadal_y_test_seq.npy
```

## 3. Deney Akışı

Her model için aşağıdaki işlemler gerçekleştirilir:

1. Train, validation ve test sequence dosyaları yüklenir.
2. LSTM veya GRU modeli oluşturulur.
3. Model beş farklı seed ile ayrı ayrı eğitilir.
4. Her seed için validation olasılıkları hesaplanır.
5. Validation kümesinde en yüksek F1-score'u sağlayan threshold seçilir.
6. Seçilen threshold test olasılıklarına uygulanır.
7. Her seed'in Accuracy, Precision, Recall ve F1-score sonuçları kaydedilir.
8. Beş modelin validation olasılıkları örnek bazında ortalanır.
9. Ensemble için ortak threshold yalnızca validation kümesinde seçilir.
10. Beş modelin test olasılıkları örnek bazında ortalanır ve ortak threshold uygulanır.
11. Tek bir final ensemble tahmini ve ensemble test sonucu üretilir.

Akışın özeti:

```text
5 farklı seed ile eğitim
        ↓
Her modelden validation ve test olasılığı
        ↓
Validation olasılıklarının ortalaması
        ↓
Validation üzerinde ensemble threshold seçimi
        ↓
Test olasılıklarının ortalaması
        ↓
Final 0/1 ensemble tahmini
        ↓
Accuracy, Precision, Recall ve F1-score
```

## 4. Threshold Seçimi

Kod, aşağıdaki aralıkta threshold araması yapmaktadır:

```python
thresholds = np.arange(0.01, 0.51, 0.01)
```

Her threshold için validation F1-score hesaplanır ve en yüksek F1-score'u sağlayan değer seçilir. Seçilen threshold daha sonra test verisine uygulanır.

Threshold test etiketlerine göre seçilmediği için test verisinden doğrudan bilgi sızıntısı yapılmaz.

## 5. Seed Ensemble Mantığı

Ensemble işleminde seed'lerin F1-score değerleri ortalanmaz. Her test örneği için beş modelin ürettiği anomali olasılıkları ortalanır.

Örneğin aynı test sequence'i için modeller şu olasılıkları üretmiş olsun:

```text
Seed 42   → 0.30
Seed 123  → 0.18
Seed 2026 → 0.25
Seed 7    → 0.40
Seed 999  → 0.32
```

Ensemble olasılığı:

```text
(0.30 + 0.18 + 0.25 + 0.40 + 0.32) / 5 = 0.29
```

Bu işlem tüm test örnekleri için ayrı ayrı gerçekleştirilir. Daha sonra validation kümesinden seçilen ortak threshold kullanılarak final sınıf tahminleri oluşturulur.

Kod içerisindeki temel işlem:

```python
ensemble_val_prob = val_probability_matrix.mean(axis=0)
ensemble_test_prob = test_probability_matrix.mean(axis=0)
```
## 6. Deney Sonuçları

### 6.1. Beş seed ortalaması

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| LSTM | 0.9108 ± 0.0260 | 0.2889 ± 0.3104 | 0.3775 ± 0.5057 | 0.3072 ± 0.4010 |
| GRU | 0.9345 ± 0.0289 | 0.6044 ± 0.1549 | 0.7625 ± 0.3398 | 0.6590 ± 0.2440 |

Bu tablo bağımsız modellerin metrik ortalamalarını göstermektedir; ensemble sonucu değildir.

### 6.2. Final seed ensemble sonuçları

| Model | Ensemble Size | Threshold | Validation F1 | Accuracy | Precision | Recall | Test F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| LSTM | 5 | 0.36 | 0.5000 | 0.9274 | 0.7083 | 0.4250 | 0.5313 |
| GRU | 5 | 0.37 | 0.5714 | 0.9601 | 0.7582 | 0.8625 | **0.8070** |

GRU seed ensemble modeli, tekil GRU modellerinin ortalama F1-score'u olan `0.6590` değerinden daha yüksek bir `0.8070` F1-score elde etmiştir. Ayrıca precision ve recall arasında daha dengeli bir sonuç üretmiştir.

LSTM ensemble modeli, tekil LSTM modellerinin ortalama F1-score'unu yükseltmiş olsa da recall değeri `0.4250` seviyesinde kalmıştır. Bu deneyde en başarılı yöntem GRU seed ensemble olmuştur.

## 7. Tekrarlanabilirlik ve Deterministik Eğitim

`run_batadal_seed_experiments.py` dosyasında şu seed ayarları bulunmaktadır:

```python
np.random.seed(seed)
tf.random.set_seed(seed)
```

Bu ayarlar rastgeleliği azaltır ancak özellikle GPU üzerinde tam deterministik eğitim için tek başına yeterli olmayabilir.


Bu çalışmada farklı rastgele başlangıçlarla eğitilen modellerin sınıf etiketleri yerine olasılık çıktıları birleştirilmiştir. Beş seed ile oluşturulan GRU ensemble modeli `0.8070` F1-score elde ederek tekil GRU modellerinin ortalama sonucunu aşmıştır. Sonuçlar, seed ensemble yönteminin BATADAL veri setinde rastgele başlangıçlara bağlı kararsızlığın etkisini azaltmak için kullanılabileceğini göstermektedir.

Ancak seed ensemble her deneyde F1-score artışını garanti etmez. Yöntemin temel avantajı, kararın tek bir rastgele başlangıca bağlı kalmamasıdır. Gerçek kullanımda aynı ensemble tahminini üretmek için eğitilen beş modelin ağırlıkları saklanmalı ve yeni verilerin olasılıkları bu beş model üzerinden ortalanmalıdır.