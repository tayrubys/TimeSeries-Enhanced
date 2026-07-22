### BATADAL LSTM/GRU Seed Ensemble Deneyi

### 1. Çalışmanın Amacı

Bu deneyde BATADAL veri seti üzerinde LSTM ve GRU modellerinin rastgele başlangıç ağırlıklarına karşı kararlılığı incelenmiştir. Aynı model beş farklı seed ile eğitilmiş, her modelin validation ve test olasılıkları kaydedilmiş ve bu olasılıklar ortalanarak bir seed ensemble tahmini oluşturulmuştur.

Kullanılan seed değerleri:

```text
42, 123, 2026, 7, 999
```

Seed ensemble yaklaşımının amacı, tek bir modelin rastgele başlangıç ağırlıklarına bağlı olarak çok iyi veya çok kötü sonuç üretmesinin etkisini azaltmaktır.

### 2. Kullanılan Veri Dosyaları

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

### 3. Deney Akışı

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

### 4. Threshold Seçimi

Kod, aşağıdaki aralıkta threshold araması yapmaktadır:

```python
thresholds = np.arange(0.01, 0.51, 0.01)
```

Her threshold için validation F1-score hesaplanır ve en yüksek F1-score'u sağlayan değer seçilir. Seçilen threshold daha sonra test verisine uygulanır.

Threshold test etiketlerine göre seçilmediği için test verisinden doğrudan bilgi sızıntısı yapılmaz.

### 5. Seed Ensemble Mantığı

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

### 6. Deney Sonuçları

### 6.1. Beş seed ortalaması

| Model |        Accuracy |       Precision |          Recall |        F1-score |
| ----- | --------------: | --------------: | --------------: | --------------: |
| LSTM  | 0.9108 ± 0.0260 | 0.2889 ± 0.3104 | 0.3775 ± 0.5057 | 0.3072 ± 0.4010 |
| GRU   | 0.9345 ± 0.0289 | 0.6044 ± 0.1549 | 0.7625 ± 0.3398 | 0.6590 ± 0.2440 |

Bu tablo bağımsız modellerin metrik ortalamalarını göstermektedir; ensemble sonucu değildir.

### 6.2. Final seed ensemble sonuçları

| Model | Ensemble Size | Threshold | Validation F1 | Accuracy | Precision | Recall |    Test F1 |
| ----- | ------------: | --------: | ------------: | -------: | --------: | -----: | ---------: |
| LSTM  |             5 |      0.36 |        0.5000 |   0.9274 |    0.7083 | 0.4250 |     0.5313 |
| GRU   |             5 |      0.37 |        0.5714 |   0.9601 |    0.7582 | 0.8625 | **0.8070** |

GRU seed ensemble modeli, tekil GRU modellerinin ortalama F1-score'u olan `0.6590` değerinden daha yüksek bir `0.8070` F1-score elde etmiştir. Ayrıca precision ve recall arasında daha dengeli bir sonuç üretmiştir.

LSTM ensemble modeli, tekil LSTM modellerinin ortalama F1-score'unu yükseltmiş olsa da recall değeri `0.4250` seviyesinde kalmıştır. Bu deneyde en başarılı yöntem GRU seed ensemble olmuştur.

### 7. Tekrarlanabilirlik ve Deterministik Eğitim

`run_batadal_seed_experiments.py` dosyasında şu seed ayarları bulunmaktadır:

```python
np.random.seed(seed)
tf.random.set_seed(seed)
```

Bu ayarlar rastgeleliği azaltır ancak özellikle GPU üzerinde tam deterministik eğitim için tek başına yeterli olmayabilir.

Bu çalışmada farklı rastgele başlangıçlarla eğitilen modellerin sınıf etiketleri yerine olasılık çıktıları birleştirilmiştir. Beş seed ile oluşturulan GRU ensemble modeli `0.8070` F1-score elde ederek tekil GRU modellerinin ortalama sonucunu aşmıştır. Sonuçlar, seed ensemble yönteminin BATADAL veri setinde rastgele başlangıçlara bağlı kararsızlığın etkisini azaltmak için kullanılabileceğini göstermektedir.

Ancak seed ensemble her deneyde F1-score artışını garanti etmez. Yöntemin temel avantajı, kararın tek bir rastgele başlangıca bağlı kalmamasıdır. Gerçek kullanımda aynı ensemble tahminini üretmek için eğitilen beş modelin ağırlıkları saklanmalı ve yeni verilerin olasılıkları bu beş model üzerinden ortalanmalıdır.

---

### SKAB LSTM/GRU Seed Ensemble Deneyi

### 1. Çalışmanın Amacı

SKAB veri setinde LSTM ve GRU modellerinin rastgele başlangıçlara bağlı değişimini azaltmak amacıyla seed ensemble yöntemi uygulanmıştır. Deneyler 5-fold değerlendirme yapısıyla gerçekleştirilmiştir.

Her fold içerisinde aynı model 42 ve 123 olmak üzere iki farklı seed ile eğitilmiştir. Modellerin validation ve test kümeleri için ürettiği anomali olasılıkları örnek bazında ortalanmıştır. Ensemble karar eşiği, her fold'un validation kümesinde en yüksek F1-score'u sağlayan değere göre belirlenmiştir. Bu eşik, ilgili fold'un ortalama test olasılıklarına uygulanarak final tahminler elde edilmiştir.

Kullanılan seed değerleri:

```text
42, 123
```

### 2. Deney Akışı

```text
Her model için
    ↓
5 farklı fold
    ↓
Her fold içerisinde 2 farklı seed ile eğitim
    ↓
Seed modellerinin validation olasılıklarını ortalama
    ↓
Validation üzerinde ortak threshold belirleme
    ↓
Seed modellerinin test olasılıklarını ortalama
    ↓
Fold için final ensemble tahmini
    ↓
5 fold sonucunun ortalama ± standart sapması
```

Farklı fold'ların test örnekleri birbirinden farklı olduğu için fold'lar arasındaki olasılıklar doğrudan ortalanmamıştır. Her fold içerisinde ayrı bir ensemble sonucu hesaplanmış, ardından beş fold'un metrikleri ortalama ve standart sapma olarak raporlanmıştır.

### 3. Fold Bazlı Ensemble Sonuçları

### 3.1. LSTM Ensemble Sonuçları

| Fold | Threshold | Validation F1 | Accuracy | Precision | Recall | Test F1 |
| ---: | --------: | ------------: | -------: | --------: | -----: | ------: |
|    1 |      0.30 |        0.8772 |   0.8981 |    0.9634 | 0.7338 |  0.8331 |
|    2 |      0.35 |        0.9137 |   0.8510 |    0.7357 | 0.8886 |  0.8050 |
|    3 |      0.25 |        0.9048 |   0.8780 |    0.7713 | 0.9198 |  0.8390 |
|    4 |      0.50 |        0.9120 |   0.9422 |    1.0000 | 0.8421 |  0.9143 |
|    5 |      0.30 |        0.9481 |   0.9174 |    0.9852 | 0.7829 |  0.8725 |

### 3.2. GRU Ensemble Sonuçları

| Fold | Threshold | Validation F1 | Accuracy | Precision | Recall | Test F1 |
| ---: | --------: | ------------: | -------: | --------: | -----: | ------: |
|    1 |      0.45 |        0.8776 |   0.8891 |    0.9764 | 0.6969 |  0.8133 |
|    2 |      0.50 |        0.9144 |   0.9110 |    0.8841 | 0.8547 |  0.8692 |
|    3 |      0.25 |        0.9059 |   0.8449 |    0.7111 | 0.9282 |  0.8053 |
|    4 |      0.45 |        0.9146 |   0.9440 |    1.0000 | 0.8471 |  0.9172 |
|    5 |      0.45 |        0.9514 |   0.9212 |    0.9945 | 0.7860 |  0.8780 |

### 4. Genel Ensemble Sonuçları

| Model         |            Accuracy |           Precision |              Recall |            F1-score |
| ------------- | ------------------: | ------------------: | ------------------: | ------------------: |
| LSTM Ensemble |     0.8973 ± 0.0351 |     0.8911 ± 0.1269 | **0.8334 ± 0.0759** |     0.8528 ± 0.0419 |
| GRU Ensemble  | **0.9020 ± 0.0376** | **0.9132 ± 0.1223** |     0.8226 ± 0.0865 | **0.8566 ± 0.0469** |

### 5. Tekil Modeller ve Ensemble Karşılaştırması

| Model |   Tekil Seed F1 |     Ensemble F1 | Değişim |
| ----- | --------------: | --------------: | ------: |
| LSTM  | 0.8505 ± 0.0388 | 0.8528 ± 0.0419 | +0.0022 |
| GRU   | 0.8586 ± 0.0438 | 0.8566 ± 0.0469 | -0.0020 |

### 6. Sonuçların Değerlendirilmesi

Seed ensemble sonucunda LSTM modeli `0.8528 ± 0.0419`, GRU modeli ise `0.8566 ± 0.0469` F1-score elde etmiştir. En yüksek ortalama F1-score GRU ensemble modeliyle elde edilmiştir. Bununla birlikte tekil modellerle ensemble modelleri arasındaki farkın oldukça düşük olduğu görülmüştür.

LSTM modelinde ensemble kullanımı F1-score'u `0.8505` değerinden `0.8528` değerine yükselterek küçük bir iyileşme sağlamıştır. GRU modelinde ise tekil modellerin ortalama F1-score'u `0.8586` iken ensemble sonucu `0.8566` olmuştur. Bu nedenle seed ensemble yaklaşımı SKAB veri setinde performansı belirgin şekilde artırmamış, ancak modellerin kararlarını tek bir rastgele başlangıç yerine birden fazla modelin ortak çıktısına dayandırmıştır.

Sonuçlar, SKAB veri setindeki modellerin başlangıçta görece kararlı olduğunu ve seed ensemble yönteminin BATADAL deneyindeki kadar büyük bir katkı sağlamadığını göstermektedir. Ensemble yönteminin etkisini daha güvenilir değerlendirmek için deneyin daha fazla seed ile tekrarlanması planlanmaktadır.

### 7. Seed Sayısı ve Deney Maliyeti

Bu deney 5-seed değil, 2-seed ensemble olarak gerçekleştirilmiştir. Her model için:

```text
2 seed × 5 fold = 10 eğitim
```

LSTM ve GRU birlikte değerlendirildiğinde toplam eğitim sayısı:

```text
2 model × 2 seed × 5 fold = 20 eğitim
```

Gerçek 5-seed ensemble deneyi için seed listesi aşağıdaki şekilde genişletilebilir:

```python
"seeds": [42, 123, 2026, 7, 999]
```

Bu durumda toplam `2 model × 5 seed × 5 fold = 50` eğitim gerçekleştirilir.