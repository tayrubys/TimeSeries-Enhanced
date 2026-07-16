# BATADAL GRU Prediction-Based Anomaly Detection Deneyi

Bu çalışmada BATADAL veri seti üzerinde **GRU tabanlı tahmin odaklı anomali tespiti** gerçekleştirilmiştir. LSTM AutoEncoder yaklaşımından farklı olarak model, giriş sequence'ini yeniden oluşturmak yerine geçmiş zaman adımlarını kullanarak bir sonraki sensör vektörünü tahmin etmektedir.

Model yalnızca normal etiketli örneklerle eğitilmiştir. Gerçek sensör değerleri ile model tarafından tahmin edilen değerler arasındaki fark **prediction error** olarak hesaplanmış ve validation kümesi üzerinde belirlenen threshold kullanılarak anomaliler tespit edilmiştir.

## Deney Akışı

```text
BATADAL verisi
      ↓
Zamana göre Train / Validation / Test ayrımı
      ↓
Ölçeklendirilmiş sensör değerleri
      ↓
Prediction sequence oluşturma
(window = 10)
      ↓
Sadece normal örneklerle GRU eğitimi
      ↓
Bir sonraki sensör vektörünün tahmini
      ↓
Sensör bazlı karesel prediction error
      ↓
Maksimum sensör hatasının seçilmesi
      ↓
Validation üzerinde percentile threshold optimizasyonu
      ↓
Test değerlendirmesi
```

## Prediction Sequence Yapısı

Model, geçmiş 10 zaman adımını kullanarak bir sonraki zaman adımındaki 43 sensör değerini tahmin etmektedir.

```text
X[t-10], X[t-9], ..., X[t-1]
                ↓
               GRU
                ↓
             X̂[t]
```

Burada:

- `X[t-10:t]`: Modele verilen geçmiş zaman penceresi
- `X[t]`: Tahmin edilmek istenen gerçek sensör vektörü
- `X̂[t]`: GRU modeli tarafından üretilen sensör tahmini

Prediction sequence boyutu aşağıdaki yapıdadır:

```text
(örnek sayısı, 10 zaman adımı, 43 özellik)
```

## Model Yapısı

- Model: GRU Predictor
- GRU katmanı: 64 unit
- Dense katmanı: 32 unit
- İlk Dropout: 0.3
- İkinci Dropout: 0.2
- Çıkış katmanı: 43 unit, linear aktivasyon
- Loss: Mean Squared Error
- Optimizer: Adam
- Gradient Clipping: `clipnorm=1.0`
- Batch size: 32
- Maksimum epoch: 50
- Sequence window size: 10
- Özellik sayısı: 43
- Early Stopping kullanılmıştır.

## Prediction Error Hesaplaması

Model tarafından tahmin edilen sensör değerleri ile gerçek sensör değerleri arasındaki karesel fark hesaplanmıştır:

```math
e_j =
\left(
X_j-\hat{X}_j
\right)^2
```

Burada:

- \(X_j\): `j` sensörünün gerçek değeri
- \(\hat{X}_j\): `j` sensörü için model tahmini
- \(e_j\): İlgili sensörün karesel tahmin hatasıdır.

Her örnek için 43 ayrı sensör hatası elde edilmektedir. Bu deneyde genel anomali skoru olarak sensör hatalarının maksimumu kullanılmıştır:

```math
\text{Prediction Error}
=
\max_{j=1,\ldots,43}
\left(
X_j-\hat{X}_j
\right)^2
```

Kod karşılığı:

```python
squared_errors = np.square(
    X_targets - X_predictions
)

errors = np.max(
    squared_errors,
    axis=1
)
```

Maksimum hata yönteminin kullanılmasının amacı, yalnızca bir veya birkaç sensörde oluşan güçlü sapmaların diğer sensörlerin düşük hataları tarafından bastırılmasını engellemektir.

## Threshold Seçimi

Validation kümesindeki prediction error değerleri kullanılarak percentile tabanlı threshold optimizasyonu gerçekleştirilmiştir.

```math
\text{Prediction Error} \ge \text{Threshold}
\Rightarrow \text{Anomaly}
```

Her seed için validation kümesinde en yüksek F1-score değerini sağlayan percentile ve threshold seçilmiş, ardından bu threshold değiştirilmeden test kümesine uygulanmıştır.

## Seed Bazlı Test Sonuçları

| Seed | Percentile | Threshold | Accuracy | Precision | Recall | F1-Score |
|-----:|-----------:|----------:|---------:|----------:|-------:|---------:|
| 42 | 98.0 | 9.8334 | 0.9467 | 0.6875 | 0.8250 | 0.7500 |
| 123 | 98.0 | 10.7950 | 0.9431 | 0.6701 | 0.8125 | 0.7345 |
| 2026 | 99.0 | 21.5789 | 0.9467 | **0.7045** | 0.7750 | 0.7381 |
| 7 | 98.0 | 11.2778 | **0.9504** | 0.6893 | **0.8875** | **0.7760** |
| 999 | 98.0 | 10.5469 | 0.9431 | 0.6701 | 0.8125 | 0.7345 |

## Ortalama Sonuçlar

| Metrik | Ortalama ± Standart Sapma |
|---|---:|
| Accuracy | **0.9460 ± 0.0030** |
| Precision | **0.6843 ± 0.0146** |
| Recall | **0.8225 ± 0.0409** |
| F1-Score | **0.7466 ± 0.0176** |
| Threshold | 12.8064 ± 4.9316 |

## Değerlendirme

GRU Predictor modeli, 5 farklı seed sonucunda ortalama **0.7466 F1-score** elde etmiştir. F1-score standart sapmasının yalnızca **0.0176** olması, modelin rastgele başlangıç ağırlıklarına karşı oldukça kararlı çalıştığını göstermektedir.

Test sonuçlarında:

- Accuracy değerinin yaklaşık `%94.60`,
- Precision değerinin yaklaşık `%68.43`,
- Recall değerinin yaklaşık `%82.25`

olduğu görülmüştür.

Recall değerinin precision değerinden yüksek olması, modelin saldırı örneklerinin büyük bölümünü yakalayabildiğini ancak bazı normal örnekleri de anomali olarak işaretlediğini göstermektedir. 

## Validation ve Test Sonuçları Arasındaki Fark

Validation F1-score değerlerinin test F1-score değerlerinden oldukça düşük olduğu görülmüştür. Örneğin validation F1-score değerleri yaklaşık `0.1481–0.1739` aralığında kalırken test F1-score değerleri `0.7345–0.7760` aralığında gerçekleşmiştir.

Bu fark aşağıdaki nedenlerden kaynaklanabilir:

- Validation ve test saldırılarının farklı davranışlara sahip olması,
- Prediction error dağılımlarının veri bölümleri arasında değişmesi,
- Validation kümesindeki anomali sayısının ve saldırı türlerinin sınırlı olması,
- Percentile tabanlı threshold'un validation kümesine tam olarak genellenememesi.

Threshold standart sapmasının yüksek olması da farklı seed değerlerinde prediction error ölçeğinin değişebildiğini göstermektedir.

## LSTM AutoEncoder ile Karşılaştırma

| Yaklaşım | Window | F1 Mean | F1 Std |
|---|---:|---:|---:|
| LSTM AutoEncoder | 20 | **0.7730** | 0.0217 |
| GRU Predictor | 10 | 0.7466 | **0.0176** |

GRU Predictor modeli, LSTM AutoEncoder modelinden daha düşük ortalama F1-score üretmiştir. Bununla birlikte GRU Predictor deneyinin standart sapması daha düşük gerçekleşmiş ve model seed değişimlerine karşı daha kararlı sonuçlar vermiştir.

Bu sonuç, reconstruction-based ve prediction-based yaklaşımların farklı anomali davranışlarını yakalayabildiğini göstermektedir.

## Genel Sonuç

GRU tabanlı prediction-based anomaly detection yaklaşımı, BATADAL veri setinde ortalama **0.7466 ± 0.0176 F1-score** elde etmiştir. Model, geçmiş 10 zaman adımını kullanarak bir sonraki sensör değerlerini tahmin etmiş ve sensörler arasındaki maksimum karesel tahmin hatasını anomali skoru olarak kullanmıştır.

Elde edilen düşük F1 standart sapması, yaklaşımın kararlı olduğunu göstermektedir. Buna karşılık validation ve test performansları arasındaki fark, threshold seçimi ve veri bölme stratejisinin ilerleyen çalışmalarda daha ayrıntılı incelenmesi gerektiğini ortaya koymuştur.