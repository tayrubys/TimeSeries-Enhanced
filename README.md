# BATADAL - Moving Average Deneyi

Bu deneyde, LSTM ve GRU modellerinin ürettiği anomali olasılıklarına (**prediction probability**) Moving Average uygulanarak karar mekanizmasının iyileştirilip iyileştirilemeyeceği araştırılmıştır.

## Deney Akışı

```text
BATADAL Zaman Serisi Verisi
           │
           ▼
   LSTM / GRU Eğitimi  ───> (Sınıf ağırlıkları 'balanced' olarak ayarlanır)
           │
           ├──> [VALIDATION AKIŞI]
           │         │
           │         ├──> Ham Olasılıkları Hesapla (y_val_pred_prob)
           │         ├──> Moving Average Uygula (window = 3 / 5 / 7)
           │         └──> En İyi Threshold'u Seç (Optimum F1)
           │
           └──> [TEST AKIŞI]
                     │
                     ├──> Ham Test Olasılıklarını Hesapla (y_test_pred_prob)
                     ├──> Aynı Pencerede Moving Average Uygula
                     └──> Validation'da Seçilen Threshold'u Uygula
```

## Moving Average Nasıl Kullanıldı?

Model eğitimi tamamlandıktan sonra validation ve test kümeleri için üretilen **olasılık değerlerine** Moving Average uygulanmıştır.

Örneğin modelin ürettiği olasılıklar:

```
0.02
0.05
0.18
0.22
0.16
```

Window = 3 için Moving Average sonrası:

```
0.02
0.035
0.083
0.150
0.186
```

Daha sonra bu yumuşatılmış olasılıklar üzerinde en iyi threshold seçilmiş ve test sonuçları hesaplanmıştır.

## Deney Sonuçları

Aşağıdaki tabloda farklı Moving Average pencere boyutları (MA Window) için 5 farklı seed üzerinde elde edilen ortalama performans ve standart sapma değerleri verilmiştir.

| Model | MA Window | Accuracy (Ort ± Std) | Precision (Ort ± Std) | Recall (Ort ± Std) | F1-Score (Ort ± Std) |
|:------:|:---------:|:--------------------:|:----------------------:|:------------------:|:--------------------:|
| **LSTM** | **3** | 0.922 ± 0.049 | 0.494 ± 0.326 | 0.440 ± 0.445 | **0.447 ± 0.403** |
| **LSTM** | **5** | 0.926 ± 0.031 | 0.401 ± 0.371 | 0.573 ± 0.524 | **0.469 ± 0.430** |
| **LSTM** | **7** | 0.931 ± 0.031 | 0.598 ± 0.349 | 0.493 ± 0.443 | **0.475 ± 0.401** |
| **GRU** | **3** | 0.934 ± 0.031 | 0.446 ± 0.408 | 0.500 ± 0.465 | **0.470 ± 0.431** |
| **GRU** | **5** | 0.931 ± 0.036 | 0.592 ± 0.199 | 0.680 ± 0.384 | **0.617 ± 0.286** |
| **GRU** | **7** | 0.941 ± 0.033 | 0.594 ± 0.342 | 0.588 ± 0.451 | **0.564 ± 0.392** |

### Gözlemler

- LSTM modeli için pencere boyutu arttıkça ortalama F1 skorunda küçük bir artış gözlenmiştir.
- GRU modeli en yüksek ortalama F1 skorunu **MA Window = 5** ile (**0.617 ± 0.286**) elde etmiştir.
- Moving Average bazı seed'lerde performansı artırırken, bazı seed'lerde önemli performans kayıplarına neden olmuştur.
- Sonuçlar genel olarak değerlendirildiğinde, Moving Average yöntemi mevcut LSTM ve GRU modelleri üzerinde **kararlı ve tutarlı bir performans artışı sağlayamamıştır.**