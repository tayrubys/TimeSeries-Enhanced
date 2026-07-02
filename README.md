## BATADAL — LSTM Modeli, ADASYN ile Dengeleme (Ön Sonuçlar)
 
Bu bölümde, BATADAL train setinin ADASYN ile (pencere seviyesinde) dengelenmesi sonrası LSTM modelinin performansı raporlanmaktadır. Eğitim, 5 farklı random seed (`42, 123, 2026, 7, 999`) ile tekrarlanmış, her seed için validation setinde F1-optimal eşik (threshold) bulunup bu eşik test setinde uygulanmıştır.
 
### Seed Bazlı Sonuçlar (Test Kümesi)
 
| Seed | Threshold | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|---|
| 42 | 0.2 | 0.8984 | 0.0000 | 0.0000 | 0.0000 |
| 123 | 0.1 | 0.9523 | 0.7158 | 0.8500 | 0.7771 |
| 2026 | 0.4 | 0.9755 | 0.8750 | 0.8750 | 0.8750 |
| 7 | 0.1 | 0.9535 | 0.6981 | 0.9250 | 0.7957 |
| 999 | 0.3 | 0.9694 | 0.8986 | 0.7750 | 0.8322 |
 
### Ortalama ± Standart Sapma
 
**Tüm seed'ler (5 seed):**
 
| Metrik | Ortalama ± Std |
|---|---|
| Accuracy | 0.9498 ± 0.0272 |
| Precision | 0.6375 ± 0.3289 |
| Recall | 0.6850 ± 0.3459 |
| F1-score | 0.6560 ± 0.3297 |
 
**Seed=42 hariç (4 seed):**
 
| Metrik | Ortalama ± Std |
|---|---|
| Accuracy | 0.9627 ± 0.0100 |
| Precision | 0.7969 ± 0.0905 |
| Recall | 0.8563 ± 0.0541 |
| F1-score | 0.8200 ± 0.0374 |
 
>  **LSTM Özeti:** 5 seed'in 4'ünde (123, 2026, 7, 999) LSTM iyi sonuç verdi (F1: 0.78–0.88). Ama seed=42'de model test setinde hiçbir anomaliyi bulamadı (precision, recall, F1 hepsi 0). Validation'da bu seed için sonuç fena değildi (F1≈0.545), ama test'e geçince tamamen çöktü.


---

## BATADAL — GRU Modeli, ADASYN ile Dengeleme (Ön Sonuçlar)

Bu bölümde, aynı deneysel protokol (pencere seviyesinde ADASYN, class_weight=0 ve aynı 5 random seed) takip edilerek eğitilen GRU modelinin performans metrikleri yer almaktadır.

### Seed Bazlı Sonuçlar (Test Kümesi)

| Seed | Threshold | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|---|
| 42 | 0.5 | 0.9021 | 0.0000 | 0.0000 | 0.0000 |
| 123 | 0.4 | 0.9058 | 0.6364 | 0.0875 | 0.1538 |
| 2026 | 0.1 | 0.9621 | 0.8267 | 0.7750 | 0.8000 |
| 7 | 0.1 | 0.9094 | 0.6875 | 0.1375 | 0.2292 |
| 999 | 0.3 | 0.9351 | 0.7368 | 0.5250 | 0.6131 |

### Ortalama ± Standart Sapma

**Tüm seed'ler (5 seed):**

| Metrik | Ortalama ± Std |
|---|---|
| Accuracy | 0.9229 ± 0.0255 |
| Precision | 0.5775 ± 0.3304 |
| Recall | 0.3050 ± 0.3308 |
| F1-score | 0.3592 ± 0.3343 |

**Seed=42 hariç (4 seed):**

| Metrik | Ortalama ± Std |
|---|---|
| Accuracy | 0.9281 ± 0.0263 |
| Precision | 0.7218 ± 0.0805 |
| Recall | 0.3813 ± 0.3204 |
| F1-score | 0.4490 ± 0.3039 |

>  **GRU Özeti:** GRU modeli genel olarak başlangıç ağırlıklarına (seed) karşı çok daha agresif bir hassasiyet sergilemiştir. LSTM'de olduğu gibi `seed=42` durumunda test kümesinde tamamen sıfır çekmiştir. Bununla birlikte, `seed=123` ve `seed=7` senaryolarında da model anomalileri yakalamakta (Recall) ciddi direnç göstermiş, yalnızca `seed=2026` altında güçlü bir genelleme başarısı (F1: 0.80) yakalayabilmiştir.
---
## BATADAL — LSTM ve GRU Parametre Optimizasyonu (Pencere Boyutu Deneyleri)

### 1. pencere boyutu = 40
Bu bölümde, ADASYN ile dengelenmiş BATADAL veri setinde `sequence_window_size` parametresi 20'den 40'a çıkarılarak, modelin daha geniş bir zaman alanını baz alması sağlanmış ve veri artırımı bu doğrultuda yenilenmiştir. Elde edilen sonuçlar varsayılan (baseline) model ile kıyaslanarak performans değişimleri test edilmiştir. Sınıf ağırlıkları (`class_weight`) sıfır tutulmuş ve eğitim parametrelerinin etkisini dürüst gözlemlemek adına süreç aynı 5 rastgele seed (`42, 123, 2026, 7, 999`) ile tekrarlanmıştır.

### Seed Bazlı Detaylı Sonuçlar (Test Kümesi)

| Model | Seed | Threshold | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|---|---|
| **LSTM** | 42 | 0.5 | 0.9034 | 0.5385 | 0.2625 | 0.3529 |
| **LSTM** | 123 | 0.5 | 0.8996 | 0.0000 | 0.0000 | 0.0000 |
| **LSTM** | 2026 | 0.5 | 0.8959 | 0.2857 | 0.0250 | 0.0460 |
| **LSTM** | 7 | 0.1 | 0.9360 | 0.7302 | 0.5750 | 0.6434 |
| **LSTM** | 999 | 0.5 | 0.8984 | 0.3333 | 0.0125 | 0.0241 |
| **GRU** | 42 | 0.5 | 0.8971 | 0.0000 | 0.0000 | 0.0000 |
| **GRU** | 123 | 0.2 | 0.9373 | 0.6154 | 1.0000 | 0.7619 |
| **GRU** | 2026 | 0.5 | 0.8971 | 0.0000 | 0.0000 | 0.0000 |
| **GRU** | 7 | 0.3 | 0.8996 | 0.5000 | 0.2000 | 0.2857 |
| **GRU** | 999 | 0.3 | 0.8984 | 0.0000 | 0.0000 | 0.0000 |

### Model Performans Özetleri (Ortalama ± Standart Sapma)

| Model | Ortalama Accuracy | Ortalama Precision | Ortalama Recall | Ortalama F1-score |
|---|---|---|---|---|
| **GRU (ADASYN)** | 0.9059 ± 0.0176 | 0.2231 ± 0.3082 | 0.2400 ± 0.4336 | 0.2095 ± 0.3327 |
| **LSTM (ADASYN)** | 0.9066 ± 0.0166 | 0.3775 ± 0.2753 | 0.1750 ± 0.2486 | 0.2133 ± 0.2801 |

>  **Deney Bulgusu:** Pencere boyutunun değiştirilmesi, modellerin rastgele başlangıç ağırlıklarına (seed) olan yüksek hassasiyetini ortadan kaldırmamıştır. Varyansın yüksek kalması (örneğin GRU modelinin Seed=123'te %100 Recall yakalarken diğer 3 seed'de %0 çekmesi), zaman serisi anomali tespitinde tek başına pencere boyutunun yeterli bir regülasyon sağlamadığını göstermektedir.
>
>  **Metodolojik Değerlendirme (Boyutun Laneti):** Pencere boyutunun 20'den 40'a çıkarılmasıyla zaman ufkunu genişletmenin tek başına yeterli bir kararlılık sağlamadığı görülmüştür. Bu durumun temel nedeni, pencere boyutu büyüdükçe ADASYN algoritmasının sentetik veri üretirken çalıştığı öznitelik uzayının da doğrusal olarak büyümesidir ($40 \times 43 = 1720$ boyut). Yüksek boyutlu uzaylarda (Curse of Dimensionality), sentetik pencereler arasındaki zamansal tutarlılık ve kronolojik korelasyon zayıflamakta, bu da üretilen yapay verinin kalitesini düşürerek modelin kararlı öğrenmesini zorlaştırmaktadır.

### Window Size = 10
Bu bölümde, ADASYN ile dengelenmiş BATADAL veri setinde `sequence_window_size` parametresi 20'den 10'a düşürülerek, modelin daha kısa vadeli ve anlık zamansal değişimleri baz alması sağlanmıştır. Amaç, pencere boyutunu küçülterek ADASYN algoritmasının sentetik veri üretirken karşılaştığı öznitelik uzayını daraltmak ve veri kalitesini artırmaktır. Sınıf ağırlıkları (`class_weight`) sıfır tutulmuş ve süreç aynı 5 rastgele seed (`42, 123, 2026, 7, 999`) ile tekrarlanmıştır.

### Seed Bazlı Detaylı Sonuçlar (Test Kümesi)

| Model | Seed | Threshold | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|---|---|
| **LSTM** | 42 | 0.5 | 0.8996 | 0.3846 | 0.0625 | 0.1075 |
| **LSTM** | 123 | 0.2 | 0.9021 | 0.3333 | 0.0125 | 0.0241 |
| **LSTM** | 2026 | 0.1 | 0.9480 | 0.6697 | 0.9125 | 0.7725 |
| **LSTM** | 7 | 0.5 | 0.9021 | 0.0000 | 0.0000 | 0.0000 |
| **LSTM** | 999 | 0.1 | 0.9686 | 0.7935 | 0.9125 | 0.8488 |
| **GRU** | 42 | 0.4 | 0.9674 | 0.8193 | 0.8500 | 0.8344 |
| **GRU** | 123 | 0.5 | 0.9045 | 1.0000 | 0.0125 | 0.0247 |
| **GRU** | 2026 | 0.3 | 0.9637 | 0.8125 | 0.8125 | 0.8125 |
| **GRU** | 7 | 0.4 | 0.9250 | 0.7813 | 0.3125 | 0.4464 |
| **GRU** | 999 | 0.1 | 0.9577 | 0.7143 | 0.9375 | 0.8108 |

### Model Performans Özetleri (Ortalama ± Standart Sapma)

| Model | Ortalama Accuracy | Ortalama Precision | Ortalama Recall | Ortalama F1-score |
|---|---|---|---|---|
| **GRU (ADASYN)** | 0.9437 ± 0.0276 | 0.8255 ± 0.1060 | 0.5850 ± 0.4026 | 0.5858 ± 0.3529 |
| **LSTM (ADASYN)** | 0.9241 ± 0.0321 | 0.4362 ± 0.3105 | 0.3800 ± 0.4867 | 0.3506 ± 0.4227 |

>  **Metodolojik Değerlendirme (Daraltılmış Öznitelik Uzayı Etkisi):** Pencere boyutunun 10'a düşürülmesi, ADASYN'in sentetik veri üretirken çalıştığı öznitelik uzayını ciddi ölçüde daraltmıştır ($10 \times 43 = 430$ boyut). Boyutun küçülmesiyle birlikte, üretilen yapay pencerelerin kalitesi ve matematiksel tutarlılığı artmıştır.
> 
> Bu durumun en somut kanıtı **GRU modelinin performansıdır**: Varsayılan modele (`window_size=20`) kıyasla GRU'nun ortalama F1-skoru **%35.92'den %58.58'e** fırlamış, daha da önemlisi ortalama Precision değeri **%57.75'ten %82.55'e** yükselmiştir. Ayrıca Precision standart sapmasının **0.1060** seviyesine düşmesi, modelin sahte alarm üretme eğiliminin kararlı bir şekilde kontrol altına alındığını kanıtlamaktadır.
>
>  **Kararsızlık Eğilimi (Seed Hassasiyeti):** Veri kalitesindeki genel artışa rağmen, modellerin rastgele başlangıç ağırlıklarına (seed) olan yüksek bağımlılığı tamamen kırılamamıştır. LSTM modeli belirli seed'lerde (Seed=7 için F1: 0.0, Seed=123 için F1: 0.02) tamamen anomali kaçırma eğilimindeyken; GRU modelinde Seed=42, 2026 ve 999 senaryolarında, LSTM modelinde ise Seed=2026 ve 999 altında **%77 ile %84 F1-skoru** bandında çok güçlü ve dengeli tepe performansları yakalanmıştır. Bu durum, model kararlılığı için sadece pencere boyutunun yeterli olmadığını, eğitim/mimari parametrelerinin de optimize edilmesi gerektiğini göstermektedir.

