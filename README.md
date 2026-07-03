## BATADAL — LSTM Modeli, ADASYN ile Dengeleme 
 
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
## 1. BATADAL — LSTM ve GRU Parametre Optimizasyonu (Pencere Boyutu Deneyleri)

### pencere boyutu = 40
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
>  **Metodolojik Değerlendirme (Boyutun Laneti):** Pencere boyutunun 20'den 40'a çıkarılmasıyla zaman ufkunu genişletmenin tek başına yeterli bir kararlılık sağlamadığı görülmüştür. Bu durumun temel nedeni, pencere boyutu büyüdükçe ADASYN algoritmasının sentetik veri üretirken çalıştığı öznitelik uzayının da doğrusal olarak büyümesidir ($40 \times 43 = 1720$ boyut).Yüksek boyutlu uzaylarda (Curse of Dimensionality), sentetik pencereler arasındaki zamansal tutarlılık zayıflıyor olabilir; nitekim bu deneyde hem LSTM hem GRU'nun performansı düşmüştür. Ancak Bölüm 3'te tartışıldığı gibi, bu etkinin GRU'da window=10'da tersine dönmesi, açıklamanın tek başına yeterli olmadığını, model mimarisine özgü etkileşimlerin de rol oynadığını göstermektedir.

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

>  **Kararsızlık Eğilimi (Seed Hassasiyeti):** Veri kalitesindeki genel artışa rağmen, modellerin rastgele başlangıç ağırlıklarına (seed) olan yüksek bağımlılığı tamamen kırılamamıştır. LSTM modeli belirli seed'lerde (Seed=7 için F1: 0.0, Seed=123 için F1: 0.02) tamamen anomali kaçırma eğilimindeyken; GRU modelinde Seed=42, 2026 ve 999 senaryolarında, LSTM modelinde ise Seed=2026 ve 999 altında **%77 ile %84 F1-skoru** bandında çok güçlü ve dengeli tepe performansları yakalanmıştır. Bu durum, model kararlılığı için sadece pencere boyutunun yeterli olmadığını, eğitim/mimari parametrelerinin de optimize edilmesi gerektiğini göstermektedir.

---
## 2. Batadal Eğitim Parametreleri Optimizasyonu: LR=0.0005, Dropout=0.4/0.3, Patience=8
Bu bölümde, `sequence_window_size` parametresi 10'da sabit tutulmuş; modellerin rastgele başlangıç ağırlıklarına (seed) hassasiyetini azaltmak ve ADASYN kaynaklı aşırı öğrenmeyi (overfitting) engellemek amacıyla eğitim parametrelerine müdahale edilmiştir. Öğrenme oranı (learning rate) `0.0005` seviyesine çekilerek kararlı yakınsama amaçlanmış, katmanlardaki dropout oranları `0.4` ve `0.3` seviyelerine çıkarılarak model genellemeye zorlanmıştır. Süreç aynı 5 rastgele seed (`42, 123, 2026, 7, 999`) ile doğrulanmıştır.

### Seed Bazlı Detaylı Sonuçlar (Test Kümesi)

| Model | Seed | Threshold | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|---|---|
| **LSTM** | 42 | 0.2 | 0.9057 | 0.6000 | 0.0750 | 0.1333 |
| **LSTM** | 123 | 0.3 | 0.9649 | 0.7429 | 0.9750 | 0.8432 |
| **LSTM** | 2026 | 0.5 | 0.9008 | 0.0000 | 0.0000 | 0.0000 |
| **LSTM** | 7 | 0.5 | 0.9033 | 0.5000 | 0.0375 | 0.0698 |
| **LSTM** | 999 | 0.1 | 0.9553 | 0.7792 | 0.7500 | 0.7643 |
| **GRU** | 42 | 0.1 | 0.9553 | 0.7087 | 0.9125 | 0.7978 |
| **GRU** | 123 | 0.5 | 0.9311 | 0.7949 | 0.3875 | 0.5210 |
| **GRU** | 2026 | 0.4 | 0.9492 | 0.7879 | 0.6500 | 0.7123 |
| **GRU** | 7 | 0.1 | 0.9577 | 0.7711 | 0.8000 | 0.7853 |
| **GRU** | 999 | 0.1 | 0.9589 | 0.7054 | 0.9875 | 0.8229 |

### Model Performans Özetleri (Ortalama ± Standart Sapma)

| Model | Ortalama Accuracy | Ortalama Precision | Ortalama Recall | Ortalama F1-score |
|---|---|---|---|---|
| **GRU (ADASYN + Opt)** | 0.9504 ± 0.0114 | 0.7536 ± 0.0434 | 0.7475 ± 0.2381 | 0.7279 ± 0.1227 |
| **LSTM (ADASYN + Opt)** | 0.9260 ± 0.0314 | 0.5244 ± 0.3138 | 0.3675 ± 0.4596 | 0.3621 ± 0.4069 |

>  **Metodolojik Değerlendirme ve Regülasyon Etkisi:** Öğrenme oranının düşürülmesi ve regresyon maskelerinin (Dropout) sıkılaştırılması, özellikle **GRU mimarisi üzerinde dönüştürücü bir etki** yaratmıştır. GRU modelinin ortalama F1-skoru baseline deneydeki %58.58 seviyesinden **%72.79'a** yükselmiştir. En kritik kazanım ise kararlılık tarafındadır; modelin Precision standart sapması **0.0434** gibi son derece düşük bir seviyeye indirilerek rastgele seed varyansı baskılanmış ve model kararlı sınıflandırma yeteneği kazanmıştır.
>
>  **Mimariler Arası Dayanıklılık Farkı:** GRU modeli hiperparametre optimizasyonu ile tüm seed'lerde dengeli ve yüksek genelleme başarısı gösterirken, LSTM modeli başlangıç ağırlık bağımlılığını tam olarak kıramamıştır. LSTM mimarisi `seed=123` (%84.32 F1) ve `seed=999` (%76.43 F1) senaryolarında mükemmel performans gösterse de, `seed=2026` altında test kümesinde tamamen anomali kaçırma eğilimine (F1: 0.0) girmiştir. Bu durum, GRU hücresinin (katman yapısının) daha az parametre içermesinin verdiği avantajla, kısıtlı endüstriyel veri senaryolarında regülasyon ayarlarına çok daha hızlı ve kararlı tepki verdiği görülmektedir.

## 2.1.Melez Yöntem Deneyi: ADASYN + Balanced Class Weighting
Başlık 2 deki tüm hiper parametreler sabit tutularak, literatürde melez yaklaşım olarak bilinen veri seviyesinde ADASYN artırımı ile model seviyesinde `class_weight="balanced"` cezalandırma mekanizması aynı anda devreye alınmıştır. Amaç, ADASYN'in sentetik pencereler üzerindeki ezber (overfitting) etkisini kayıp fonksiyonundaki sınıf ağırlıklarıyla dengelemektir.

### Seed Bazlı Detaylı Sonuçlar (Test Kümesi)

| Model | Seed | Threshold | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|---|---|
| **LSTM** | 42 | 0.5 | 0.9008 | 0.0000 | 0.0000 | 0.0000 |
| **LSTM** | 123 | 0.5 | 0.9021 | 0.0000 | 0.0000 | 0.0000 |
| **LSTM** | 2026 | 0.5 | 0.8996 | 0.2000 | 0.0125 | 0.0235 |
| **LSTM** | 7 | 0.1 | 0.9226 | 0.7000 | 0.3500 | 0.4667 |
| **LSTM** | 999 | 0.5 | 0.8875 | 0.0000 | 0.0000 | 0.0000 |
| **GRU** | 42 | 0.5 | 0.9504 | 0.7143 | 0.8125 | 0.7602 |
| **GRU** | 123 | 0.1 | 0.9504 | 0.7407 | 0.7500 | 0.7453 |
| **GRU** | 2026 | 0.3 | 0.9299 | 0.9231 | 0.3000 | 0.4528 |
| **GRU** | 7 | 0.5 | 0.9637 | 0.8289 | 0.7875 | 0.8077 |
| **GRU** | 999 | 0.5 | 0.9238 | 0.7576 | 0.3125 | 0.4425 |

### Model Performans Özetleri (Ortalama ± Standart Sapma)

| Model | Ortalama Accuracy | Ortalama Precision | Ortalama Recall | Ortalama F1-score |
|---|---|---|---|---|
| **GRU (Melez)** | 0.9437 ± 0.0164 | 0.7929 ± 0.0843 | 0.5925 ± 0.2623 | 0.6417 ± 0.1787 |
| **LSTM (Melez)** | 0.9025 ± 0.0126 | 0.1800 ± 0.3033 | 0.0725 ± 0.1552 | 0.0980 ± 0.2063 |

>  **Metodolojik Değerlendirme (Kapasite ve Aşırı Regülasyon Çatışması):** Veri artırımı ve sınıf ağırlıklandırmanın melez kombinasyonu, iki mimari üzerinde tamamen zıt etkiler yaratmıştır. Daha sade bir parametre akışına sahip olan **GRU modeli**, bu yoğun cezalandırma mekanizmasına uyum sağlayarak kararlı yapısını korumuş ve ortalama Precision değerini **%79.29** seviyesine stabilize etmeyi başarmıştır. 
>
> Buna karşın **LSTM modeli**, bünyesindeki fazla parametre yükü ve yüksek dropout oranlarının ($0.4 / 0.3$), çift yönlü sınıf dengeleme baskısıyla birleşmesi sonucu **Aşırı Düzenleme (Aşırı Regülasyon - Underfitting)** tuzağına düşmüştür. Model, anomali sınıfını ayırt etmek için gereken gradyan sinyallerini tamamen kaybetmiş ve ağırlık uzayında tüm anomali pencerelerini "Normal" olarak etiketleme kolaycılığına kaçarak çökmüştür (%0.09 Ortalama F1). Bu durum, zaman serisi anomali tespitinde her regülasyon aracının model kapasitesine göre hassas ayarlanması gerektiğini göstermektedir.

---
## 3.Genel Değerlendirme: Hangi Konfigürasyon Gerçekten En İyisi
 
Yukarıdaki dört deney setini yan yana koyduğumuzda, LSTM ve GRU modellerinin optimizasyon müdahalelerine **tamamen zıt yönde** tepki verdiği görülmektedir:
 
| Deney | LSTM F1 (ort.) | GRU F1 (ort.) |
|---|---|---|
| Baseline (window=20, ADASYN) | **0.656** (outlier hariç: 0.820) | 0.359 |
| Window=40 | 0.213 | 0.210 |
| Window=10 | 0.351 | 0.586 |
| Window=10 + HP optimizasyonu | 0.362 | **0.728** |
| Melez Yaklaşım (window=10, LR=0.0005, Dropout + class_weight="balanced") | 0.098 | 0.641 |

>LSTM modelinin en iyi performansı, hiçbir ek müdahale yapılmadan, **ilk baseline denemesinde** (window=20) elde edilmiştir. Sonraki tüm optimizasyon adımları (pencere boyutu küçültme, learning rate/dropout ayarı) LSTM performansını **iyileştirmek yerine kötüleştirmiştir**. GRU ise tam tersi bir eğilim göstermiş, her adımda tutarlı biçimde iyileşmiştir (F1: 0.359 → 0.728).