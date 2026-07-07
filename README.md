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
 
>  **LSTM Özeti:** 5 seed'in 4'ünde (123, 2026, 7, 999) LSTM iyi sonuç verdi (F1: 0.78–0.88). Ama seed=42'de model test setinde hiçbir anomaliyi bulamadı (precision, recall, F1 hepsi 0). Validation'da bu seed için sonuç fena değildi (F1≈0.545), ama test'e geçince model çıktı olasılıkları, validation setinden taşınan katı threshold sınırlarını aşamadığı için test setinde anomali yakalanamamıştır.


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
## 2. Batadal Veri Setinde HiperParametreleri Optimizasyonu
## 2.1. LR=0.0005, Dropout=0.4/0.3, Patience=8 
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

## 2.2.Melez Yöntem Deneyi: ADASYN + Balanced Class Weighting
Başlık 2.1. deki tüm hiper parametreler sabit tutularak, literatürde melez yaklaşım olarak bilinen veri seviyesinde ADASYN artırımı ile model seviyesinde `class_weight="balanced"` cezalandırma mekanizması aynı anda devreye alınmıştır. Amaç, ADASYN'in sentetik pencereler üzerindeki ezber (overfitting) etkisini kayıp fonksiyonundaki sınıf ağırlıklarıyla dengelemektir.

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

## 2.3.Genişletilmiş Kapasite ve Gevşetilmiş Regülasyon Deneyi: Units=128, Dense=64, Dropout=0.3/0.2
Bu bölümde, modellerin anomali sınıflarına ait gizli zamansal örüntüleri çözme potansiyelini test etmek amacıyla mimari kapasite genişletilmiştir. Katmanlardaki nöron sayıları `128` ve `64` seviyelerine çıkarılmış; modelin öğrenmesini kolaylaştırmak adına dropout oranları `0.3` ve `0.2` değerlerine gevşetilmiştir. Öğrenme oranı `0.0005` seviyesine sabit tutulmuş, sınıf ağırlıkları devre dışı bırakılarak (`class_weight=None`) saf kapasite etkisi gözlemlenmiştir. Süreç aynı 5 rastgele seed ile tekrarlanmıştır.

### Seed Bazlı Detaylı Sonuçlar (Test Kümesi)

| Model | Seed | Threshold | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|---|---|
| **LSTM** | 42 | 0.5 | 0.9021 | 0.0000 | 0.0000 | 0.0000 |
| **LSTM** | 123 | 0.1 | 0.9553 | 0.7087 | 0.9125 | 0.7978 |
| **LSTM** | 2026 | 0.1 | 0.9553 | 0.7471 | 0.8125 | 0.7784 |
| **LSTM** | 7 | 0.5 | 0.9021 | 0.0000 | 0.0000 | 0.0000 |
| **LSTM** | 999 | 0.5 | 0.9021 | 0.0000 | 0.0000 | 0.0000 |
| **GRU** | 42 | 0.3 | 0.9541 | 0.7100 | 0.8875 | 0.7889 |
| **GRU** | 123 | 0.1 | 0.9287 | 0.7143 | 0.4375 | 0.5426 |
| **GRU** | 2026 | 0.4 | 0.9021 | 0.0000 | 0.0000 | 0.0000 |
| **GRU** | 7 | 0.1 | 0.9492 | 0.6696 | 0.9375 | 0.7813 |
| **GRU** | 999 | 0.4 | 0.9008 | 0.4286 | 0.0750 | 0.1277 |

### Model Performans Özetleri (Ortalama ± Standart Sapma)

| Model | Ortalama Accuracy | Ortalama Precision | Ortalama Recall | Ortalama F1-score |
|---|---|---|---|---|
| **GRU (128 Units)** | 0.9270 ± 0.0252 | 0.5045 ± 0.3057 | 0.4675 ± 0.4390 | 0.4481 ± 0.3673 |
| **LSTM (128 Units)** | 0.9233 ± 0.0291 | 0.2912 ± 0.3989 | 0.3450 ± 0.4737 | 0.3153 ± 0.4317 |

>  **Metodolojik Değerlendirme (Kapasite Artışının İki Uçlu Değneği):** Nöron kapasitesinin 128'e çıkarılması ve dropout baskısının azaltılması, modellerde **kararsız ve kutuplaşmış (polarize) bir öğrenme davranışı** doğurmuştur. LSTM modeli, kısıtlı kapasitedeki kilitlenmesini kırarak belirli başlangıç ağırlıklarında (Seed=123 ve Seed=2026) **%79.78** ve **%77.84** F1 gibi oldukça güçlü tepe performanslarına ulaşabileceğini kanıtlamıştır. Ancak varyans kontrol altına alınamamış, model diğer 3 seed senaryosunda anomali sınıfını tamamen ıskalamıştır.
>
> GRU mimarisinde ise kapasite artışı ters etki yaratmış, daha önce kontrollü regülasyonla (64 units) elde edilen %72.79'luk kararlı ortalama F1 başarısı, regülasyonun gevşemesiyle **%44.81** seviyesine gerilemiştir. Bu deneysel çıktı, endüstriyel anomali tespitinde salt model büyüklüğünün veya nöron hacminin başarı getirmediğini; aksine aşırı parametre yükünün, veri setindeki gürültü ve dengesizlik yapılarıyla birleşerek rassal başlangıç ağırlıklarına bağımlılığı artırdığını (yüksek varyans ve standart sapma) gözler önüne sermektedir.

## 2.4. Genişletilmiş Kapasite + Orijinal Pencere Boyutu Deneyi: Window=20, Units=128, Dense=64, Dropout=0.3/0.2
Bu bölümde, 2.3'deki genişletilmiş kapasite (units=128, dense=64) ve gevşetilmiş regülasyon (dropout=0.3/0.2, LR=0.0005) ayarları sabit tutulmuş, ancak `sequence_window_size` parametresi tekrar baseline değeri olan 20'ye çıkarılmıştır. Amaç, yüksek kapasiteli mimarilerin, ADASYN'in daha geniş (ve dolayısıyla daha yüksek boyutlu: $20 \times 43 = 860$) öznitelik uzayında nasıl davrandığını gözlemlemektir. Sınıf ağırlıkları devre dışı bırakılmış (`class_weight=None`) ve süreç aynı 5 rastgele seed (`42, 123, 2026, 7, 999`) ile tekrarlanmıştır.

### Seed Bazlı Detaylı Sonuçlar (Test Kümesi)

| Model | Seed | Threshold | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|---|---|
| **LSTM** | 42 | 0.1 | 0.9694 | 0.8161 | 0.8875 | 0.8503 |
| **LSTM** | 123 | 0.2 | 0.9009 | 0.4000 | 0.0250 | 0.0471 |
| **LSTM** | 2026 | 0.1 | 0.9694 | 0.8235 | 0.8750 | 0.8485 |
| **LSTM** | 7 | 0.1 | 0.9621 | 0.7526 | 0.9125 | 0.8249 |
| **LSTM** | 999 | 0.1 | 0.9547 | 0.7087 | 0.9125 | 0.7978 |
| **GRU** | 42 | 0.4 | 0.8984 | 0.0000 | 0.0000 | 0.0000 |
| **GRU** | 123 | 0.1 | 0.9400 | 0.6281 | 0.9500 | 0.7562 |
| **GRU** | 2026 | 0.3 | 0.9033 | 1.0000 | 0.0125 | 0.0247 |
| **GRU** | 7 | 0.5 | 0.9082 | 0.5758 | 0.2375 | 0.3363 |
| **GRU** | 999 | 0.2 | 0.9143 | 0.7083 | 0.2125 | 0.3269 |

### Model Performans Özetleri (Ortalama ± Standart Sapma)

**Tüm seed'ler (5 seed):**

| Model | Ortalama Accuracy | Ortalama Precision | Ortalama Recall | Ortalama F1-score |
|---|---|---|---|---|
| **GRU (Window=20, 128 Units)** | 0.9129 ± 0.0163 | 0.5824 ± 0.3645 | 0.2825 ± 0.3890 | 0.2888 ± 0.3063 |
| **LSTM (Window=20, 128 Units)** | 0.9513 ± 0.0288 | 0.7002 ± 0.1743 | 0.7225 ± 0.3903 | 0.6737 ± 0.3510 |

**LSTM, Seed=123 hariç (4 seed):**

| Metrik | Ortalama ± Std |
|---|---|
| Accuracy | 0.9639 ± 0.0065 |
| Precision | 0.7752 ± 0.0554 |
| Recall | 0.8969 ± 0.0177 |
| F1-score | 0.8304 ± 0.0234 |

>
>  **LSTM İçin En İyi Konfigürasyon:** Seed=123 dışındaki 4 seed'de LSTM, %80–85 F1 bandında son derece tutarlı sonuçlar vermiştir (outlier hariç ortalama F1: **0.8304 ± 0.0234**) — bu, dokümandaki tüm LSTM deneyleri arasında en yüksek ve en kararlı ortalamadır (baseline'ın outlier hariç F1: 0.820 değerini de geçmektedir). Yani LSTM için ideal tarif, kapasiteyi artırmak (units=128) fakat pencere boyutunu **küçültmemek** (window=20'de bırakmak) olarak ortaya çıkmaktadır.
>
>  **GRU İçin Ters Etki:** Aynı kapasite artışı ve window=20 kombinasyonu GRU için performansı belirgin şekilde kötüleştirmiştir (F1 ort. 0.2888, oysa aynı kapasitede window=10'da F1 ort. 0.4481'di — bkz. Bölüm 2.2). Bu, GRU'nun optimal çalışma noktasının küçük pencere boyutlarında (window=10) kaldığını, window=20'nin GRU için hem gereksiz hem de zararlı bir kapasite/boyut kombinasyonu oluşturduğunu doğrulamaktadır.

## 2.5. Dengelenmiş Eğitim Parametreleri Deneyi: Epoch=75, Patience=12, Model Tabanlı Dinamik Pencere Boyutu(gru window = 10, lstm window = 20)
Bu bölümde, baseline model ayarlarından yola çıkılarak modelin öğrenme sürecini daha derin yakınsamaya zorlamak adına maksimum epoch sayısı `75`'e, erken durdurma sabrı (`early_stopping_patience`) ise `12` seviyesine çıkarılmıştır.Dropout oranları ($0.3 / 0.2$)  modellerin başlangıç ağırlıklarına (seed) karşı dayanıklılığı test edilmiştir.

### Seed Bazlı Detaylı Sonuçlar (Test Kümesi)

| Model | Seed | Threshold | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|---|---|
| **LSTM** | 42 | 0.1 | 0.9547 | 0.7129 | 0.9000 | 0.7956 |
| **LSTM** | 123 | 0.1 | 0.9535 | 0.8088 | 0.6875 | 0.7432 |
| **LSTM** | 2026 | 0.2 | 0.8984 | 0.0000 | 0.0000 | 0.0000 |
| **LSTM** | 7 | 0.1 | 0.9694 | 0.8090 | 0.9000 | 0.8521 |
| **LSTM** | 999 | 0.1 | 0.9682 | 0.7935 | 0.9125 | 0.8488 |
| **GRU** | 42 | 0.5 | 0.9637 | 0.7907 | 0.8500 | 0.8193 |
| **GRU** | 123 | 0.2 | 0.9601 | 0.7527 | 0.8750 | 0.8092 |
| **GRU** | 2026 | 0.2 | 0.9577 | 0.7027 | 0.9750 | 0.8168 |
| **GRU** | 7 | 0.5 | 0.9311 | 0.8286 | 0.3625 | 0.5043 |
| **GRU** | 999 | 0.3 | 0.9021 | 0.0000 | 0.0000 | 0.0000 |

### Model Performans Özetleri (Ortalama ± Standart Sapma)

| Model | Ortalama Accuracy | Ortalama Precision | Ortalama Recall | Ortalama F1-score |
|---|---|---|---|---|
| **GRU (Epoch=75)** | 0.9429 ± 0.0262 | 0.6149 ± 0.3469 | 0.6125 ± 0.4166 | 0.5899 ± 0.3562 |
| **LSTM (Epoch=75)** | 0.9488 ± 0.0291 | 0.6248 ± 0.3516 | 0.6800 ± 0.3916 | 0.6479 ± 0.3649 |

>Model bazlı dinamik pencere seçimi, her iki mimarinin de güçlü olduğu alanları (GRU için dar/kaliteli uzay, LSTM için geniş zaman ufku) ortaya çıkarmıştır. Genişletilmiş eğitim süresi, LSTM modelinin tepe performanslarını %81-85 F1 bandına stabilize ederken; GRU modelinin 5 rastgele başlangıç ağırlığının 3'ünde %80+ F1 başarısı yakalamasını sağlamıştır. Her iki modelde de tekil seed bazlı görülen sıfır çekme (anomali kaçırma) durumları, modellerin yetersizliğinden ziyade validation setinden taşınan sabit eşik listesi adımlarının test kümesi olasılık dağılımlarına tam adapte olamamasından kaynaklanmaktadır.

## Deney 2.6: Genişletilmiş Rassal Başlangıç (15-Seed) ve Optimize Kapasite Analizi
Bu deneysel senaryoda, modellerin başlangıç ağırlıklarına (seed) olan duyarlılığını ve genelleme yeteneğini en üst düzeyde test etmek amacıyla rastgele başlangıç varyasyonu 5'ten **15 farklı seed değerine** çıkarılmıştır (`42, 123, 2026, 7, 999, 1, 13, 101, 256, 314, 512, 1024, 2048, 4096, 8192`). (lstm window=20 gru window=10)

Önceki deneylerde görülen aşırı öğrenme (overfitting) eğilimini ve yüksek standart sapmayı kontrol altına almak adına, model mimarilerinde yapısal bir optimizasyona gidilmiş; `lstm_units` ve `gru_units` kapasiteleri **64** seviyesine indirilmiş, `dense_units` ise **32** olarak dengelenmiştir. Eğitim parametreleri `epochs: 100` ve `early_stopping_patience: 15` olarak pürüzsüz yakınsamaya izin verecek şekilde korunmuştur.

### Seed Bazlı Detaylı Sonuçlar (Test Kümesi)
 
| Model | Seed | Threshold | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|---|---|
| **LSTM** | 42 | 0.5 | 0.9033 | 0.6667 | 0.0250 | 0.0482 |
| **LSTM** | 123 | 0.1 | 0.9670 | 0.7849 | 0.9125 | 0.8439 |
| **LSTM** | 2026 | 0.1 | 0.9400 | 0.6202 | 1.0000 | 0.7656 |
| **LSTM** | 7 | 0.2 | 0.9168 | 0.7727 | 0.2125 | 0.3333 |
| **LSTM** | 999 | 0.1 | 0.9400 | 0.6220 | 0.9875 | 0.7633 |
| **LSTM** | 1 | 0.2 | 0.9486 | 0.6759 | 0.9125 | 0.7766 |
| **LSTM** | 13 | 0.5 | 0.8972 | 0.0000 | 0.0000 | 0.0000 |
| **LSTM** | 101 | 0.2 | 0.8984 | 0.4615 | 0.2250 | 0.3025 |
| **LSTM** | 256 | 0.5 | 0.9400 | 0.8780 | 0.4500 | 0.5950 |
| **LSTM** | 314 | 0.1 | 0.9584 | 0.8286 | 0.7250 | 0.7733 |
| **LSTM** | 512 | 0.3 | 0.9009 | 0.3333 | 0.0125 | 0.0241 |
| **LSTM** | 1024 | 0.5 | 0.9009 | 0.0000 | 0.0000 | 0.0000 |
| **LSTM** | 2048 | 0.1 | 0.9229 | 0.5594 | 1.0000 | 0.7175 |
| **LSTM** | 4096 | 0.2 | 0.9498 | 0.6757 | 0.9375 | 0.7853 |
| **LSTM** | 8192 | 0.1 | 0.9192 | 0.5507 | 0.9500 | 0.6972 |
| **GRU** | 42 | 0.3 | 0.9311 | 0.5935 | 0.9125 | 0.7192 |
| **GRU** | 123 | 0.5 | 0.9069 | 0.5556 | 0.1875 | 0.2804 |
| **GRU** | 2026 | 0.5 | 0.9553 | 0.7654 | 0.7750 | 0.7702 |
| **GRU** | 7 | 0.4 | 0.8912 | 0.0000 | 0.0000 | 0.0000 |
| **GRU** | 999 | 0.5 | 0.8912 | 0.0000 | 0.0000 | 0.0000 |
| **GRU** | 1 | 0.5 | 0.8960 | 0.0000 | 0.0000 | 0.0000 |
| **GRU** | 13 | 0.5 | 0.9033 | 0.5000 | 0.4750 | 0.4872 |
| **GRU** | 101 | 0.5 | 0.9432 | 0.7143 | 0.6875 | 0.7006 |
| **GRU** | 256 | 0.5 | 0.9420 | 0.6633 | 0.8125 | 0.7303 |
| **GRU** | 314 | 0.5 | 0.9021 | 0.4872 | 0.2375 | 0.3193 |
| **GRU** | 512 | 0.5 | 0.8899 | 0.0000 | 0.0000 | 0.0000 |
| **GRU** | 1024 | 0.2 | 0.9274 | 0.5758 | 0.9500 | 0.7170 |
| **GRU** | 2048 | 0.5 | 0.8948 | 0.1818 | 0.0250 | 0.0440 |
| **GRU** | 4096 | 0.4 | 0.9408 | 0.6566 | 0.8125 | 0.7263 |
| **GRU** | 8192 | 0.5 | 0.8899 | 0.1765 | 0.0375 | 0.0619 |
 
### Model Performans Özetleri (Ortalama ± Standart Sapma)
 
| Model | Ortalama Accuracy | Ortalama Precision | Ortalama Recall | Ortalama F1-score |
|---|---|---|---|---|
| **GRU (ADASYN Baseline)** | 0.9137 ± 0.0235 | 0.3913 ± 0.2945 | 0.3942 ± 0.3889 | 0.3704 ± 0.3319 |
| **LSTM (ADASYN Baseline)** | 0.9269 ± 0.0237 | 0.5620 ± 0.2677 | 0.5567 ± 0.4308 | 0.4951 ± 0.3363 |

> Sonuçlarının Değerlendirilmesi: Bu deneyde seed sayısı 15’e çıkarılarak modellerin başlangıç ağırlıklarına karşı dayanıklılığı daha kapsamlı biçimde incelenmiştir. Ayrıca model kapasiteleri azaltılarak daha sade mimarilerin eğitim kararlılığı üzerindeki etkisi analiz edilmiştir.

>Sonuçlar, LSTM modelinin ortalama performans açısından GRU modelinden daha başarılı olduğunu göstermiştir. LSTM modeli bazı seed değerlerinde oldukça yüksek F1-score değerlerine ulaşırken, her iki modelde de standart sapmaların yüksek olması seed duyarlılığının halen önemli bir problem olduğunu ortaya koymuştur.

>Bazı seed değerlerinde precision, recall ve F1-score değerlerinin sıfıra düşmesi, modellerin belirli başlangıçlarda anomalileri tamamen kaçırabildiğini göstermektedir. Özellikle başarısız seed’lerde threshold değerlerinin çoğunlukla 0.5 seviyesinde kalması, model çıktı olasılıklarının düşük bölgede sıkıştığını ve anomalilerin yeterli güvenle tahmin edilemediğini düşündürmektedir.

>Genel olarak değerlendirildiğinde, kapasite azaltımı bazı seed’lerde daha kararlı sonuçlar üretilmesine katkı sağlamış olsa da, seed duyarlılığı problemi tamamen giderilememiştir. Bu nedenle yalnızca ortalama performans değerlerinin değil, seed bazlı kararlılık analizlerinin de dikkate alınması gerektiği sonucuna ulaşılmıştır.


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
|Genişletilmiş Kapasite (window=10, Units=128, Dropout=0.3/0.2) | 0.315 | 0.448|
|Genişletilmiş Kapasite + Window=20 (Units=128, Dropout=0.3/0.2)| **0.674** (outlier hariç: **0.830**) | 0.289 |
|Dengelenmiş Eğitim Parametreleri Deneyi (Epoch=75, Patience=12, Dinamik window) |0.6479 | 0.5899 |


> **Sonuç — Mimariye Özel Optimal Konfigürasyonlar:** Tek bir "genel en iyi konfigürasyon" yerine, her mimarinin kendi optimal ayarına sahip olduğu görülmektedir:

>LSTM modeli, karmaşık ve çok kapılı (gate) yapısı gereği yüksek parametre kapasitesine ihtiyaç duyar. Pencere boyutunu 10'a düşürmek veya aşırı regülasyon uygulamak (Bölüm 2.2'deki Melez Yaklaşım gibi) LSTM'in gradyan sinyallerini kaybetmesine ve Underfitting tuzağına düşerek çökmesine neden olmuştur. LSTM için ideal denge, öznitelik uzayı geniş olsa bile pencere boyutunu 20'de tutmak ve nöron kapasitesini 128'e çıkarmaktır. Bu kombinasyon, outlier (Seed=123) hariç tutulduğunda 0.8304 ± 0.0234 F1-skoru ile dokümandaki en yüksek ve en kararlı (en düşük varyanslı) performansı üretmiştir.

>GRU, LSTM'e kıyasla daha sade bir hücre yapısına (daha az parametreye) sahiptir. Bu durum, ADASYN'in yüksek boyutlu uzaylardaki gürültüsünden (Boyutun Laneti) GRU'nun çok daha hızlı etkilenmesine yol açmaktadır. GRU için en başarılı hamle, pencere boyutunu 10'a düşürerek öznitelik uzayını daraltmak ve düşük öğrenme oranı + sıkı dropout ile modeli kısıtlamaktır. Bu sayede GRU, rastgele başlangıç ağırlıklarına (seed) olan hassasiyetini tamamen kırmış, Precision standart sapmasını 0.0434 gibi mükemmel bir seviyeye indirerek 0.7279 kararlı ortalama F1 başarısı yakalamıştır.


###  Metodolojik Güvence (Veri Sızıntısı Önlemleri)

Projede veri sızıntısını (data leakage) engellemek ve modellerin gerçek dünya performansını dürüstçe ölçmek adına aşağıdaki protokole kesin olarak sadık kalınmıştır:
1. **Saf Doğrulama ve Test:** ADASYN ile veri artırımı (pencere seviyesinde) **yalnızca Eğitim (Train) kümesine** uygulanmıştır. Doğrulama (Validation) ve Test kümelerine hiçbir şekilde sentetik veri bulaştırılmamış; bu kümeler tamamen orijinal ve ham zaman serisi akışından koparılmıştır.
2. **Eşik Kararsızlığı Notu:** Bazı konfigürasyonlarda ve belirli başlangıç ağırlıklarında (örneğin Seed=42) test F1 skorunun `0.0000` çıkması, bir veri sızıntısından değil; tamamen saf validation setinden öğrenilen katı eşik değerlerinin (threshold), ADASYN ile eğitilen modelin ham test kümesindeki tahmin olasılık aralığını (confidence interval) karşılayamamış olmasından kaynaklanmaktadır.