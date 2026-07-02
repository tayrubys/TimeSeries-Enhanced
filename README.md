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
