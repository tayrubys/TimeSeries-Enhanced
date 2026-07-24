# Probabilistic Suffix Tree Automata Deneyi

## Amaç

Bu branch, zaman serisi anomali tespiti için kullanılan probabilistic automata yaklaşımını geliştirmek amacıyla Probabilistic Suffix Tree (PST) tabanlı değişken uzunluklu context modelini test eder.

Önceki Markov tabanlı automata yaklaşımı sabit uzunluklu geçmiş durumları kullanırken, PST yaklaşımı farklı uzunluklardaki suffix context’leri öğrenerek bir sonraki pattern’in olasılığını hesaplar. Böylece modelin yalnızca sabit order değerine bağlı kalmadan daha esnek geçiş yapıları öğrenmesi hedeflenmiştir.

## Model Özeti

PST modeli eğitim sırasında zaman serisinden elde edilen sembolik pattern dizileri üzerinde çalışır. Her geçiş için farklı uzunluklardaki geçmiş context’ler kaydedilir.

Örnek context yapıları:

```text
<root>
A
A -> B
A -> B -> C
```

Test sırasında model, mevcut geçmişe göre en uzun güvenilir suffix context’i seçer. Eğer uzun context yeterince görülmemişse daha kısa context’e geri düşer. Seçilen context üzerinden bir sonraki pattern’in olasılığı hesaplanır.

Karar mantığı:

```text
P(next_pattern | selected_suffix_context) < threshold ise anomaly
P(next_pattern | selected_suffix_context) >= threshold ise normal
```

## Eklenen Temel Parametreler

```text
suffix_max_order
suffix_min_context_count
suffix_smoothing_alpha
anomaly_threshold
```

## Dataset-Specific Ayarlar

PST modelinin SKAB ve BATADAL veri setlerinde farklı davranması nedeniyle veri setine özel ayarlar kullanılmıştır.

### SKAB Final PST Ayarı

```text
window_size = 5
alphabet_size = 5
suffix_max_order = 2
suffix_min_context_count = 1
suffix_smoothing_alpha = 0.1
anomaly_threshold = 0.9
```

### BATADAL Final PST Ayarı

```text
window_size = 4
alphabet_size = 5
suffix_max_order = 2
suffix_min_context_count = 2
suffix_smoothing_alpha = 0.1
anomaly_threshold = 0.001
```

## Yapılan Deneyler

Bu branch üzerinde aşağıdaki deney aşamaları uygulanmıştır:

1. İlk PST modelinin eklenmesi
2. Veri setine özel PST parametre desteği
3. Threshold ve smoothing alpha taraması
4. SKAB için top candidate doğrulaması
5. SKAB fold-aware refined search
6. Final PST ayarlarının ana deney akışına taşınması
7. 5 fold x 5 seed final doğrulama

## Final Sonuçlar

### SKAB

```text
Original F1       = 0.580373
Gaussian Noise F1 = 0.579320
Unseen Data F1    = 0.255882
```

Önceki Markov-final SKAB sonucu:

```text
Original F1 = 0.574023
```

Sonuç:

```text
PST, SKAB veri setinde Markov-final modeline göre küçük fakat doğrulanmış bir F1 iyileşmesi sağlamıştır.
```

### BATADAL

```text
Original F1       = 0.173913
Gaussian Noise F1 = 0.166015
Unseen Data F1    = 0.153846
```

Önceki Markov-final BATADAL sonucu:

```text
Original F1 = 0.444444
```

Sonuç:

```text
PST, BATADAL veri setinde Markov-final modelinin gerisinde kalmıştır.
```

## Genel Değerlendirme

Probabilistic Suffix Tree yaklaşımı SKAB veri setinde yüksek recall üreterek sınırlı bir F1 iyileşmesi sağlamıştır. Ancak BATADAL veri setinde aynı başarıyı gösterememiştir.

Bu sonuç, değişken uzunluklu context modellemenin bazı zaman serisi anomaly detection senaryolarında faydalı olabileceğini, fakat performansın veri setinin yapısına duyarlı olduğunu göstermektedir.

## Öne Çıkan Gözlemler

- SKAB üzerinde PST, Markov-final modelini az farkla geçmiştir.
- SKAB’de model yüksek recall üretmiştir.
- BATADAL’da raw probability threshold yaklaşımı yetersiz kalmıştır.
- BATADAL performansı threshold tuning sonrası toparlansa da Markov-final seviyesine ulaşamamıştır.
- Fold1 üzerinde iyi görünen bazı adaylar 5 fold doğrulamasında düşmüştür.
- Fold-aware refined search, daha güvenilir final adayının seçilmesini sağlamıştır.

## Üretilen Önemli Çıktılar

```text
results/outputs/automata_advanced_all_scenarios_metrics.csv
results/outputs/automata_skab_fold_summary.csv
results/outputs/automata_batadal_seed_summary.csv
results/outputs/statistical_test_results.csv
results/outputs/pst_ablation/pst_skab_fold_aware_refined_search.csv
results/outputs/pst_ablation/pst_skab_fold_aware_refined_summary.csv
results/outputs/pst_ablation/pst_skab_fold_aware_refined_best_candidates.csv
results/outputs/pst_ablation/pst_skab_fold_aware_refined_stability.csv
```
