# PST Adaptive Threshold Ablation Deneyi

## Amaç

Bu branch’in amacı, Probabilistic Suffix Tree (PST) tabanlı automata modelinde anomaly threshold değerini sabit seçmek yerine, eğitim skor dağılımından otomatik olarak üretmektir.

Önceki PST-NLL ve PST Windowed-NLL deneylerinde threshold değerleri elle belirlenmişti. Bu branch’te ise threshold değerleri train anomaly score dağılımı üzerinden adaptive olarak hesaplanmıştır.

Temel fikir şudur:

```text
threshold = percentile(train_scores, p)
```

veya:

```text
threshold = mean(train_scores) + k * std(train_scores)
```

## Hipotez

Sabit threshold değerleri her veri seti için optimal olmayabilir. Eğitim verisindeki anomaly score dağılımından otomatik threshold üretmek, özellikle BATADAL gibi veri setlerinde daha uygun karar sınırı sağlayabilir.

Bu nedenle adaptive threshold yaklaşımının PST performansını artırabileceği varsayılmıştır.

## Denenen Stratejiler

Bu branch’te iki adaptive threshold stratejisi denenmiştir:

```text
1. Percentile tabanlı threshold
2. Mean + standard deviation tabanlı threshold
```

Percentile değerleri:

```text
[90, 95, 97, 99]
```

Mean + std multiplier değerleri:

```text
[1.0, 2.0, 3.0]
```

Score window değerleri:

```text
[1, 3, 5, 10]
```

## Kullanılan Temel PST Ayarları

### BATADAL

```text
window_size              = 4
alphabet_size            = 5
suffix_max_order         = 2
suffix_min_context_count = 2
suffix_smoothing_alpha   = 0.1
scoring_mode             = negative_log
```

### SKAB

```text
window_size              = 5
alphabet_size            = 5
suffix_max_order         = 2
suffix_min_context_count = 1
suffix_smoothing_alpha   = 0.1
scoring_mode             = negative_log
```

## Baseline Sonuçlar

Adaptive threshold tuning öncesinde PST-NLL baseline sonuçları şu şekildedir:

### SKAB

```text
Original F1      = 0.580373
Gaussian F1      = 0.579320
Unseen Data F1   = 0.255882
```

### BATADAL

```text
Original F1      = 0.173913
Gaussian F1      = 0.166015
Unseen Data F1   = 0.153846
```

## Adaptive Threshold Sonuçları

### BATADAL

Adaptive threshold, BATADAL’da PST-NLL baseline’a göre küçük bir iyileştirme sağlamıştır.

En iyi BATADAL sonucu:

```text
score_window = 3
strategy     = percentile
value        = 99
F1           ≈ 0.196078
```

Karşılaştırma:

```text
PST-NLL baseline BATADAL          = 0.173913
PST adaptive threshold BATADAL    ≈ 0.196078
PST Windowed-NLL best BATADAL     = 0.285714
Markov-final BATADAL              = 0.444444
```

Bu sonuç, adaptive threshold yaklaşımının BATADAL’da küçük bir katkı sağladığını, ancak önceki `feature/pst-windowed-negative-log` branch’inden daha zayıf kaldığını göstermektedir.

### SKAB

Adaptive threshold, SKAB üzerinde ciddi performans düşüşüne neden olmuştur.

En iyi SKAB adaptive threshold sonucu yaklaşık olarak:

```text
score_window = 10
strategy     = mean_std
value        = 1.0
F1           ≈ 0.3253
```

Karşılaştırma:

```text
PST-NLL / PST-final SKAB       = 0.580373
PST adaptive threshold SKAB    ≈ 0.3253
```

Bu sonuç, SKAB veri setinde train-score dağılımından otomatik threshold üretmenin sabit threshold yaklaşımından belirgin şekilde daha kötü olduğunu göstermektedir.

## Genel Değerlendirme

Bu branch performans açısından başarılı bir feature değildir.

BATADAL’da küçük bir iyileştirme sağlanmıştır, ancak bu iyileştirme önceki Windowed-NLL branch’inin gerisinde kalmıştır. SKAB tarafında ise adaptive threshold ciddi bir performans düşüşüne yol açmıştır.

Bu nedenle branch’in sonucu şu şekilde yorumlanmalıdır:

```text
Adaptive threshold, bu haliyle PST modeline genel bir katkı sağlamamıştır.
BATADAL’da küçük bir artış üretmiştir.
SKAB’de ciddi performans kaybına neden olmuştur.
Bu nedenle negatif ablation sonucu olarak değerlendirilebilir.
```

## Üretilen Çıktı Dosyaları

Bu branch kapsamında aşağıdaki adaptive threshold tuning dosyaları üretilmiştir:

```text
results/outputs/pst_ablation/pst_adaptive_threshold_tuning_results.csv
results/outputs/pst_ablation/pst_adaptive_threshold_tuning_summary.csv
results/outputs/pst_ablation/pst_adaptive_threshold_tuning_best_original.csv
results/outputs/pst_ablation/pst_adaptive_threshold_tuning_stability.csv
```

Ayrıca standart experiment çıktıları da güncellenmiştir:

```text
results/outputs/automata_advanced_all_scenarios_metrics.csv
results/outputs/automata_batadal_advanced_explainability.json
results/outputs/statistical_test_results.csv
```