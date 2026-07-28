# PST Windowed Negative Log Scoring Deneyi

## Amaç

Bu branch’in amacı, Probabilistic Suffix Tree (PST) tabanlı automata modelinde anomaly kararını tek geçişlik negative log skoruna göre değil, son birkaç geçişin ortalama negative log skoruna göre vermektir.

Önceki `feature/pst-negative-log-scoring` branch’inde kullanılan karar mantığı şu şekildeydi:

```text
score_t = -log(P(next_pattern | context))
```

Bu yaklaşım, tek bir geçişin olasılığını anomaly kararı için kullanıyordu.

Bu branch’te ise karar skoru şu hâle getirildi:

```text
score_t = average(-log(P_i)) over last k transitions
```

Buradaki `k`, `score_window` parametresidir.

## Hipotez

Tek bir düşük olasılıklı geçiş her zaman gerçek anomaly anlamına gelmeyebilir. Ancak ardışık birkaç geçiş boyunca düşük olasılıklı davranış görülüyorsa, bu durum daha güçlü bir anomaly sinyali olabilir.

Bu nedenle windowed negative log scoring yaklaşımının özellikle BATADAL gibi saldırı senaryolarında daha stabil ve anlamlı sonuç verebileceği varsayılmıştır.

## Yapılan Kod Değişiklikleri

Bu branch’te aşağıdaki değişiklikler yapılmıştır:

```text
1. ProbabilisticAutomata modeline score_window parametresi eklendi.
2. Negative log scoring modunda son score_window adet geçiş skorunun ortalaması alınarak anomaly_score üretildi.
3. Anomaly kararı tek geçiş skoru yerine windowed anomaly_score üzerinden verildi.
4. Counterfactual açıklamalarında alternatif geçiş için windowed skor hesaplaması eklendi.
5. runner.py içine pst_score_window desteği eklendi.
6. build_suffix_automata fonksiyonu score_window parametresini modele gönderecek şekilde güncellendi.
7. Dataset-specific config üretiminde batadal_pst_score_window ve skab_pst_score_window desteği eklendi.
8. Windowed NLL için score_window + threshold tuning fonksiyonu eklendi.
9. Deney çıktıları pst_ablation klasörüne ayrı CSV dosyaları olarak kaydedildi.
```

## Kullanılan Temel Ayarlar

### Genel Scoring Ayarları

```json
{
    "pst_scoring_mode": "negative_log",
    "pst_nll_epsilon": 1e-12,
    "pst_score_window": 1,
    "pst_score_windows": [1, 3, 5, 10]
}
```

### BATADAL PST Ayarları

```json
{
    "batadal_window_size": 4,
    "batadal_alphabet_size": 5,
    "batadal_suffix_max_order": 2,
    "batadal_suffix_min_context_count": 2,
    "batadal_suffix_smoothing_alpha": 0.1,
    "batadal_anomaly_threshold": 6.907755
}
```

### SKAB PST Ayarları

```json
{
    "skab_window_size": 5,
    "skab_alphabet_size": 5,
    "skab_suffix_max_order": 2,
    "skab_suffix_min_context_count": 1,
    "skab_suffix_smoothing_alpha": 0.1,
    "skab_anomaly_threshold": 0.1053605
}
```

## Tuning Aralığı

BATADAL için kullanılan threshold değerleri:

```text
[3.0, 4.0, 5.0, 6.0, 6.907755, 7.5, 8.0, 9.0, 10.0]
```

SKAB için kullanılan threshold değerleri:

```text
[0.05, 0.1053605, 0.2, 0.35, 0.5, 0.7, 0.9, 1.2, 1.5]
```

Score window değerleri:

```text
[1, 3, 5, 10]
```

## Deney Akışı

Deney şu sırayla çalıştırılmıştır:

```text
1. BATADAL üzerinde 5 farklı seed ile final PST run.
2. SKAB üzerinde 5 fold x 5 seed ile final PST run.
3. BATADAL için score_window + threshold tuning.
4. SKAB için score_window + threshold tuning.
5. Statistical test pipeline.
```

Runner çıktısında windowed tuning’in doğru şekilde çalıştığı doğrulanmıştır:

```text
--- PST WINDOWED NEGATIVE LOG THRESHOLD TUNING BAŞLATILIYOR ---
```

## Sonuçlar

### Final PST-NLL Başlangıç Sonuçları

Windowed tuning öncesinde `score_window=1` ile elde edilen temel sonuçlar:

#### SKAB

```text
Original F1      = 0.580373
Gaussian F1      = 0.579320
Unseen Data F1   = 0.255882
```

#### BATADAL

```text
Original F1      = 0.173913
Gaussian F1      = 0.166015
Unseen Data F1   = 0.153846
```

## Windowed-NLL Tuning Sonuçları

### BATADAL

BATADAL üzerinde windowed negative log scoring performansı artırmıştır.

En iyi BATADAL sonucu:

```text
score_window = 3
threshold    = 6.907755
F1           = 0.285714
precision    = 0.285714
recall       = 0.285714
```

Önceki PST-NLL sonucuna göre gelişim:

```text
PST-NLL baseline F1       = 0.173913
PST Windowed-NLL best F1  = 0.285714
```

Bu sonuç, windowed negative log scoring yaklaşımının BATADAL veri setinde PST modelinin zayıf kaldığı noktayı kısmen iyileştirdiğini göstermektedir.

Ancak BATADAL için Markov-final sonucu hâlâ daha yüksektir:

```text
Markov-final BATADAL F1 = 0.444444
PST Windowed-NLL F1     = 0.285714
```

Bu nedenle BATADAL tarafında PST Windowed-NLL, PST ailesi içinde ilerleme sağlasa da genel automata sonuçları içinde Markov-final modelini geçememiştir.

### SKAB

SKAB üzerinde windowed scoring iyileştirme sağlamamıştır.

En iyi SKAB sonucu:

```text
score_window = 1
threshold    = 0.1053605
F1           = 0.580373
```

Window büyüdükçe SKAB performansı hafif düşmüştür:

```text
score_window = 1   -> F1 = 0.580373
score_window = 3   -> F1 = 0.578582
score_window = 5   -> F1 = 0.576905
score_window = 10  -> F1 = 0.574182
```

Bu sonuç, SKAB veri setinde tek geçişlik PST-NLL kararının windowed ortalamadan daha uygun olduğunu göstermektedir.

## Karşılaştırmalı Değerlendirme

### BATADAL

```text
PST-NLL baseline       = 0.173913
PST Windowed-NLL best  = 0.285714
Markov-final           = 0.444444
```

Yorum:

```text
Windowed-NLL, BATADAL’da PST performansını artırmıştır.
Ancak Markov-final hâlâ daha güçlüdür.
```

### SKAB

```text
Markov-final           = 0.574023
PST-final              = 0.580373
PST-NLL                = 0.580373
PST Windowed-NLL best  = 0.580373
```

Yorum:

```text
SKAB’de PST yaklaşımı Markov-final’dan az farkla daha iyidir.
Ancak windowed-NLL ek iyileştirme sağlamamıştır.
```

## Üretilen Çıktı Dosyaları

Bu branch kapsamında aşağıdaki deney çıktıları üretilmiştir:

```text
results/outputs/automata_advanced_all_scenarios_metrics.csv
results/outputs/automata_skab_fold_summary.csv
results/outputs/automata_batadal_seed_summary.csv
results/outputs/automata_batadal_advanced_explainability.json
results/outputs/statistical_test_results.csv
```

Windowed-NLL tuning çıktıları:

```text
results/outputs/pst_ablation/pst_windowed_nll_tuning_results.csv
results/outputs/pst_ablation/pst_windowed_nll_tuning_summary.csv
results/outputs/pst_ablation/pst_windowed_nll_tuning_best_original.csv
results/outputs/pst_ablation/pst_windowed_nll_tuning_stability.csv
```

## Genel Sonuç

Bu branch başarılı bir deneydir.

Windowed negative log scoring, BATADAL veri setinde PST modelinin performansını artırmıştır. En iyi BATADAL sonucu `score_window=3` ve `threshold=6.907755` ile elde edilmiştir.

SKAB veri setinde ise en iyi sonuç `score_window=1` ile korunmuştur. Daha büyük window değerleri SKAB performansını hafif düşürmüştür.

Bu nedenle nihai yorum şu şekildedir:

```text
PST Windowed-NLL, BATADAL için faydalı bir iyileştirme sağlamıştır.
SKAB için ise ek katkı sağlamamıştır.
Dataset-specific karar mekanizması açısından anlamlı bir bulgudur.
```
