# feature/ensemble-markov

Bu branch, Markov tabanlı probabilistic automata modelinde farklı Markov order değerlerini birlikte kullanarak anomaly detection performansını artırmayı hedefler.

Önceki feature denemelerinde tek modelin karar/skor mekanizmasını değiştiren yaklaşımlar sınırlı veya olumsuz sonuçlar vermişti. Bu nedenle bu branch’te tek bir Markov modelini iyileştirmek yerine, farklı context uzunluklarına sahip modellerin birlikte kullanılması denenmiştir.

Bu branch henüz `markov-base` branch’ine merge edilmemiştir. Tüm feature branch’leri denendikten sonra final değerlendirme yapılacak ve gerekirse merge kararı verilecektir.

Şu anki deney sonuçlarına göre `feature/ensemble-markov`, denenmiş feature’lar arasında en güçlü adaydır.

---

## Amaç

Mevcut `markov-base` modelinde final otomata ayarları büyük ölçüde `order=3` üzerinden çalışmaktadır.

Ancak farklı Markov order değerleri farklı davranışlar gösterebilir:

```text
order=2 -> daha kısa context, daha genel transition davranışı
order=3 -> mevcut base model için en iyi denge
order=4 -> daha uzun context, daha detaylı ama daha sparse transition yapısı
```

Bu branch’in amacı, bu farklı order modellerinin skorlarını birleştirerek daha dengeli bir anomaly score üretmektir.

Temel fikir:

```text
Tek bir Markov order yerine birden fazla order kullan.
Her order için transition probability / anomaly score hesapla.
Bu skorları mean, weighted_mean veya max aggregation ile birleştir.
```

---

## Eklenen Parametreler

`ProbabilisticAutomata` sınıfına ensemble desteği için aşağıdaki parametreler eklendi:

```python
ensemble_enabled=False
ensemble_orders=None
ensemble_weights=None
ensemble_aggregation="mean"
```

### Parametre Açıklamaları

| Parametre | Açıklama |
|---|---|
| `ensemble_enabled` | Ensemble Markov mekanizmasını açar veya kapatır. |
| `ensemble_orders` | Birlikte kullanılacak Markov order listesini belirtir. Örnek: `[2, 3, 4]`. |
| `ensemble_weights` | Weighted ensemble için order ağırlıklarını belirtir. |
| `ensemble_aggregation` | Order skorlarının nasıl birleştirileceğini belirler. |

Desteklenen aggregation modları:

| Aggregation | Açıklama |
|---|---|
| `mean` | Order skorlarının düz ortalamasını alır. |
| `weighted_mean` | Order skorlarını verilen ağırlıklara göre ortalar. |
| `max` | En yüksek anomaly skorunu kullanır. |
| `min` | En düşük anomaly skorunu kullanır. |

---

## Modelde Yapılan Değişiklikler

Bu branch’te modelin temel transition learning yapısı korunmuştur.

Mevcut model zaten `1..order` aralığındaki transition context’lerini öğrenmektedir. Bu yapı ensemble için uygun olduğu için model tamamen yeniden yazılmamıştır.

Eklenen ensemble mekanizması, her order seviyesi için ayrı skor hesaplayıp bu skorları seçilen aggregation stratejisiyle birleştirir.

Örnek ensemble yapı:

```text
ensemble_orders = [2, 3, 4]
ensemble_aggregation = "max"
```

Bu durumda model:

```text
order=2 skorunu hesaplar
order=3 skorunu hesaplar
order=4 skorunu hesaplar
son karar skorunu max aggregation ile üretir
```

---

## Runner Değişiklikleri

`runner.py` dosyasına ensemble parametreleri eklendi.

Final koşulara şu alanlar dahil edildi:

```python
ensemble_enabled
ensemble_orders
ensemble_weights
ensemble_aggregation
```

Bu parametreler:

- model oluşturulurken `ProbabilisticAutomata` sınıfına geçirildi,
- deney sonuçlarına metadata olarak eklendi,
- SKAB ve BATADAL summary çıktılarında raporlandı,
- ensemble sweep fonksiyonuna dahil edildi.

Ayrıca `run_ensemble_markov_sweep` parametresi eklendi.

---

## Eklenen Sweep Mekanizması

Bu branch’te farklı ensemble kombinasyonlarını test etmek için ayrı bir sweep mekanizması eklendi.

Sweep sırasında şu adaylar test edildi:

```text
base_disabled
orders_2_3_mean
orders_2_3_4_mean
orders_3_4_mean
orders_2_3_4_weighted_center
orders_2_3_4_weighted_high
orders_2_3_4_max
```

BATADAL için test edilen score threshold değerleri:

```text
[3.5066, 3.9120, 4.2, 4.5]
```

SKAB için test edilen score threshold değerleri:

```text
[0.1054, 0.2231, 0.3567, 0.5108]
```

---

## İlk Ensemble Deneyi

İlk final koşuda ensemble global olarak açık bırakıldı.

Kullanılan ayar:

```json
{
  "ensemble_enabled": true,
  "ensemble_orders": [2, 3, 4],
  "ensemble_weights": null,
  "ensemble_aggregation": "mean"
}
```

### İlk Final Sonuçları

| Dataset | Scenario | F1 |
|---|---|---:|
| BATADAL | original | 0.4211 |
| SKAB | original | 0.4911 |

Bu sonuçlar, global ensemble kullanımının BATADAL tarafında performansı düşürdüğünü gösterdi.

BATADAL base sonucu:

```text
F1 = 0.4444
```

Global ensemble sonucu:

```text
F1 = 0.4211
```

Bu nedenle ensemble’ın her dataset için aynı şekilde açılmasının doğru olmadığı görüldü.

---

## Ensemble Sweep Sonuçları

### BATADAL

BATADAL tarafında en iyi sonuç base model ile veya base’e çok yakın ayarlarla elde edildi.

En iyi BATADAL sonucu:

| Candidate | Ensemble | Orders | Aggregation | Score Threshold | Precision | Recall | F1 |
|---|---|---|---|---:|---:|---:|---:|
| `base_disabled` | false | `[3]` | mean | 3.9120 | 0.3636 | 0.5714 | 0.4444 |

Bazı ensemble ayarları BATADAL’da base’e yaklaşsa da base’i geçemedi.

Örneğin:

| Candidate | Orders | Aggregation | Score Threshold | F1 |
|---|---|---|---:|---:|
| `orders_2_3_mean` | `[2, 3]` | mean | 3.9120 | 0.4444 |
| `orders_2_3_4_mean` | `[2, 3, 4]` | mean | 3.9120 | 0.4211 |
| `orders_3_4_mean` | `[3, 4]` | mean | 3.9120 | 0.4211 |
| `orders_2_3_4_max` | `[2, 3, 4]` | max | 3.9120 | 0.4211 |

BATADAL yorumu:

```text
BATADAL için order=3 base model en iyi dengeyi korumaktadır.
Ensemble kullanımı genel olarak performansı artırmamış, bazı ayarlarda düşürmüştür.
Bu nedenle BATADAL final koşuda ensemble kapalı bırakılmıştır.
```

---

### SKAB

SKAB tarafında ensemble daha olumlu sonuç vermiştir.

En iyi SKAB sonucu:

| Candidate | Ensemble | Orders | Aggregation | Score Threshold | Precision | Recall | F1 |
|---|---|---|---|---:|---:|---:|---:|
| `orders_2_3_4_max` | true | `[2, 3, 4]` | max | 0.2231 | 0.4950 | 0.6090 | 0.4958 |

Base SKAB sonucu:

| Candidate | Ensemble | Score Threshold | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|
| `base_disabled` | false | 0.2231 | 0.4951 | 0.5975 | 0.4911 |

SKAB tarafında ensemble ile elde edilen değişim:

```text
Base F1      = 0.4911
Ensemble F1  = 0.4958
Delta        = +0.0047
```

Bu artış büyük değildir; ancak önceki feature branch’lere kıyasla daha temizdir çünkü BATADAL performansı dataset-specific ayarla korunabilmektedir.

---

## Refined Final Deneyi

İlk ensemble deneyi sonrasında dataset-specific final ayar denenmiştir.

Bu refined final koşuda kullanılan strateji:

```text
BATADAL -> ensemble kapalı
SKAB    -> ensemble açık, orders=[2,3,4], aggregation=max
```

### Refined Final Config

```json
{
  "run_ensemble_markov_sweep": false,

  "batadal_final_ensemble_enabled": false,
  "batadal_final_ensemble_orders": [3],
  "batadal_final_ensemble_weights": null,
  "batadal_final_ensemble_aggregation": "mean",

  "skab_final_ensemble_enabled": true,
  "skab_final_ensemble_orders": [2, 3, 4],
  "skab_final_ensemble_weights": null,
  "skab_final_ensemble_aggregation": "max"
}
```

Final threshold değerleri değiştirilmemiştir:

```json
{
  "batadal_final_score_threshold": 3.9120,
  "skab_final_score_threshold": 0.2231
}
```

---

## Refined Final Sonuçları

### BATADAL

BATADAL’da ensemble kapalı bırakılarak base performans korunmuştur.

| Scenario | Precision | Recall | F1 |
|---|---:|---:|---:|
| original | 0.3636 | 0.5714 | 0.4444 |
| gaussian_noise | 0.3121 | 0.4857 | 0.3796 |
| unseen_data | 0.2500 | 1.0000 | 0.4000 |

BATADAL yorumu:

```text
Dataset-specific ayar sayesinde BATADAL performansı bozulmadan korunmuştur.
```

---

### SKAB

SKAB’da ensemble açık bırakılmış ve `max` aggregation kullanılmıştır.

| Scenario | Precision | Recall | F1 |
|---|---:|---:|---:|
| original | 0.4950 | 0.6090 | 0.4958 |
| gaussian_noise | 0.4797 | 0.6071 | 0.4915 |
| unseen_data | 0.1500 | 0.2000 | 0.1714 |

SKAB yorumu:

```text
SKAB üzerinde ensemble max aggregation, recall değerini artırarak küçük bir F1 iyileşmesi sağlamıştır.
```

---

## Genel Karşılaştırma

| Dataset | Base F1 | Ensemble Final F1 | Değişim |
|---|---:|---:|---:|
| BATADAL | 0.4444 | 0.4444 | 0.0000 |
| SKAB | 0.4911 | 0.4958 | +0.0047 |

Bu sonuçlar, ensemble yaklaşımının global olarak değil, dataset-specific şekilde kullanılması gerektiğini göstermektedir.

---

## Deep Learning Karşılaştırması

Final statistical test çıktısında Automata mean değerleri şu şekildedir:

| Dataset | Model | Mean F1 |
|---|---|---:|
| SKAB | LSTM | 0.8404 |
| SKAB | GRU | 0.8589 |
| SKAB | Automata | 0.4958 |
| BATADAL | LSTM | 0.5609 |
| BATADAL | GRU | 0.5585 |
| BATADAL | Automata | 0.4444 |

Wilcoxon testlerinde p-value değerleri anlamlılık eşiğinin altında değildir:

| Dataset | Karşılaştırma | p-value | Anlamlı |
|---|---|---:|---|
| SKAB | LSTM vs GRU | 0.0625 | Hayır |
| SKAB | LSTM vs Automata | 0.0625 | Hayır |
| SKAB | GRU vs Automata | 0.0625 | Hayır |
| BATADAL | LSTM vs GRU | 1.0000 | Hayır |
| BATADAL | LSTM vs Automata | 0.3125 | Hayır |
| BATADAL | GRU vs Automata | 0.1875 | Hayır |

Bu sonuçlar, otomata modelinin deep learning modellerini geçmediğini göstermektedir. Ancak ensemble branch, otomata tarafında şimdiye kadarki en temiz küçük iyileştirmeyi sağlamıştır.

---

## Önceki Feature Branch’lerle Karşılaştırma

| Feature Branch | BATADAL Etkisi | SKAB Etkisi | Karar |
|---|---|---|---|
| `feature/entropy-threshold` | Ciddi düşüş | Belirgin katkı yok | Zayıf aday |
| `feature/transition-confidence` | Düşüş | Çok küçük artış | Zayıf aday |
| `feature/dirichlet-smoothing` | Base’i ancak yakaladı | Çok küçük artış | Neutral / zayıf aday |
| `feature/ensemble-markov` | Base korundu | Küçük artış | Şu an en güçlü aday |

---

## Teknik Yorum

Ensemble Markov yaklaşımı, tek bir transition probability estimator’ı değiştirmek yerine farklı context uzunluklarını birleştirdiği için daha güvenli bir yaklaşım sunmuştur.

BATADAL tarafında:

```text
order=3 base model zaten en iyi precision/recall dengesini sağlamaktadır.
order=4 veya çoklu order ensemble eklenince bu denge bozulabilmektedir.
Bu yüzden BATADAL’da ensemble kapalı bırakılmıştır.
```

SKAB tarafında:

```text
Farklı order skorlarının max aggregation ile birleştirilmesi recall değerini artırmıştır.
Recall artışı F1 skoruna küçük ama pozitif katkı sağlamıştır.
```

Bu sonuçlar, ensemble mekanizmasının dataset-independent değil, dataset-specific uygulanması gerektiğini göstermektedir.

---

## Önerilen Final Ayar

Şu anki en iyi dataset-specific ayar:

```json
{
  "batadal_final_ensemble_enabled": false,
  "batadal_final_ensemble_orders": [3],
  "batadal_final_ensemble_weights": null,
  "batadal_final_ensemble_aggregation": "mean",

  "skab_final_ensemble_enabled": true,
  "skab_final_ensemble_orders": [2, 3, 4],
  "skab_final_ensemble_weights": null,
  "skab_final_ensemble_aggregation": "max"
}
```

---

## Üretilen Çıktılar

Bu branch’te aşağıdaki çıktı dosyaları üretilmiştir:

```text
results/outputs/automata_ensemble_markov_sweep_metrics.csv
results/outputs/automata_ensemble_markov_sweep_summary.csv
results/outputs/automata_ensemble_markov_best_candidates.csv

results/outputs/automata_advanced_all_scenarios_metrics.csv
results/outputs/automata_skab_fold_summary.csv
results/outputs/automata_batadal_seed_summary.csv
results/outputs/statistical_test_results.csv
```

---

## Sonuç

`feature/ensemble-markov` branch’i şu ana kadarki en umut verici feature branch’tir.

Final sonuç:

```text
BATADAL:
- Ensemble kapalı
- Base F1 korunur: 0.4444

SKAB:
- Ensemble açık
- orders=[2,3,4]
- aggregation=max
- F1: 0.4911 -> 0.4958
```

Bu nedenle branch, tüm feature denemeleri tamamlandıktan sonra merge için yeniden değerlendirilecektir.