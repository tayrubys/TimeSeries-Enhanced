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

---

## Window Size + Score Threshold Ablation

Ensemble-Markov branch üzerinde ek olarak `window_size` ve `score_threshold` değerlerinin performansa etkisi incelendi.

Bu çalışma için ayrı bir ablation script’i kullanıldı:

```text
src/experiments/window_threshold_ablation.py
```

Bu script ile farklı `window_size` ve `score_threshold` kombinasyonları denenerek ensemble Markov mekanizmasının farklı pencere boyutlarında nasıl davrandığı gözlemlendi.

---

### Deney Kurulumu

Ablation çalışması şu branch üzerinde yürütüldü:

```text
feature/ensemble-markov
```

Script dry-run aşamasında branch’i doğru şekilde algıladı:

```text
Branch slug: ensemble_markov
```

Model tarafında ensemble Markov parametreleri desteklenen model parametreleri içinde görüldü:

```text
ensemble_enabled
ensemble_orders
ensemble_weights
ensemble_aggregation
```

Bu nedenle mevcut `window_threshold_ablation.py` script’inin ensemble Markov branch ile uyumlu olduğu görüldü.

Kod kontrolünde ensemble mekanizmasının hem `calculate_scores()` hem de `predict()` içinde uygulandığı görüldü. Bu nedenle `Fast predict=True` ayarının ensemble etkisini bypass etmediği değerlendirildi.

Ablation çalışması şu ayarla yürütüldü:

```text
Fast predict: True
```

Test edilen window size aralığı:

```text
window_size = [3, 4, 5, 6, 7, 8, 9, 10]
```

Main ablation senaryoları:

```text
original
gaussian_noise
```

Main ablation tamamlandıktan sonra `original` F1 skoruna göre en iyi 3 aday seçildi. Daha sonra yalnızca bu en iyi 3 aday için `unseen_data` değerlendirmesi yapıldı.

Çıktılar şu klasör altında üretildi:

```text
results/outputs/window_threshold_ablation/ensemble_markov/
```

---

### BATADAL Window Size Sonuçları

BATADAL tarafında en iyi sonuç şu kombinasyonla elde edildi:

```text
window_size = 4
score_threshold = 3.9120
precision = 0.363636
recall = 0.571429
F1 = 0.444444
accuracy = 0.951220
```

En iyi 3 aday:

```text
1. window_size = 4
   score_threshold = 3.9120
   precision = 0.363636
   recall = 0.571429
   F1 = 0.444444

2. window_size = 4
   score_threshold = 3.5066
   precision = 0.148148
   recall = 0.571429
   F1 = 0.235294

3. window_size = 4
   score_threshold = 4.2000
   precision = 0.500000
   recall = 0.142857
   F1 = 0.222222
```

BATADAL için en iyi pencere boyutu yine `window_size=4` oldu.

Baseline BATADAL sonucu:

```text
Baseline BATADAL F1 = 0.444444
```

Ensemble-Markov window ablation sonrası en iyi BATADAL sonucu:

```text
Ensemble-Markov best BATADAL F1 = 0.444444
```

Bu nedenle ensemble Markov branch, BATADAL üzerinde baseline performansını korudu. Window size ve threshold araması sonrasında BATADAL tarafında ek bir iyileşme görülmedi, ancak önceki bazı feature branch’lerin aksine performans düşüşü de oluşmadı.

---

### BATADAL Unseen Data Sonuçları

En iyi original F1 adayının unseen-data sonucu:

```text
window_size = 4
score_threshold = 3.9120
unseen_precision = 0.250000
unseen_recall = 1.000000
unseen_F1 = 0.400000
unseen_accuracy = 0.812500
```

Diğer adayların unseen-data sonuçları:

```text
window_size = 4
score_threshold = 3.5066
unseen_F1 = 0.222222

window_size = 4
score_threshold = 4.2000
unseen_F1 = 0.000000
```

Baseline BATADAL unseen F1 değeri daha önce `0.400000` seviyesindeydi. Ensemble-Markov window ablation sonucunda bu değer korunmuştur.

---

### BATADAL Yorumu

BATADAL tarafında `window_size=4` açık şekilde en iyi sonucu verdi.

`window_size=3` düşük threshold değerlerinde recall’u yükseltti, ancak precision çok düşük kaldı. Bu durum modelin çok fazla false positive ürettiğini gösterdi.

`window_size=5` ve üzerindeki değerlerde ise F1 skorları genel olarak `0.0` kaldı. Bu sonuç, BATADAL için daha büyük window size değerlerinin ensemble Markov branch’inde de işe yaramadığını gösterdi.

BATADAL için genel sonuç:

```text
Ensemble-Markov branch, BATADAL üzerinde baseline performansını korudu.
En iyi sonuç window_size=4 ve score_threshold=3.9120 ile elde edildi.
Original F1 = 0.444444
Unseen-data F1 = 0.400000
```

Bu sonuç, ensemble Markov yaklaşımının BATADAL tarafında zarar vermediğini, ancak base modelin üzerine çıkamadığını göstermektedir.

---

### SKAB Window Size Sonuçları

SKAB tarafında en iyi original F1 sonucu şu kombinasyonla elde edildi:

```text
window_size = 5
score_threshold = 0.05
precision = 0.371420
recall = 1.000000
F1 = 0.541259
accuracy = 0.387222
```

En iyi 3 aday:

```text
1. window_size = 5
   score_threshold = 0.05
   precision = 0.371420
   recall = 1.000000
   F1 = 0.541259

2. window_size = 4
   score_threshold = 0.05
   precision = 0.391446
   recall = 0.863659
   F1 = 0.537854

3. window_size = 5
   score_threshold = 0.00
   precision = 0.361601
   recall = 1.000000
   F1 = 0.531125
```

Önceki referans SKAB ayarı:

```text
window_size = 4
score_threshold = 0.2231
F1 ≈ 0.4911
```

Window ablation sonrası en iyi SKAB original sonucu:

```text
window_size = 5
score_threshold = 0.05
F1 = 0.541259
```

Bu sonuç SKAB original senaryoda sayısal olarak F1 artışı olduğunu gösterdi. Ancak bu artışın temel nedeni recall değerinin `1.0` seviyesine çıkmasıdır.

En iyi adayda:

```text
recall = 1.000000
precision = 0.371420
accuracy = 0.387222
```

Bu durum modelin anomalileri yakaladığını, ancak çok fazla false positive ürettiğini göstermektedir.

---

### SKAB Unseen Data Sonuçları

En iyi adayların unseen-data sonuçları:

```text
window_size = 5
score_threshold = 0.05
unseen_precision = 0.145000
unseen_recall = 0.200000
unseen_F1 = 0.168116
unseen_accuracy = 0.748837
```

```text
window_size = 4
score_threshold = 0.05
unseen_precision = 0.150000
unseen_recall = 0.200000
unseen_F1 = 0.171429
unseen_accuracy = 0.556522
```

```text
window_size = 5
score_threshold = 0.00
unseen_precision = 0.145000
unseen_recall = 0.200000
unseen_F1 = 0.168116
unseen_accuracy = 0.748837
```

En iyi unseen-data sonucu şu adayda elde edildi:

```text
window_size = 4
score_threshold = 0.05
unseen_F1 = 0.171429
```

Önceki SKAB unseen referansı yaklaşık `0.1667` seviyesindeydi. Bu nedenle ensemble Markov window ablation, SKAB unseen-data tarafında çok küçük bir artış sağlamıştır.

```text
Baseline/reference unseen F1 ≈ 0.1667
Ensemble-Markov unseen F1 = 0.171429
Delta ≈ +0.0047
```

Ancak bu fark çok küçük olduğu için güçlü bir genelleme iyileşmesi olarak değerlendirilmedi.

---

### SKAB Yorumu

SKAB tarafında `window_size=5` ve düşük threshold değerleri original F1 skorunu artırdı.

Ancak bu artış yüksek recall ve düşük precision kaynaklıdır. Accuracy değerinin düşük kalması da modelin fazla false positive ürettiğini desteklemektedir.

SKAB unseen-data tarafında çok küçük bir F1 artışı görülse de bu artış sınırlıdır.

SKAB için genel sonuç:

```text
Ensemble-Markov branch, SKAB original senaryoda recall ağırlıklı bir F1 artışı sağladı.
Unseen-data tarafında çok küçük bir iyileşme görüldü.
Ancak precision ve accuracy düşük kaldığı için bu artış güçlü ve dengeli bir model geliştirmesi olarak değerlendirilmedi.
```

---

### Window Size Ablation Genel Yorumu

Window size ablation sonuçları ensemble Markov branch için şu tabloyu ortaya koydu:

```text
BATADAL:
- En iyi window_size yine 4 oldu.
- En iyi original F1 = 0.444444
- Baseline performansı korundu.
- Unseen-data F1 = 0.400000 ile baseline seviyesi korundu.
- Büyük window size değerleri anlamlı sonuç üretmedi.

SKAB:
- En iyi original F1 window_size=5 ve threshold=0.05 ile elde edildi.
- En iyi original F1 = 0.541259
- Bu artış yüksek recall kaynaklıydı.
- Precision ve accuracy düşük kaldı.
- Unseen-data tarafında çok küçük bir artış görüldü.
```

Önceki feature branch’lerle karşılaştırıldığında ensemble Markov daha olumlu görünmektedir. Bunun nedeni BATADAL tarafında performansı bozmaması ve SKAB tarafında küçük de olsa pozitif etki üretmesidir.

Ancak window size ablation sonucu, SKAB original tarafındaki yüksek F1 değerinin büyük ölçüde recall artışından kaynaklandığını göstermiştir. Bu nedenle sonuç dengeli ve güçlü bir iyileşme olarak değerlendirilmemelidir.

---

### Branch Kararı

Window size ve score threshold araması ensemble Markov branch’in genel değerlendirmesini tamamen değiştirmedi, ancak mevcut yorumu güçlendirdi.

```text
feature/ensemble-markov = sınırlı pozitif / en güçlü adaylardan biri
```

Daha açıklayıcı karar:

```text
Ensemble-Markov branch, BATADAL üzerinde baseline performansını korumuş, SKAB original senaryosunda recall ağırlıklı bir F1 artışı üretmiş ve SKAB unseen-data üzerinde çok küçük bir iyileşme sağlamıştır. Bu nedenle önceki feature branch’lere göre daha güçlü bir adaydır. Ancak SKAB artışının düşük precision ve düşük accuracy ile gelmesi nedeniyle final model için tek başına güçlü kanıt olarak değerlendirilmemelidir.
```

Bu branch, tüm feature denemeleri tamamlandıktan sonra final değerlendirmede yeniden ele alınmalıdır.

---

### Üretilen Window Ablation Çıktıları

Window size ve score threshold ablation çalışması sonucunda aşağıdaki dosyalar üretildi:

```text
results/outputs/window_threshold_ablation/ensemble_markov/01_main_ablation/ensemble_markov_batadal_main_metrics.csv
results/outputs/window_threshold_ablation/ensemble_markov/01_main_ablation/ensemble_markov_batadal_main_summary.csv
results/outputs/window_threshold_ablation/ensemble_markov/01_main_ablation/ensemble_markov_batadal_main_best_candidates.csv
results/outputs/window_threshold_ablation/ensemble_markov/02_unseen_top_candidates/ensemble_markov_batadal_unseen_top_candidates_metrics.csv
results/outputs/window_threshold_ablation/ensemble_markov/02_unseen_top_candidates/ensemble_markov_batadal_unseen_top_candidates_summary.csv
results/outputs/window_threshold_ablation/ensemble_markov/03_reports/ensemble_markov_batadal_final_comparison.csv

results/outputs/window_threshold_ablation/ensemble_markov/01_main_ablation/ensemble_markov_skab_main_metrics.csv
results/outputs/window_threshold_ablation/ensemble_markov/01_main_ablation/ensemble_markov_skab_main_summary.csv
results/outputs/window_threshold_ablation/ensemble_markov/01_main_ablation/ensemble_markov_skab_main_best_candidates.csv
results/outputs/window_threshold_ablation/ensemble_markov/02_unseen_top_candidates/ensemble_markov_skab_unseen_top_candidates_metrics.csv
results/outputs/window_threshold_ablation/ensemble_markov/02_unseen_top_candidates/ensemble_markov_skab_unseen_top_candidates_summary.csv
results/outputs/window_threshold_ablation/ensemble_markov/03_reports/ensemble_markov_skab_final_comparison.csv
```