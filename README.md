# feature/time-decay

Bu branch, probabilistic automata tabanlı high-order Markov modeline time-decay ağırlıklandırması eklemek için oluşturulmuştur.

Time-decay yaklaşımının amacı, eğitim sırasında görülen transition kayıtlarını zamansal konumlarına göre ağırlıklandırmaktır. Klasik Markov sayımında her transition eşit ağırlığa sahiptir. Bu branch’te ise daha yeni transition’ların daha yüksek, daha eski transition’ların daha düşük ağırlıkla modele katkı vermesi denenmiştir.

Bu branch henüz `markov-base` branch’ine merge edilmemiştir. Tüm feature branch’leri denendikten sonra final değerlendirme yapılacak ve gerekirse merge kararı verilecektir.

Mevcut sonuçlara göre `feature/time-decay`, şu ana kadar denenmiş feature branch’leri içinde en güçlü adaydır.

---

## Amaç

Mevcut `markov-base` modelinde transition sayımları şu şekilde yapılmaktadır:

```text
Her görülen transition +1 olarak sayılır.
Eski transition = yeni transition
```

Time-decay yaklaşımı ise bu mantığı değiştirir:

```text
Yeni transition'lar daha yüksek ağırlık alır.
Eski transition'ların etkisi decay rate oranına göre azaltılır.
```

Temel amaç:

```text
Zamansal olarak daha güncel transition davranışlarını modele daha güçlü yansıtmak.
```

Bu özellikle zaman serisi anomaly detection problemlerinde faydalı olabilir; çünkü bazı sistemlerde normal davranış zaman içinde değişebilir.

---

## Eklenen Parametreler

`ProbabilisticAutomata` sınıfına aşağıdaki parametreler eklenmiştir:

```python
time_decay_enabled=False
time_decay_rate=0.99
time_decay_min_weight=0.01
```

### Parametre Açıklamaları

| Parametre | Açıklama |
|---|---|
| `time_decay_enabled` | Time-decay mekanizmasını açar veya kapatır. |
| `time_decay_rate` | Eski transition ağırlıklarının ne kadar hızlı azalacağını belirler. |
| `time_decay_min_weight` | Çok eski transition'ların düşebileceği minimum ağırlığı belirler. |

---

## Time-Decay Mantığı

Time-decay kapalıyken model base davranışını korur:

```text
time_decay_enabled=False
transition_weight = 1.0
```

Time-decay açıkken transition ağırlığı şu mantıkla hesaplanır:

```text
age = son_transition_index - mevcut_transition_index
weight = time_decay_rate ** age
weight = max(weight, time_decay_min_weight)
```

Yani:

```text
En yeni transition ağırlığı yaklaşık 1.0 olur.
Daha eski transition'ların ağırlığı kademeli olarak azalır.
Ağırlık time_decay_min_weight değerinin altına düşmez.
```

Örnek:

```python
ProbabilisticAutomata(
    smoothing=True,
    order=3,
    smoothing_alpha=0.5,
    time_decay_enabled=True,
    time_decay_rate=0.999,
    time_decay_min_weight=0.01
)
```

---

## Modelde Yapılan Değişiklikler

Bu branch’te modelin temel high-order Markov yapısı korunmuştur.

Değişiklik sadece transition sayım aşamasına eklenmiştir.

Base modelde transition update:

```python
self.transitions[state][next_pattern] += 1
self.total_exits[state] += 1
```

Time-decay branch’inde:

```python
self.transitions[state][next_pattern] += transition_weight
self.total_exits[state] += transition_weight
```

Böylece transition probability hesaplaması aynı kalır; ancak probability değerleri artık decay uygulanmış ağırlıklı transition sayımlarına göre hesaplanır.

---

## Runner Değişiklikleri

`runner.py` dosyasına time-decay parametreleri eklenmiştir.

Final koşulara şu alanlar dahil edilmiştir:

```python
time_decay_enabled
time_decay_rate
time_decay_min_weight
```

Bu parametreler:

- model oluşturulurken `ProbabilisticAutomata` sınıfına geçirilmiştir,
- deney sonuçlarına metadata olarak eklenmiştir,
- SKAB ve BATADAL summary çıktılarında raporlanmıştır,
- time-decay sweep fonksiyonuna dahil edilmiştir.

Ayrıca şu parametre eklenmiştir:

```python
run_time_decay_sweep
```

---

## Eklenen Sweep Mekanizması

Bu branch’te farklı time-decay kombinasyonlarını test etmek için ayrı bir sweep mekanizması eklenmiştir.

Sweep sırasında test edilen decay rate değerleri:

```text
[0.999, 0.995, 0.99, 0.98, 0.95]
```

Test edilen minimum weight değerleri:

```text
[0.001, 0.01, 0.05]
```

BATADAL için test edilen score threshold değerleri:

```text
[3.5066, 3.9120, 4.2, 4.5]
```

SKAB için test edilen score threshold değerleri:

```text
[0.1054, 0.2231, 0.3567, 0.5108]
```

Sweep sonucunda şu dosyalar üretilmiştir:

```text
results/outputs/automata_time_decay_sweep_metrics.csv
results/outputs/automata_time_decay_sweep_summary.csv
results/outputs/automata_time_decay_best_candidates.csv
```

---

## İlk Time-Decay Deneyi

İlk final koşuda time-decay global olarak açık bırakılmıştır.

Kullanılan ayar:

```json
{
  "time_decay_enabled": true,
  "time_decay_rate": 0.99,
  "time_decay_min_weight": 0.01
}
```

### İlk Final Sonuçları

#### SKAB

| Scenario | Precision | Recall | F1 |
|---|---:|---:|---:|
| original | 0.3657 | 0.9885 | 0.5334 |
| gaussian_noise | 0.3645 | 0.9839 | 0.5313 |
| unseen_data | 0.1429 | 0.2000 | 0.1667 |

SKAB için base F1:

```text
0.4911
```

Global time-decay ile SKAB F1:

```text
0.5334
```

Bu sonuç SKAB tarafında ciddi bir artış göstermiştir.

#### BATADAL

| Scenario | Precision | Recall | F1 |
|---|---:|---:|---:|
| original | 0.0363 | 1.0000 | 0.0700 |
| gaussian_noise | 0.0347 | 0.9429 | 0.0670 |
| unseen_data | 0.0769 | 1.0000 | 0.1429 |

BATADAL için base F1:

```text
0.4444
```

Global time-decay ile BATADAL F1:

```text
0.0700
```

Bu sonuç, global time-decay kullanımının BATADAL tarafında ciddi performans düşüşüne neden olduğunu göstermiştir.

---

## İlk Deney Yorumu

İlk global time-decay deneyi iki farklı etki göstermiştir:

```text
SKAB    -> ciddi iyileşme
BATADAL -> ciddi bozulma
```

BATADAL’da recall değeri 1.0 olmasına rağmen precision çok düşük kalmıştır. Bu, modelin çok fazla false positive ürettiğini göstermektedir.

Bu nedenle time-decay mekanizmasının global olarak açılması uygun bulunmamıştır.

---

## Sweep Sonuçları

### BATADAL

BATADAL tarafında time-decay açıkken base modeli geçen bir ayar bulunamamıştır.

En iyi BATADAL sonucu yine base-disabled ayar ile elde edilmiştir:

| Candidate | Time-Decay | Rate | Min Weight | Score Threshold | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| `base_disabled` | false | 0.99 | 0.01 | 3.9120 | 0.3636 | 0.5714 | 0.4444 |

Time-decay açık ayarlarda bazı sonuçlar base’e yaklaşsa da base’i geçememiştir.

Örnek:

| Candidate | Rate | Min Weight | Score Threshold | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| `decay_rate_0.999_min_0.001` | 0.999 | 0.001 | 3.9120 | 0.2500 | 0.4286 | 0.3158 |
| `decay_rate_0.999_min_0.01` | 0.999 | 0.01 | 3.9120 | 0.2500 | 0.4286 | 0.3158 |
| `decay_rate_0.99_min_0.05` | 0.99 | 0.05 | 3.9120 | 0.0822 | 0.8571 | 0.1500 |

BATADAL yorumu:

```text
BATADAL için time-decay mekanizması transition dağılımını bozmuş ve false positive oranını artırmıştır.
Bu nedenle BATADAL final koşuda time-decay kapalı bırakılmalıdır.
```

---

### SKAB

SKAB tarafında time-decay güçlü sonuç vermiştir.

En iyi SKAB sweep sonucu:

| Candidate | Time-Decay | Rate | Min Weight | Score Threshold | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| `decay_rate_0.999_min_0.001` | true | 0.999 | 0.001 | 0.3567 | 0.5503 | 0.7404 | 0.5740 |
| `decay_rate_0.999_min_0.01` | true | 0.999 | 0.01 | 0.3567 | 0.5503 | 0.7404 | 0.5740 |

Base SKAB sonucu:

| Candidate | Time-Decay | Score Threshold | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|
| `base_disabled` | false | 0.2231 | 0.4951 | 0.5975 | 0.4911 |

SKAB tarafında sweep ile elde edilen iyileşme:

```text
Base F1       = 0.4911
Time-decay F1 = 0.5740
Delta         = +0.0829
```

Bu artış, daha önceki feature branch’lerde elde edilen iyileşmelerden belirgin şekilde daha yüksektir.

---

## Refined Final Deneyi

İlk global deneyden sonra dataset-specific final ayar denenmiştir.

Refined final strateji:

```text
BATADAL -> time-decay kapalı
SKAB    -> time-decay açık
```

Kullanılan final config:

```json
{
  "run_time_decay_sweep": false,

  "time_decay_enabled": false,
  "time_decay_rate": 0.99,
  "time_decay_min_weight": 0.01,

  "batadal_final_time_decay_enabled": false,
  "batadal_final_time_decay_rate": 0.99,
  "batadal_final_time_decay_min_weight": 0.01,

  "skab_final_time_decay_enabled": true,
  "skab_final_time_decay_rate": 0.999,
  "skab_final_time_decay_min_weight": 0.01,

  "batadal_final_score_threshold": 3.9120,
  "skab_final_score_threshold": 0.3567
}
```

---

## Refined Final Sonuçları

### BATADAL

BATADAL’da time-decay kapalı bırakılmış ve base performans korunmuştur.

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

SKAB’da time-decay açık bırakılmıştır.

Kullanılan ayar:

```text
time_decay_enabled=True
time_decay_rate=0.999
time_decay_min_weight=0.01
score_threshold=0.3567
```

Final sonuç:

| Scenario | Precision | Recall | F1 |
|---|---:|---:|---:|
| original | 0.5503 | 0.7404 | 0.5740 |
| gaussian_noise | 0.5363 | 0.7392 | 0.5706 |
| unseen_data | 0.1429 | 0.2000 | 0.1667 |

SKAB yorumu:

```text
Time-decay, SKAB üzerinde hem precision hem de recall dengesini base modele göre iyileştirmiştir.
F1 değeri 0.4911'den 0.5740'a yükselmiştir.
```

---

## Genel Karşılaştırma

| Dataset | Base F1 | Time-Decay Final F1 | Değişim |
|---|---:|---:|---:|
| BATADAL | 0.4444 | 0.4444 | 0.0000 |
| SKAB | 0.4911 | 0.5740 | +0.0829 |

Bu sonuçlar, time-decay mekanizmasının global değil, dataset-specific şekilde kullanılması gerektiğini göstermektedir.

---

## Önceki Feature Branch’lerle Karşılaştırma

| Feature Branch | BATADAL F1 | SKAB F1 | Yorum |
|---|---:|---:|---|
| `markov-base` | 0.4444 | 0.4911 | Baseline |
| `feature/entropy-threshold` | ciddi düşüş | belirgin katkı yok | Zayıf aday |
| `feature/transition-confidence` | düşüş | çok küçük artış | Zayıf aday |
| `feature/dirichlet-smoothing` | base’i yakaladı | çok küçük artış | Neutral / zayıf aday |
| `feature/ensemble-markov` | 0.4444 | 0.4958 | Önceki en güçlü aday |
| `feature/time-decay` | 0.4444 | 0.5740 | Şu an en güçlü aday |

Time-decay branch, ensemble branch’e göre SKAB tarafında çok daha güçlü iyileşme sağlamıştır.

---

## Deep Learning Karşılaştırması

Final statistical test çıktısında Automata mean değerleri şu şekildedir:

| Dataset | Model | Mean F1 |
|---|---|---:|
| SKAB | LSTM | 0.8404 |
| SKAB | GRU | 0.8589 |
| SKAB | Automata | 0.5740 |
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

Deep learning modelleri hâlâ otomata modelinden daha yüksek F1 değerlerine sahiptir. Ancak time-decay branch, otomata modelini özellikle SKAB tarafında önceki baseline’a göre belirgin şekilde iyileştirmiştir.

---

## Teknik Yorum

Time-decay mekanizması SKAB ve BATADAL üzerinde farklı davranmıştır.

BATADAL tarafında:

```text
Time-decay transition dağılımını fazla agresif değiştirmiştir.
Model çok fazla anomaly üretmiştir.
Recall yükselmiş ancak precision ciddi şekilde düşmüştür.
Bu nedenle F1 skoru belirgin şekilde azalmıştır.
```

SKAB tarafında:

```text
Time-decay daha güncel transition davranışlarını öne çıkarmış ve anomaly detection performansını artırmıştır.
Özellikle rate=0.999 gibi yumuşak decay kullanıldığında precision/recall dengesi iyileşmiştir.
```

Bu sonuç, time-decay’in veri setinin temporal yapısına duyarlı olduğunu göstermektedir.

---

## Önerilen Final Ayar

Şu anki en iyi dataset-specific ayar:

```json
{
  "batadal_final_time_decay_enabled": false,
  "batadal_final_time_decay_rate": 0.99,
  "batadal_final_time_decay_min_weight": 0.01,

  "skab_final_time_decay_enabled": true,
  "skab_final_time_decay_rate": 0.999,
  "skab_final_time_decay_min_weight": 0.01,

  "batadal_final_score_threshold": 3.9120,
  "skab_final_score_threshold": 0.3567
}
```

---

## Üretilen Çıktılar

Bu branch’te aşağıdaki çıktı dosyaları üretilmiştir:

```text
results/outputs/automata_time_decay_sweep_metrics.csv
results/outputs/automata_time_decay_sweep_summary.csv
results/outputs/automata_time_decay_best_candidates.csv

results/outputs/automata_advanced_all_scenarios_metrics.csv
results/outputs/automata_skab_fold_summary.csv
results/outputs/automata_batadal_seed_summary.csv
results/outputs/statistical_test_results.csv
```

---

## Window Size + Score Threshold Ablation

Time-decay branch üzerinde ek olarak `window_size` ve `score_threshold` değerlerinin performansa etkisi incelendi.

Bu çalışma için ayrı bir ablation script’i kullanıldı:

```text
src/experiments/window_threshold_ablation.py
```

Bu script ile farklı `window_size` ve `score_threshold` kombinasyonları denenerek time-decay mekanizmasının farklı pencere boyutlarında nasıl davrandığı gözlemlendi.

---

### Deney Kurulumu

Ablation çalışması şu branch üzerinde yürütüldü:

```text
feature/time-decay
```

Script dry-run aşamasında branch’i doğru şekilde algıladı:

```text
Branch slug: time_decay
```

Model tarafında time-decay parametreleri desteklenen model parametreleri içinde görüldü:

```text
time_decay_enabled
time_decay_rate
time_decay_min_weight
```

Bu nedenle mevcut `window_threshold_ablation.py` script’inin time-decay branch ile uyumlu olduğu görüldü.

Kod kontrolünde time-decay mekanizmasının transition öğrenme aşamasında uygulandığı görüldü:

```text
transition_weight = self._get_time_decay_weight(age)
```

Bu nedenle `Fast predict=True` ayarının time-decay etkisini bypass etmediği değerlendirildi. Çünkü time-decay etkisi `predict()` aşamasında sonradan eklenen bir gate değil, transition ağırlıklarının öğrenilmesi sırasında uygulanmaktadır.

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
results/outputs/window_threshold_ablation/time_decay/
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

Time-decay window ablation sonrası en iyi BATADAL sonucu:

```text
Time-decay best BATADAL F1 = 0.444444
```

Bu nedenle time-decay branch, BATADAL üzerinde baseline performansını korudu. Window size ve threshold araması sonrasında BATADAL tarafında ek bir iyileşme görülmedi, ancak performans düşüşü de oluşmadı.

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

Baseline BATADAL unseen F1 değeri daha önce `0.400000` seviyesindeydi. Time-decay window ablation sonucunda bu değer korunmuştur.

---

### BATADAL Yorumu

BATADAL tarafında `window_size=4` açık şekilde en iyi sonucu verdi.

`window_size=3` düşük threshold değerlerinde recall’u yükseltti, ancak precision çok düşük kaldı. Bu durum modelin çok fazla false positive ürettiğini gösterdi.

`window_size=5` ve üzerindeki değerlerde ise F1 skorları genel olarak `0.0` kaldı. Bu sonuç, BATADAL için daha büyük window size değerlerinin time-decay branch’inde de işe yaramadığını gösterdi.

BATADAL için genel sonuç:

```text
Time-decay branch, BATADAL üzerinde baseline performansını korudu.
En iyi sonuç window_size=4 ve score_threshold=3.9120 ile elde edildi.
Original F1 = 0.444444
Unseen-data F1 = 0.400000
```

Bu sonuç, time-decay yaklaşımının BATADAL tarafında zarar vermediğini, ancak base modelin üzerine çıkamadığını göstermektedir.

---

### SKAB Window Size Sonuçları

SKAB tarafında en iyi original F1 sonucu şu kombinasyonla elde edildi:

```text
window_size = 4
score_threshold = 0.3567
precision = 0.550311
recall = 0.740352
F1 = 0.574023
accuracy = 0.641103
```

En iyi 3 aday:

```text
1. window_size = 4
   score_threshold = 0.3567
   precision = 0.550311
   recall = 0.740352
   F1 = 0.574023

2. window_size = 5
   score_threshold = 0.0000
   precision = 0.361195
   recall = 1.000000
   F1 = 0.530688

3. window_size = 5
   score_threshold = 0.0010
   precision = 0.361195
   recall = 1.000000
   F1 = 0.530688
```

Önceki referans SKAB ayarı:

```text
window_size = 4
score_threshold = 0.2231
F1 ≈ 0.4911
```

Window ablation sonrası en iyi SKAB original sonucu:

```text
window_size = 4
score_threshold = 0.3567
F1 = 0.574023
```

Bu sonuç SKAB original senaryoda anlamlı bir F1 artışı olduğunu gösterdi.

Önceki bazı feature branch’lerde SKAB tarafındaki F1 artışı çoğunlukla `recall=1.0` ve düşük precision ile oluşmuştu. Time-decay branch’te ise en iyi sonuç daha dengeli bir precision/recall dağılımı sağlamıştır:

```text
precision = 0.550311
recall = 0.740352
accuracy = 0.641103
```

Bu nedenle time-decay branch, SKAB original senaryoda önceki feature branch’lere göre daha dengeli bir iyileşme üretmiştir.

---

### SKAB Unseen Data Sonuçları

En iyi adayların unseen-data sonuçları:

```text
window_size = 4
score_threshold = 0.3567
unseen_precision = 0.142857
unseen_recall = 0.200000
unseen_F1 = 0.166667
unseen_accuracy = 0.547826
```

```text
window_size = 5
score_threshold = 0.0000
unseen_precision = 0.141463
unseen_recall = 0.200000
unseen_F1 = 0.165714
unseen_accuracy = 0.610853
```

```text
window_size = 5
score_threshold = 0.0010
unseen_precision = 0.141463
unseen_recall = 0.200000
unseen_F1 = 0.165714
unseen_accuracy = 0.610853
```

SKAB unseen-data tarafında anlamlı bir iyileşme elde edilmedi.

Baseline SKAB unseen F1 değeri yaklaşık şu seviyedeydi:

```text
Baseline/reference SKAB unseen F1 ≈ 0.166667
```

Time-decay window ablation sonucunda en iyi unseen F1 değeri:

```text
Time-decay SKAB unseen F1 = 0.166667
```

Bu nedenle SKAB original senaryodaki güçlü artış unseen-data tarafına taşınmamıştır.

---

### SKAB Yorumu

SKAB tarafında time-decay branch, şimdiye kadarki feature branch’ler içinde en dikkat çekici original F1 artışını sağlamıştır.

En önemli fark, bu artışın yalnızca recall değerinin `1.0` seviyesine çıkmasından kaynaklanmamasıdır. En iyi adayda precision, recall ve accuracy değerleri önceki recall-ağırlıklı sonuçlara göre daha dengelidir.

SKAB için genel sonuç:

```text
Time-decay branch, SKAB original senaryoda güçlü ve daha dengeli bir F1 artışı sağlamıştır.
Ancak unseen-data tarafında iyileşme sağlamamıştır.
Bu nedenle sonuç umut verici olsa da genelleme açısından dikkatli değerlendirilmelidir.
```

---

### Window Size Ablation Genel Yorumu

Window size ablation sonuçları time-decay branch için şu tabloyu ortaya koydu:

```text
BATADAL:
- En iyi window_size yine 4 oldu.
- En iyi original F1 = 0.444444
- Baseline performansı korundu.
- Unseen-data F1 = 0.400000 ile baseline seviyesi korundu.
- Büyük window size değerleri anlamlı sonuç üretmedi.

SKAB:
- En iyi original F1 window_size=4 ve threshold=0.3567 ile elde edildi.
- En iyi original F1 = 0.574023
- Precision, recall ve accuracy değerleri önceki feature branch’lere göre daha dengeli kaldı.
- Unseen-data tarafında iyileşme görülmedi.
```

Bu sonuçlara göre time-decay branch, önceki feature branch’lere kıyasla daha güçlü bir adaydır. Çünkü BATADAL performansını bozmamış, SKAB original senaryoda ise belirgin ve daha dengeli bir F1 artışı üretmiştir.

Ancak SKAB unseen-data üzerinde iyileşme sağlamadığı için final karar verilirken genelleme performansı ayrıca dikkate alınmalıdır.

---

### Önceki Feature Branch’lerle Karşılaştırma

Window ablation sonuçlarına göre time-decay branch’in konumu şu şekildedir:

```text
1. time-decay
   - BATADAL korunuyor
   - SKAB original belirgin ve daha dengeli iyileşiyor
   - SKAB unseen değişmiyor

2. ensemble-markov
   - BATADAL korunuyor
   - SKAB original artıyor fakat daha recall-ağırlıklı
   - SKAB unseen çok küçük artıyor

3. dirichlet-smoothing
   - BATADAL baseline altında veya ancak baseline seviyesinde
   - SKAB çok sınırlı artıyor

4. transition-confidence
   - BATADAL düşüyor
   - SKAB recall ağırlıklı artıyor

5. entropy-threshold
   - BATADAL ciddi düşüyor
   - Genelde güçlü aday değil
```

Bu sıralamada time-decay branch, şu ana kadarki en güçlü feature adayı olarak değerlendirilebilir.

---

### Branch Kararı

Window size ve score threshold araması time-decay branch için olumlu bir tablo ortaya koydu.

```text
feature/time-decay = güçlü aday / şu ana kadarki en iyi feature adayı
```

Daha açıklayıcı karar:

```text
Time-decay branch, BATADAL üzerinde baseline performansını korumuş, SKAB original senaryosunda ise belirgin ve daha dengeli bir F1 artışı sağlamıştır. SKAB unseen-data tarafında iyileşme görülmemiş olsa da, önceki feature branch’lerle karşılaştırıldığında en güçlü aday olarak değerlendirilmektedir.
```

Bu branch, final model değerlendirmesinde özellikle dikkate alınmalıdır.

---

### Üretilen Window Ablation Çıktıları

Window size ve score threshold ablation çalışması sonucunda aşağıdaki dosyalar üretildi:

```text
results/outputs/window_threshold_ablation/time_decay/01_main_ablation/time_decay_batadal_main_metrics.csv
results/outputs/window_threshold_ablation/time_decay/01_main_ablation/time_decay_batadal_main_summary.csv
results/outputs/window_threshold_ablation/time_decay/01_main_ablation/time_decay_batadal_main_best_candidates.csv
results/outputs/window_threshold_ablation/time_decay/02_unseen_top_candidates/time_decay_batadal_unseen_top_candidates_metrics.csv
results/outputs/window_threshold_ablation/time_decay/02_unseen_top_candidates/time_decay_batadal_unseen_top_candidates_summary.csv
results/outputs/window_threshold_ablation/time_decay/03_reports/time_decay_batadal_final_comparison.csv

results/outputs/window_threshold_ablation/time_decay/01_main_ablation/time_decay_skab_main_metrics.csv
results/outputs/window_threshold_ablation/time_decay/01_main_ablation/time_decay_skab_main_summary.csv
results/outputs/window_threshold_ablation/time_decay/01_main_ablation/time_decay_skab_main_best_candidates.csv
results/outputs/window_threshold_ablation/time_decay/02_unseen_top_candidates/time_decay_skab_unseen_top_candidates_metrics.csv
results/outputs/window_threshold_ablation/time_decay/02_unseen_top_candidates/time_decay_skab_unseen_top_candidates_summary.csv
results/outputs/window_threshold_ablation/time_decay/03_reports/time_decay_skab_final_comparison.csv
```