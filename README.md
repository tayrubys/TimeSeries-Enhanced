# feature/transition-confidence

## Amaç

Bu branch, Markov-chain tabanlı probabilistic automata modeline **transition-confidence** mekanizması eklemek için oluşturuldu.

Temel motivasyon şuydu:

> Aynı transition probability değerine sahip iki geçişin güvenilirliği aynı olmayabilir. Çünkü az gözlenmiş bir transition ile çok sık gözlenmiş bir transition istatistiksel olarak aynı güven seviyesinde değildir.

Örneğin:

```text
count=1 olan transition  -> düşük güven
count=1000 olan transition -> yüksek güven
```

Bu nedenle bu branch’te transition probability değerinin yanında, ilgili state-transition çiftinin ne kadar güvenilir olduğu da hesaba katılmaya çalışıldı.

---

## Eklenen Özellikler

`ProbabilisticAutomata` sınıfına transition-confidence destekli karar mekanizması eklendi.

Eklenen temel parametreler:

```python
transition_confidence_enabled=False
transition_confidence_k=10.0
transition_confidence_weight=1.0
transition_confidence_mode="additive"
transition_confidence_use_transition_count=True
```

### Parametreler

| Parametre | Açıklama |
|---|---|
| `transition_confidence_enabled` | Transition-confidence mekanizmasını açar veya kapatır. |
| `transition_confidence_k` | Confidence eğrisinin ne kadar hızlı doyuma ulaşacağını belirler. |
| `transition_confidence_weight` | Confidence penalty etkisinin ağırlığını belirler. |
| `transition_confidence_mode` | Confidence skorunun anomaly skoruna nasıl uygulanacağını belirler. |
| `transition_confidence_use_transition_count` | Confidence hesaplanırken spesifik transition count değerinin kullanılıp kullanılmayacağını belirler. |

---

## Confidence Mantığı

Bu branch’te kullanılan temel fikir:

```text
Düşük gözlem sayısı -> düşük confidence
Düşük confidence -> anomaly skoruna kontrollü ceza
```

Entropy-threshold branch’ten farklı olarak burada sert bir karar kapısı kullanılmadı.

Yani şu yaklaşım tercih edilmedi:

```text
uncertain transition => doğrudan anomaly
```

Bunun yerine daha yumuşak bir yaklaşım denendi:

```text
low confidence => anomaly score üzerinde kontrollü penalty
```

Bu nedenle transition-confidence, entropy-threshold’a göre daha az agresif bir yöntem olarak tasarlandı.

---

## Desteklenen Modlar

Kod tarafında aşağıdaki modlar desteklenecek şekilde tasarlandı:

```text
additive
multiplicative
probability_damping
```

Bu deneyde yalnızca aşağıdaki mod test edildi:

```text
transition_confidence_mode="additive"
```

---

## Deney Kurulumu

Base Markov automata ile transition-confidence branch’i aynı final otomata parametreleriyle karşılaştırıldı.

### BATADAL final parametreleri

```text
order=3
smoothing_alpha=0.1
decision_mode=avg_negative_log
score_window=5
score_threshold=3.912
anomaly_threshold=0.05
```

### SKAB final parametreleri

```text
order=3
smoothing_alpha=0.5
decision_mode=avg_negative_log
score_window=10
score_threshold=0.2231
anomaly_threshold=0.90
```

Transition-confidence default final koşuda şu ayarlarla çalıştırıldı:

```text
transition_confidence_enabled=True
transition_confidence_k=10.0
transition_confidence_weight=0.5
transition_confidence_mode=additive
transition_confidence_use_transition_count=True
```

Ayrıca sweep çalıştırıldı:

```text
transition_confidence_ks = [1.0, 2.0, 5.0, 10.0, 20.0]
transition_confidence_weights = [0.05, 0.1, 0.25, 0.5]
transition_confidence_modes = ["additive"]
transition_confidence_use_transition_count_options = [true]
```

---

## Base vs Transition-Confidence Sonuçları

### BATADAL

| Scenario | Base F1 | Transition-Confidence F1 | Sonuç |
|---|---:|---:|---|
| original | 0.4444 | 0.2174 | Gerileme |
| gaussian_noise | 0.3796 | 0.1903 | Gerileme |
| unseen_data | 0.4000 | 0.2222 | Gerileme |

Original BATADAL karşılaştırması:

| Model | Precision | Recall | F1 |
|---|---:|---:|---:|
| Base Markov Automata | 0.3636 | 0.5714 | 0.4444 |
| Transition-Confidence default | 0.1282 | 0.7143 | 0.2174 |

Transition-confidence default ayarı BATADAL’da recall’u artırdı fakat precision’ı ciddi şekilde düşürdü. Bu nedenle F1 skoru base modele göre belirgin şekilde geriledi.

---

## BATADAL Transition-Confidence Sweep

| Mode | Use Transition Count | k | Weight | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|
| additive | true | 1.0 | 0.05 | 0.3077 | 0.5714 | 0.4000 |
| additive | true | 1.0 | 0.10 | 0.3077 | 0.5714 | 0.4000 |
| additive | true | 1.0 | 0.25 | 0.2857 | 0.5714 | 0.3810 |
| additive | true | 1.0 | 0.50 | 0.1905 | 0.5714 | 0.2857 |
| additive | true | 2.0 | 0.05 | 0.3077 | 0.5714 | 0.4000 |
| additive | true | 2.0 | 0.10 | 0.3077 | 0.5714 | 0.4000 |
| additive | true | 2.0 | 0.25 | 0.2857 | 0.5714 | 0.3810 |
| additive | true | 2.0 | 0.50 | 0.1600 | 0.5714 | 0.2500 |
| additive | true | 5.0 | 0.05 | 0.3077 | 0.5714 | 0.4000 |
| additive | true | 5.0 | 0.10 | 0.3077 | 0.5714 | 0.4000 |
| additive | true | 5.0 | 0.25 | 0.2667 | 0.5714 | 0.3636 |
| additive | true | 5.0 | 0.50 | 0.1389 | 0.7143 | 0.2326 |
| additive | true | 10.0 | 0.05 | 0.3077 | 0.5714 | 0.4000 |
| additive | true | 10.0 | 0.10 | 0.3077 | 0.5714 | 0.4000 |
| additive | true | 10.0 | 0.25 | 0.2667 | 0.5714 | 0.3636 |
| additive | true | 10.0 | 0.50 | 0.1282 | 0.7143 | 0.2174 |
| additive | true | 20.0 | 0.05 | 0.3077 | 0.5714 | 0.4000 |
| additive | true | 20.0 | 0.10 | 0.3077 | 0.5714 | 0.4000 |
| additive | true | 20.0 | 0.25 | 0.2667 | 0.5714 | 0.3636 |
| additive | true | 20.0 | 0.50 | 0.1111 | 0.7143 | 0.1923 |

### BATADAL sweep yorumu

BATADAL için en iyi transition-confidence sonucu:

```text
Precision=0.3077
Recall=0.5714
F1=0.4000
```

Bu sonuç birden fazla ayarda elde edildi:

```text
k = 1.0, 2.0, 5.0, 10.0, 20.0
weight = 0.05 veya 0.10
mode = additive
```

Ancak base modelin BATADAL original F1 skoru:

```text
Base F1 = 0.4444
```

olduğu için, sweep sonucunda bulunan en iyi transition-confidence ayarı bile base modelin altında kaldı.

---

## SKAB Sonuçları

| Scenario | Base F1 | Transition-Confidence F1 | Sonuç |
|---|---:|---:|---|
| original | 0.4911 | 0.4940 | Çok küçük iyileşme |
| gaussian_noise | 0.4870 | 0.4898 | Çok küçük iyileşme |
| unseen_data | 0.1667 | 0.1667 | Etkisiz |

Original SKAB karşılaştırması:

| Model | Precision | Recall | F1 |
|---|---:|---:|---:|
| Base Markov Automata | 0.4951 | 0.5975 | 0.4911 |
| Transition-Confidence default | 0.4960 | 0.6035 | 0.4940 |

SKAB’da transition-confidence çok küçük bir iyileşme sağladı. Ancak fark oldukça sınırlı kaldı.

---

## SKAB Transition-Confidence Sweep

| Mode | Use Transition Count | k | Weight | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|
| additive | true | 1.0 | 0.05 | 0.4951 | 0.5975 | 0.4911 |
| additive | true | 1.0 | 0.10 | 0.4951 | 0.5975 | 0.4911 |
| additive | true | 1.0 | 0.25 | 0.4951 | 0.5975 | 0.4911 |
| additive | true | 1.0 | 0.50 | 0.4950 | 0.5975 | 0.4910 |
| additive | true | 2.0 | 0.05 | 0.4951 | 0.5975 | 0.4911 |
| additive | true | 2.0 | 0.10 | 0.4951 | 0.5975 | 0.4911 |
| additive | true | 2.0 | 0.25 | 0.4950 | 0.5975 | 0.4910 |
| additive | true | 2.0 | 0.50 | 0.4946 | 0.5980 | 0.4907 |
| additive | true | 5.0 | 0.05 | 0.4951 | 0.5975 | 0.4911 |
| additive | true | 5.0 | 0.10 | 0.4950 | 0.5975 | 0.4910 |
| additive | true | 5.0 | 0.25 | 0.4946 | 0.5980 | 0.4907 |
| additive | true | 5.0 | 0.50 | 0.4951 | 0.5995 | 0.4916 |
| additive | true | 10.0 | 0.05 | 0.4950 | 0.5975 | 0.4910 |
| additive | true | 10.0 | 0.10 | 0.4946 | 0.5980 | 0.4907 |
| additive | true | 10.0 | 0.25 | 0.4951 | 0.5995 | 0.4916 |
| additive | true | 10.0 | 0.50 | 0.4960 | 0.6035 | 0.4940 |
| additive | true | 20.0 | 0.05 | 0.4950 | 0.5975 | 0.4910 |
| additive | true | 20.0 | 0.10 | 0.4952 | 0.5990 | 0.4915 |
| additive | true | 20.0 | 0.25 | 0.4963 | 0.6020 | 0.4936 |
| additive | true | 20.0 | 0.50 | 0.4958 | 0.6045 | 0.4942 |

### SKAB sweep yorumu

SKAB için en iyi transition-confidence sonucu:

```text
k=20.0
weight=0.5
mode=additive
F1=0.4942
```

Base SKAB F1 skoru:

```text
Base F1 = 0.4911
```

İyileşme:

```text
+0.0031 F1
```

Bu iyileşme çok küçük olduğu için güçlü bir model geliştirmesi olarak değerlendirilmedi.

---

## Genel Karşılaştırma

| Dataset | Base F1 | Best Transition-Confidence F1 | Delta |
|---|---:|---:|---:|
| BATADAL | 0.4444 | 0.4000 | -0.0444 |
| SKAB | 0.4911 | 0.4942 | +0.0031 |

Transition-confidence SKAB üzerinde çok küçük bir artış sağladı, ancak BATADAL üzerinde base performansın altında kaldı.

BATADAL’daki düşüş, SKAB’daki küçük kazançtan çok daha belirgin olduğu için feature genel model açısından başarılı kabul edilmedi.

---

## Sonuç

Bu branch’te test edilen transition-confidence mekanizması mevcut haliyle final modele alınmadı.

### Genel karar

```text
feature/transition-confidence = rejected / not merged
```

### Gerekçe

```text
BATADAL base F1:                  0.4444
BATADAL best transition F1:       0.4000
BATADAL default transition F1:    0.2174

SKAB base F1:                     0.4911
SKAB best transition F1:          0.4942
SKAB default transition F1:       0.4940
```

Özetle:

- BATADAL’da base modelden daha düşük performans verdi.
- SKAB’da çok küçük bir iyileşme sağladı.
- Datasetler arası tutarlı ve güçlü bir katkı göstermedi.
- Default ayar BATADAL’da ciddi performans kaybına neden oldu.
- Bu nedenle ana modele merge edilmemesine karar verildi.

---

## Teknik Yorum

Transition-confidence yaklaşımı teorik olarak mantıklı olsa da, mevcut additive penalty formu BATADAL’da false positive sayısını artırmış olabilir.

Özellikle yüksek `weight` değerlerinde BATADAL precision ciddi şekilde düştü.

Örneğin:

```text
k=10.0, weight=0.5
Precision=0.1282
Recall=0.7143
F1=0.2174
```

Bu durum, confidence penalty’nin bazı normal fakat düşük gözlemli transition’ları anomaly tarafına ittiğini gösteriyor olabilir.

Daha düşük `weight` değerlerinde performans daha dengeli hale geldi:

```text
k=10.0, weight=0.05
Precision=0.3077
Recall=0.5714
F1=0.4000
```

Ancak bu ayar bile base modeldeki `0.4444` F1 skorunu geçemedi.

---

## Üretilen Dosyalar

Transition-confidence sweep çalıştırıldığında aşağıdaki dosyalar oluşturuldu:

```text
results/outputs/automata_transition_confidence_sweep_metrics.csv
results/outputs/automata_transition_confidence_sweep_summary.csv
results/outputs/automata_transition_confidence_best_candidates.csv
```

Ana final koşu çıktıları:

```text
results/outputs/automata_advanced_all_scenarios_metrics.csv
results/outputs/statistical_test_results.csv
```
---

## Window Size + Score Threshold Ablation

Transition-confidence branch üzerinde ek olarak `window_size` ve `score_threshold` değerlerinin performansa etkisi incelendi.

Bu çalışma için ayrı bir ablation script’i kullanıldı:

```text
src/experiments/window_threshold_ablation.py
```

Bu script ile farklı `window_size` ve `score_threshold` kombinasyonları denenerek transition-confidence mekanizmasının farklı pencere boyutlarında nasıl davrandığı gözlemlendi.

---

### Deney Kurulumu

Ablation çalışması şu branch üzerinde yürütüldü:

```text
feature/transition-confidence
```

Script dry-run aşamasında branch’i doğru şekilde algıladı:

```text
Branch slug: transition_confidence
```

Model tarafında transition-confidence parametreleri desteklenen model parametreleri içinde görüldü:

```text
transition_confidence_enabled
transition_confidence_k
transition_confidence_weight
transition_confidence_mode
transition_confidence_use_transition_count
```

Bu nedenle mevcut `window_threshold_ablation.py` script’inin transition-confidence branch ile uyumlu olduğu görüldü.

Bu branch’te transition-confidence mekanizması hem `calculate_scores()` hem de `predict()` içinde uygulandığı için `Fast predict=True` ayarı feature’ı bypass etmedi. Bu nedenle ablation çalışması aşağıdaki ayarla yürütüldü:

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
results/outputs/window_threshold_ablation/transition_confidence/
```

---

### BATADAL Window Size Sonuçları

BATADAL tarafında en iyi sonuç şu kombinasyonla elde edildi:

```text
window_size = 4
score_threshold = 4.2000
precision = 0.285714
recall = 0.571429
F1 = 0.380952
accuracy = 0.936585
```

En iyi 3 aday:

```text
1. window_size = 4
   score_threshold = 4.2000
   precision = 0.285714
   recall = 0.571429
   F1 = 0.380952

2. window_size = 4
   score_threshold = 3.9120
   precision = 0.128205
   recall = 0.714286
   F1 = 0.217391

3. window_size = 4
   score_threshold = 3.5066
   precision = 0.069444
   recall = 0.714286
   F1 = 0.126582
```

BATADAL için en iyi pencere boyutu yine `window_size=4` oldu.

Baseline BATADAL sonucu:

```text
Baseline BATADAL F1 = 0.444444
```

Transition-confidence window ablation sonrası en iyi BATADAL sonucu:

```text
Transition-confidence best BATADAL F1 = 0.380952
```

Bu nedenle transition-confidence branch, window size ve threshold araması sonrasında da BATADAL üzerinde baseline performansını geçemedi.

---

### BATADAL Unseen Data Sonuçları

En iyi original F1 adayının unseen-data sonucu:

```text
window_size = 4
score_threshold = 4.2000
unseen_precision = 0.142857
unseen_recall = 1.000000
unseen_F1 = 0.250000
unseen_accuracy = 0.625000
```

Diğer adayların unseen-data sonuçları:

```text
window_size = 4
score_threshold = 3.9120
unseen_F1 = 0.222222

window_size = 4
score_threshold = 3.5066
unseen_F1 = 0.153846
```

Baseline BATADAL unseen F1 değeri daha önce `0.400000` seviyesindeydi. Bu nedenle transition-confidence window ablation, unseen-data tarafında da iyileşme sağlamadı.

---

### BATADAL Yorumu

BATADAL tarafında `window_size=4` açık şekilde en iyi sonucu verdi.

`window_size=3` düşük threshold değerlerinde recall’u yükseltti, ancak precision çok düşük kaldı. Bu durum modelin çok fazla false positive ürettiğini gösterdi.

`window_size=5` ve üzerindeki değerlerde ise F1 skorları genel olarak `0.0` kaldı. Bu sonuç, BATADAL için daha büyük window size değerlerinin transition-confidence branch’inde de işe yaramadığını gösterdi.

BATADAL için genel sonuç:

```text
Transition-confidence branch, BATADAL üzerinde window size ve threshold aramasıyla baseline performansını geçemedi.
En iyi sonuç window_size=4 ve score_threshold=4.2 ile elde edildi.
Ancak F1 skoru baseline değerinin altında kaldı.
```

---

### SKAB Window Size Sonuçları

SKAB tarafında en iyi original F1 sonucu şu kombinasyonla elde edildi:

```text
window_size = 5
score_threshold = 0.05
precision = 0.371625
recall = 1.000000
F1 = 0.541434
accuracy = 0.387431
```

En iyi 3 aday:

```text
1. window_size = 5
   score_threshold = 0.05
   precision = 0.371625
   recall = 1.000000
   F1 = 0.541434

2. window_size = 4
   score_threshold = 0.05
   precision = 0.391427
   recall = 0.863158
   F1 = 0.537760

3. window_size = 5
   score_threshold = 0.00
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
window_size = 5
score_threshold = 0.05
F1 = 0.541434
```

Bu sonuç SKAB original senaryoda sayısal olarak F1 artışı olduğunu gösterdi. Ancak bu artışın temel nedeni recall değerinin `1.0` seviyesine çıkmasıdır.

En iyi adayda:

```text
recall = 1.000000
precision = 0.371625
accuracy = 0.387431
```

Bu durum modelin anomalileri yakaladığını, ancak çok fazla false positive ürettiğini göstermektedir.

---

### SKAB Unseen Data Sonuçları

En iyi adayların unseen-data sonuçları:

```text
window_size = 5
score_threshold = 0.05
unseen_precision = 0.141463
unseen_recall = 0.200000
unseen_F1 = 0.165714
unseen_accuracy = 0.610853
```

```text
window_size = 4
score_threshold = 0.05
unseen_precision = 0.142857
unseen_recall = 0.200000
unseen_F1 = 0.166667
unseen_accuracy = 0.547826
```

```text
window_size = 5
score_threshold = 0.00
unseen_precision = 0.141463
unseen_recall = 0.200000
unseen_F1 = 0.165714
unseen_accuracy = 0.610853
```

Unseen-data tarafında anlamlı bir iyileşme elde edilmedi. En iyi unseen F1 değeri yaklaşık `0.1667` seviyesinde kaldı.

---

### SKAB Yorumu

SKAB tarafında `window_size=5` ve düşük threshold değerleri original F1 skorunu artırdı.

Ancak bu artış yüksek recall ve düşük precision kaynaklıdır. Accuracy değerinin düşük kalması da modelin fazla false positive ürettiğini desteklemektedir.

Bu nedenle SKAB original senaryodaki F1 artışı güçlü ve kararlı bir iyileşme olarak değerlendirilmedi.

SKAB için genel sonuç:

```text
Transition-confidence branch, SKAB original senaryoda recall ağırlıklı bir F1 artışı sağladı.
Ancak unseen-data tarafında iyileşme sağlamadı.
Precision ve accuracy düşük kaldığı için bu artış güçlü bir model geliştirmesi olarak değerlendirilmedi.
```

---

### Window Size Ablation Genel Yorumu

Window size ablation sonuçları transition-confidence branch için şu tabloyu ortaya koydu:

```text
BATADAL:
- En iyi window_size yine 4 oldu.
- En iyi F1 = 0.380952
- Bu değer baseline F1 = 0.444444 değerinin altında kaldı.
- Büyük window size değerleri anlamlı sonuç üretmedi.

SKAB:
- En iyi original F1 window_size=5 ve threshold=0.05 ile elde edildi.
- En iyi original F1 = 0.541434
- Bu artış yüksek recall kaynaklıydı.
- Precision ve accuracy düşük kaldı.
- Unseen-data tarafında iyileşme görülmedi.
```

Sonuç olarak, window size ve score threshold araması transition-confidence branch’in genel kararını değiştirmedi.

```text
feature/transition-confidence = rejected / not merged
```

Bu branch, SKAB original senaryoda sınırlı ve recall-ağırlıklı bir F1 artışı üretmiştir. Ancak BATADAL üzerinde baseline performansının altında kalmış, SKAB unseen-data üzerinde anlamlı bir iyileşme sağlamamış ve datasetler arası tutarlı bir katkı göstermemiştir. Bu nedenle final model adayı olarak seçilmemiştir.