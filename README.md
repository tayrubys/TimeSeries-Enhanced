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