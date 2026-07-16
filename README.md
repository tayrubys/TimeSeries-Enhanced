# feature/entropy-threshold

## Amaç

Bu branch, Markov-chain tabanlı probabilistic automata modeline **entropy-threshold** mekanizması eklemek için oluşturuldu.

Temel fikir şuydu:

> Eğer mevcut state için transition dağılımı yüksek entropiye sahipse, modelin bir sonraki pattern tahminindeki belirsizliği yüksektir. Bu belirsizlik anomaly kararına dahil edilerek model daha hassas hale getirilebilir.

Bu nedenle state-level transition entropy hesaplanmış ve belirli bir eşik değerinin üzerinde kalan durumlar anomaly kararını etkileyebilecek şekilde modele entegre edilmiştir.

---

## Eklenen Özellikler

Bu branch’te `ProbabilisticAutomata` modeli aşağıdaki parametrelerle genişletildi:

```python
entropy_threshold=None
entropy_mode="normalized"
entropy_min_total_exits=1
```

### Parametreler

| Parametre | Açıklama |
|---|---|
| `entropy_threshold` | Entropy gate için eşik değeri. `None` olduğunda feature kapalıdır. |
| `entropy_mode` | Entropy hesaplama modu. Bu deneyde `"normalized"` kullanıldı. |
| `entropy_min_total_exits` | Entropy kararının uygulanması için minimum transition çıkış sayısı. |

---

## Deney Kurulumu

Base Markov automata ile entropy-threshold branch’i aynı final otomata parametreleriyle karşılaştırıldı.

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

Entropy branch’te ek olarak:

```text
entropy_threshold=0.85
entropy_mode=normalized
```

Ayrıca entropy threshold sweep çalıştırıldı:

```text
entropy_thresholds = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
```

---

## Base vs Entropy-Threshold Sonuçları

### BATADAL

| Scenario | Base F1 | Entropy F1 | Sonuç |
|---|---:|---:|---|
| original | 0.4444 | 0.0805 | Gerileme |
| gaussian_noise | 0.3796 | 0.0742 | Gerileme |
| unseen_data | 0.4000 | 0.3333 | Hafif gerileme |

Original BATADAL karşılaştırması:

| Model | Precision | Recall | F1 |
|---|---:|---:|---:|
| Base Markov Automata | 0.3636 | 0.5714 | 0.4444 |
| Entropy-Threshold | 0.0423 | 0.8571 | 0.0805 |

Entropy-threshold recall’u artırdı fakat precision’ı çok ciddi şekilde düşürdü. Bu durum çok fazla false positive üretildiğini gösteriyor.

---

## BATADAL Entropy Threshold Sweep

| Entropy Threshold | Precision | Recall | F1 |
|---:|---:|---:|---:|
| 0.70 | 0.0331 | 0.8571 | 0.0638 |
| 0.75 | 0.0357 | 0.8571 | 0.0686 |
| 0.80 | 0.0382 | 0.8571 | 0.0732 |
| 0.85 | 0.0423 | 0.8571 | 0.0805 |
| 0.90 | 0.0632 | 0.8571 | 0.1176 |
| 0.95 | 0.0617 | 0.7143 | 0.1136 |

En iyi entropy threshold değeri `0.90` olmasına rağmen F1 skoru `0.1176` seviyesinde kaldı. Bu değer base modeldeki `0.4444` F1 skorunun oldukça altında kaldı.

---

## SKAB Sonuçları

SKAB tarafında entropy-threshold feature’ı sonuçları değiştirmedi.

| Scenario | Base F1 | Entropy F1 | Sonuç |
|---|---:|---:|---|
| original | 0.4911 | 0.4911 | Etkisiz |
| gaussian_noise | 0.4870 | 0.4870 | Etkisiz |
| unseen_data | 0.1667 | 0.1667 | Etkisiz |

SKAB entropy sweep sonucunda tüm threshold değerlerinde aynı performans elde edildi:

| Entropy Threshold | Precision | Recall | F1 |
|---:|---:|---:|---:|
| 0.70 | 0.4951 | 0.5975 | 0.4911 |
| 0.75 | 0.4951 | 0.5975 | 0.4911 |
| 0.80 | 0.4951 | 0.5975 | 0.4911 |
| 0.85 | 0.4951 | 0.5975 | 0.4911 |
| 0.90 | 0.4951 | 0.5975 | 0.4911 |
| 0.95 | 0.4951 | 0.5975 | 0.4911 |

---

## Sonuç

Bu branch’te test edilen entropy-threshold mekanizması mevcut haliyle başarılı bulunmadı.

### Genel karar

```text
feature/entropy-threshold = rejected / not merged
```

### Gerekçe

BATADAL üzerinde entropy-threshold mekanizması recall’u artırsa da precision’ı ciddi biçimde düşürdü.

```text
Base BATADAL F1:     0.4444
Entropy BATADAL F1:  0.0805
Best sweep F1:       0.1176
```

SKAB üzerinde ise feature performansı değiştirmedi.

Dolayısıyla bu branch ana modele merge edilmeyecek, ancak ileride referans olması için bağımsız deney branch’i olarak korunacaktır.

---

## Teknik Yorum

Bu feature’ın temel varsayımı şuydu:

```text
high entropy state => uncertain transition => anomaly
```

Fakat BATADAL sonuçları bu varsayımın mevcut karar mekanizmasıyla iyi çalışmadığını gösterdi. Yüksek entropy yalnızca anomaly durumlarını değil, normal fakat çok seçenekli transition state’lerini de işaretlemiş olabilir.

Bu da false positive sayısını artırarak precision değerini düşürmüştür.

Özellikle BATADAL original senaryoda:

```text
Base precision:    0.3636
Entropy precision: 0.0423
```

Bu düşüş, entropy gate’in çok agresif anomaly üretmesine neden olduğunu göstermektedir.

---

## Üretilen Dosyalar

Entropy-threshold sweep çalıştırıldığında aşağıdaki dosyalar oluşturuldu:

```text
results/outputs/automata_entropy_threshold_sweep_metrics.csv
results/outputs/automata_entropy_threshold_sweep_summary.csv
results/outputs/automata_entropy_threshold_best_candidates.csv
```

Ana final koşu çıktıları:

```text
results/outputs/automata_advanced_all_scenarios_metrics.csv
results/outputs/statistical_test_results.csv
```

---

## Reproduction

Entropy sweep’i yeniden çalıştırmak için `src/config/settings.json` içinde:

```json
{
  "automata": {
    "run_entropy_threshold_sweep": true,
    "entropy_threshold": 0.85,
    "batadal_final_entropy_threshold": 0.85,
    "skab_final_entropy_threshold": 0.85,
    "entropy_thresholds": [0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
    "entropy_mode": "normalized",
    "entropy_min_total_exits": 1
  }
}
```

Ardından:

```bash
python src/experiments/runner.py
```

Baseline davranışa dönmek için entropy threshold değerleri `null` yapılabilir:

```json
{
  "automata": {
    "entropy_threshold": null,
    "batadal_final_entropy_threshold": null,
    "skab_final_entropy_threshold": null,
    "run_entropy_threshold_sweep": false
  }
}
```

---

## Window Size + Score Threshold Ablation

Entropy-threshold branch üzerinde ek olarak `window_size` ve `score_threshold` değerlerinin performansa etkisi incelendi.

Bu çalışma için ayrı bir ablation script’i kullanıldı:

~~~text
src/experiments/window_threshold_ablation.py
~~~

Bu script ile farklı `window_size` ve `score_threshold` kombinasyonları denenerek entropy-threshold mekanizmasının farklı pencere boyutlarında nasıl davrandığı gözlemlendi.

### Deney Kurulumu

Entropy-threshold etkisinin gerçekten ölçülebilmesi için bu branch’te `Fast predict` kapatıldı:

~~~text
Fast predict: False
~~~

Böylece modelin `predict()` fonksiyonu kullanıldı ve entropy-threshold karar mekanizmasının devreye girmesi sağlandı.

Test edilen window size aralığı:

~~~text
window_size = [3, 4, 5, 6, 7]
~~~

Başlangıçta `[3, 4, 5, 6, 7, 8, 9, 10]` aralığı denenmişti. Ancak özellikle SKAB tarafında `window_size=8`, `window_size=9` ve `window_size=10` değerleri anlamlı katkı sağlamadığı ve koşu süresini uzattığı için final çalışmada aralık `[3, 4, 5, 6, 7]` ile sınırlandırıldı.

Main ablation senaryoları:

~~~text
original
gaussian_noise
~~~

Main ablation tamamlandıktan sonra `original` F1 skoruna göre en iyi 3 aday seçildi. Daha sonra yalnızca bu en iyi 3 aday için `unseen_data` değerlendirmesi yapıldı.

Çıktılar şu klasör altında üretildi:

~~~text
results/outputs/window_threshold_ablation/entropy_threshold/
~~~

---

### BATADAL Window Size Sonuçları

BATADAL tarafında en iyi sonuç şu kombinasyonla elde edildi:

~~~text
window_size = 4
score_threshold = 2.9957
precision = 0.042683
recall = 1.000000
F1 = 0.081871
accuracy = 0.234146
~~~

En iyi 3 aday:

~~~text
1. window_size = 4
   score_threshold = 2.9957
   precision = 0.042683
   recall = 1.000000
   F1 = 0.081871

2. window_size = 4
   score_threshold = 3.9120
   precision = 0.042254
   recall = 0.857143
   F1 = 0.080537

3. window_size = 4
   score_threshold = 3.5066
   precision = 0.041096
   recall = 0.857143
   F1 = 0.078431
~~~

BATADAL için `window_size=4` yine en iyi pencere boyutu olarak kaldı. Ancak entropy-threshold mekanizması aktifken bu sonuç baseline performansının oldukça altında kaldı.

Karşılaştırma:

~~~text
Baseline BATADAL F1:                  0.444444
Entropy-threshold window ablation F1: 0.081871
~~~

Bu sonuç, window size değiştirmenin BATADAL üzerinde entropy-threshold branch’ini iyileştirmediğini gösterdi.

`window_size=3` düşük threshold değerlerinde recall’u artırdı, ancak precision çok düşük kaldı. Bu durum modelin çok fazla false positive ürettiğini gösterdi.

`window_size=5` ve üzerindeki değerlerde ise anlamlı bir iyileşme elde edilmedi.

BATADAL için genel yorum:

~~~text
Entropy-threshold branch’inde window size değişimi BATADAL performansını iyileştirmedi.
En iyi sonuç yine window_size=4 ile elde edildi, ancak F1 skoru baseline’ın çok altında kaldı.
~~~

---

### BATADAL Unseen Data Sonuçları

En iyi original F1 adayının unseen-data sonucu:

~~~text
window_size = 4
score_threshold = 2.9957
unseen_precision = 0.083333
unseen_recall = 1.000000
unseen_F1 = 0.153846
unseen_accuracy = 0.312500
~~~

Diğer adayların unseen-data sonuçları:

~~~text
window_size = 4
score_threshold = 3.9120
unseen_F1 = 0.333333

window_size = 4
score_threshold = 3.5066
unseen_F1 = 0.222222
~~~

Her ne kadar `score_threshold=3.9120` unseen-data üzerinde daha yüksek F1 üretmiş olsa da, original senaryodaki performansı çok düşük kaldığı için genel aday olarak güçlü görülmedi.

---

### SKAB Window Size Sonuçları

SKAB tarafında en iyi original F1 sonucu şu kombinasyonla elde edildi:

~~~text
window_size = 5
score_threshold = 0.05
precision = 0.371625
recall = 1.000000
F1 = 0.541434
accuracy = 0.387431
~~~

En iyi 3 aday:

~~~text
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
~~~

SKAB tarafında düşük threshold değerleriyle original F1 skorunda artış gözlendi. Ancak bu artışın temel nedeni recall değerinin çok yükselmesidir.

En iyi adayda:

~~~text
recall = 1.000000
precision = 0.371625
accuracy = 0.387431
~~~

Bu durum modelin anomalileri yakaladığını, fakat çok fazla false positive ürettiğini göstermektedir.

Önceki referans ayar:

~~~text
window_size = 4
score_threshold = 0.2231
F1 ≈ 0.4911
~~~

Window ablation sonrası en iyi original sonuç:

~~~text
window_size = 5
score_threshold = 0.05
F1 = 0.541434
~~~

Bu nedenle SKAB original senaryoda sayısal olarak F1 artışı vardır. Ancak precision ve accuracy düşük kaldığı için bu iyileşme güçlü ve kararlı bir iyileşme olarak değerlendirilmemelidir.

---

### SKAB Unseen Data Sonuçları

En iyi adayların unseen-data sonuçları:

~~~text
window_size = 5
score_threshold = 0.05
unseen_precision = 0.141463
unseen_recall = 0.200000
unseen_F1 = 0.165714
unseen_accuracy = 0.610853
~~~

~~~text
window_size = 4
score_threshold = 0.05
unseen_precision = 0.142857
unseen_recall = 0.200000
unseen_F1 = 0.166667
unseen_accuracy = 0.547826
~~~

~~~text
window_size = 5
score_threshold = 0.00
unseen_precision = 0.141463
unseen_recall = 0.200000
unseen_F1 = 0.165714
unseen_accuracy = 0.610853
~~~

Unseen-data tarafında anlamlı bir iyileşme elde edilmedi. En iyi unseen F1 değeri yaklaşık `0.1667` seviyesinde kaldı.

---

### Window Size Ablation Genel Yorumu

Window size ablation sonuçları entropy-threshold branch’i için şu tabloyu ortaya koydu:

~~~text
BATADAL:
- En iyi window_size yine 4 oldu.
- Ancak entropy-threshold aktifken F1 ciddi şekilde düştü.
- Window size değişimi BATADAL performansını iyileştirmedi.

SKAB:
- En iyi original F1 window_size=5 ve threshold=0.05 ile elde edildi.
- F1 artışı yüksek recall kaynaklıydı.
- Precision ve accuracy düşük kaldı.
- Unseen-data tarafında iyileşme görülmedi.
~~~

Sonuç olarak, window size ve score threshold araması entropy-threshold branch’inin genel kararını değiştirmedi.

~~~text
feature/entropy-threshold = rejected / not merged
~~~

Bu branch’te window size değişimi SKAB original senaryoda sınırlı ve recall-ağırlıklı bir artış sağladı. Ancak BATADAL’daki ciddi performans düşüşü ve unseen-data tarafındaki zayıf sonuçlar nedeniyle entropy-threshold mekanizması final model adayı olarak seçilmedi.
