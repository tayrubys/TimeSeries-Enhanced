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
