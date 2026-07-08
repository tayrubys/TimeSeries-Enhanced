# feature/dirichlet-smoothing

Bu branch, Markov tabanlı probabilistic automata modelinde transition probability tahminini daha stabil hale getirmek amacıyla geliştirilmiştir.

Amaç, özellikle sparse transition durumlarında klasik additive smoothing yerine Dirichlet prior tabanlı daha kontrollü bir olasılık tahmini kullanarak anomaly detection performansını artırmaktır.

---

## Amaç

`markov-base` modelinde transition probability hesaplaması additive smoothing ile yapılmaktadır:

```python
P(next | state) = (transition_count + alpha) / (total_output + alpha * vocab_size)
```

Bu branch’te amaç, bu yapıyı Dirichlet prior tabanlı bir forma genişletmektir:

```python
P(next | state) =
(count(state, next) + alpha * prior(next)) /
(total_exits(state) + alpha)
```

Böylece düşük gözlemli veya sparse state-transition durumlarında probability estimation’ın daha stabil olması hedeflenmiştir.

---

## Eklenen Parametreler

`ProbabilisticAutomata` sınıfına aşağıdaki parametreler eklendi:

```python
dirichlet_smoothing_enabled=False
dirichlet_alpha=None
dirichlet_prior_mode="uniform"
```

### Parametre Açıklamaları

| Parametre | Açıklama |
|---|---|
| `dirichlet_smoothing_enabled` | Dirichlet smoothing mekanizmasını açar/kapatır. |
| `dirichlet_alpha` | Dirichlet prior ağırlığını belirler. |
| `dirichlet_prior_mode` | Kullanılacak prior dağılımını belirler. |

Desteklenen prior modları:

| Prior Mode | Açıklama |
|---|---|
| `uniform` | Tüm pattern’lere eşit prior verir. |
| `unigram` | Eğitim verisindeki genel pattern frekanslarını prior olarak kullanır. |
| `backoff` | Alt dereceli Markov dağılımını prior olarak kullanır; uygun değilse unigram prior’a düşer. |

---

## Modelde Yapılan Değişiklikler

Bu branch’te anomaly karar mekanizmasına doğrudan penalty veya gate eklenmedi.

Önceki branch’lerde test edilen:

- `entropy-threshold`
- `transition-confidence`

gibi mekanizmalar anomaly skoruna doğrudan müdahale ettiği için özellikle BATADAL’da false positive sayısını artırmıştı.

Bu nedenle `feature/dirichlet-smoothing` branch’inde daha yumuşak bir yaklaşım tercih edildi:

```text
Karar mekanizması korunur.
Transition probability estimation iyileştirilmeye çalışılır.
```

Yani modelin scoring mantığı aynı kalırken, `get_transition_probability()` fonksiyonu Dirichlet smoothing destekleyecek şekilde genişletildi.

---

## Runner Değişiklikleri

`runner.py` dosyasına Dirichlet smoothing parametreleri eklendi.

Final koşulara şu alanlar dahil edildi:

```python
dirichlet_smoothing_enabled
dirichlet_alpha
dirichlet_prior_mode
```

Bu parametreler:

- model oluşturulurken `ProbabilisticAutomata` sınıfına geçirildi,
- deney sonuçlarına metadata olarak eklendi,
- SKAB ve BATADAL summary çıktılarında raporlandı,
- sweep fonksiyonlarına dahil edildi.

---

## İlk Dirichlet Sweep Deneyi

İlk deneyde aşağıdaki ayarlar tarandı:

```json
"dirichlet_alphas": [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
"dirichlet_prior_modes": ["uniform", "unigram", "backoff"]
```

Başlangıç default ayarı:

```json
"dirichlet_smoothing_enabled": true,
"dirichlet_alpha": 0.5,
"dirichlet_prior_mode": "unigram"
```

### İlk Default Sonuçları

| Dataset | Base F1 | Dirichlet Default F1 |
|---|---:|---:|
| BATADAL | 0.4444 | 0.1299 |
| SKAB | 0.4911 | 0.4789 |

İlk default ayar özellikle BATADAL veri setinde ciddi performans düşüşüne neden oldu.

BATADAL’da recall yüksek kalmasına rağmen precision ciddi biçimde düştü. Bu durum modelin çok fazla false positive ürettiğini gösterdi.

---

## İlk Sweep Sonuçları

### BATADAL

İlk sweep sonucunda BATADAL için en iyi Dirichlet ayarı:

| Prior | Alpha | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| uniform | 5.0 | 0.3077 | 0.5714 | 0.4000 |

Bu sonuç, default Dirichlet ayarına göre daha iyi olsa da `markov-base` sonucunu geçemedi.

| Model | BATADAL F1 |
|---|---:|
| Markov-base | 0.4444 |
| Best Dirichlet initial sweep | 0.4000 |

### SKAB

İlk sweep sonucunda SKAB için en iyi Dirichlet sonucu yaklaşık:

| Prior | Alpha | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| uniform | 5.0 | 0.4950 | 0.5775 | 0.4831 |

Bu sonuç da `markov-base` değerinin altında kaldı.

| Model | SKAB F1 |
|---|---:|
| Markov-base | 0.4911 |
| Best Dirichlet initial sweep | 0.4831 |

---

## Refined Sweep Deneyi

İlk sweep sonucunda BATADAL’da `uniform` prior ve yüksek alpha değerlerinin daha iyi çalıştığı görüldü.

Bu nedenle ikinci bir refined sweep deneyi yapıldı.

Refined sweep ayarları:

```json
"run_dirichlet_refined_sweep": true,

"dirichlet_refined_alphas": [5.0, 7.5, 10.0, 15.0, 20.0, 30.0],
"dirichlet_refined_prior_modes": ["uniform"],

"batadal_dirichlet_score_thresholds": [3.9120, 4.2, 4.5, 4.8, 5.0, 5.3, 5.6],
"skab_dirichlet_score_thresholds": [0.2231, 0.3, 0.4, 0.5, 0.7, 0.9]
```

Bu deneyde amaç:

```text
1. BATADAL’da alpha=5.0 sonrası değerleri test etmek
2. Threshold değişimiyle precision/recall dengesini iyileştirmek
3. SKAB için Dirichlet açık/kapalı durumunu karşılaştırmak
```

---

## Refined Sweep Sonuçları

### BATADAL

Refined sweep sonucunda BATADAL için en iyi ayar:

| Dirichlet | Prior | Alpha | Score Threshold | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|
| enabled | uniform | 7.5 | 3.9120 | 0.3636 | 0.5714 | 0.4444 |

Bu sonuç `markov-base` ile aynıdır.

| Model | BATADAL F1 |
|---|---:|
| Markov-base | 0.4444 |
| Best refined Dirichlet | 0.4444 |

BATADAL’da Dirichlet smoothing, refined tuning sonrasında ilk kötü sonucu toparlamış olsa da base modelin üzerine çıkamamıştır.

Threshold yükseltildiğinde precision artsa bile recall ciddi biçimde düşmüştür. Örneğin `score_threshold=4.2` civarında precision yükselmiş, ancak recall `0.1429` seviyesine düşmüş ve F1 skoru zayıf kalmıştır.

Bu nedenle BATADAL için temel problem sadece threshold ayarı değildir. Dirichlet smoothing, skor dağılımını base modelden daha ayırt edici hale getirememiştir.

---

### SKAB

Refined sweep sonucunda SKAB için en iyi sonuç:

| Dirichlet | Prior | Alpha | Score Threshold | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|
| enabled | uniform | 30.0 | 0.2231 | 0.4957 | 0.6045 | 0.4941 |

Dirichlet kapalı base SKAB sonucu:

| Dirichlet | F1 |
|---|---:|
| disabled | 0.4911 |

SKAB’da Dirichlet smoothing çok küçük bir iyileşme sağlamıştır:

```text
0.4911 -> 0.4941
delta = +0.0030
```

Ancak bu artış çok sınırlıdır ve final modele dahil etmek için yeterince güçlü görülmemiştir.

---

## Gaussian Noise ve Unseen Data Sonuçları

### BATADAL

Refined Dirichlet final ayarında BATADAL sonuçları:

| Scenario | Precision | Recall | F1 |
|---|---:|---:|---:|
| original | 0.3077 | 0.5714 | 0.4000 |
| gaussian_noise | 0.2387 | 0.4857 | 0.3195 |
| unseen_data | 0.1429 | 1.0000 | 0.2500 |

Refined sweep içinde BATADAL için en iyi original sonuç `alpha=7.5` ile `0.4444` olsa da final default koşuda `alpha=5.0` kullanıldığında original F1 `0.4000` seviyesinde kalmıştır.

### SKAB

SKAB final koşuda Dirichlet kapalı bırakıldığında base sonuç korunmuştur:

| Scenario | Precision | Recall | F1 |
|---|---:|---:|---:|
| original | 0.4951 | 0.5975 | 0.4911 |
| gaussian_noise | 0.4801 | 0.5961 | 0.4870 |
| unseen_data | 0.1429 | 0.2000 | 0.1667 |

---

## Genel Karşılaştırma

| Dataset | Base F1 | Best Dirichlet F1 | Değişim |
|---|---:|---:|---:|
| BATADAL | 0.4444 | 0.4444 | 0.0000 |
| SKAB | 0.4911 | 0.4941 | +0.0030 |

Dirichlet smoothing:

- BATADAL’da base performansı yakaladı ancak geçemedi.
- SKAB’da çok küçük bir F1 artışı sağladı.
- Gaussian noise ve unseen data senaryolarında belirgin ve tutarlı bir iyileşme üretmedi.
- İlk default ayarda ciddi performans düşüşüne neden oldu.
- Refined tuning ile toparlandı ancak final model için yeterli katkı sağlamadı.

---

## Teknik Yorum

Dirichlet smoothing teorik olarak sparse transition problemine uygun bir yöntemdir. Ancak bu projedeki deneylerde probability distribution’ın fazla düzleşmesi, bazı transition skorlarını base modele göre daha az ayırt edici hale getirmiş olabilir.

Özellikle BATADAL’da şu davranış gözlendi:

```text
Düşük threshold:
- Recall korunuyor
- Precision düşüyor
- False positive artıyor

Yüksek threshold:
- Precision artıyor
- Recall çöküyor
- F1 düşüyor
```

Bu durum, Dirichlet smoothing’in tek başına anomaly-normal ayrımını iyileştirmediğini göstermektedir.

SKAB tarafında ise Dirichlet smoothing çok küçük bir iyileşme sağlamış olsa da bu kazanım final modele dahil edilecek kadar güçlü değildir.

---

## Karar

Bu branch için karar:

```text
feature/dirichlet-smoothing = not merged
```

Daha açıklayıcı karar:

```text
Rejected as final improvement, documented as neutral/slightly-positive ablation.
```

Branch tamamen başarısız değildir; çünkü refined tuning sonrasında BATADAL’da base performansı yakalamış, SKAB’da ise çok küçük bir iyileşme sağlamıştır.

Ancak iyileşme:

- yeterince büyük değildir,
- iki dataset üzerinde tutarlı değildir,
- robustness senaryolarında belirgin katkı sağlamamıştır.

Bu nedenle final Markov automata modeline dahil edilmemiştir.

---

## Reproduction Config

Refined sweep için kullanılan temel ayarlar:

```json
{
  "run_dirichlet_smoothing_sweep": false,
  "run_dirichlet_refined_sweep": true,

  "dirichlet_smoothing_enabled": true,
  "dirichlet_alpha": 5.0,
  "dirichlet_prior_mode": "uniform",

  "dirichlet_refined_alphas": [5.0, 7.5, 10.0, 15.0, 20.0, 30.0],
  "dirichlet_refined_prior_modes": ["uniform"],

  "batadal_dirichlet_score_thresholds": [3.9120, 4.2, 4.5, 4.8, 5.0, 5.3, 5.6],
  "skab_dirichlet_score_thresholds": [0.2231, 0.3, 0.4, 0.5, 0.7, 0.9],

  "batadal_final_dirichlet_smoothing_enabled": true,
  "batadal_final_dirichlet_alpha": 5.0,
  "batadal_final_dirichlet_prior_mode": "uniform",

  "skab_final_dirichlet_smoothing_enabled": false,
  "skab_final_dirichlet_alpha": 5.0,
  "skab_final_dirichlet_prior_mode": "uniform"
}
```

---

## Üretilen Çıktılar

Bu branch’te aşağıdaki deney çıktıları üretildi:

```text
results/outputs/automata_dirichlet_smoothing_sweep_metrics.csv
results/outputs/automata_dirichlet_smoothing_sweep_summary.csv
results/outputs/automata_dirichlet_smoothing_best_candidates.csv

results/outputs/automata_dirichlet_refined_sweep_metrics.csv
results/outputs/automata_dirichlet_refined_sweep_summary.csv
results/outputs/automata_dirichlet_refined_best_candidates.csv
```

