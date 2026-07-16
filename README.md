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

---

## Window Size + Score Threshold Ablation

Dirichlet-smoothing branch üzerinde ek olarak `window_size` ve `score_threshold` değerlerinin performansa etkisi incelendi.

Bu çalışma için ayrı bir ablation script’i kullanıldı:

```text
src/experiments/window_threshold_ablation.py
```

Bu script ile farklı `window_size` ve `score_threshold` kombinasyonları denenerek Dirichlet smoothing mekanizmasının farklı pencere boyutlarında nasıl davrandığı gözlemlendi.

---

### Deney Kurulumu

Ablation çalışması şu branch üzerinde yürütüldü:

```text
feature/dirichlet-smoothing
```

Script dry-run aşamasında branch’i doğru şekilde algıladı:

```text
Branch slug: dirichlet_smoothing
```

Model tarafında Dirichlet smoothing parametreleri desteklenen model parametreleri içinde görüldü:

```text
dirichlet_smoothing_enabled
dirichlet_alpha
dirichlet_prior_mode
```

Bu nedenle mevcut `window_threshold_ablation.py` script’inin Dirichlet-smoothing branch ile uyumlu olduğu görüldü.

Dirichlet smoothing, karar aşamasına doğrudan bir gate veya penalty eklemediği için `Fast predict=True` ayarıyla çalıştırıldı. Bu branch’te smoothing etkisi transition probability / score hesaplama seviyesinde uygulandığından, hızlı skor hesaplama yolu feature’ı bypass etmedi.

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
results/outputs/window_threshold_ablation/dirichlet_smoothing/
```

---

### BATADAL Window Size Sonuçları

BATADAL tarafında en iyi sonuç şu kombinasyonla elde edildi:

```text
window_size = 4
score_threshold = 3.9120
precision = 0.307692
recall = 0.571429
F1 = 0.400000
accuracy = 0.941463
```

En iyi 3 aday:

```text
1. window_size = 4
   score_threshold = 3.9120
   precision = 0.307692
   recall = 0.571429
   F1 = 0.400000

2. window_size = 4
   score_threshold = 3.5066
   precision = 0.166667
   recall = 0.571429
   F1 = 0.258065

3. window_size = 4
   score_threshold = 4.2000
   precision = 0.333333
   recall = 0.142857
   F1 = 0.200000
```

BATADAL için en iyi pencere boyutu yine `window_size=4` oldu.

Baseline BATADAL sonucu:

```text
Baseline BATADAL F1 = 0.444444
```

Dirichlet-smoothing window ablation sonrası en iyi BATADAL sonucu:

```text
Dirichlet-smoothing best BATADAL F1 = 0.400000
```

Bu nedenle Dirichlet-smoothing branch, window size ve threshold araması sonrasında da BATADAL üzerinde baseline performansını geçemedi.

---

### BATADAL Unseen Data Sonuçları

En iyi original F1 adayının unseen-data sonucu:

```text
window_size = 4
score_threshold = 3.9120
unseen_precision = 0.142857
unseen_recall = 1.000000
unseen_F1 = 0.250000
unseen_accuracy = 0.625000
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

Baseline BATADAL unseen F1 değeri daha önce `0.400000` seviyesindeydi. Bu nedenle Dirichlet-smoothing window ablation, unseen-data tarafında da iyileşme sağlamadı.

---

### BATADAL Yorumu

BATADAL tarafında `window_size=4` açık şekilde en iyi sonucu verdi.

`window_size=3` düşük threshold değerlerinde recall’u yüksek tuttu, ancak precision çok düşük kaldı. Bu durum modelin çok fazla false positive ürettiğini gösterdi.

`window_size=5` ve üzerindeki değerlerde ise F1 skorları genel olarak `0.0` kaldı. Bu sonuç, BATADAL için daha büyük window size değerlerinin Dirichlet-smoothing branch’inde de işe yaramadığını gösterdi.

BATADAL için genel sonuç:

```text
Dirichlet-smoothing branch, BATADAL üzerinde window size ve threshold aramasıyla baseline performansını geçemedi.
En iyi sonuç window_size=4 ve score_threshold=3.9120 ile elde edildi.
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
Dirichlet-smoothing branch, SKAB original senaryoda recall ağırlıklı bir F1 artışı sağladı.
Ancak unseen-data tarafında iyileşme sağlamadı.
Precision ve accuracy düşük kaldığı için bu artış güçlü bir model geliştirmesi olarak değerlendirilmedi.
```

---

### Window Size Ablation Genel Yorumu

Window size ablation sonuçları Dirichlet-smoothing branch için şu tabloyu ortaya koydu:

```text
BATADAL:
- En iyi window_size yine 4 oldu.
- En iyi F1 = 0.400000
- Bu değer baseline F1 = 0.444444 değerinin altında kaldı.
- Unseen-data tarafında iyileşme görülmedi.
- Büyük window size değerleri anlamlı sonuç üretmedi.

SKAB:
- En iyi original F1 window_size=5 ve threshold=0.05 ile elde edildi.
- En iyi original F1 = 0.541434
- Bu artış yüksek recall kaynaklıydı.
- Precision ve accuracy düşük kaldı.
- Unseen-data tarafında iyileşme görülmedi.
```

Sonuç olarak, window size ve score threshold araması Dirichlet-smoothing branch’in genel kararını değiştirmedi.

```text
feature/dirichlet-smoothing = not merged
```

Bu branch, SKAB original senaryoda sınırlı ve recall-ağırlıklı bir F1 artışı üretmiştir. Ancak BATADAL üzerinde baseline performansının altında kalmış, unseen-data üzerinde anlamlı bir iyileşme sağlamamış ve datasetler arası tutarlı bir katkı göstermemiştir. Bu nedenle final model adayı olarak seçilmemiştir.

---

### Üretilen Window Ablation Çıktıları

Window size ve score threshold ablation çalışması sonucunda aşağıdaki dosyalar üretildi:

```text
results/outputs/window_threshold_ablation/dirichlet_smoothing/01_main_ablation/dirichlet_smoothing_batadal_main_metrics.csv
results/outputs/window_threshold_ablation/dirichlet_smoothing/01_main_ablation/dirichlet_smoothing_batadal_main_summary.csv
results/outputs/window_threshold_ablation/dirichlet_smoothing/01_main_ablation/dirichlet_smoothing_batadal_main_best_candidates.csv
results/outputs/window_threshold_ablation/dirichlet_smoothing/02_unseen_top_candidates/dirichlet_smoothing_batadal_unseen_top_candidates_metrics.csv
results/outputs/window_threshold_ablation/dirichlet_smoothing/02_unseen_top_candidates/dirichlet_smoothing_batadal_unseen_top_candidates_summary.csv
results/outputs/window_threshold_ablation/dirichlet_smoothing/03_reports/dirichlet_smoothing_batadal_final_comparison.csv

results/outputs/window_threshold_ablation/dirichlet_smoothing/01_main_ablation/dirichlet_smoothing_skab_main_metrics.csv
results/outputs/window_threshold_ablation/dirichlet_smoothing/01_main_ablation/dirichlet_smoothing_skab_main_summary.csv
results/outputs/window_threshold_ablation/dirichlet_smoothing/01_main_ablation/dirichlet_smoothing_skab_main_best_candidates.csv
results/outputs/window_threshold_ablation/dirichlet_smoothing/02_unseen_top_candidates/dirichlet_smoothing_skab_unseen_top_candidates_metrics.csv
results/outputs/window_threshold_ablation/dirichlet_smoothing/02_unseen_top_candidates/dirichlet_smoothing_skab_unseen_top_candidates_summary.csv
results/outputs/window_threshold_ablation/dirichlet_smoothing/03_reports/dirichlet_smoothing_skab_final_comparison.csv
```