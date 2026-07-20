# markov-final

Bu branch, Markov tabanlı probabilistic automata modeli için yapılan tüm feature denemeleri sonrasında seçilen final otomata modelini temsil eder.

`markov-final` branch’i, `feature/time-decay` branch’inin güncel ve temiz hali üzerinden oluşturulmuştur. Bu branch, `markov-base` modeline kıyasla BATADAL performansını korurken SKAB original senaryosunda daha dengeli ve belirgin bir F1 artışı sağlamıştır.

Final model tek bir global ayardan oluşmamaktadır. Bunun yerine dataset-specific final ayar kullanılmıştır:

```text
BATADAL -> base Markov ayarı korunur, time-decay kapalıdır.
SKAB    -> time-decay aktif kullanılır.
```

---

## Amaç

Bu branch’in amacı, proje kapsamında geliştirilen Markov tabanlı probabilistic automata modelinin final halini belirlemek ve önceki feature branch’lerle karşılaştırmalı olarak raporlamaktır.

Proje kapsamında temel hedef, deep learning modelleri ile probabilistic automata tabanlı yaklaşımları anomaly detection problemi üzerinde karşılaştırmak ve automata modelini mümkün olduğunca iyileştirmektir.

Başlangıç noktası `markov-base` branch’idir. Bu branch üzerinde çalışan temel model, yüksek dereceli Markov chain mantığına dayanan bir probabilistic automata modelidir.

Bu final branch’te amaç şudur:

```text
Markov-base modelini referans al.
Farklı feature branch sonuçlarını karşılaştır.
En iyi ve en dengeli otomata geliştirmesini seç.
Seçilen modeli markov-final olarak sabitle.
Deep learning modelleriyle final karşılaştırmaya hazır hale getir.
```

---

## Final Model Seçimi

Yapılan deneyler sonucunda final Markov modeli olarak `feature/time-decay` branch’i temel alınmıştır.

Final branch şu branch üzerinden oluşturulmuştur:

```text
feature/time-decay -> markov-final
```

Ancak final modelde time-decay her dataset için aynı şekilde uygulanmamıştır. Final runner sonucunda kullanılan yapı dataset-specific olarak belirlenmiştir:

```text
BATADAL:
time_decay_enabled = False

SKAB:
time_decay_enabled = True
```

Bu tercih yapılmıştır çünkü time-decay BATADAL üzerinde base modelin üzerine çıkmamış, ancak SKAB original senaryoda belirgin ve daha dengeli bir iyileşme sağlamıştır.

Time-decay yaklaşımı final model için seçilmiştir çünkü:

```text
BATADAL original performansını bozmadı.
BATADAL unseen-data performansını korudu.
SKAB original senaryoda belirgin F1 artışı sağladı.
SKAB gaussian_noise senaryosunda da original sonuca yakın performans üretti.
SKAB original artışı önceki feature branch’lere göre daha dengeli precision/recall dağılımıyla geldi.
```

Ancak sonuçlar değerlendirilirken şu sınırlılık dikkate alınmalıdır:

```text
SKAB unseen-data tarafında anlamlı bir iyileşme görülmemiştir.
```

Bu nedenle `markov-final`, tüm problemlerde deep learning modellerini geçen bir model olarak değil, Markov tabanlı automata ailesi içinde en güçlü ve en dengeli final aday olarak değerlendirilmiştir.

---

## Test Edilen Feature Branch'ler

Markov tabanlı modeli iyileştirmek için aşağıdaki feature branch’ler test edilmiştir:

| Branch | Amaç |
|---|---|
| `feature/state-similarity` | Görülmeyen veya nadir state geçişlerinde benzer state/pattern mantığıyla daha esnek skor üretmek |
| `feature/entropy-threshold` | Yüksek belirsizlik içeren transition durumlarını entropy ile filtrelemek |
| `feature/transition-confidence` | Transition güvenilirliğini skora dahil ederek daha güvenli karar üretmek |
| `feature/dirichlet-smoothing` | Transition probability tahminini Dirichlet smoothing ile daha kararlı hale getirmek |
| `feature/ensemble-markov` | Farklı Markov order değerlerinden gelen skorları birleştirmek |
| `feature/time-decay` | Daha güncel transition’lara daha yüksek ağırlık vererek zamansal değişimi modele dahil etmek |

Bu branch’lerin tamamı `markov-base` modelini iyileştirmek amacıyla ayrı ayrı denenmiştir.

---

## Base Markov Sonuçları

Referans olarak kullanılan temel model `markov-base` branch’idir.

Base model genel olarak şu yapıdadır:

```text
Model: ProbabilisticAutomata
Temel yaklaşım: High-order Markov Chain
Order: 3
Decision mode: avg_negative_log
```

Base modelin final referans ayarları:

```text
BATADAL:
window_size = 4
order = 3
smoothing_alpha = 0.1
score_window = 5
score_threshold = 3.9120
anomaly_threshold = 0.05

SKAB:
window_size = 4
order = 3
smoothing_alpha = 0.5
score_window = 10
score_threshold = 0.2231
anomaly_threshold = 0.90
```

Base Markov sonuçları:

| Dataset | Scenario | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| BATADAL | original | 0.951220 | 0.363636 | 0.571429 | 0.444444 |
| BATADAL | gaussian_noise | 0.946341 | 0.312121 | 0.485714 | 0.379635 |
| BATADAL | unseen_data | 0.812500 | 0.250000 | 1.000000 | 0.400000 |
| SKAB | original | 0.593595 | 0.495065 | 0.597499 | 0.491061 |
| SKAB | gaussian_noise | 0.587431 | 0.480094 | 0.596081 | 0.486964 |
| SKAB | unseen_data | 0.547826 | 0.142857 | 0.200000 | 0.166667 |

Base model, BATADAL üzerinde makul bir sonuç üretmiş; ancak SKAB tarafında deep learning modellerine göre geride kalmıştır.

---

## Feature Branch Karşılaştırması

Yapılan feature denemelerinin genel etkisi aşağıdaki gibidir:

| Feature Branch | BATADAL Etkisi | SKAB Etkisi | Genel Karar |
|---|---|---|---|
| `feature/state-similarity` | Base performansını büyük ölçüde korudu | Original F1 arttı fakat recall ağırlıklıydı | Sınırlı / kararsız iyileşme |
| `feature/entropy-threshold` | Ciddi performans düşüşü oluşturdu | Belirgin ve dengeli katkı sağlamadı | Zayıf aday |
| `feature/transition-confidence` | BATADAL F1 düştü | SKAB original F1 arttı fakat recall ağırlıklıydı | Zayıf / kararsız aday |
| `feature/dirichlet-smoothing` | BATADAL’da base’i geçemedi | SKAB’de sınırlı artış sağladı | Neutral / zayıf aday |
| `feature/ensemble-markov` | BATADAL base performansını korudu | SKAB original F1 arttı fakat artış daha çok recall ağırlıklıydı | Sınırlı pozitif aday |
| `feature/time-decay` | BATADAL base performansını korudu | SKAB original F1 belirgin ve daha dengeli arttı | En güçlü aday |

Özet sıralama:

```text
1. feature/time-decay
2. feature/ensemble-markov
3. feature/state-similarity
4. feature/dirichlet-smoothing
5. feature/transition-confidence
6. feature/entropy-threshold
```

Bu sıralamada `feature/time-decay` branch’i, hem BATADAL tarafında performans kaybı oluşturmaması hem de SKAB original tarafında daha dengeli bir artış sağlaması nedeniyle final model olarak seçilmiştir.

---

## Neden Time-Decay Seçildi?

Time-decay yaklaşımı, transition öğrenme aşamasında daha güncel transition’lara daha yüksek ağırlık vermeyi amaçlar.

Temel fikir:

```text
Daha yeni transition'lar daha yüksek ağırlık alır.
Daha eski transition'ların ağırlığı zamanla azalır.
Ancak çok eski transition'lar tamamen yok sayılmaz.
```

Time-decay etkisi transition öğrenme aşamasında uygulanır:

```text
transition_weight = self._get_time_decay_weight(age)
```

Bu nedenle time-decay, prediction aşamasında sonradan eklenen bir filtre değil, modelin transition probability öğrenme sürecini etkileyen bir mekanizmadır.

Final runner sonucuna göre time-decay sadece SKAB tarafında aktif kullanılmıştır:

```text
BATADAL:
time_decay_enabled = False

SKAB:
time_decay_enabled = True
time_decay_rate = 0.999
time_decay_min_weight = 0.01
```

Time-decay’in seçilme nedenleri:

```text
BATADAL original F1 değerini düşürmedi.
BATADAL unseen-data F1 değerini düşürmedi.
SKAB original F1 değerini yaklaşık 0.4911 seviyesinden 0.5740 seviyesine çıkardı.
SKAB gaussian_noise F1 değerini yaklaşık 0.4870 seviyesinden 0.5706 seviyesine çıkardı.
SKAB’de precision, recall ve accuracy değerleri önceki feature branch’lere göre daha dengeli kaldı.
```

Özellikle SKAB tarafındaki en iyi original sonuç:

```text
window_size = 4
score_threshold = 0.3567
precision = 0.550311
recall = 0.740352
F1 = 0.574023
accuracy = 0.641103
```

Bu sonuç, önceki bazı branch’lerde görülen sadece `recall=1.0` ile F1 yükselmesi durumundan daha değerlidir. Çünkü time-decay’de precision ve accuracy değerleri de daha makul seviyede kalmıştır.

---

## Markov-Final Parametreleri

`markov-final` branch’i dataset-specific final automata ayarlarını içerir.

Final karar tek bir global time-decay ayarı kullanmak yerine dataset-specific şekilde yapılmıştır:

```text
BATADAL -> time-decay kapalı, base Markov ayarı korunur
SKAB    -> time-decay açık, score_threshold=0.3567 kullanılır
```

Bu tercih yapılmıştır çünkü time-decay BATADAL üzerinde base modelin üzerine çıkmamış, ancak SKAB original senaryoda belirgin ve daha dengeli bir iyileşme sağlamıştır.

Genel model yapısı:

```text
Model: ProbabilisticAutomata
Base structure: High-order Markov Chain
Selected feature: Dataset-specific Time Decay
```

BATADAL final ayarları:

```text
order = 3
smoothing_alpha = 0.1
decision_mode = avg_negative_log
score_window = 5
score_threshold = 3.9120
time_decay_enabled = False
time_decay_rate = 0.99
time_decay_min_weight = 0.01
anomaly_threshold = 0.05
```

SKAB final ayarları:

```text
order = 3
smoothing_alpha = 0.5
decision_mode = avg_negative_log
score_window = 10
score_threshold = 0.3567
time_decay_enabled = True
time_decay_rate = 0.999
time_decay_min_weight = 0.01
anomaly_threshold = 0.90
```

Bu final yapı, BATADAL için base performansı korurken SKAB için daha iyi original ve gaussian_noise F1 sonucu vermektedir.

---

## BATADAL Final Sonuçları

BATADAL üzerinde yapılan window size ve score threshold ablation sonucunda en iyi sonuç şu kombinasyonla elde edilmiştir:

```text
window_size = 4
score_threshold = 3.9120
```

Final runner sonucuna göre BATADAL tarafında time-decay kapalıdır:

```text
time_decay_enabled = False
```

BATADAL final sonucu:

| Scenario | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| original | 0.951220 | 0.363636 | 0.571429 | 0.444444 |
| gaussian_noise | 0.946341 | 0.312121 | 0.485714 | 0.379635 |
| unseen_data | 0.812500 | 0.250000 | 1.000000 | 0.400000 |

BATADAL original için karşılaştırma:

| Model | Precision | Recall | F1 | Değişim |
|---|---:|---:|---:|---:|
| Markov-base | 0.363636 | 0.571429 | 0.444444 | - |
| Markov-final | 0.363636 | 0.571429 | 0.444444 | 0.000000 |

BATADAL unseen-data için karşılaştırma:

| Model | Precision | Recall | F1 | Değişim |
|---|---:|---:|---:|---:|
| Markov-base | 0.250000 | 1.000000 | 0.400000 | - |
| Markov-final | 0.250000 | 1.000000 | 0.400000 | 0.000000 |

BATADAL yorumu:

```text
Time-decay, BATADAL üzerinde ek bir F1 artışı sağlamamıştır.
Bu nedenle final runner’da BATADAL için time-decay kapalı bırakılmış ve base Markov ayarı korunmuştur.
Bu tercih sayesinde BATADAL original, gaussian_noise ve unseen-data performansı bozulmadan korunmuştur.
BATADAL için en iyi pencere boyutu yine window_size=4 olmuştur.
window_size=5 ve üzerindeki değerlerde F1 skorları genel olarak 0.0 seviyesinde kalmıştır.
```

---

## SKAB Final Sonuçları

SKAB üzerinde yapılan window size ve score threshold ablation sonucunda en iyi original F1 şu kombinasyonla elde edilmiştir:

```text
window_size = 4
score_threshold = 0.3567
```

Final runner sonucuna göre SKAB tarafında time-decay aktiftir:

```text
time_decay_enabled = True
time_decay_rate = 0.999
time_decay_min_weight = 0.01
```

SKAB final sonuçları:

| Scenario | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| original | 0.641103 | 0.550311 | 0.740352 | 0.574023 |
| gaussian_noise | 0.636692 | 0.543078 | 0.739234 | 0.570633 |
| unseen_data | 0.547826 | 0.142857 | 0.200000 | 0.166667 |

SKAB original karşılaştırması:

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Markov-base | 0.593595 | 0.495065 | 0.597499 | 0.491061 |
| Markov-final | 0.641103 | 0.550311 | 0.740352 | 0.574023 |

SKAB original değişim:

| Metric | Markov-base | Markov-final | Değişim |
|---|---:|---:|---:|
| Accuracy | 0.593595 | 0.641103 | +0.047508 |
| Precision | 0.495065 | 0.550311 | +0.055246 |
| Recall | 0.597499 | 0.740352 | +0.142853 |
| F1 | 0.491061 | 0.574023 | +0.082962 |

SKAB gaussian_noise karşılaştırması:

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Markov-base | 0.587431 | 0.480094 | 0.596081 | 0.486964 |
| Markov-final | 0.636692 | 0.543078 | 0.739234 | 0.570633 |

SKAB gaussian_noise değişim:

| Metric | Markov-base | Markov-final | Değişim |
|---|---:|---:|---:|
| Accuracy | 0.587431 | 0.636692 | +0.049261 |
| Precision | 0.480094 | 0.543078 | +0.062984 |
| Recall | 0.596081 | 0.739234 | +0.143153 |
| F1 | 0.486964 | 0.570633 | +0.083669 |

SKAB unseen-data sonucu:

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Markov-base | 0.547826 | 0.142857 | 0.200000 | 0.166667 |
| Markov-final | 0.547826 | 0.142857 | 0.200000 | 0.166667 |

SKAB yorumu:

```text
Time-decay, SKAB original senaryoda belirgin ve daha dengeli bir iyileşme sağlamıştır.
F1 skoru yaklaşık 0.4911 seviyesinden 0.5740 seviyesine çıkmıştır.
Gaussian_noise senaryosunda da F1 skoru yaklaşık 0.4870 seviyesinden 0.5706 seviyesine yükselmiştir.
Precision, recall ve accuracy değerlerinin birlikte artması, bu sonucu önceki recall-ağırlıklı feature sonuçlarından daha güçlü kılmaktadır.
Ancak unseen-data tarafında iyileşme görülmemiştir.
```

---

## Deep Learning Modelleriyle Karşılaştırma

Projedeki deep learning baseline modelleri LSTM ve GRU’dur.

Final runner sonucunda Automata modeli `markov-final` değerleriyle karşılaştırmaya dahil edilmiştir.

Final karşılaştırma için kullanılan değerler:

| Dataset | Model | Mean F1 |
|---|---|---:|
| BATADAL | LSTM | 0.560850 |
| BATADAL | GRU | 0.558519 |
| BATADAL | Markov-base | 0.444444 |
| BATADAL | Markov-final | 0.444444 |
| SKAB | LSTM | 0.840446 |
| SKAB | GRU | 0.858942 |
| SKAB | Markov-base | 0.491061 |
| SKAB | Markov-final | 0.574023 |

Genel yorum:

```text
Markov-final, Markov-base modeline göre SKAB original ve gaussian_noise senaryolarında daha iyi sonuç vermiştir.
Ancak deep learning modelleri, özellikle SKAB üzerinde hâlâ belirgin şekilde daha yüksek F1 skorlarına sahiptir.
BATADAL üzerinde Markov-final, Markov-base performansını korumuş ancak LSTM/GRU modellerinin gerisinde kalmıştır.
```

Final runner sonrası Wilcoxon testleri:

| Dataset | Karşılaştırma | Mean A | Mean B | p-value | Anlamlı |
|---|---|---:|---:|---:|---|
| SKAB | LSTM vs GRU | 0.840446 | 0.858942 | 0.0625 | Hayır |
| SKAB | LSTM vs Automata | 0.840446 | 0.574023 | 0.0625 | Hayır |
| SKAB | GRU vs Automata | 0.858942 | 0.574023 | 0.0625 | Hayır |
| BATADAL | LSTM vs GRU | 0.560850 | 0.558519 | 1.0000 | Hayır |
| BATADAL | LSTM vs Automata | 0.560850 | 0.444444 | 0.3125 | Hayır |
| BATADAL | GRU vs Automata | 0.558519 | 0.444444 | 0.1875 | Hayır |

Bu sonuçlar, örneklem sayısı ve fold/seed yapısı nedeniyle istatistiksel anlamlılık üretmemiştir. Ancak ortalama F1 değerleri açısından deep learning modelleri genel olarak daha güçlü görünmektedir.

---

## Sınırlılıklar

`markov-final` branch’i önemli bir iyileştirme sağlasa da bazı sınırlılıklar devam etmektedir.

Başlıca sınırlılıklar:

```text
SKAB unseen-data tarafında iyileşme görülmemiştir.
BATADAL üzerinde base modelin üzerine çıkılamamıştır.
Time-decay katkısı dataset-dependent görünmektedir.
BATADAL için time-decay aktif kullanılmamış, base Markov ayarı korunmuştur.
SKAB original tarafındaki iyileşme güçlü olsa da farklı unseen kurulumlarda tekrar doğrulanmalıdır.
Deep learning modelleri hâlâ özellikle SKAB üzerinde daha yüksek F1 skorlarına sahiptir.
```

Ayrıca Markov tabanlı modellerin doğası gereği:

```text
Uzun temporal bağımlılıkları öğrenme kapasitesi sınırlıdır.
Çok büyük window size değerleri sparse transition problemine yol açabilmektedir.
State discretization / representation kalitesi model performansını doğrudan etkilemektedir.
Threshold seçimi performans üzerinde oldukça belirleyicidir.
```

Bu nedenle `markov-final`, deep learning modellerinin yerine geçen nihai çözüm olarak değil, probabilistic automata yaklaşımının iyileştirilmiş ve deneysel olarak en güçlü hali olarak değerlendirilmelidir.

---

## Sonuç

Bu branch, proje kapsamında geliştirilen Markov tabanlı probabilistic automata modelinin final halini temsil eder.

Final karar:

```text
markov-final = dataset-specific final automata modeli
```

Final yapı:

```text
BATADAL:
- Base Markov ayarı korunmuştur.
- time_decay_enabled = False

SKAB:
- Time-decay aktif kullanılmıştır.
- time_decay_enabled = True
- time_decay_rate = 0.999
- score_threshold = 0.3567
```

Ana bulgular:

```text
BATADAL:
- Markov-final, Markov-base performansını korudu.
- Original F1 = 0.444444
- Gaussian-noise F1 = 0.379635
- Unseen-data F1 = 0.400000

SKAB:
- Markov-final, Markov-base modeline göre belirgin iyileşme sağladı.
- Original F1 = 0.491061 -> 0.574023
- Gaussian-noise F1 = 0.486964 -> 0.570633
- Precision, recall ve accuracy birlikte arttı.
- Unseen-data F1 değişmedi.
```

Genel sonuç:

```text
Time-decay mekanizması, Markov tabanlı automata modeli için test edilen feature’lar arasında en güçlü ve en dengeli aday olmuştur. Final model dataset-specific şekilde yapılandırılmıştır: BATADAL tarafında base Markov ayarı korunmuş, SKAB tarafında time-decay aktif kullanılmıştır. Bu yapı BATADAL performansını bozmadan korumuş, SKAB original ve gaussian_noise senaryolarında anlamlı bir performans artışı sağlamıştır. Bununla birlikte, unseen-data sonuçları genelleme katkısının sınırlı olduğunu göstermektedir.
```

Bu nedenle `markov-final`, paper/rapor aşamasında automata modelinin final versiyonu olarak kullanılabilir.

---

## Üretilen Çıktılar

`markov-final` branch’i, `feature/time-decay` üzerinden türetildiği için time-decay window ablation çıktıları final değerlendirmeye dahil edilmiştir.

Window size ve score threshold ablation çıktıları:

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

Final karşılaştırmada kullanılması önerilen ana dosyalar:

```text
results/outputs/window_threshold_ablation/time_decay/03_reports/time_decay_batadal_final_comparison.csv
results/outputs/window_threshold_ablation/time_decay/03_reports/time_decay_skab_final_comparison.csv
```

Final runner çıktıları:

```text
results/outputs/automata_advanced_all_scenarios_metrics.csv
results/outputs/automata_batadal_seed_summary.csv
results/outputs/automata_skab_fold_summary.csv
results/outputs/statistical_test_results.csv
```