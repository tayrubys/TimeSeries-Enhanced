# Otomata Tabanlı Model İyileştirmeleri

Bu bölümde BATADAL ve SKAB veri setleri üzerinde otomata tabanlı anomali tespit modelinin performansını artırmak amacıyla yapılan iyileştirmeler raporlanmaktadır.

## 1. Weighted Transition Probability

Klasik otomata modelinde geçiş olasılığı yalnızca geçiş frekansına göre hesaplanmaktaydı. Bu çalışmada geçiş olasılıkları ağırlıklandırılarak sık görülen geçişlerin etkisi artırılmış, nadir geçişlerin etkisi daha kontrollü hale getirilmiştir.

Kullanılan temel hesaplama:

```text
base_prob = count / total_output
weight = log1p(weight_sharpness * base_prob)
raw_score = base_prob * weight
weighted_probability = raw_score / total_raw_score
```

## 2. Parametre Taraması

Model performansını artırmak için `window_size`, `alphabet_size` ve `weight_sharpness` parametreleri grid search mantığıyla taranmıştır.

| Dataset | En İyi Window Size | En İyi Alphabet Size | En İyi Weight Sharpness |
|---|---|---|---|
| BATADAL | 4 | 3 | 6.0 |
| SKAB | 4 | 4 | 2.0 |

## 3. Threshold Duyarlılık Analizi

Weighted Transition Probability sonrasında anomali karar eşiği de yeniden incelenmiştir. Farklı threshold değerleri denenmiş ve F1-score'a göre en iyi değerler seçilmiştir.

| Dataset | En İyi Threshold | F1-score |
|---|---|---|
| BATADAL | 0.01 | 0.1667 |
| SKAB | 0.99 | 0.5414 |

## 4. Similarity Penalty Deneyi
 
Weighted Transition Probability sonrasında, özellikle eğitim verisinde görülmeyen (`unseen`) sembolik pattern'ların daha kontrollü değerlendirilmesi için Levenshtein distance tabanlı Similarity Penalty yöntemi denenmiştir.
 
Bu yöntemde test sırasında gelen pattern eğitimde yoksa, önce eğitimdeki en yakın bilinen pattern'a eşlenmiştir. Daha sonra bu eşleşmenin güvenilirliğini ölçmek için Levenshtein mesafesi kullanılmıştır. Mesafe arttıkça geçiş olasılığı düşürülerek uzak eşleşmelerin daha şüpheli değerlendirilmesi amaçlanmıştır.
 
Kullanılan temel hesaplama:
 
```text
similarity_score = 1 / (1 + similarity_penalty_strength * distance)
final_prob = weighted_transition_probability * similarity_score
```
 
Farklı `similarity_penalty_strength` değerleri denenmiştir:
 
```text
[0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
```
 
| Dataset | En İyi Similarity Penalty Strength | Original F1-score | Unseen F1-score |
|---|---|---|---|
| BATADAL | 0.0 | 0.1667 | 0.0000 |
| SKAB | 0.1 | 0.5416 | 0.3027 |
 
Deney sonuçlarına göre Similarity Penalty yöntemi BATADAL veri setinde performans artışı sağlamamış, en iyi sonuç penalty uygulanmadığında (`strength=0.0`) elde edilmiştir. SKAB veri setinde ise `strength=0.1` değerinde çok sınırlı bir artış görülmüştür. Ancak bu artış oldukça küçük olduğu için yöntem ana iyileştirme olarak değil, ek analiz deneyi olarak değerlendirilmiştir.

## 5. Relative Transition Score Deneyi

Similarity Penalty sonrasında, geçiş olasılıklarını daha hassas değerlendirmek amacıyla Relative Transition Score yaklaşımı denenmiştir.

Bu yöntemde mevcut geçiş olasılığı, aynı state üzerinden çıkabilecek en yüksek geçiş olasılığı ile karşılaştırılmıştır.

Kullanılan temel hesaplama:

```text
relative_score = transition_probability / best_transition_probability
```
Amaç, bir geçişin yalnızca mutlak olasılığına değil, aynı state içindeki en güçlü geçişe göre ne kadar zayıf kaldığına bakmaktı. Böylece düşük göreli skora sahip geçişlerin anomali olarak daha kolay yakalanması hedeflenmiştir.

Ancak yapılan deneylerde Relative Transition Score yaklaşımının hem BATADAL hem de SKAB veri setlerinde performansı düşürdüğü gözlemlenmiştir. Bu nedenle bu yöntem final modelde kullanılmamış ve kod akışından tamamen kaldırılmıştır.

-----
Son durumda final otomata modeli şu yapı ile devam etmektedir:

- Weighted Transition Probability aktif
- Threshold duyarlılık analizi ile seçilen eşik değerleri kullanılıyor
- Similarity Penalty ana deneyde kapalı
- Relative Transition Score final modelden kaldırıldı
## 6. Final Sonuçlar

| Dataset | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| BATADAL | 0.9512 | 0.2000 | 0.1429 | 0.1667 |
| SKAB | 0.5261 | 0.4104 | 0.8056 | 0.5414 |
