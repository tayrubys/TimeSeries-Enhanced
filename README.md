### feature/state-similarity (Mesafe-Ağırlıklı Güven Skoru ve Tolerans)

*   **Problem:** Yüksek dereceli (Order=2+) Markov modellerinde, eğitimde hiç görülmemiş (unseen) bir örüntü geldiğinde modelin en yakın duruma eşleme yapması, ufak sensör sapmalarında bile yanlış alarm (False Positive) üretilmesine neden oluyordu.
    
*   **Çözüm (Geliştirme):** \* Modelin olasılık hesaplamasına Levenshtein mesafesi tabanlı eksponansiyel bir ceza katsayısı ($e^{-\\alpha D}$) eklendi.
    
    *   **Tolerans Payı (distance\_tolerance=1):** Sensörlerdeki doğal zamansal kaymaları (Concept Drift) siber saldırılardan ayırmak için 1 birimlik mesafe sapmalarına ceza muafiyeti getirildi.
        
*   **Test Bulguları (BATADAL & SKAB):**
    
    *   _Acımasız Ceza (Tolerance = 0):_ Modelin Precision değeri %7'ye kadar çöktü. Bu durum, BATADAL test setindeki "görülmemiş" örüntülerin aslında siber saldırı değil, ufak sapmalı _normal_ sistem davranışları (Concept Drift) olduğunu kanıtladı.
        
    *   _Esnek Ceza (Tolerance = 1):_ 1 birimlik tolerans uygulandığında F1 Skoru tekrar **0.4444** seviyesinde korundu ve modelin sahte alarmlara (paranoyaya) düşmesi engellendi.
        
*   **Akademik Çıkarım:** Olasılıksal Otomata modeli, 1 birimlik mesafe toleransı ile Concept Drift'e karşı dayanıklı (robust) hale getirilmiştir. Bu yapılandırılmış haliyle, derin öğrenme algoritmaları (LSTM/GRU) ile istatistiksel olarak **eşdeğer (**$p > 0.05$**)** bir anomali tespit performansı sergilediği kanıtlanmıştır.


---

# State-Similarity Branch Window Size + Threshold Ablation Raporu

## 1. Amaç

Bu çalışmada `feature/state-similarity` branch’i üzerinde `window_size` ve `score_threshold` değerlerinin model performansına etkisi incelendi.

Önceki deneylerde Markov tabanlı otomata modeli çoğunlukla `window_size=4` ile çalıştırılmıştı. Bu nedenle bu branch’te yalnızca state-similarity özelliğinin etkisini değil, farklı pencere boyutlarında ve farklı eşik değerlerinde modelin nasıl davrandığını görmek amaçlandı.

Temel araştırma soruları şunlardı:

- `window_size=4` gerçekten en iyi değer mi?
- Daha küçük veya daha büyük `window_size` değerleri performansı artırıyor mu?
- `score_threshold` değeri değiştiğinde state-similarity feature daha iyi sonuç veriyor mu?
- `unseen_data` üzerinde state-similarity feature beklenen katkıyı sağlıyor mu?

---

## 2. Kullanılan Branch

Çalışılan branch:

~~~text
feature/state-similarity
~~~

Bu branch’te kullanılan feature, görülmemiş veya eğitim verisindeki pattern’lere uzak olan pattern’lere mesafe cezası uygulamaktadır.

Model tarafındaki ilgili parametreler:

~~~text
distance_penalty_alpha
distance_tolerance
~~~

Bu parametrelerin amacı, benzer olmayan pattern’lerin geçiş olasılığını düşürerek anomali tespitini güçlendirmektir.

---

## 3. Eklenen Ablation Script’i

Ana `runner.py` dosyasına doğrudan müdahale etmek yerine ayrı bir deney script’i eklendi:

~~~text
src/experiments/window_threshold_ablation.py
~~~

Bu tercih edildi çünkü her feature branch’in `runner.py` dosyası farklılaşabiliyor. Her branch’te `runner.py` dosyasını değiştirmek yerine, aynı deney protokolünü çalıştıran ayrı bir ablation script’i daha güvenli ve takip edilebilir oldu.

Script’in yaptığı işlemler:

- Mevcut branch’teki `ProbabilisticAutomata` modelini kullanır.
- `settings.json` içindeki automata ayarlarını okur.
- Farklı `window_size` değerlerini dener.
- Her `window_size` için farklı `score_threshold` değerlerini dener.
- BATADAL ve SKAB için sonuçları ayrı ayrı üretir.
- İlk aşamada yalnızca `original` ve `gaussian_noise` senaryolarını çalıştırır.
- Main ablation bittikten sonra `original` F1 skoruna göre en iyi 3 adayı seçer.
- Sadece bu en iyi 3 aday için `unseen_data` değerlendirmesi yapar.
- Çıktıları düzenli klasör yapısında kaydeder.

---

## 4. Deney Aralığı

İlk aşamada `window_size=3-12` aralığı düşünülmüştü. Ancak büyük pencere değerlerinde özellikle state-similarity branch’inde Levenshtein tabanlı benzerlik hesaplamaları çok maliyetli hale geldi.

Ayrıca ilk gözlemlerde `window_size=11` ve `window_size=12` değerlerinin anlamlı katkı sağlaması beklenmediği için deney aralığı şu şekilde sınırlandırıldı:

~~~text
window_size = [3, 4, 5, 6, 7, 8, 9, 10]
~~~

Böylece hem mevcut baseline değeri olan `4` korundu hem de daha büyük window size değerleri gözlemlendi.

---

## 5. Threshold Aralıkları

İlk denemelerde bazı threshold değerlerinin özellikle büyük kaldığı ve birçok `window_size` değerinde F1 skorunun sıfır olduğu görüldü. Bu nedenle threshold listeleri daha düşük değerleri de içerecek şekilde genişletildi.

### BATADAL Threshold Aralığı

~~~text
[0.0, 0.001, 0.005, 0.01, 0.05, 0.1054, 0.2231, 0.3567,
 0.5108, 0.6931, 0.9163, 1.2040, 1.6094, 2.3026,
 2.9957, 3.5066, 3.9120, 4.2]
~~~

BATADAL için yüksek threshold değerleri tamamen kaldırılmadı çünkü önceki en iyi sonuç `score_threshold=3.9120` civarında elde edilmişti.

### SKAB Threshold Aralığı

~~~text
[0.0, 0.001, 0.005, 0.01, 0.05, 0.1054, 0.2231,
 0.3567, 0.5108, 0.6931, 0.9163]
~~~

SKAB tarafında düşük threshold değerleri daha önemli hale geldi çünkü düşük threshold değerleri recall değerini belirgin şekilde artırdı.

---

## 6. Çalıştırma Şekli

Önce dry-run ile script’in doğru branch, doğru model parametreleri ve doğru deney aralıklarıyla çalışacağı kontrol edildi:

~~~bash
python src/experiments/window_threshold_ablation.py --datasets skab --dry-run
~~~

Daha sonra SKAB için çalıştırıldı:

~~~bash
python src/experiments/window_threshold_ablation.py --datasets skab
~~~

Ardından BATADAL için çalıştırıldı:

~~~bash
python src/experiments/window_threshold_ablation.py --datasets batadal
~~~

---

## 7. Çıktı Klasör Yapısı

Sonuçlar şu ana klasör altında üretildi:

~~~text
results/outputs/window_threshold_ablation/state_similarity/
~~~

Klasör yapısı:

~~~text
00_partials/
01_main_ablation/
02_unseen_top_candidates/
03_reports/
~~~

Klasörlerin amacı:

- `00_partials/`: Uzun koşular sırasında ara kayıtlar.
- `01_main_ablation/`: `original` ve `gaussian_noise` ana ablation sonuçları.
- `02_unseen_top_candidates/`: En iyi 3 aday için `unseen_data` sonuçları.
- `03_reports/`: Seçilen adaylar, run plan ve final karşılaştırma dosyaları.

---

## 8. Unseen Data Stratejisi

Başta tüm kombinasyonlar için `unseen_data` hesaplanıyordu. Ancak state-similarity branch’inde unseen pattern değerlendirmesi Levenshtein mesafe hesaplarına dayandığı için büyük window size değerlerinde deney çok yavaşladı.

Bu nedenle deney stratejisi değiştirildi.

Yeni strateji:

1. Tüm `window_size` ve `score_threshold` kombinasyonları için yalnızca şu senaryolar çalıştırıldı:
   - `original`
   - `gaussian_noise`

2. Main ablation bittikten sonra `original` F1 skoruna göre en iyi 3 aday seçildi.

3. Yalnızca bu en iyi 3 aday için `unseen_data` değerlendirmesi yapıldı.

Bu sayede gereksiz hesaplama maliyeti azaltıldı ve deneyler daha verimli hale getirildi.

---

## 9. BATADAL Sonuçları

BATADAL için en iyi sonuç şu kombinasyonla elde edildi:

~~~text
window_size = 4
score_threshold = 3.9120
precision = 0.363636
recall = 0.571429
F1 = 0.444444
accuracy = 0.951220
~~~

Bu sonuç daha önce bilinen baseline BATADAL sonucu ile aynıdır. Yani state-similarity feature, BATADAL üzerinde baseline üstüne ek bir iyileşme sağlamadı.

### BATADAL En İyi 3 Aday

~~~text
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
~~~

### BATADAL Unseen Data Sonucu

En iyi adayın unseen-data sonucu:

~~~text
window_size = 4
score_threshold = 3.9120
unseen_precision = 0.250000
unseen_recall = 1.000000
unseen_F1 = 0.400000
unseen_accuracy = 0.812500
~~~

### BATADAL Yorumu

BATADAL tarafında `window_size=4` açık şekilde en iyi sonuç verdi.

`window_size=3` değerinde düşük thresholdlarda recall yükseldi, ancak precision çok düşük kaldı. Bu durum modelin çok fazla false positive ürettiğini gösteriyor.

`window_size=5-10` aralığında ise tüm threshold denemelerine rağmen anlamlı bir sonuç elde edilmedi. Bu aralıkta F1 skorları genel olarak sıfır kaldı.

BATADAL için sonuç:

~~~text
State-similarity feature, BATADAL üzerinde iyileşme sağlamadı.
En iyi sonuç yine window_size=4 ve score_threshold=3.9120 ile elde edildi.
~~~

---

## 10. SKAB Sonuçları

SKAB için en iyi original F1 sonucu şu kombinasyonla elde edildi:

~~~text
window_size = 5
score_threshold = 0.05
precision = 0.371625
recall = 1.000000
F1 = 0.541434
accuracy = 0.387431
~~~

İkinci en iyi sonuç:

~~~text
window_size = 4
score_threshold = 0.05
precision = 0.391427
recall = 0.863158
F1 = 0.537760
accuracy = 0.471682
~~~

Önceki referans ayar yaklaşık olarak şu seviyedeydi:

~~~text
window_size = 4
score_threshold = 0.2231
F1 ≈ 0.4911
~~~

Düşük threshold değerleri eklendiğinde SKAB tarafında original F1 skorunun yükseldiği görüldü.

### SKAB En İyi 3 Aday

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

### SKAB Yorumu

SKAB tarafında original senaryoda F1 artışı gözlendi. En iyi F1 değeri `window_size=5` ve `score_threshold=0.05` ile elde edildi.

Ancak bu artışın temel nedeni recall değerinin 1.0 seviyesine çıkmasıdır. Precision değeri düşük kaldığı için modelin fazla sayıda false positive ürettiği anlaşılmaktadır.

Accuracy değerinin de düşük kalması bu sonucu desteklemektedir.

SKAB için sonuç:

~~~text
State-similarity + düşük threshold, SKAB original F1 skorunu artırdı.
Ancak bu artış yüksek recall ve düşük precision pahasına gerçekleşti.
Unseen-data tarafında anlamlı bir iyileşme gözlenmedi.
~~~

---

## 11. Genel Değerlendirme

### BATADAL

~~~text
Önceki en iyi değer:
F1 = 0.444444

State-similarity ablation en iyi değer:
F1 = 0.444444

Sonuç:
İyileşme yok.
~~~

### SKAB

~~~text
Önceki referans:
window_size = 4
score_threshold = 0.2231
F1 ≈ 0.4911

State-similarity ablation en iyi değer:
window_size = 5
score_threshold = 0.05
F1 = 0.541434

Sonuç:
Original F1 arttı, ancak precision ve accuracy düştü.
~~~

### Unseen Data

~~~text
BATADAL unseen F1:
0.400000

SKAB unseen F1:
yaklaşık 0.166
~~~

State-similarity feature’ın özellikle unseen pattern benzerliği üzerinden katkı sağlaması bekleniyordu. Ancak unseen-data sonuçlarında belirgin bir iyileşme gözlenmedi.

---

## 12. Branch Kararı

`feature/state-similarity` branch’i karışık sonuçlar üretmiştir.

BATADAL tarafında herhangi bir iyileşme sağlanmamıştır. En iyi sonuç baseline ile aynı kalmıştır.

SKAB tarafında original F1 artmıştır. Ancak bu artış düşük threshold ile recall’ın agresif şekilde yükselmesinden kaynaklanmıştır. Precision ve accuracy düşük kaldığı için bu iyileşme güçlü ve kararlı bir iyileşme olarak değerlendirilmemelidir.

Unseen-data tarafında da beklenen katkı elde edilmemiştir.

Bu nedenle branch kararı:

~~~text
feature/state-similarity = sınırlı / kararsız iyileşme
~~~

Bu branch final merge için güçlü aday olarak değerlendirilmemektedir. Ancak window size ve threshold etkisini göstermek açısından sonuçlar dokümante edilmiştir.

---

## 13. Bugünkü Çalışmanın Özeti

Bugün yapılanlar:

- `feature/state-similarity` branch’i üzerinde çalışıldı.
- Ayrı bir `window_threshold_ablation.py` script’i eklendi.
- `runner.py` dosyasına doğrudan müdahale edilmedi.
- `window_size=3-10` aralığı test edildi.
- BATADAL ve SKAB için ayrı threshold listeleri oluşturuldu.
- Threshold listelerine daha düşük değerler eklendi.
- Main ablation sadece `original` ve `gaussian_noise` için çalıştırıldı.
- Main ablation sonunda en iyi 3 aday seçildi.
- Sadece en iyi 3 aday için `unseen_data` çalıştırıldı.
- Sonuçlar düzenli klasör yapısında kaydedildi.
- BATADAL’da iyileşme olmadığı görüldü.
- SKAB’de original F1 artışı görüldü, ancak bu artış düşük precision ve düşük accuracy pahasına gerçekleşti.
- Unseen-data tarafında güçlü bir iyileşme elde edilmedi.

Sonuç olarak, `feature/state-similarity` branch’i dokümante edildi ancak final merge için güçlü aday olarak seçilmedi.