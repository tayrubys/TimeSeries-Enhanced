# TimeSeries-Enhanced

Bu proje, zaman serilerindeki anomalileri tespit etmek için kullanılan Derin Öğrenme (Deep Learning) ve Otomata tabanlı yaklaşımların iyileştirilmesi ve optimize edilmesi amacıyla oluşturulmuştur. 

## ALERGIA-inspired Probabilistic State-Merging Denemesi

Bu branch üzerinde otomata modelinin BATADAL performansını artırmak amacıyla ALERGIA-inspired probabilistic state-merging yöntemini denedim. Model, ADASYN ile dengelenmiş PC1 eğitim verisi üzerinde PAA ve SAX dönüşümleri uygulanarak oluşturuldu.

### İlk State-Merging Denemesi

İlk aşamada benzer geçiş davranışlarına sahip SAX pattern state’leri istatistiksel olarak birleştirildi. Ancak normal ve anomalili geçişlerin aynı otomata içinde öğrenilmesi sınıflar arasındaki ayrımı azalttı. Bu sürümde test F1 değeri `0.0714` olarak elde edildi.

### Dual ALERGIA Yaklaşımı

Normal ve anomalili geçişlerin birbirinden ayrı öğrenilebilmesi için iki farklı state-merging modeli oluşturuldu:

```text
Normal geçişler  → Normal ALERGIA modeli
Anomali geçişleri → Anomali ALERGIA modeli
```
Bir geçişin iki modelde ürettiği olasılık değerleri karşılaştırılarak anomali skoru hesaplandı. İlk sürümde istenen performans elde edilemediği için state birleştirme yapısı daha ayrıntılı hâle getirildi.

Pattern-Aware State-Merging

İlk modelde state’ler yalnızca sonraki SAX sembolüne göre karşılaştırılıyordu. Bu durum farklı pattern’ların gereğinden fazla birleştirilmesine neden oldu.

Bu problemi azaltmak için state’ler, sonraki tek sembol yerine sonraki tam SAX pattern dağılımına göre karşılaştırıldı. Böylece geçiş davranışı farklı olan pattern’ların aynı state altında toplanması azaltıldı.

Elde edilen en iyi sonuçlar:

| Senaryo        | Precision | Recall |     F1 |
| -------------- | --------: | -----: | -----: |
| Original       |    0.1818 | 0.2143 | 0.1967 |
| Gaussian Noise |    0.1765 | 0.2143 | 0.1935 |
| Unseen Data    |    0.0000 | 0.0000 | 0.0000 |

Pattern-aware geliştirme sonrasında test F1 değeri 0.0400 seviyesinden 0.1967 seviyesine yükseldi.

Ek Denemeler

State’lerin birleştirilmesi sırasında farklı Levenshtein mesafe sınırları denendi. Ayrıca unseen pattern’ların anomali skoruna mesafeye bağlı ek ceza verilmesi test edildi.

Bu ek denemeler test performansında kararlı bir artış sağlamadığı için final yapıya dahil edilmedi.

Genel Sonuç

ALERGIA-inspired state-merging yöntemi içerisinde en iyi sonuç, state’lerin sonraki tam pattern dağılımına göre karşılaştırıldığı pattern-aware sürümde elde edildi.

Yöntem state sayısını azaltarak daha genel bir otomata oluşturdu ve ilk ALERGIA sürümlerine göre performansı artırdı. Ancak unseen pattern’ların belirlenmesinde yeterli başarı sağlanamadığı görüldü


## SKAB Dual ALERGIA Denemesi
PAA ve SAX dönüşümleri her kaynak dosya için ayrı uygulanarak farklı dosyalar arasında sahte geçiş oluşması engellendi. Deneyler 5 farklı seed ile tekrarlandı.

| Senaryo | Ortalama Precision | Ortalama Recall | Ortalama F1 |
|---|---:|---:|---:|
| Original | 0.3554 | 0.7194 | 0.4541 |
| Gaussian Noise | 0.3583 | 0.7180 | 0.4550 |
| Unseen Data | 0.0000 | 0.0000 | 0.0000 |

Original ve Gaussian noise sonuçlarının birbirine yakın olması, modelin eklenen gürültüden fazla etkilenmediğini gösterdi. Fold sonuçları arasında farklılık bulunduğu için modelin SKAB üzerindeki performansının dosyalara bağlı olarak değişebildiği görüldü.

### Levenshtein Uzaklık Cezası

Unseen pattern eşleştirmelerini daha güvenilir hâle getirmek için Levenshtein uzaklığı surprise skoruna ceza olarak eklendi. Ceza katsayısı diğer model parametreleriyle birlikte validation seti üzerinden seçildi. Ayrıca state-merging işlemi tekrarlanabilir sonuçlar üretecek şekilde düzenlendi.

Bu düzenleme sonucunda BATADAL test F1 değeri yaklaşık `0.284`, SKAB ortalama F1 değeri ise yaklaşık `0.507` olarak elde edildi.

### Geçiş Güvenilirliği Denemesi

Az görülen state geçişlerinin etkisini azaltmak amacıyla dual surprise skorları geçiş desteğine göre ağırlıklandırıldı. Farklı güvenilirlik katsayıları validation setinde denendi ancak en iyi sonuç ağırlıklandırmanın kapalı olduğu durumda elde edildi. Bu nedenle yöntem final modele eklenmedi.

### Temporal Persistence Filter

Levenshtein uzaklık cezası sonrasında modelin çok sayıda normal örneği anomaly olarak işaretlediği görüldü. Kısa süreli ve tek başına kalan anomaly tahminlerini temizlemek amacıyla Temporal Persistence Filter eklendi.

Threshold ve minimum ardışık anomaly uzunluğu yalnızca validation seti üzerinden seçildi. Filtrenin gerçek anomalileri fazla silmesini önlemek için validation recall değerinin en az `0.70` olması şartı kullanıldı. En iyi ayarda `min_anomaly_run=8` seçildi.

| Senaryo        | Precision | Recall |     F1 |
| -------------- | --------: | -----: | -----: |
| Original       |    0.2907 | 0.8929 | 0.4386 |
| Gaussian Noise |    0.2907 | 0.8929 | 0.4386 |
| Unseen Data    |    0.2667 | 0.4000 | 0.3077 |

Bu düzenleme ile BATADAL üzerindeki yanlış anomaly tahminleri azaltılırken recall büyük ölçüde korundu. `0.4386` değeri mevcut en iyi deney sonucu olarak kaydedildi.

#### SKAB Temporal Persistence Filter Denemesi

BATADAL üzerinde başarılı olan Temporal Persistence Filter, SKAB veri setinde de source file sınırları korunarak denendi. Threshold ve minimum ardışık anomaly uzunluğu her fold ve seed için inner validation üzerinden seçildi. Ayrıca gerçek anomalilerin fazla silinmemesi için validation recall değerinin en az `0.70` olması şartı kullanıldı.

Deney sonucunda SKAB ortalama F1 değeri yaklaşık `0.507` seviyesinden `0.501` seviyesine düştü. Precision değeri çok az artarken recall değerinin düşmesi, filtrenin SKAB üzerindeki bazı kısa veya kesintili anomaly bölgelerini kaldırdığını gösterdi.

Bu nedenle Temporal Persistence Filter BATADAL modelinde korunurken SKAB final modeline dahil edilmedi.

### SKAB Hysteresis Thresholding Denemesi

SKAB tarafında tek threshold kullanımının oluşturduğu kararsız anomaly geçişlerini azaltmak amacıyla çift eşikli Hysteresis Thresholding yöntemi denendi. Yöntem her `source_file` için ayrı uygulandı. Yüksek eşik anomaly durumunu başlatırken düşük eşik başlamış anomaly durumunun devam etmesini sağladı. Eşikler ve hysteresis delta değeri her fold ve seed için inner validation üzerinden seçildi.

Bazı validation bölümlerinde iyileşme görülmesine rağmen 5-fold × 5-seed genel sonucunda ortalama F1 değeri yaklaşık `0.507` seviyesinden `0.499` seviyesine düştü. Ayrıca birçok fold ve seed için en iyi ayar `delta=0.0` olarak seçildi. Bu durum, hysteresis mekanizmasının çoğu deneyde mevcut tek-threshold yöntemine ek katkı sağlamadığını gösterdi.

Bu nedenle Hysteresis Thresholding yöntemi SKAB final modeline dahil edilmedi.

### One-Class ALERGIA Denemesi

BATADAL üzerinde Dual ALERGIA yapısına alternatif olarak yalnızca normal geçişleri öğrenen One-Class ALERGIA yöntemi denendi. Bu yöntemde anomaly geçişler otomatanın eğitiminde kullanılmadı ve geçişlerin normal modele göre surprise değerleri anomaly skoru olarak değerlendirildi.

İlk deneyde yüksek surprise değerleri anomaly olarak kabul edildi. Ancak validation sonuçlarında anomalili geçişlerin ortalama surprise değerinin normal geçişlerden daha düşük olduğu görüldü. Bu nedenle ikinci aşamada threshold yönü de validation üzerinden seçilecek şekilde hem yüksek hem düşük skor yönleri denendi.

Çift yönlü threshold seçimi validation sonucunu bir miktar artırsa da test sonucunda Precision `0.1176`, Recall `0.1429` ve F1-score `0.1290` olarak elde edildi. Bu sonuç mevcut Dual ALERGIA tabanlı modelin gerisinde kaldığı için One-Class ALERGIA final modele dahil edilmedi.

### Class-Specific Dual ALERGIA Denemesi

BATADAL üzerinde normal ve anomaly otomatalarının farklı davranış yapılarına sahip olabileceği düşünülerek Class-Specific Dual ALERGIA yöntemi denendi. Mevcut Dual ALERGIA yapısında iki otomata aynı state-merging parametrelerini kullanırken bu deneyde `merge_alpha`, `min_state_count` ve `smoothing_alpha` değerleri normal ve anomaly modelleri için ayrı ayrı seçildi.

Toplam 27 normal ve 27 anomaly model eğitildi. Eğitilen modellerin bütün kombinasyonları kullanılarak `27 × 27 = 729` model çifti validation seti üzerinde değerlendirildi. Seçilen model çifti üzerinde normal ve anomaly Levenshtein cezaları, threshold ve Temporal Persistence parametreleri ayrıca validation üzerinden tarandı. Parametre seçiminde validation recall değerinin en az `0.70` olması şartı korundu.

Validation sonucunda normal model için `merge_alpha=0.01`, `min_state_count=10`, `smoothing_alpha=0.5`; anomaly model için ise `merge_alpha=0.1`, `min_state_count=5`, `smoothing_alpha=1.0` seçildi. Final deneyde her iki model için distance penalty `2.0`, minimum anomaly run değeri `6` olarak belirlendi.

Test sonucunda Precision `0.1827`, Recall `0.6786` ve F1-score `0.2879` olarak elde edildi. Bu sonuç mevcut Dual ALERGIA + Levenshtein + Temporal Persistence sonucunun gerisinde kaldığı için Class-Specific Dual ALERGIA final modele dahil edilmedi.

### Soft Top-k Levenshtein Mapping Denemesi

BATADAL üzerinde unseen SAX pattern’larının yalnızca tek bir en yakın train pattern’ına eşlenmesi yerine, en yakın birkaç pattern’ın geçiş olasılıklarını ağırlıklı olarak kullanan Soft Top-k Levenshtein Mapping yöntemi denendi.

Deneyde mevcut en iyi Dual ALERGIA state-merging yapısı sabit tutuldu. `top_k=[1,3,5]` ve `gamma=[0.5,1.0,2.0]` değerleri farklı Levenshtein distance penalty, threshold ve Temporal Persistence ayarlarıyla validation seti üzerinde değerlendirildi.

`top_k=3` kullanıldığında validation recall değeri `1.0` seviyesine yükseldi ancak yanlış pozitif tahminlerin artması nedeniyle precision düştü. `top_k=5` kullanımında ise sınıflar arasındaki ayrım daha fazla azaldı. Validation sonucunda en iyi ayar tekrar `top_k=1` olarak seçildi.

Test sonucunda Precision `0.2907`, Recall `0.8929` ve F1-score `0.4386` olarak elde edildi. Sonuç mevcut tek-komşu Levenshtein eşleme yöntemiyle aynı kaldığı için Soft Top-k yaklaşımı final modele dahil edilmedi.

### Support-Aware Dual ALERGIA Denemesi

BATADAL üzerinde düşük geçiş sayısına sahip tahminlerin daha temkinli değerlendirilmesi amacıyla Support-Aware Dual ALERGIA yöntemi denendi. Bu yöntemde normal ve anomaly otomatalarındaki geçiş sayıları kullanılarak her geçiş için bir support güven değeri hesaplandı. Eğitimde az görülen geçişlerin anomaly skorunu azaltmak için farklı `support_weight` ve `tau` değerleri validation seti üzerinde tarandı.

Deney sırasında mevcut en iyi Dual ALERGIA yapısı sabit tutuldu. `support_weight=[0.0,0.25,0.5,1.0,2.0]` ve `tau=[1,2,5,10]` değerleri, threshold ve Temporal Persistence parametreleriyle birlikte validation üzerinden değerlendirildi. Validation recall değerinin en az `0.70` olması şartı korundu.

Analiz sonucunda normal geçişlerin ortalama support değeri `17.4974`, anomaly geçişlerin ortalama support değeri ise `9.3846` olarak bulundu. Anomaly geçişlerinin doğal olarak daha düşük desteğe sahip olması nedeniyle support cezası gerçek anomalilerin skorlarını da düşürdü. Support cezası kullanılan ayarlarda validation F1-score yaklaşık `0.136` seviyesinde kaldı.

Validation sonucunda en iyi ayar tekrar `support_weight=0.0` olarak seçildi. Test sonucunda Precision `0.2907`, Recall `0.8929` ve F1-score `0.4386` olarak elde edildi. Sonuç mevcut Dual ALERGIA modeliyle aynı kaldığı için Support-Aware yaklaşım final modele dahil edilmedi.

### Class-Relative Support Ratio Denemesi

Geçiş sayılarının normal veya anomaly modelini ne ölçüde desteklediğini dikkate alan Class-Relative Support Ratio yaklaşımı denenmiştir. Mevcut başarılı ALERGIA yapısı sabit tutulmuş; yalnızca `support_weight`, `beta` ve `tau` parametreleri validation üzerinden taranmıştır.

Support kullanılan en iyi ayarda validation F1 `0.2824` olarak elde edilmiş, ancak mevcut modelin `0.2857` sonucunu geçememiştir. Validation tekrar `support_weight=0.0` değerini seçmiş ve test F1 `0.4386` olarak değişmeden kalmıştır. Bu nedenle yöntem final modele dahil edilmemiş ve deney kodları kaldırılmıştır.

### Class-Normalized Raw Transition Support Denemesi

Normal ve anomaly sınıflarındaki aynı ham pattern geçişleri karşılaştırılarak, geçiş sayılarının sınıf içindeki toplam çıkış sayılarına göre normalize edildiği Class-Normalized Raw Transition Support yaklaşımı denenmiştir.

Mevcut başarılı Dual ALERGIA yapısı sabit tutulmuş ve yalnızca `support_weight`, `beta` ve `tau` parametreleri validation üzerinden taranmıştır. Support yön doğruluğu yaklaşık `0.49` seviyesinde kalmış ve geçiş desteğinin normal ile anomaly sınıflarını yeterince ayıramadığı görülmüştür.

Support kullanılan ayarlar mevcut `0.2857` validation F1 değerini geçememiştir. Validation tekrar `support_weight=0.0` değerini seçmiş ve test F1 `0.4386` olarak değişmeden kalmıştır. Bu nedenle yöntem final modele dahil edilmemiş ve deney kodları kaldırılmıştır.

### Dual-Threshold Hysteresis Denemesi

False positive anomaly bloklarını azaltmak amacıyla anomaly durumunu başlatmak ve devam ettirmek için iki farklı eşik kullanan Dual-Threshold Hysteresis yaklaşımı denenmiştir. Mevcut Dual ALERGIA model yapısı sabit tutulmuş; `high_threshold`, `low_threshold` ve `min_anomaly_run` değerleri validation üzerinden seçilmiştir.

Yöntem validation F1 değerini `0.2857` seviyesinden `0.3333` seviyesine yükseltmiştir. Ancak test sonucunda recall değeri `0.8929` seviyesinden `0.7143` seviyesine düşmüş ve test F1 değeri `0.4386` yerine `0.4082` olarak elde edilmiştir.

Yapılan tanı analizinde hysteresis yönteminin validation üzerinde 17 false positive tahmini kaldırırken yalnızca 1 true positive tahmini kaybettiği görülmüştür. Test üzerinde ise 11 false positive kaldırılmış, ancak 5 true positive kaybedilmiş ve iki gerçek anomaly bloğundan biri tamamen kaçırılmıştır. Validation verisinde yalnızca bir anomaly bloğu bulunduğu için seçilen hysteresis yapısının testteki farklı anomaly davranışına genelleşmediği değerlendirilmiştir.

Bu nedenle yöntem final modele dahil edilmemiş ve mevcut `0.4386` test F1 sonucuna sahip Dual ALERGIA modeli korunmuştur.

### SKAB Dual-Threshold Hysteresis Denemesi

SKAB üzerinde false positive tahminleri azaltmak amacıyla Dual-Threshold Hysteresis yöntemi denendi. Anomaly başlangıcı için `high_threshold`, devamı için `low_threshold` kullanıldı. Parametreler yalnızca inner validation üzerinden seçildi ve hysteresis durumu her `source_file` başlangıcında sıfırlandı.

#### Genel Sonuçlar

| Yöntem                    | Senaryo        | Accuracy | Precision | Recall | F1-score | F1 Std. Sapma |
| ------------------------- | -------------- | -------: | --------: | -----: | -------: | ------------: |
| Tek Threshold Baseline    | Original       |   0.4402 |    0.3766 | 0.8162 |   0.5069 |        0.0779 |
| Dual-Threshold Hysteresis | Original       |   0.4802 |    0.3905 | 0.7379 |   0.5083 |        0.0507 |
| Tek Threshold Baseline    | Gaussian Noise |   0.4405 |    0.3767 | 0.8160 |   0.5071 |        0.0792 |
| Dual-Threshold Hysteresis | Gaussian Noise |   0.4795 |    0.3896 | 0.7362 |   0.5073 |        0.0505 |

#### Original Senaryo Değişimi

| Metrik        | Baseline | Hysteresis | Değişim |
| ------------- | -------: | ---------: | ------: |
| Accuracy      |   0.4402 |     0.4802 | +0.0400 |
| Precision     |   0.3766 |     0.3905 | +0.0139 |
| Recall        |   0.8162 |     0.7379 | -0.0783 |
| F1-score      |   0.5069 |     0.5083 | +0.0014 |
| F1 Std. Sapma |   0.0779 |     0.0507 | -0.0272 |

Hysteresis yöntemi precision ve accuracy değerlerini artırırken recall değerini düşürmüştür. F1-score `0.5069` seviyesinden `0.5083` seviyesine yükselmiş, ancak artış oldukça sınırlı kalmıştır. F1 standart sapmasının azalması yöntemin fold ve seed sonuçlarında daha kararlı olabileceğini göstermektedir. Bu nedenle yöntem henüz final modele dahil edilmemiştir.



