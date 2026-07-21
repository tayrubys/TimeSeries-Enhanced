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

