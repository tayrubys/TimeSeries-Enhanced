## VOMM/PST Tabanlı Otomata İyileştirmesi

Mevcut otomata tabanlı anomali tespitini geliştirmek için Değişken Dereceli Markov Modeli (VOMM) ve Probabilistic Suffix Tree (PST) denedim. Klasik otomatada sabit pencere uzunluğu kullanılırken, VOMM/PST geçmiş bağlam uzunluğunu dinamik tutuyor; yani model bazen kısa bazen uzun geçmişe bakarak karar verebiliyor.

İki dosyadan oluşuyor:
- `pst_model.py`: eğitim verisindeki dizilerden PST ağacını kurup bağlama göre geçiş olasılıklarını hesaplıyor.
- `vomm_model.py`: PST'yi kullanarak test dizisi üzerinde anomali tahmini üretiyor.

Karar basit: `probability < threshold` ise anomali, değilse normal.

### Threshold Duyarlılık Analizi

Threshold yüksek olunca model daha fazlasını anomali sayıyor, recall artıyor ama precision düşüyor. Düşük olunca model daha seçici oluyor, false positive azalıyor. Bunu görmek için BATADAL ve SKAB'da farklı threshold değerleri denedim.

**BATADAL:** en iyi F1-score **0.005** threshold'da (F1: 0.2222). Threshold düştükçe model daha seçici davranıp false positive'i azalttı.

**SKAB:** tam tersi — en iyi F1-score **0.5** threshold'da (F1: 0.6373). Yani threshold'u sabitlemek yanlış, veri setine göre ayrı ayarlanması gerekiyor.

> **Kısaca öğrendiğim şey:** VOMM/PST çalışıyor ama performansı tamamen threshold seçimine bağlı. BATADAL için düşük, SKAB için yüksek threshold daha iyi.

### Final VOMM/PST Sonuçları

Bu analiz sonucunda BATADAL için `threshold=0.005`, SKAB için `threshold=0.5` seçildi. Final sonuçlar:

| Dataset | Window Size | Alphabet Size | Threshold | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|---:|---:|
| BATADAL | 4 | 3 | 0.005 | 0.1500 | 0.4286 | 0.2222 |
| SKAB | 4 | 4 | 0.5 | 0.4197 | 0.6444 | 0.5031 |

BATADAL'da düşük threshold false positive'i azaltıp F1'i artırdı; SKAB'da yüksek threshold precision-recall dengesini daha iyi kurdu.

### Validation Tabanlı Threshold Seçimi
 
VOMM/PST, SAX/PAA ile elde edilen sembolik pattern geçişleri üzerinde çalıştığı için tahminler ham zaman noktalarıyla birebir aynı uzunlukta değil. Bu yüzden nokta bazlı etiketleri pattern seviyesine hizaladım: bir pattern'in kapsadığı aralıkta en az bir anomalili nokta varsa, o pattern anomalili sayıldı.
 
Threshold'u doğrudan test setinden seçmek performansı olduğundan iyi gösterebileceği için, bu sefer threshold'u validation setinden seçtim:
 
```text
Train set -> VOMM/PST eğitildi
Validation set -> en iyi threshold F1-score'a göre seçildi
Test set -> seçilen threshold ile final performans ölçüldü
```
 
BATADAL'da ayrı validation dosyaları (`batadal_X_val_pc1.csv`, `batadal_y_val.csv`) kullanıldı. SKAB'da ise GroupKFold yapısı korunarak her fold'un train bölümü kendi içinde train/validation olarak ayrıldı; fold test verisi threshold seçiminde hiç kullanılmadı.
 
**Validation Tabanlı Final Sonuçlar**
 
| Dataset | Evaluation Level | Threshold Selection | Precision | Recall | F1-score |
|---|---|---|---:|---:|---:|
| BATADAL | Pattern-level | Validation | 0.0793 | 0.5909 | 0.1398 |
| SKAB | Pattern-level | Validation | 0.3319 | 0.5193 | 0.4010 |
 
Bu sonuçlar threshold seçiminin performansı ne kadar etkilediğini bir kez daha gösteriyor. Test setinden değil validation setinden threshold seçildiği için değerlendirme artık daha güvenilir.
 
### Validation Tabanlı Hiperparametre Seçimi

Validation tabanlı threshold seçiminden sonra, VOMM/PST modelinde yalnızca threshold'un değil, `window_size` ve `alphabet_size` parametrelerinin de performansı ciddi şekilde etkilediği görüldü. Bu nedenle SKAB veri setinde her fold için validation seti üzerinde `window_size`, `alphabet_size` ve `threshold` birlikte seçildi.

SKAB için her fold'un train bölümü tekrar train/validation olarak ayrıldı. Model yalnızca train kısmı ile eğitildi, validation kısmında en iyi hiperparametre kombinasyonu F1-score'a göre seçildi ve final sonuçlar fold test setinde ölçüldü. Böylece test fold'u hiperparametre seçiminde kullanılmadı.

SKAB tarafında fold bazlı seçilen ayarlar şu şekilde oldu:

| Fold | Window Size | Alphabet Size | Threshold | Validation F1 |
|---|---:|---:|---:|---:|
| Fold 1 | 5 | 5 | 0.1 | 0.5608 |
| Fold 2 | 6 | 6 | 0.03 | 0.5623 |
| Fold 3 | 6 | 6 | 0.02 | 0.5696 |
| Fold 4 | 6 | 3 | 0.2 | 0.5504 |
| Fold 5 | 6 | 4 | 0.2 | 0.5640 |

Bu ayarlarla SKAB için final test sonucu aşağıdaki gibi elde edildi:

| Dataset | Selection Strategy | Precision | Recall | F1-score |
|---|---|---:|---:|---:|
| SKAB | Fold bazlı validation hiperparametre seçimi | 0.3472 | 0.6013 | 0.4341 |

### BATADAL Özel Parametre İyileştirmesi

BATADAL veri setinde validation setindeki anomalili pattern sayısı sınırlı olduğu için `window_size`, `alphabet_size` ve `threshold` parametrelerinin tamamını validation üzerinden seçmek test performansında kararsız sonuçlar üretti. Bu nedenle BATADAL için daha kontrollü bir deney yapıldı.

Önceki validation tabanlı ayarda `window_size=4`, `alphabet_size=3` kullanılmış ve test setinde F1-score `0.1398` elde edilmişti. Daha sonra parametre duyarlılık analizinde öne çıkan `window_size=6`, `alphabet_size=4` kombinasyonu BATADAL için ayrıca denendi. Threshold yine validation seti üzerinden seçildi ve `0.005` olarak belirlendi.

Bu özel ayar BATADAL performansını iyileştirdi:

| Ayar | Precision | Recall | F1-score |
|---|---:|---:|---:|
| `window=4`, `alphabet=3`, validation threshold | 0.0793 | 0.5909 | 0.1398 |
| `window=6`, `alphabet=4`, validation threshold | 0.1085 | 0.9333 | 0.1944 |

Bu sonuç BATADAL tarafında daha uzun sembolik pencere kullanımının anomalileri yakalamada daha etkili olduğunu göstermektedir. Özellikle recall değerinin `0.5909` seviyesinden `0.9333` seviyesine çıkması, modelin saldırı örüntülerini daha başarılı yakaladığını göstermektedir.

### BATADAL Min Count ve Smoothing Analizi

BATADAL tarafında `window_size=6`, `alphabet_size=4` ayarı ile elde edilen performansı daha da iyileştirmek için VOMM/PST modelinde `min_count` ve `smoothing_alpha` parametreleri taranmıştır.

`min_count`, bir bağlamın güvenilir kabul edilmesi için eğitim verisinde en az kaç kez görülmesi gerektiğini belirler. `smoothing_alpha` ise görülmeyen veya nadir geçişlere verilen olasılığı kontrol eder.

Yapılan son taramada en iyi sonuç `min_count=3`, `smoothing_alpha=1.0` ve validation üzerinden seçilen `threshold=0.005` ile elde edilmiştir.
| Ayar | Precision | Recall | F1-score |
|---|---:|---:|---:|
| `window=6`, `alphabet=4`, önceki ayar `min_count=3`, `alpha=0.1` | 0.1163 | 1.0000 | 0.2083 |
| `window=6`, `alphabet=4`, güncel ayar `min_count=3`, `alpha=1.0` | 0.1923 | 1.0000 | 0.3226 |

Bu güncelleme sonrasında ana BATADAL çalışmasında da validation seçilen threshold `0.005` olmuş ve final test sonucunda F1-score `0.3226` seviyesine yükselmiştir. Gaussian noise senaryosunda da F1-score `0.3281` olarak ölçülmüştür.

Bu sonuç, bağlam güvenilirliğini kontrol eden `min_count` parametresinin artırılmasının BATADAL üzerinde false positive davranışını kısmen azaltabildiğini ve modelin F1-score değerini iyileştirdiğini göstermektedir.

SKAB için min_count ve smoothing_alpha değerleri de validation tabanlı hiperparametre seçimine dahil edilmiştir. Bu tarama sonucunda fold bazlı farklı parametreler seçilmesine rağmen, test F1-score değerinde önceki ayara göre belirgin bir artış gözlenmemiştir. Bu nedenle bu adım, modelin regularization hassasiyetini inceleyen ek bir deney olarak değerlendirilmiştir.

Ek olarak BATADAL için daha ince threshold aralıkları (`0.00001` - `0.01`) denenmiştir. Ancak F1-score değerinde ek bir artış gözlenmemiştir. Bu sonuç, mevcut aşamada performans artışının yalnızca threshold seçimiyle sınırlı kalmadığını; sembolik temsil, bağlam seçimi ve regularization parametrelerinin daha belirleyici olduğunu göstermektedir.
### Ardışık Anomali Filtresi Denemesi

BATADAL tarafında VOMM/PST modelinin recall değerinin oldukça yüksek, precision değerinin ise düşük olduğu gözlemlenmişti. Bu durum modelin anomalileri yakalayabildiğini, ancak normal örneklerin bir kısmını da anomali olarak işaretlediğini göstermektedir. Bu nedenle false positive oranını azaltmak amacıyla VOMM/PST tahminleri üzerinde ardışık anomali filtresi denenmiştir.

Bu yöntemde tekil anomali tahminleri doğrudan kabul edilmemiş, yalnızca art arda belirli sayıda anomali tahmini geldiğinde bu bölge anomali olarak korunmuştur. Örneğin `min_consecutive=2` için tek başına kalan anomali tahminleri silinirken, en az iki ardışık anomali tahmini korunmaktadır. Amaç, gerçek siber-fiziksel anomalilerin genellikle zaman içinde devam eden olaylar olması varsayımından yararlanarak izole false positive tahminlerini azaltmaktır.

Deneyde `min_consecutive` değeri validation seti üzerinde `[1, 2, 3, 4]` aralığında denenmiştir. Ancak validation sonucunda en iyi F1-score değeri `min_consecutive=1` iken elde edilmiştir. Bu değer filtrenin uygulanmadığı temel duruma karşılık gelmektedir. Dolayısıyla ardışık anomali filtresi BATADAL performansında ek bir iyileştirme sağlamamıştır.

Bu sonuç, BATADAL tarafındaki false positive tahminlerin çoğunun tekil ve izole noktalardan oluşmadığını; daha çok ardışık bloklar halinde ortaya çıktığını göstermektedir. Bu nedenle ardışık filtre final modele dahil edilmemiştir.

### Pattern-Level Label Alignment Düzeltmesi

VOMM/PST modeli ham zaman noktaları yerine SAX/PAA ile oluşturulan sembolik pattern’ler üzerinde çalıştığı için, tahminler doğrudan nokta bazlı etiketlerle aynı uzunlukta değildir. Bu nedenle pattern-level label hizalama mantığı güncellendi.
Önceki yöntemde her pattern yalnızca tek bir `window_size` aralığıyla eşleştiriliyordu. Ancak bir pattern birden fazla SAX sembolünden oluştuğu için daha geniş bir zaman aralığını temsil etmektedir. Yeni düzenlemede pattern’in kapsadığı tüm zaman aralığı dikkate alındı ve bu aralıkta en az bir anomalili nokta varsa ilgili pattern anomalili kabul edildi.
Bu değişiklik modelin eğitim yapısını değiştirmemiş, yalnızca validation ve test aşamasındaki değerlendirmeyi daha doğru hale getirmiştir. Düzeltilmiş hizalama sonrasında BATADAL tarafında F1-score yaklaşık `0.20` seviyesinden `0.31` seviyesine yükselmiştir.

| Dataset | Scenario       | Precision | Recall | F1-score |
| ------- | -------------- | --------: | -----: | -------: |
| BATADAL | Original       |    0.1860 | 0.9600 |   0.3117 |
| BATADAL | Gaussian Noise |    0.1899 | 0.9600 |   0.3171 |

Ayrıca BATADAL min_count ve smoothing taramasında en iyi sonuç `min_count=3`, `smoothing_alpha=1.0` ve `threshold=0.005` ile elde edilmiştir:

| Precision | Recall | F1-score |
| --------: | -----: | -------: |
|    0.1923 | 1.0000 |   0.3226 |

### Dual VOMM/PST Denemesi

BATADAL tarafında precision değerinin düşük kalması nedeniyle normal ve anomalili geçişleri ayrı ayrı öğrenen Dual VOMM/PST yaklaşımı denenmiştir. Bu yöntemde normal geçişler bir PST ağacında, anomalili geçişler ise ayrı bir PST ağacında tutulmuş ve test aşamasında geçişin hangi modele daha yakın olduğuna bakılmıştır.

Eğitim sırasında `382` normal geçiş ve `415` anomalili geçiş öğrenilmiştir. Ayrıca ADASYN sonrası oluşan yapay sınıf dengesi etkisini azaltmak için validation setindeki gerçek sınıf dağılımına göre prior correction uygulanmıştır. Validation setinde `prior_normal=0.9098` ve `prior_anomaly=0.0902` olarak hesaplanmış, en iyi skor eşiği `-20` seçilmiştir.

| Model                            | Precision | Recall | F1-score |
| -------------------------------- | --------: | -----: | -------: |
| VOMM/PST + regularization        |    0.1923 | 1.0000 |   0.3226 |
| Dual VOMM/PST + prior correction |    0.1880 | 1.0000 |   0.3165 |

Dual VOMM/PST yaklaşımı çalışmış olsa da mevcut en iyi VOMM/PST + regularization sonucunu geçememiştir. Bu nedenle final BATADAL sonucu için regularization uygulanmış tek VOMM/PST modeli daha iyi seçenek olarak değerlendirilmiştir.

### Context Reliability Filter Denemesi

BATADAL tarafında false positive tahminleri azaltmak amacıyla context reliability filter denenmiştir. Bu yöntemde modelin anomali kararı verebilmesi için düşük geçiş olasılığına ek olarak kararın belirli bir minimum context derinliğine dayanması istenmiştir.

Validation seti üzerinde `min_context_depth` değerleri `[0, 1, 2, 3]` olarak denenmiştir. `min_context_depth=0`, filtrenin uygulanmadığı temel duruma karşılık gelmektedir. Deney sonucunda en iyi değer yine `0` seçilmiş, bu nedenle filtre final modele dahil edilmemiştir.

| Yöntem                    | Selected Threshold | Selected Min Context Depth | Precision | Recall | F1-score |
| ------------------------- | -----------------: | -------------------------: | --------: | -----: | -------: |
| VOMM/PST + regularization |              0.005 |                          0 |    0.1923 | 1.0000 |   0.3226 |

Bu sonuç, BATADAL tarafındaki false positive tahminlerin yalnızca düşük context derinliğinden kaynaklanmadığını göstermektedir.

### Context Count Reliability Filter Denemesi

Context reliability filter sonrasında, kararın dayandığı bağlamın eğitim verisinde yeterli sayıda görülüp görülmediğini kontrol eden `min_context_count` filtresi de denenmiştir.

Bu amaçla `min_context_count` değerleri `[0, 1, 2, 3, 5, 10, 20, 30]` aralığında validation seti üzerinde taranmıştır. Ancak en iyi sonuç yine `min_context_count=0` ile elde edilmiştir. Bu nedenle context count reliability filter da final modele dahil edilmemiş, deneysel analiz olarak bırakılmıştır.

Bu sonuç, BATADAL tarafındaki false positive probleminin yalnızca az görülen context’lerden kaynaklanmadığını; sembolik temsil, threshold seçimi ve genel karar yapısının precision üzerinde daha belirleyici olduğunu göstermektedir.

### Multi-step Score Smoothing Denemesi

BATADAL tarafında false positive tahminleri azaltmak için VOMM/PST modeline multi-step score smoothing eklenmiştir. Bu yöntemde tek bir geçişe göre karar vermek yerine, son birkaç geçişin ortalama anomali skoru dikkate alınmıştır.

`smooth_window` değerleri `[1, 2, 3, 5, 7]` aralığında validation seti üzerinde denenmiştir. Deney sonucunda en iyi değer `smooth_window=1` olarak seçilmiştir. Bu değer yumuşatma uygulanmayan temel duruma karşılık geldiği için yöntem final modele dahil edilmemiştir.

Bu sonuç, BATADAL tarafındaki false positive tahminlerin yalnızca ani ve tekil skor sıçramalarından kaynaklanmadığını göstermektedir

### Quantile-based Thresholding Denemesi

BATADAL tarafında false positive tahminleri azaltmak amacıyla quantile-based thresholding yaklaşımı denenmiştir. Bu yöntemde sabit olasılık eşiği kullanmak yerine, VOMM/PST tarafından üretilen anomaly skorlarına göre en şüpheli belirli orandaki pattern’ler anomaly olarak seçilmiştir.

Validation seti üzerinde farklı anomaly oranları ve skor yönleri denenmiştir. Ancak normal ve anomalili pattern’lerin skor ortalamalarının birbirine çok yakın olduğu görülmüş, bu nedenle yöntem sınıflar arasında yeterli ayrım sağlayamamıştır.

Deney sonucunda F1-score mevcut VOMM/PST sonucunun altında kaldığı için quantile-based thresholding final modele dahil edilmemiştir.
### Nearest-Pattern Backoff Denemesi

BATADAL tarafında VOMM/PST modelinin görülmeyen pattern’lere aynı veya çok benzer olasılıklar vermesi nedeniyle nearest-pattern backoff yaklaşımı denenmiştir. Bu yöntemde validation veya test sırasında eğitim kümesinde bulunmayan bir SAX pattern’i, Levenshtein uzaklığına göre eğitimdeki en yakın pattern ile eşleştirilmiştir. Eşleştirilen pattern daha sonra VOMM/PST olasılık hesabında kullanılmıştır.

Yöntemin amacı, görülmeyen pattern’lerin doğrudan smoothing olasılığı alması yerine eğitimde bulunan benzer pattern’lerden yararlanmasını sağlamaktır. Eşleştirmenin yalnızca belirli bir uzaklığa kadar yapılabilmesi için `max_nearest_distance` değerleri `1`, `2`, `3` ve sınırsız olarak validation seti üzerinde denenmiştir.

Validation sonuçları aşağıdaki gibi elde edilmiştir:

| Yöntem | Max Distance | Precision | Recall | F1-score | Farklı Skor Sayısı |
|---|---:|---:|---:|---:|---:|
| Mevcut VOMM/PST | Kapalı | 0.0902 | 1.0000 | 0.1655 | 1 |
| Nearest Backoff | 1 | 0.0909 | 1.0000 | 0.1667 | 21 |
| Nearest Backoff | 2 | 0.1000 | 0.7500 | 0.1765 | 32 |
| Nearest Backoff | 3 | 0.1098 | 0.7500 | 0.1915 | 35 |
| Nearest Backoff | Sınırsız | 0.1098 | 0.7500 | 0.1915 | 35 |

Validation seti üzerinde en iyi sonuç `max_nearest_distance=3` ve `threshold=0.005` ile elde edilmiştir. Bu ayar, bütün validation pattern’lerini anomalili tahmin eden temel davranışı azaltmış ve farklı anomali skorlarının oluşmasını sağlamıştır. Tahmin edilen anomalili pattern sayısı `133/133` değerinden `82/133` değerine düşmüştür.

Validation üzerinden seçilen ayar test setine uygulandığında aşağıdaki sonuç elde edilmiştir:

| Yöntem | Precision | Recall | F1-score |
|---|---:|---:|---:|
| VOMM/PST + Nearest-Pattern Backoff | 0.1829 | 0.6000 | 0.2804 |
| Mevcut VOMM/PST + Regularization | 0.1923 | 1.0000 | 0.3226 |

Nearest-pattern backoff yaklaşımı görülmeyen pattern’lere verilen skorların çeşitlenmesini sağlamış olsa da gerçek anomalili pattern’lerin bilinen pattern’lere eşlenmesi recall değerinin `1.0000` seviyesinden `0.6000` seviyesine düşmesine neden olmuştur. Final F1-score mevcut VOMM/PST + regularization sonucunun altında kaldığı için yöntem final modele dahil edilmemiştir.

### Interpolated VOMM/PST Denemesi

BATADAL tarafında seyrek context’lerin oluşturduğu false positive tahminleri azaltmak amacıyla Interpolated VOMM/PST yaklaşımı denenmiştir. Bu yöntemde yalnızca en uzun context’i kullanmak yerine, kısa ve uzun context’lerden elde edilen geçiş olasılıkları ağırlıklı olarak birleştirilmiştir.

Validation setinde `beta=[0.5, 1.0, 2.0, 5.0, 10.0]` değerleri denenmiştir. Ancak bütün beta değerlerinde aynı sonuç elde edilmiş ve validation F1-score `0.1655` seviyesinde kalmıştır. Model tüm validation pattern’lerini anomalili tahmin ettiği için interpolation ek bir ayrım sağlayamamıştır.

Bu nedenle yöntem final modele dahil edilmemiş, mevcut VOMM/PST + regularization sonucu korunmuştur.

| Yöntem | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Interpolated VOMM/PST – Validation | 0.0902 | 1.0000 | 0.1655 |
| Mevcut VOMM/PST – Test | 0.1923 | 1.0000 | 0.3226 |

### State Aggregation Denemesi

SAX/PAA sonucunda oluşan benzer pattern’leri ortak durumlarda birleştirerek durum uzayındaki seyrekliği azaltmak amacıyla State Aggregation yaklaşımı denenmiştir.

`merge_distance=[0, 1, 2]` değerleri validation setinde karşılaştırılmıştır. `merge_distance=2` ile durum sayısı `158` değerinden `21` değerine düşmüş ve farklı anomali skoru sayısı artmıştır. Ancak test F1-score `0.3117` olarak elde edilmiş ve mevcut VOMM/PST sonucu olan `0.3226` değerinin altında kalmıştır.

Bu nedenle State Aggregation yaklaşımı final modele dahil edilmemiştir.

### Uzun Window Size ve Context Derinliği Analizi

VOMM-PST modelinde daha uzun geçmiş kullanımının performansa etkisini incelemek amacıyla BATADAL veri setinde daha büyük window_size ve max_depth değerleri denenmiştir.
Ancak mevcut SAX/PAA yapısında aynı `window_size` hem PAA segment boyutunu hem de pattern uzunluğunu belirlediği için, window büyüdükçe pattern’lerin kapsadığı zaman aralığı da genişlemiştir.
Bir pattern’in içinde tek bir anomalili nokta olması tüm pattern’in anomalili kabul edilmesine neden olduğundan, büyük window değerlerinde normal noktalar da anomalili pattern’lerin içine girmiş ve F1 değeri yapay olarak yükselmiştir.

| Window Size | Validation Anomaly | Validation Normal | Validation F1 |
|---:|---:|---:|---:|
| 12 | 15 | 42 | 0.4167 |
| 16 | 18 | 18 | 0.6667 |
| 20 | 19 | 2 | 0.9500 |

`window_size=20` için elde edilen yüksek sonuç bu nedenle gerçek bir iyileşme olarak kabul edilmemiştir.

Daha adil bir karşılaştırma için `window_size=6` sabit tutulmuş ve yalnızca `max_depth` değiştirilmiştir.

| Max Depth | PST Durum Sayısı | Validation F1 |
|---:|---:|---:|
| 6 | 2.111 | 0.1655 |
| 12 | 6.012 | 0.1655 |
| 20 | 11.966 | 0.1655 |

`max_depth` arttıkça model karmaşıklığı yükselmiş ancak performans değişmemiştir. Bu nedenle final modelde `window_size=6` ve `max_depth=6` korunmuştur.

### SKAB Window Size Analizi

SKAB üzerinde daha uzun pattern uzunluklarının etkisini incelemek amacıyla `window_size=[3,4,5,6,7,8,9,10,12]` değerleri validation seti üzerinde taranmıştır. Mevcut model yapısında VOMM/PST `max_depth` değeri de `window_size` ile aynı tutulmuştur.

Beş fold’un tamamında en iyi değer `window_size=12` olarak seçilmiştir. Fold bazlı threshold değerleri kullanılarak test setinde aşağıdaki ortalama sonuç elde edilmiştir:

| Window Size | Precision | Recall | F1-score |
|---:|---:|---:|---:|
| 12 | 0.4938 | 0.7952 | 0.6051 |

Bu nedenle SKAB final modelinde `window_size=12` kullanılmıştır. Window büyüdükçe pattern seviyesindeki sınıf dağılımının değiştiği de sonuçlar değerlendirilirken dikkate alınmıştır.

