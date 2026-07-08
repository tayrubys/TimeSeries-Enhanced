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

Yapılan taramada en iyi sonuç `min_count=3`, `smoothing_alpha=0.1` ve validation üzerinden seçilen `threshold=0.001` ile elde edilmiştir.

| Ayar | Precision | Recall | F1-score |
|---|---:|---:|---:|
| `window=6`, `alphabet=4`, varsayılan `min_count=2`, `alpha=1.0` | 0.1085 | 0.9333 | 0.1944 |
| `window=6`, `alphabet=4`, `min_count=3`, `alpha=0.1` | 0.1163 | 1.0000 | 0.2083 |

Bu sonuç, bağlam güvenilirliğini kontrol eden `min_count` parametresinin artırılmasının BATADAL üzerinde false positive davranışını kısmen azaltabildiğini ve modelin F1-score değerini iyileştirdiğini göstermektedir.


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
