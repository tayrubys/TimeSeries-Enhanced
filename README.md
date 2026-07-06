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
 
### Sonraki İyileştirme
 
window_size + alphabet_size + threshold hepsini validation üzerinden seçmek çünkü parametre taramasında bazı `window_size` ve `alphabet_size` kombinasyonlarının daha yüksek F1-score ürettiği görüldü. 
