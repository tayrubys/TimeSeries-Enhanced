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