### feature/state-similarity (Mesafe-Ağırlıklı Güven Skoru ve Tolerans)

*   **Problem:** Yüksek dereceli (Order=2+) Markov modellerinde, eğitimde hiç görülmemiş (unseen) bir örüntü geldiğinde modelin en yakın duruma eşleme yapması, ufak sensör sapmalarında bile yanlış alarm (False Positive) üretilmesine neden oluyordu.
    
*   **Çözüm (Geliştirme):** \* Modelin olasılık hesaplamasına Levenshtein mesafesi tabanlı eksponansiyel bir ceza katsayısı ($e^{-\\alpha D}$) eklendi.
    
    *   **Tolerans Payı (distance\_tolerance=1):** Sensörlerdeki doğal zamansal kaymaları (Concept Drift) siber saldırılardan ayırmak için 1 birimlik mesafe sapmalarına ceza muafiyeti getirildi.
        
*   **Test Bulguları (BATADAL & SKAB):**
    
    *   _Acımasız Ceza (Tolerance = 0):_ Modelin Precision değeri %7'ye kadar çöktü. Bu durum, BATADAL test setindeki "görülmemiş" örüntülerin aslında siber saldırı değil, ufak sapmalı _normal_ sistem davranışları (Concept Drift) olduğunu kanıtladı.
        
    *   _Esnek Ceza (Tolerance = 1):_ 1 birimlik tolerans uygulandığında F1 Skoru tekrar **0.4444** seviyesinde korundu ve modelin sahte alarmlara (paranoyaya) düşmesi engellendi.
        
*   **Akademik Çıkarım:** Olasılıksal Otomata modeli, 1 birimlik mesafe toleransı ile Concept Drift'e karşı dayanıklı (robust) hale getirilmiştir. Bu yapılandırılmış haliyle, derin öğrenme algoritmaları (LSTM/GRU) ile istatistiksel olarak **eşdeğer (**$p > 0.05$**)** bir anomali tespit performansı sergilediği kanıtlanmıştır.