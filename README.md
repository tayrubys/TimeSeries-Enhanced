## BATADAL — LSTM Modeli, ADASYN ile Dengeleme (Ön Sonuçlar)
 
Bu bölümde, BATADAL train setinin ADASYN ile (pencere seviyesinde) dengelenmesi sonrası LSTM modelinin performansı raporlanmaktadır. Eğitim, 5 farklı random seed (`42, 123, 2026, 7, 999`) ile tekrarlanmış, her seed için validation setinde F1-optimal eşik (threshold) bulunup bu eşik test setinde uygulanmıştır.
 
### Seed Bazlı Sonuçlar (Test Kümesi)
 
| Seed | Threshold | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|---|
| 42 | 0.2 | 0.8984 | 0.0000 | 0.0000 | 0.0000 |
| 123 | 0.1 | 0.9523 | 0.7158 | 0.8500 | 0.7771 |
| 2026 | 0.4 | 0.9755 | 0.8750 | 0.8750 | 0.8750 |
| 7 | 0.1 | 0.9535 | 0.6981 | 0.9250 | 0.7957 |
| 999 | 0.3 | 0.9694 | 0.8986 | 0.7750 | 0.8322 |
 
### Ortalama ± Standart Sapma
 
**Tüm seed'ler (5 seed):**
 
| Metrik | Ortalama ± Std |
|---|---|
| Accuracy | 0.9498 ± 0.0272 |
| Precision | 0.6375 ± 0.3289 |
| Recall | 0.6850 ± 0.3459 |
| F1-score | 0.6560 ± 0.3297 |
 
**Seed=42 hariç (4 seed):**
 
| Metrik | Ortalama ± Std |
|---|---|
| Accuracy | 0.9627 ± 0.0100 |
| Precision | 0.7969 ± 0.0905 |
| Recall | 0.8563 ± 0.0541 |
| F1-score | 0.8200 ± 0.0374 |
 
>> 5 seed'in 4'ünde (123, 2026, 7, 999) LSTM iyi sonuç verdi (F1: 0.78–0.88). Ama seed=42'de model test setinde hiçbir anomaliyi bulamadı (precision, recall, F1 hepsi 0). Validation'da bu seed için sonuç fena değildi (F1≈0.545), ama test'e geçince tamamen çöktü.

![Lstm confusion matrix](results\figures\batadal_lstm_confusion_matrix.png)
![Lstm precission-recall curve](results\figures\batadal_lstm_pr_curve.png)
>> cizimlerde seed olarak 3 alınmıştır. 
