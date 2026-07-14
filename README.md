# TimeSeries-Enhanced

Bu proje, zaman serilerindeki anomalileri tespit etmek için kullanılan Derin Öğrenme (Deep Learning) ve Otomata tabanlı yaklaşımların iyileştirilmesi ve optimize edilmesi amacıyla oluşturulmuştur. 


Otomata modelinin BATADAL performansını artırmak amacıyla bu branch üzerinde Likelihood-Ratio ve CUSUM tabanlı yöntemler denedim.

### Likelihood-Ratio Denemesi
Sistemi geliştirmek adına, normal ve anomalili geçişleri birbirinden ayırıp karşılaştıran bir likelihood-ratio modeli kurdum. Ancak veriyi derinlemesine incelediğimde kritik bir darboğazla karşılaştım: ADASYN ile üretilen 2312 adet sentetik anomali verisi, eğitim setinin sonuna tek bir blok halinde eklenmişti. Bu örnekler gerçek bir zaman sırası taşımadığından, geçiş tabanlı modellerde yanıltıcı sonuçlara yol açtılar. Validation sonuçlarında anomali skorları olması gerektiği gibi ayrışmadı ve model her şeyi anomali olarak işaretleme eğilimi gösterdi. Dolayısıyla bu rotadan vazgeçtim

Validation sonucunda anomali skorları beklenen şekilde ayrışmamış ve model çok fazla örneği anomali olarak tahmin etmiştir. Bu nedenle transition likelihood-ratio yaklaşımına devam edilmemiştir.

### Transition Surprise ve CUSUM

İkinci aşamada, otomata geçiş olasılıklarını şu formülle "surprise" skoruna dönüştürmeyi denedim:

```text
surprise = -log(transition_probability)
````
Buradaki amacım, düşük olasılıklı geçişlerin daha yüksek "surprise" değeri üretmesini sağlamak ve ardından CUSUM ile bu skorları biriktirerek bir anomali sinyali yakalamaktı. Sonuçlar ise şöyle oldu:

| Yöntem                      | Validation F1 | Test F1 |
| --------------------------- | ------------: | ------: |
| Raw Transition Surprise     |        0.2500 |  0.0000 |
| Transition Surprise + CUSUM |        0.2500 |  0.0000 |

Ve CUSUM, validation performansını iyileştirmediği gibi test verisinde de beklediğim başarıyı getirmedi.

Genel Sonuç

Bu branch üzerinde yapılan deneyler sonucunda likelihood-ratio ve CUSUM yöntemlerinin mevcut BATADAL otomata yapısına yeterli katkı sağlamadığı görülmüştür. Deneyler olumsuz sonuçlansa da ADASYN verisinin zaman sırası açısından incelenmesi ve geçiş tabanlı yöntemlerde sentetik örnek sırasının önemli olduğunun görülmesi açısından faydalı olmuştur.

Özetle, bu branch üzerinde gerçekleştirdiğim çalışmalar sonucunda Likelihood-Ratio ve CUSUM yöntemlerinin, mevcut BATADAL otomata yapısına istatistiksel anlamda beklediğim katkıyı sağlamadığını gözlemledim.