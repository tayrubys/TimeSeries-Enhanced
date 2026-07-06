from src.models.pst_model import ProbabilisticSuffixTree

class VariableOrderMarkovModel:
    def __init__(self, max_depth=5, alphabet_size=3, smoothing=True):
        self.max_depth = max_depth
        self.alphabet_size = alphabet_size
        self.smoothing = smoothing
        
        # Gerçek zekayı barındıran ağacımız
        self.pst = ProbabilisticSuffixTree(max_depth=self.max_depth, alphabet_size=self.alphabet_size)
        
        # Runner.py çökmesin diye uyumluluk değişkenleri
        self._trained_patterns = set()

    def fit(self, train_patterns):
        if len(train_patterns) < 2:
            raise ValueError("VOMM eğitimi için en az 2 pattern gereklidir.")
            
        self._trained_patterns = set(train_patterns)
        
        # Veriyi PST'nin fit fonksiyonuna besliyoruz
        self.pst.fit(train_patterns)

    def predict(self, test_patterns, anomaly_threshold=0.05):
        predictions = []
        explainability_logs = []  # Şimdilik boş bırakıyoruz, pipeline kırılmasın
        
        # Test dizisi üzerinde kayan pencere ile tahminleme yapacağız
        for i in range(len(test_patterns)):
            # Geçmiş bağlamı (context) çıkar
            start_idx = max(0, i - self.max_depth)
            context = test_patterns[start_idx:i]
            target_symbol = test_patterns[i]
            
            # Ağaçtan bu bağlama göre olasılık iste
            prob = self.pst.predict_probability(context, target_symbol)
            
            # Anomali kararı
            if prob < anomaly_threshold:
                predictions.append(1) # Anomali
            else:
                predictions.append(0) # Normal
                
        return predictions, explainability_logs

    # Runner.py'ın "num_states" hesabı için sahte özellik (property)
    @property
    def trained_patterns(self):
        return self._trained_patterns

    # Runner.py'ın "num_transitions" hesabı için sahte özellik (property)
    @property
    def transitions(self):
        # Geçici bir çözüm: Ağacın boyutunu simüle eden boş bir sözlük yapısı döndürebiliriz
        # veya ileride PST'nin düğüm sayısını buraya bağlayabiliriz.
        return {pattern: {pattern: 1} for pattern in self._trained_patterns}