import numpy as np
from collections import defaultdict
from src.models.explainability import AutomataExplainer
import pandas as pd

class ProbabilisticAutomata:
    def __init__(self, smoothing=True):
        self.smoothing = smoothing
        self.transitions = defaultdict(lambda: defaultdict(int))
        self.total_exits = defaultdict(int)
        self.trained_patterns = set()

    def fit(self, train_patterns):
        if len(train_patterns) < 2:
            raise ValueError("Otomata eğitimi için en az 2 pattern gereklidir.")
        self.trained_patterns = set(train_patterns)
        for i in range(len(train_patterns) - 1):
            current_state = train_patterns[i]
            next_state = train_patterns[i+1]
            self.transitions[current_state][next_state] += 1
            self.total_exits[current_state] += 1

    def get_transition_probability(self, current_state, next_state):
        total_output = self.total_exits[current_state]
        if total_output == 0:
            return 1.0 / (len(self.trained_patterns) if self.smoothing else 1.0)
        transition_count = self.transitions[current_state][next_state]
        if self.smoothing:
            return (transition_count + 1) / (total_output + len(self.trained_patterns))
        return transition_count / total_output

    def _calculate_levenshtein(self, s1, s2):
        if len(s1) < len(s2): return self._calculate_levenshtein(s2, s1)
        if len(s2) == 0: return len(s1)
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]

    def _find_nearest_pattern(self, unseen_pattern):
        best_distance = float('inf')
        nearest_pattern = None
        for trained_pattern in sorted(self.trained_patterns):
            dist = self._calculate_levenshtein(unseen_pattern, trained_pattern)
            if dist < best_distance:
                best_distance = dist
                nearest_pattern = trained_pattern
        return nearest_pattern, best_distance

    #dinamik eşikleme ile tespit
    def predict(self, X, anomaly_threshold=None, **kwargs):
        """
        Dinamik Eşikleme (Dynamic Thresholding) destekli anomali tespiti.
        Gelen X nesnesi zaten sembolize edilmiş (harf dizisi) durumdadır.
        Son adımlardaki geçiş olasılıklarının hareketli standart sapmasına göre anlık eşik belirler.
        """
        # Eğer dışarıdan bir threshold beslenmediyse sınıftaki varsayılanı kullan
        current_base_threshold = anomaly_threshold if anomaly_threshold is not None else 0.05
        
        test_sequence = X
        predictions = []
        explainability_logs = [] # Orijinal runner.py'ın beklediği log yapısı için boş liste tutuyoruz
        
        # Dinamik eşikleme için son olasılıkları tutacağımız hareketli bir hafıza havuzu
        recent_probabilities = []
        moving_window_limit = 10  # Son 10 adımın oynaklığına bakılacak

        # Sembolik dizide kayarak ilerle
        for i in range(len(test_sequence) - 1):
            current_state = test_sequence[i]
            next_state = test_sequence[i+1]

            # Eğitimde bu geçiş var mı kontrol et, varsa olasılığını al
            prob = self.get_transition_probability(current_state, next_state)

            # Olasılık havuzunu güncelle (son 10 olasılığı sakla)
            recent_probabilities.append(prob)
            if len(recent_probabilities) > moving_window_limit:
                recent_probabilities.pop(0)

            # Dinamik Eşik Hesaplama:
            if len(recent_probabilities) >= 3:
                local_std = np.std(recent_probabilities)
                # Oynaklık yüksekse eşiği esneterek yanlış alarmları (False Positive) önler
                dynamic_threshold = max(0.005, current_base_threshold - (0.5 * local_std))
            else:
                dynamic_threshold = current_base_threshold

            # Karar Mekanizması
            if prob < dynamic_threshold:
                predictions.append(1) # Anomali
            else:
                predictions.append(0) # Normal

        # Veri boyutunu girdi boyutuyla eşitlemek için listenin sonuna 1 eleman ekle
        while len(predictions) < len(X):
            predictions.append(0)

        # Orijinal runner.py hem tahminleri hem de açıklanabilirlik loglarını bekliyor
        return np.array(predictions)[:len(X)], explainability_logs