import numpy as np
from collections import defaultdict
from src.models.explainability import AutomataExplainer

class ProbabilisticAutomata:

    #yüksek dereceli olasılıksal otomata modelini başlatır
    def __init__(self, smoothing=True, order=2, learning_rate=0.0):
        self.smoothing = smoothing
        self.order = order
        self.learning_rate = learning_rate
        self.transitions = defaultdict(lambda: defaultdict(float))
        self.total_exits = defaultdict(float)
        self.trained_patterns = set()

    #modeli verilen eğitim örüntüleriyle eğitir ve geçiş frekanslarını kaydeder
    def fit(self, train_patterns):
        if len(train_patterns) < self.order + 1:
            raise ValueError(f"Otomata eğitimi için en az {self.order + 1} pattern gereklidir.")
            
        self.trained_patterns = set(train_patterns)
        
        for ord_idx in range(1, self.order + 1):
            for i in range(len(train_patterns) - ord_idx):
                state = tuple(train_patterns[i : i + ord_idx])
                next_pattern = train_patterns[i + ord_idx]
                self.transitions[state][next_pattern] += 1
                self.total_exits[state] += 1

    #katz back-off algoritması ile bir sonraki durumun geçiş olasılığını hesaplar
    def get_transition_probability(self, current_state, next_pattern):
        state_to_check = current_state
        
        while len(state_to_check) > 0:
            total_output = self.total_exits[state_to_check]
            if total_output > 0:
                transition_count = self.transitions[state_to_check][next_pattern]
                if self.smoothing:
                    return (transition_count + 1) / (total_output + len(self.trained_patterns))
                return transition_count / total_output
            
            state_to_check = state_to_check[1:]
            
        if self.smoothing and len(self.trained_patterns) > 0:
            return 1.0 / len(self.trained_patterns)
        return 0.0

    #karar normal çıktığında geçiş ağırlıklarını güncelleyerek adaptif öğrenmeyi sağlar
    def _update_transition(self, current_state, next_state):
        if self.learning_rate <= 0.0:
            return
            
        state_to_update = current_state
        while len(state_to_update) > 0:
            base_weight = max(1.0, self.total_exits[state_to_update])
            self.transitions[state_to_update][next_state] += self.learning_rate * base_weight
            self.total_exits[state_to_update] += self.learning_rate * base_weight
            state_to_update = state_to_update[1:]
            
        if next_state not in self.trained_patterns:
            self.trained_patterns.add(next_state)

    #iki dizi arasındaki Levenshtein mesafesini hesaplar
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

    #eğitilmiş örüntüler arasında verilen görülmemiş örüntüye en yakın olanı bulur
    def _find_nearest_pattern(self, unseen_pattern):
        best_distance = float('inf')
        nearest_pattern = None
        for trained_pattern in sorted(self.trained_patterns):
            dist = self._calculate_levenshtein(unseen_pattern, trained_pattern)
            if dist < best_distance:
                best_distance = dist
                nearest_pattern = trained_pattern
        return nearest_pattern, best_distance

    #test verisi üzerinde kayan pencere ile anomali tahmini yapar ve sonuçları döndürür
    def predict(self, test_patterns, anomaly_threshold=0.05):
        if len(test_patterns) < self.order:
            return [0] * (len(test_patterns) - 1), []

        predictions = []
        explainability_logs = []
        
        mapped_initial = []
        for i in range(self.order):
            pat = test_patterns[i]
            if pat not in self.trained_patterns:
                pat_mapped, _ = self._find_nearest_pattern(pat)
                mapped_initial.append(pat_mapped)
            else:
                mapped_initial.append(pat)
                
        current_state = tuple(mapped_initial)
        transition_history = list(current_state)
        cumulative_path_prob = 1.0

        for t in range(self.order, len(test_patterns)):
            incoming_pattern = test_patterns[t]
            status = "seen"
            mapped_to = incoming_pattern
            distance = 0
            
            similarity_report = []
            if incoming_pattern not in self.trained_patterns:
                status = "unseen"
                mapped_to, distance = self._find_nearest_pattern(incoming_pattern)
                distances = [(tp, self._calculate_levenshtein(incoming_pattern, tp)) for tp in self.trained_patterns]
                distances.sort(key=lambda x: x[1])
                similarity_report = [{"pattern": p, "distance": d} for p, d in distances[:3]]

            prob = self.get_transition_probability(current_state, mapped_to)
            cumulative_path_prob *= prob
            path_probability = float(cumulative_path_prob) 
            decision = "anomaly" if prob < anomaly_threshold else "normal"
            confidence_score = float(prob)
            
            if decision == "normal" and self.learning_rate > 0.0:
                self._update_transition(current_state, mapped_to)
            
            counterfactuals = []
            total_exits_for_state = self.total_exits[current_state]
            if total_exits_for_state > 0 or self.smoothing:
                possible_transitions = [
                    (p_next, self.get_transition_probability(current_state, p_next)) 
                    for p_next in self.trained_patterns
                ]
                possible_transitions.sort(key=lambda x: x[1], reverse=True)
                
                for alt_pattern, alt_prob in possible_transitions[:3]:
                    if alt_pattern != mapped_to: 
                        alt_decision = "normal" if alt_prob >= anomaly_threshold else "anomaly"
                        counterfactuals.append({
                            "pattern": alt_pattern,
                            "probability": float(alt_prob),
                            "would_be_anomaly": alt_decision == "anomaly"
                        })

            transition_history.append(mapped_to)

            state_str = "->".join(current_state)

            log_entry = AutomataExplainer.generate_log(
                t, state_str, incoming_pattern, status, mapped_to, distance,
                prob, path_probability, decision, confidence_score,
                transition_history.copy(), total_exits_for_state,
                counterfactuals, similarity_report
            )
            explainability_logs.append(log_entry)
            predictions.append(1 if decision == "anomaly" else 0)
            
            current_state = current_state[1:] + (mapped_to,)

        padding_count = self.order - 1
        predictions = [0] * padding_count + predictions

        return predictions, explainability_logs