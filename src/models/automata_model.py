import numpy as np
from collections import defaultdict
from src.models.explainability import AutomataExplainer

class ProbabilisticAutomata:
    def __init__(self, smoothing=True,weight_sharpness=1.0,use_similarity_penalty=False,similarity_penalty_strength=0.0,use_dynamic_threshold=False,dynamic_threshold_quantile=0.05,min_dynamic_threshold_samples=3):
        self.smoothing = smoothing
        self.weight_sharpness = weight_sharpness
        self.use_similarity_penalty = use_similarity_penalty
        self.similarity_penalty_strength = similarity_penalty_strength#cezanın ne kadar güçlü uygulanacağını belirler
        self.transitions = defaultdict(lambda: defaultdict(int))
        self.total_exits = defaultdict(int)
        self.trained_patterns = set()
        self.weighted_probabilities = defaultdict(lambda: defaultdict(float))
        self.use_dynamic_threshold = use_dynamic_threshold
        self.dynamic_threshold_quantile = dynamic_threshold_quantile
        self.min_dynamic_threshold_samples = min_dynamic_threshold_samples
        self.state_dynamic_thresholds = {}
        self.global_dynamic_threshold = None
    # Weighted probability değerleri hesaplandıktan sonra her state için ayrı threshold değerleri hazırlama
    def fit(self, train_patterns):
        if len(train_patterns) < 2:
            raise ValueError("Otomata eğitimi için en az 2 pattern gereklidir.")
        #model yeniden eğitilirse eski geçiş ve threshold bilgileri temizle
        self.transitions.clear()
        self.total_exits.clear()
        self.weighted_probabilities.clear()
        self.state_dynamic_thresholds.clear()
        self.global_dynamic_threshold = None
        self.trained_patterns = set(train_patterns)

        for i in range(len(train_patterns) - 1):
            current_state = train_patterns[i]
            next_state = train_patterns[i + 1]
            self.transitions[current_state][next_state] += 1
            self.total_exits[current_state] += 1

        for current_state, exits in self.transitions.items():
            total_output = self.total_exits[current_state]
            raw_scores = {}
            total_raw_score = 0.0
            for next_state, count in exits.items():
                # Weighted Transition Probability: sik gorulen gecislerin etkisini artirir
                base_prob = count / total_output
                weight = np.log1p(self.weight_sharpness * base_prob)
                raw_score = base_prob * weight

                raw_scores[next_state] = raw_score
                total_raw_score += raw_score

            if total_raw_score > 0:
               for next_state, raw_score in raw_scores.items():
                   #normalize yaptık
                   self.weighted_probabilities[current_state][next_state] = raw_score / total_raw_score
        self._calculate_dynamic_thresholds(train_patterns)
#eğitim verisindeki geçiş olasılıklarına bakarak state bazlı threshold hesaplar sabit threshold'dan daha iyi olmadığı için kapatıldı
    def _calculate_dynamic_thresholds(self, train_patterns):
        state_probs = defaultdict(list)
        all_probs = []
        quantile = float(np.clip(self.dynamic_threshold_quantile, 0.0, 1.0))

        for i in range(len(train_patterns) - 1):
            current_state = train_patterns[i]
            next_state = train_patterns[i + 1]
            prob = self.get_transition_probability(current_state, next_state)
            state_probs[current_state].append(prob)
            all_probs.append(prob)

        if all_probs:
            self.global_dynamic_threshold = float(np.quantile(all_probs, quantile))

        for state, probs in state_probs.items():
            if len(probs) >= self.min_dynamic_threshold_samples:
                self.state_dynamic_thresholds[state] = float(np.quantile(probs, quantile)) 

    #tahmin sırasında hangi threshold kullanılacağını secer ve dynamic threshold kapalıysa normal anomaly_threshold kullanılır            
    def _get_decision_threshold(self, current_state, fallback_threshold):
        if not self.use_dynamic_threshold:
            return fallback_threshold, "fixed"

        if current_state in self.state_dynamic_thresholds:
            return self.state_dynamic_thresholds[current_state], "state_dynamic"

        if self.global_dynamic_threshold is not None:
            return self.global_dynamic_threshold, "global_dynamic"

        return fallback_threshold, "fixed_fallback"
                           
    def get_transition_probability(self, current_state, next_state):
        total_output = self.total_exits[current_state]

        if total_output == 0:
            return 1.0 / (len(self.trained_patterns) if self.smoothing else 1.0)

        if next_state in self.weighted_probabilities[current_state]:
            return self.weighted_probabilities[current_state][next_state]

        if self.smoothing:
            #görülmemiş geçiş varsa yine smoothing
            return 1.0 / (total_output + len(self.trained_patterns))

        return 0.0

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
    
    def _calculate_similarity_score(self, distance):
        return 1.0 / (1.0 + self.similarity_penalty_strength * distance)
    
    def _find_nearest_pattern(self, unseen_pattern):
        best_distance = float('inf')
        nearest_pattern = None
        for trained_pattern in sorted(self.trained_patterns):
            dist = self._calculate_levenshtein(unseen_pattern, trained_pattern)
            if dist < best_distance:
                best_distance = dist
                nearest_pattern = trained_pattern
        return nearest_pattern, best_distance

    def predict(self, test_patterns, anomaly_threshold=0.05):
        predictions = []
        explainability_logs = []
        current_state = test_patterns[0]

        # Test dizisinin ilk pattern'ı eğitimde yoksa, otomatanın başlayabilmesi için
        # bu pattern en yakın bilinen eğitim pattern'ına dönüştürülür.
        if current_state not in self.trained_patterns:
            current_state, _ = self._find_nearest_pattern(current_state)

        transition_history = [current_state]
        cumulative_path_prob = 1.0

        for t in range(1, len(test_patterns)):
            incoming_pattern = test_patterns[t]
            status = "seen"
            mapped_to = incoming_pattern
            distance = 0
            
            similarity_report = []

            # Gelen pattern eğitimde görülmediyse "unseen" olarak işaretlenir.
            # Model bu pattern'ı en yakın bilinen pattern'a eşleyerek tahmine devam eder.
            # Böylece model, hiç görmediği bir pattern yüzünden çökmez veya tahmini durdurmaz.
            if incoming_pattern not in self.trained_patterns:
                status = "unseen"
                mapped_to, distance = self._find_nearest_pattern(incoming_pattern)

                # Açıklanabilirlik için sadece seçilen en yakın pattern değil,
                # en yakın ilk 3 aday ve mesafeleri de log'a eklenir.
                distances = [(tp, self._calculate_levenshtein(incoming_pattern, tp)) for tp in self.trained_patterns]
                distances.sort(key=lambda x: x[1])
                similarity_report = [{"pattern": p, "distance": d} for p, d in distances[:3]]

            # Olasılık hesabında incoming_pattern yerine mapped_to kullanılır.
            # Similarity penalty ana deneyde kapalıdır sadece ayrıca denenmek istenirse açılır.
            base_prob = self.get_transition_probability(current_state, mapped_to)
            similarity_score = 1.0
            if self.use_similarity_penalty:
                similarity_score = self._calculate_similarity_score(distance)
            prob = base_prob * similarity_score
            cumulative_path_prob *= prob
            path_probability = float(cumulative_path_prob) 
            #burada sabit threshold yerine, eğer açıksa dynamic threshold kullanılıyor.
            #final deneyde use_dynamic_threshold=False olduğu için sabit threshold ile devam eder
            decision_threshold, threshold_source = self._get_decision_threshold(current_state, anomaly_threshold)
            decision = "anomaly" if prob < decision_threshold else "normal"
            confidence_score = float(prob)
            
            counterfactuals = []
            if self.total_exits[current_state] > 0 or self.smoothing:
                possible_transitions = [(p_next, self.get_transition_probability(current_state, p_next)) for p_next in self.trained_patterns]
                possible_transitions.sort(key=lambda x: x[1], reverse=True)
                
                for alt_pattern, alt_prob in possible_transitions[:3]:
                    if alt_pattern != mapped_to: 
                        alt_decision = "normal" if alt_prob >= decision_threshold else "anomaly"
                        counterfactuals.append({
                            "pattern": alt_pattern,
                            "probability": float(alt_prob),
                            "would_be_anomaly": alt_decision == "anomaly"
                        })

            transition_history.append(mapped_to)

            log_entry = AutomataExplainer.generate_log(
                t, current_state, incoming_pattern, status, mapped_to, distance,
                prob, path_probability, decision, confidence_score,
                transition_history, self.total_exits[current_state],
                counterfactuals, similarity_report
            )
            log_entry["base_transition_probability"] = float(base_prob)
            log_entry["similarity_score"] = float(similarity_score)
            log_entry["penalized_transition_probability"] = float(prob)
            log_entry["decision_threshold"] = float(decision_threshold)
            log_entry["threshold_source"] = threshold_source
            log_entry["use_dynamic_threshold"] = self.use_dynamic_threshold
            explainability_logs.append(log_entry)
            predictions.append(1 if decision == "anomaly" else 0)
            current_state = mapped_to

        return predictions, explainability_logs