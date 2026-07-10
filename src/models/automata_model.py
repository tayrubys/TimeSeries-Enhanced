import numpy as np
from collections import defaultdict
from src.models.explainability import AutomataExplainer


class ProbabilisticAutomata:

    # yüksek dereceli olasılıksal otomata modelini başlatır
    def __init__(
        self,
        smoothing=True,
        order=2,
        learning_rate=0.0,
        smoothing_alpha=1.0,
        time_decay_enabled=False,
        time_decay_rate=0.99,
        time_decay_min_weight=0.01,
    ):
        self.smoothing = smoothing
        self.order = order
        self.learning_rate = learning_rate
        self.smoothing_alpha = smoothing_alpha

        # Time-decay parametreleri varsayılan olarak kapalıdır.
        # Kapalıyken transition sayımları markov-base ile aynı kalır.
        self.time_decay_enabled = time_decay_enabled
        self.time_decay_rate = time_decay_rate
        self.time_decay_min_weight = time_decay_min_weight

        self._validate_time_decay_params()

        self.transitions = defaultdict(lambda: defaultdict(float))
        self.total_exits = defaultdict(float)
        self.trained_patterns = set()

    def _validate_time_decay_params(self):
        if self.order < 1:
            raise ValueError("order en az 1 olmalıdır.")

        if self.smoothing_alpha < 0:
            raise ValueError("smoothing_alpha negatif olamaz.")

        if self.time_decay_enabled:
            if not (0.0 < self.time_decay_rate <= 1.0):
                raise ValueError("time_decay_rate 0 ile 1 arasında olmalıdır: 0 < rate <= 1")

            if not (0.0 <= self.time_decay_min_weight <= 1.0):
                raise ValueError("time_decay_min_weight 0 ile 1 arasında olmalıdır.")

    def _get_time_decay_weight(self, age):
        """
        Transition ağırlığını hesaplar.

        age=0 en yeni transition anlamına gelir ve ağırlık 1.0 olur.
        age büyüdükçe ağırlık time_decay_rate ** age şeklinde azalır.
        time_decay_min_weight, çok eski transition'ların tamamen etkisizleşmesini engeller.
        """
        if not self.time_decay_enabled:
            return 1.0

        weight = self.time_decay_rate ** age
        return float(max(self.time_decay_min_weight, weight))

    # modeli verilen eğitim örüntüleriyle eğitir ve geçiş frekanslarını kaydeder
    def fit(self, train_patterns):
        if len(train_patterns) < self.order + 1:
            raise ValueError(f"Otomata eğitimi için en az {self.order + 1} pattern gereklidir.")

        self.trained_patterns = set(train_patterns)

        for ord_idx in range(1, self.order + 1):
            num_transitions_for_order = len(train_patterns) - ord_idx

            for i in range(num_transitions_for_order):
                state = tuple(train_patterns[i : i + ord_idx])
                next_pattern = train_patterns[i + ord_idx]

                # Daha yeni transition'lara daha yüksek ağırlık verilir.
                # En sondaki transition age=0, en eski transition age büyük olur.
                age = (num_transitions_for_order - 1) - i
                transition_weight = self._get_time_decay_weight(age)

                self.transitions[state][next_pattern] += transition_weight
                self.total_exits[state] += transition_weight

    # katz back-off algoritması ile bir sonraki durumun geçiş olasılığını hesaplar
    def get_transition_probability(self, current_state, next_pattern):
        state_to_check = current_state

        while len(state_to_check) > 0:
            total_output = self.total_exits[state_to_check]
            if total_output > 0:
                transition_count = self.transitions[state_to_check][next_pattern]
                if self.smoothing:
                    alpha = self.smoothing_alpha
                    vocab_size = len(self.trained_patterns)
                    return (transition_count + alpha) / (total_output + alpha * vocab_size)
                return transition_count / total_output

            state_to_check = state_to_check[1:]

        if self.smoothing and len(self.trained_patterns) > 0:
            return 1.0 / len(self.trained_patterns)
        return 0.0

    # karar normal çıktığında geçiş ağırlıklarını güncelleyerek adaptif öğrenmeyi sağlar
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

    # iki dizi arasındaki Levenshtein mesafesini hesaplar
    def _calculate_levenshtein(self, s1, s2):
        if len(s1) < len(s2):
            return self._calculate_levenshtein(s2, s1)
        if len(s2) == 0:
            return len(s1)
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

    # eğitilmiş örüntüler arasında verilen görülmemiş örüntüye en yakın olanı bulur
    def _find_nearest_pattern(self, unseen_pattern):
        best_distance = float('inf')
        nearest_pattern = None
        for trained_pattern in sorted(self.trained_patterns):
            dist = self._calculate_levenshtein(unseen_pattern, trained_pattern)
            if dist < best_distance:
                best_distance = dist
                nearest_pattern = trained_pattern
        return nearest_pattern, best_distance

    # verilen pattern dizisi için karar vermeden olasılık/anomali skorlarını hesaplar
    def calculate_scores(
        self,
        patterns,
        decision_mode="avg_negative_log",
        score_window=1,
        max_mapping_distance=None,
    ):
        if len(patterns) < self.order:
            return []

        if decision_mode not in {"probability", "negative_log", "avg_negative_log"}:
            raise ValueError("decision_mode 'probability', 'negative_log' veya 'avg_negative_log' olmalıdır.")

        if score_window < 1:
            raise ValueError("score_window en az 1 olmalıdır.")

        eps = 1e-12
        mapped_initial = []
        for i in range(self.order):
            pat = patterns[i]
            if pat not in self.trained_patterns:
                pat_mapped, _ = self._find_nearest_pattern(pat)
                mapped_initial.append(pat_mapped if pat_mapped is not None else pat)
            else:
                mapped_initial.append(pat)

        current_state = tuple(mapped_initial)
        recent_negative_log_scores = []
        scores = []

        for t in range(self.order, len(patterns)):
            incoming_pattern = patterns[t]
            mapped_to = incoming_pattern
            distance = 0
            forced_distance_anomaly = False

            if incoming_pattern not in self.trained_patterns:
                mapped_to, distance = self._find_nearest_pattern(incoming_pattern)
                if mapped_to is None:
                    mapped_to = incoming_pattern
                    distance = float("inf")
                if max_mapping_distance is not None and distance > max_mapping_distance:
                    forced_distance_anomaly = True

            prob = self.get_transition_probability(current_state, mapped_to)
            negative_log_score = float(-np.log(prob + eps))
            recent_negative_log_scores.append(negative_log_score)
            if len(recent_negative_log_scores) > score_window:
                recent_negative_log_scores.pop(0)

            if decision_mode == "probability":
                score = float(prob)
            elif decision_mode == "negative_log":
                score = negative_log_score
            else:
                score = float(np.mean(recent_negative_log_scores))

            if forced_distance_anomaly and decision_mode in {"negative_log", "avg_negative_log"}:
                score = max(float(score), float(distance))

            scores.append(score)
            current_state = current_state[1:] + (mapped_to,)

        return scores

    # test verisi üzerinde kayan pencere ile anomali tahmini yapar ve sonuçları döndürür
    def predict(
        self,
        test_patterns,
        anomaly_threshold=0.05,
        decision_mode="probability",
        score_threshold=None,
        score_window=1,
        max_mapping_distance=None,
    ):
        if len(test_patterns) < self.order:
            return [0] * (len(test_patterns) - 1), []

        if decision_mode not in {"probability", "negative_log", "avg_negative_log"}:
            raise ValueError("decision_mode 'probability', 'negative_log' veya 'avg_negative_log' olmalıdır.")

        if score_window < 1:
            raise ValueError("score_window en az 1 olmalıdır.")

        if max_mapping_distance is not None and max_mapping_distance < 0:
            raise ValueError("max_mapping_distance negatif olamaz.")

        eps = 1e-12
        if score_threshold is None:
            score_threshold = -np.log(anomaly_threshold + eps)

        predictions = []
        explainability_logs = []

        mapped_initial = []
        for i in range(self.order):
            pat = test_patterns[i]
            if pat not in self.trained_patterns:
                pat_mapped, _ = self._find_nearest_pattern(pat)
                mapped_initial.append(pat_mapped if pat_mapped is not None else pat)
            else:
                mapped_initial.append(pat)

        current_state = tuple(mapped_initial)
        transition_history = list(current_state)
        cumulative_path_prob = 1.0
        recent_negative_log_scores = []

        for t in range(self.order, len(test_patterns)):
            incoming_pattern = test_patterns[t]
            status = "seen"
            mapped_to = incoming_pattern
            distance = 0
            forced_distance_anomaly = False

            similarity_report = []
            if incoming_pattern not in self.trained_patterns:
                status = "unseen"
                mapped_to, distance = self._find_nearest_pattern(incoming_pattern)
                if mapped_to is None:
                    mapped_to = incoming_pattern
                    distance = float("inf")

                if max_mapping_distance is not None and distance > max_mapping_distance:
                    forced_distance_anomaly = True

                distances = [(tp, self._calculate_levenshtein(incoming_pattern, tp)) for tp in self.trained_patterns]
                distances.sort(key=lambda x: x[1])
                similarity_report = [{"pattern": p, "distance": d} for p, d in distances[:3]]

            prob = self.get_transition_probability(current_state, mapped_to)
            cumulative_path_prob *= prob
            path_probability = float(cumulative_path_prob)
            negative_log_score = float(-np.log(prob + eps))
            recent_negative_log_scores.append(negative_log_score)
            if len(recent_negative_log_scores) > score_window:
                recent_negative_log_scores.pop(0)
            avg_negative_log_score = float(np.mean(recent_negative_log_scores))

            if decision_mode == "negative_log":
                decision = "anomaly" if negative_log_score > score_threshold else "normal"
                confidence_score = negative_log_score
            elif decision_mode == "avg_negative_log":
                decision = "anomaly" if avg_negative_log_score > score_threshold else "normal"
                confidence_score = avg_negative_log_score
            else:
                decision = "anomaly" if prob < anomaly_threshold else "normal"
                confidence_score = float(prob)

            if forced_distance_anomaly:
                decision = "anomaly"
                if decision_mode in {"negative_log", "avg_negative_log"}:
                    confidence_score = max(float(confidence_score), float(distance))

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
                        if decision_mode in {"negative_log", "avg_negative_log"}:
                            alt_score = float(-np.log(alt_prob + eps))
                            alt_decision = "anomaly" if alt_score > score_threshold else "normal"
                        else:
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