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
        dirichlet_smoothing_enabled=False,
        dirichlet_alpha=None,
        dirichlet_prior_mode="uniform",
    ):
        self.smoothing = smoothing
        self.order = order
        self.learning_rate = learning_rate
        self.smoothing_alpha = smoothing_alpha

        # Dirichlet smoothing ayarları.
        self.dirichlet_smoothing_enabled = dirichlet_smoothing_enabled #varsayılan kapalı
        self.dirichlet_alpha = smoothing_alpha if dirichlet_alpha is None else dirichlet_alpha
        self.dirichlet_prior_mode = dirichlet_prior_mode

        if self.order < 1:
            raise ValueError("order en az 1 olmalıdır.")

        if self.smoothing_alpha < 0:
            raise ValueError("smoothing_alpha negatif olamaz.")

        if self.dirichlet_alpha < 0:
            raise ValueError("dirichlet_alpha negatif olamaz.")

        if self.dirichlet_prior_mode not in {"uniform", "unigram", "backoff"}:
            raise ValueError("dirichlet_prior_mode 'uniform', 'unigram' veya 'backoff' olmalıdır.")

        self.transitions = defaultdict(lambda: defaultdict(float))
        self.total_exits = defaultdict(float)
        self.trained_patterns = set()

        # Unigram prior için eğitim pattern frekansları.
        self.pattern_counts = defaultdict(float)
        self.total_pattern_count = 0.0

    # modeli verilen eğitim örüntüleriyle eğitir ve geçiş frekanslarını kaydeder
    def fit(self, train_patterns):
        if len(train_patterns) < self.order + 1:
            raise ValueError(f"Otomata eğitimi için en az {self.order + 1} pattern gereklidir.")

        # Aynı model nesnesi tekrar fit edilirse eski geçişlerin karışmasını engeller.
        self.transitions = defaultdict(lambda: defaultdict(float))
        self.total_exits = defaultdict(float)
        self.pattern_counts = defaultdict(float)
        self.total_pattern_count = 0.0

        self.trained_patterns = set(train_patterns)

        for pattern in train_patterns:
            self.pattern_counts[pattern] += 1.0
            self.total_pattern_count += 1.0

        for ord_idx in range(1, self.order + 1):
            for i in range(len(train_patterns) - ord_idx):
                state = tuple(train_patterns[i : i + ord_idx])
                next_pattern = train_patterns[i + ord_idx]
                self.transitions[state][next_pattern] += 1.0
                self.total_exits[state] += 1.0

    def _get_transition_count(self, state, next_pattern):
        if state in self.transitions:
            return float(self.transitions[state].get(next_pattern, 0.0))
        return 0.0

    def _resolve_backoff_state(self, current_state):
        state_to_check = current_state
        while len(state_to_check) > 0:
            total_output = float(self.total_exits.get(state_to_check, 0.0))
            if total_output > 0:
                return state_to_check, total_output
            state_to_check = state_to_check[1:]
        return tuple(), 0.0

    def _uniform_prior_probability(self, next_pattern):
        vocab_size = len(self.trained_patterns)
        if vocab_size <= 0:
            return 0.0
        if next_pattern not in self.trained_patterns:
            return 0.0
        return 1.0 / vocab_size

    def _unigram_prior_probability(self, next_pattern):
        if self.total_pattern_count <= 0:
            return self._uniform_prior_probability(next_pattern)
        return float(self.pattern_counts.get(next_pattern, 0.0)) / float(self.total_pattern_count)

    def _dirichlet_probability_for_state(self, state, next_pattern):
        total_output = float(self.total_exits.get(state, 0.0))
        transition_count = self._get_transition_count(state, next_pattern)
        prior_probability = self._get_dirichlet_prior_probability(next_pattern, state)
        alpha = float(self.dirichlet_alpha)

        denominator = total_output + alpha
        if denominator <= 0:
            return float(prior_probability)

        return float((transition_count + alpha * prior_probability) / denominator)

    def _get_dirichlet_prior_probability(self, next_pattern, state=None):
        """
        Dirichlet prior dağılımını hesaplar.

        uniform:
            Tüm eğitim pattern'lerine eşit prior verir.

        unigram:
            Eğitim setindeki global pattern frekanslarını prior olarak kullanır.

        backoff:
            Mümkünse bir alt dereceli Markov dağılımını prior olarak kullanır.
            Alt dereceli state yoksa unigram prior'a düşer.
        """
        if self.dirichlet_prior_mode == "uniform":
            return self._uniform_prior_probability(next_pattern)

        if self.dirichlet_prior_mode == "unigram":
            return self._unigram_prior_probability(next_pattern)

        # backoff prior
        if state is not None and len(state) > 1:
            lower_state = state[1:]
            if float(self.total_exits.get(lower_state, 0.0)) > 0:
                return self._dirichlet_probability_for_state(lower_state, next_pattern)

        # En düşük derecede veya alt state yoksa global unigram prior kullanılır.
        return self._unigram_prior_probability(next_pattern)

    def _get_probability_details(self, current_state, next_pattern):
        resolved_state, total_output = self._resolve_backoff_state(current_state)
        transition_count = self._get_transition_count(resolved_state, next_pattern) if total_output > 0 else 0.0

        details = {
            "resolved_state": resolved_state,
            "resolved_order": len(resolved_state),
            "resolved_total_exits": float(total_output),
            "resolved_transition_count": float(transition_count),
            "smoothing_strategy": "none",
            "dirichlet_smoothing_enabled": bool(self.dirichlet_smoothing_enabled),
            "dirichlet_alpha": float(self.dirichlet_alpha),
            "dirichlet_prior_mode": self.dirichlet_prior_mode,
            "dirichlet_prior_probability": None,
        }

        if total_output > 0:
            if self.dirichlet_smoothing_enabled:
                prior_probability = self._get_dirichlet_prior_probability(next_pattern, resolved_state)
                probability = self._dirichlet_probability_for_state(resolved_state, next_pattern)
                details.update({
                    "probability": float(probability),
                    "smoothing_strategy": "dirichlet",
                    "dirichlet_prior_probability": float(prior_probability),
                })
                return details

            if self.smoothing:
                alpha = self.smoothing_alpha
                vocab_size = len(self.trained_patterns)
                probability = (transition_count + alpha) / (total_output + alpha * vocab_size)
                details.update({
                    "probability": float(probability),
                    "smoothing_strategy": "additive",
                })
                return details

            probability = transition_count / total_output
            details.update({
                "probability": float(probability),
                "smoothing_strategy": "mle",
            })
            return details

        # Hiçbir backoff state bulunamadığında fallback dağılımı.
        if self.dirichlet_smoothing_enabled:
            prior_probability = self._get_dirichlet_prior_probability(next_pattern, tuple())
            details.update({
                "probability": float(prior_probability),
                "smoothing_strategy": "dirichlet_fallback",
                "dirichlet_prior_probability": float(prior_probability),
            })
            return details

        if self.smoothing and len(self.trained_patterns) > 0:
            probability = 1.0 / len(self.trained_patterns)
            details.update({
                "probability": float(probability),
                "smoothing_strategy": "uniform_fallback",
            })
            return details

        details.update({
            "probability": 0.0,
            "smoothing_strategy": "zero_fallback",
        })
        return details

    # katz back-off algoritması ile bir sonraki durumun geçiş olasılığını hesaplar
    def get_transition_probability(self, current_state, next_pattern):
        return self._get_probability_details(current_state, next_pattern)["probability"]

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

        self.pattern_counts[next_state] += 1.0
        self.total_pattern_count += 1.0

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

            probability_details = self._get_probability_details(current_state, mapped_to)
            prob = probability_details["probability"]
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
            total_exits_for_state = self.total_exits.get(current_state, 0.0)
            if total_exits_for_state > 0 or self.smoothing or self.dirichlet_smoothing_enabled:
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

            if isinstance(log_entry, dict):
                log_entry.update({
                    "smoothing_strategy": probability_details.get("smoothing_strategy"),
                    "dirichlet_smoothing_enabled": probability_details.get("dirichlet_smoothing_enabled"),
                    "dirichlet_alpha": probability_details.get("dirichlet_alpha"),
                    "dirichlet_prior_mode": probability_details.get("dirichlet_prior_mode"),
                    "dirichlet_prior_probability": probability_details.get("dirichlet_prior_probability"),
                    "resolved_state": "->".join(probability_details.get("resolved_state", tuple())),
                    "resolved_order": probability_details.get("resolved_order"),
                    "resolved_total_exits": probability_details.get("resolved_total_exits"),
                    "resolved_transition_count": probability_details.get("resolved_transition_count"),
                })

            explainability_logs.append(log_entry)
            predictions.append(1 if decision == "anomaly" else 0)

            current_state = current_state[1:] + (mapped_to,)

        padding_count = self.order - 1
        predictions = [0] * padding_count + predictions

        return predictions, explainability_logs
