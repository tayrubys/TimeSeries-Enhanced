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
        ensemble_enabled=False,
        ensemble_orders=None,
        ensemble_weights=None,
        ensemble_aggregation="mean",
    ):
        self.smoothing = smoothing
        self.base_order = order
        self.learning_rate = learning_rate
        self.smoothing_alpha = smoothing_alpha
        self.ensemble_enabled = ensemble_enabled
        self.ensemble_orders = self._prepare_ensemble_orders(order, ensemble_orders)
        self.ensemble_weights = self._prepare_ensemble_weights(ensemble_weights)
        self.ensemble_aggregation = ensemble_aggregation

        if self.ensemble_aggregation not in {"mean", "weighted_mean", "max", "min"}:
            raise ValueError("ensemble_aggregation 'mean', 'weighted_mean', 'max' veya 'min' olmalıdır.")

        # Ensemble açıkken başlangıç state uzunluğu en büyük order olur.
        # Feature kapalıyken eski davranış korunur.
        self.order = max(self.ensemble_orders) if self.ensemble_enabled else order

        self.transitions = defaultdict(lambda: defaultdict(float))
        self.total_exits = defaultdict(float)
        self.trained_patterns = set()

    def _prepare_ensemble_orders(self, order, ensemble_orders):
        if ensemble_orders is None:
            return [int(order)]

        prepared_orders = sorted({int(o) for o in ensemble_orders})
        if not prepared_orders:
            raise ValueError("ensemble_orders boş olamaz.")
        if any(o < 1 for o in prepared_orders):
            raise ValueError("ensemble_orders içindeki değerler en az 1 olmalıdır.")
        return prepared_orders

    def _prepare_ensemble_weights(self, ensemble_weights):
        if ensemble_weights is None:
            return None

        weights = [float(w) for w in ensemble_weights]
        if len(weights) != len(self.ensemble_orders):
            raise ValueError("ensemble_weights uzunluğu ensemble_orders uzunluğu ile aynı olmalıdır.")
        if any(w < 0 for w in weights):
            raise ValueError("ensemble_weights negatif değer içeremez.")
        if sum(weights) <= 0:
            raise ValueError("ensemble_weights toplamı sıfırdan büyük olmalıdır.")
        return weights

    # modeli verilen eğitim örüntüleriyle eğitir ve geçiş frekanslarını kaydeder
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

    # katz back-off algoritması ile bir sonraki durumun geçiş olasılığını hesaplar
    def get_transition_probability(self, current_state, next_pattern):
        return self._get_transition_probability_for_state(tuple(current_state), next_pattern)

    def _get_transition_probability_for_state(self, state, next_pattern):
        state_to_check = tuple(state)

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

    def _get_order_state(self, current_state, order):
        current_state = tuple(current_state)
        if len(current_state) < order:
            return current_state
        return current_state[-order:]

    def _get_ensemble_probabilities(self, current_state, next_pattern):
        return {
            order: self._get_transition_probability_for_state(
                self._get_order_state(current_state, order),
                next_pattern,
            )
            for order in self.ensemble_orders
        }

    def _aggregate_values(self, values_by_order):
        values = [float(values_by_order[order]) for order in self.ensemble_orders]

        if self.ensemble_aggregation == "max":
            return float(np.max(values))
        if self.ensemble_aggregation == "min":
            return float(np.min(values))
        if self.ensemble_aggregation == "weighted_mean":
            weights = self.ensemble_weights
            if weights is None:
                weights = [1.0] * len(values)
            return float(np.average(values, weights=weights))

        return float(np.mean(values))

    def _get_effective_probability(self, current_state, next_pattern):
        if not self.ensemble_enabled:
            prob = self.get_transition_probability(current_state, next_pattern)
            return prob, {self.base_order: prob}

        probabilities = self._get_ensemble_probabilities(current_state, next_pattern)
        aggregated_probability = self._aggregate_values(probabilities)
        return aggregated_probability, probabilities

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
        recent_negative_log_scores_by_order = {order: [] for order in self.ensemble_orders}
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

            prob, probabilities_by_order = self._get_effective_probability(current_state, mapped_to)

            if self.ensemble_enabled and decision_mode in {"negative_log", "avg_negative_log"}:
                negative_log_by_order = {
                    order: float(-np.log(probabilities_by_order[order] + eps))
                    for order in self.ensemble_orders
                }
                for order, order_score in negative_log_by_order.items():
                    recent_negative_log_scores_by_order[order].append(order_score)
                    if len(recent_negative_log_scores_by_order[order]) > score_window:
                        recent_negative_log_scores_by_order[order].pop(0)

                if decision_mode == "negative_log":
                    score = self._aggregate_values(negative_log_by_order)
                else:
                    avg_negative_log_by_order = {
                        order: float(np.mean(recent_negative_log_scores_by_order[order]))
                        for order in self.ensemble_orders
                    }
                    score = self._aggregate_values(avg_negative_log_by_order)
            else:
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
        recent_negative_log_scores_by_order = {order: [] for order in self.ensemble_orders}

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

            prob, probabilities_by_order = self._get_effective_probability(current_state, mapped_to)
            cumulative_path_prob *= prob
            path_probability = float(cumulative_path_prob)

            if self.ensemble_enabled and decision_mode in {"negative_log", "avg_negative_log"}:
                negative_log_by_order = {
                    order: float(-np.log(probabilities_by_order[order] + eps))
                    for order in self.ensemble_orders
                }
                for order, order_score in negative_log_by_order.items():
                    recent_negative_log_scores_by_order[order].append(order_score)
                    if len(recent_negative_log_scores_by_order[order]) > score_window:
                        recent_negative_log_scores_by_order[order].pop(0)

                if decision_mode == "negative_log":
                    confidence_score = self._aggregate_values(negative_log_by_order)
                    decision = "anomaly" if confidence_score > score_threshold else "normal"
                else:
                    avg_negative_log_by_order = {
                        order: float(np.mean(recent_negative_log_scores_by_order[order]))
                        for order in self.ensemble_orders
                    }
                    confidence_score = self._aggregate_values(avg_negative_log_by_order)
                    decision = "anomaly" if confidence_score > score_threshold else "normal"
            else:
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
                possible_transitions = []
                for p_next in self.trained_patterns:
                    alt_prob, alt_probabilities_by_order = self._get_effective_probability(current_state, p_next)
                    possible_transitions.append((p_next, alt_prob, alt_probabilities_by_order))

                possible_transitions.sort(key=lambda x: x[1], reverse=True)

                for alt_pattern, alt_prob, alt_probabilities_by_order in possible_transitions[:3]:
                    if alt_pattern != mapped_to:
                        if decision_mode in {"negative_log", "avg_negative_log"}:
                            if self.ensemble_enabled:
                                alt_scores_by_order = {
                                    order: float(-np.log(alt_probabilities_by_order[order] + eps))
                                    for order in self.ensemble_orders
                                }
                                alt_score = self._aggregate_values(alt_scores_by_order)
                            else:
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

            if isinstance(log_entry, dict) and self.ensemble_enabled:
                log_entry.update({
                    "ensemble_enabled": self.ensemble_enabled,
                    "ensemble_orders": self.ensemble_orders,
                    "ensemble_aggregation": self.ensemble_aggregation,
                    "ensemble_probabilities": {
                        str(order): float(probabilities_by_order[order])
                        for order in probabilities_by_order
                    },
                    "ensemble_aggregated_probability": float(prob),
                })

            explainability_logs.append(log_entry)
            predictions.append(1 if decision == "anomaly" else 0)

            current_state = current_state[1:] + (mapped_to,)

        padding_count = self.order - 1
        predictions = [0] * padding_count + predictions

        return predictions, explainability_logs
