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
        transition_confidence_enabled=False,
        transition_confidence_k=10.0,
        transition_confidence_weight=1.0,
        transition_confidence_mode="additive",
        transition_confidence_use_transition_count=True,
    ):
        self.smoothing = smoothing
        self.order = order
        self.learning_rate = learning_rate
        self.smoothing_alpha = smoothing_alpha

        # Transition-confidence parametreleri.
        # Varsayılan olarak kapalıdır; böylece eski Markov-base davranışı birebir korunur.
        self.transition_confidence_enabled = transition_confidence_enabled
        self.transition_confidence_k = transition_confidence_k
        self.transition_confidence_weight = transition_confidence_weight
        self.transition_confidence_mode = transition_confidence_mode
        self.transition_confidence_use_transition_count = transition_confidence_use_transition_count

        self.transitions = defaultdict(lambda: defaultdict(float))
        self.total_exits = defaultdict(float)
        self.trained_patterns = set()

        self._validate_transition_confidence_params()

    def _validate_transition_confidence_params(self):
        valid_modes = {"additive", "multiplicative", "probability_damping"}
        if self.transition_confidence_mode not in valid_modes:
            raise ValueError(
                "transition_confidence_mode 'additive', 'multiplicative' veya "
                "'probability_damping' olmalıdır."
            )

        if self.transition_confidence_k <= 0:
            raise ValueError("transition_confidence_k pozitif olmalıdır.")

        if self.transition_confidence_weight < 0:
            raise ValueError("transition_confidence_weight negatif olamaz.")

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

    # Katz back-off sürecinde hangi state'in kullanıldığını, transition sayısını ve toplam çıkışı döndürür
    def _resolve_transition_context(self, current_state, next_pattern):
        state_to_check = current_state

        while len(state_to_check) > 0:
            total_output = self.total_exits[state_to_check]
            if total_output > 0:
                transition_count = self.transitions[state_to_check][next_pattern]
                return {
                    "resolved_state": state_to_check,
                    "resolved_order": len(state_to_check),
                    "total_exits": float(total_output),
                    "transition_count": float(transition_count),
                    "used_uniform_fallback": False,
                }

            state_to_check = state_to_check[1:]

        return {
            "resolved_state": tuple(),
            "resolved_order": 0,
            "total_exits": 0.0,
            "transition_count": 0.0,
            "used_uniform_fallback": True,
        }

    # katz back-off algoritması ile bir sonraki durumun geçiş olasılığını hesaplar
    def get_transition_probability(self, current_state, next_pattern):
        context = self._resolve_transition_context(current_state, next_pattern)

        if not context["used_uniform_fallback"]:
            total_output = context["total_exits"]
            transition_count = context["transition_count"]

            if self.smoothing:
                alpha = self.smoothing_alpha
                vocab_size = len(self.trained_patterns)
                return (transition_count + alpha) / (total_output + alpha * vocab_size)

            return transition_count / total_output

        if self.smoothing and len(self.trained_patterns) > 0:
            return 1.0 / len(self.trained_patterns)

        return 0.0

    # transition olasılığının gözlem sayısına göre ne kadar güvenilir olduğunu hesaplar
    def get_transition_confidence(self, current_state, next_pattern):
        context = self._resolve_transition_context(current_state, next_pattern)

        total_exits = context["total_exits"]
        transition_count = context["transition_count"]
        k = self.transition_confidence_k

        if context["used_uniform_fallback"]:
            state_confidence = 0.0
            transition_confidence = 0.0
        else:
            # State güveni: bu context'in eğitimde ne kadar gözlendiği.
            state_confidence = total_exits / (total_exits + k)

            # Transition güveni: spesifik next_pattern geçişinin ne kadar gözlendiği.
            # Smoothing olasılık verebilir; ama gerçek gözlem sayısı düşükse confidence düşük kalır.
            if self.transition_confidence_use_transition_count:
                transition_confidence = transition_count / (transition_count + k)
            else:
                transition_confidence = state_confidence

        if self.transition_confidence_use_transition_count:
            combined_confidence = float(np.sqrt(state_confidence * transition_confidence))
        else:
            combined_confidence = float(state_confidence)

        combined_confidence = float(np.clip(combined_confidence, 0.0, 1.0))
        uncertainty = float(1.0 - combined_confidence)

        return {
            "confidence": combined_confidence,
            "uncertainty": uncertainty,
            "state_confidence": float(state_confidence),
            "transition_confidence": float(transition_confidence),
            "resolved_state": context["resolved_state"],
            "resolved_order": context["resolved_order"],
            "total_exits": float(total_exits),
            "transition_count": float(transition_count),
            "used_uniform_fallback": bool(context["used_uniform_fallback"]),
        }

    # transition-confidence mekanizması ile skoru yumuşak şekilde ayarlar
    def _apply_transition_confidence_to_scores(self, prob, negative_log_score, current_state, next_pattern):
        confidence_info = self.get_transition_confidence(current_state, next_pattern)

        if (
            not self.transition_confidence_enabled
            or self.transition_confidence_weight <= 0
        ):
            confidence_info.update({
                "adjusted_probability": float(prob),
                "adjusted_negative_log_score": float(negative_log_score),
                "confidence_penalty": 0.0,
            })
            return float(prob), float(negative_log_score), confidence_info

        eps = 1e-12
        confidence = confidence_info["confidence"]
        uncertainty = confidence_info["uncertainty"]
        weight = self.transition_confidence_weight

        if self.transition_confidence_mode == "additive":
            # En güvenli mod: negative-log skora lineer belirsizlik cezası ekler.
            # Düşük confidence doğrudan anomaly kararı vermez; sadece skoru kontrollü yükseltir.
            penalty = weight * uncertainty
            adjusted_negative_log_score = float(negative_log_score + penalty)
            adjusted_probability = float(np.exp(-adjusted_negative_log_score))

        elif self.transition_confidence_mode == "multiplicative":
            # Daha agresif mod: negative-log skoru confidence'a göre büyütür.
            penalty = weight * uncertainty
            adjusted_negative_log_score = float(negative_log_score * (1.0 + penalty))
            adjusted_probability = float(np.exp(-adjusted_negative_log_score))

        else:
            # probability_damping: probability'yi confidence'a göre düşürür.
            # probability decision_mode için anlamlıdır; negative_log skor buna göre yeniden hesaplanır.
            damping_power = weight * uncertainty
            adjusted_probability = float(prob * (confidence + eps) ** damping_power)
            adjusted_probability = float(np.clip(adjusted_probability, eps, 1.0))
            adjusted_negative_log_score = float(-np.log(adjusted_probability + eps))
            penalty = float(adjusted_negative_log_score - negative_log_score)

        confidence_info.update({
            "adjusted_probability": float(adjusted_probability),
            "adjusted_negative_log_score": float(adjusted_negative_log_score),
            "confidence_penalty": float(penalty),
        })
        return adjusted_probability, adjusted_negative_log_score, confidence_info

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

    def _temporarily_override_transition_confidence(
        self,
        transition_confidence_enabled,
        transition_confidence_k,
        transition_confidence_weight,
        transition_confidence_mode,
        transition_confidence_use_transition_count,
    ):
        original_settings = {
            "transition_confidence_enabled": self.transition_confidence_enabled,
            "transition_confidence_k": self.transition_confidence_k,
            "transition_confidence_weight": self.transition_confidence_weight,
            "transition_confidence_mode": self.transition_confidence_mode,
            "transition_confidence_use_transition_count": self.transition_confidence_use_transition_count,
        }

        if transition_confidence_enabled is not None:
            self.transition_confidence_enabled = transition_confidence_enabled
        if transition_confidence_k is not None:
            self.transition_confidence_k = transition_confidence_k
        if transition_confidence_weight is not None:
            self.transition_confidence_weight = transition_confidence_weight
        if transition_confidence_mode is not None:
            self.transition_confidence_mode = transition_confidence_mode
        if transition_confidence_use_transition_count is not None:
            self.transition_confidence_use_transition_count = transition_confidence_use_transition_count

        self._validate_transition_confidence_params()
        return original_settings

    def _restore_transition_confidence_settings(self, original_settings):
        self.transition_confidence_enabled = original_settings["transition_confidence_enabled"]
        self.transition_confidence_k = original_settings["transition_confidence_k"]
        self.transition_confidence_weight = original_settings["transition_confidence_weight"]
        self.transition_confidence_mode = original_settings["transition_confidence_mode"]
        self.transition_confidence_use_transition_count = original_settings["transition_confidence_use_transition_count"]

    # verilen pattern dizisi için karar vermeden olasılık/anomali skorlarını hesaplar
    def calculate_scores(
        self,
        patterns,
        decision_mode="avg_negative_log",
        score_window=1,
        max_mapping_distance=None,
        transition_confidence_enabled=None,
        transition_confidence_k=None,
        transition_confidence_weight=None,
        transition_confidence_mode=None,
        transition_confidence_use_transition_count=None,
    ):
        if len(patterns) < self.order:
            return []

        if decision_mode not in {"probability", "negative_log", "avg_negative_log"}:
            raise ValueError("decision_mode 'probability', 'negative_log' veya 'avg_negative_log' olmalıdır.")

        if score_window < 1:
            raise ValueError("score_window en az 1 olmalıdır.")

        original_confidence_settings = self._temporarily_override_transition_confidence(
            transition_confidence_enabled,
            transition_confidence_k,
            transition_confidence_weight,
            transition_confidence_mode,
            transition_confidence_use_transition_count,
        )

        try:
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

                raw_prob = self.get_transition_probability(current_state, mapped_to)
                raw_negative_log_score = float(-np.log(raw_prob + eps))
                adjusted_prob, adjusted_negative_log_score, _ = self._apply_transition_confidence_to_scores(
                    raw_prob,
                    raw_negative_log_score,
                    current_state,
                    mapped_to,
                )

                recent_negative_log_scores.append(adjusted_negative_log_score)
                if len(recent_negative_log_scores) > score_window:
                    recent_negative_log_scores.pop(0)

                if decision_mode == "probability":
                    score = float(adjusted_prob)
                elif decision_mode == "negative_log":
                    score = adjusted_negative_log_score
                else:
                    score = float(np.mean(recent_negative_log_scores))

                if forced_distance_anomaly and decision_mode in {"negative_log", "avg_negative_log"}:
                    score = max(float(score), float(distance))

                scores.append(score)
                current_state = current_state[1:] + (mapped_to,)

            return scores

        finally:
            self._restore_transition_confidence_settings(original_confidence_settings)

    # test verisi üzerinde kayan pencere ile anomali tahmini yapar ve sonuçları döndürür
    def predict(
        self,
        test_patterns,
        anomaly_threshold=0.05,
        decision_mode="probability",
        score_threshold=None,
        score_window=1,
        max_mapping_distance=None,
        transition_confidence_enabled=None,
        transition_confidence_k=None,
        transition_confidence_weight=None,
        transition_confidence_mode=None,
        transition_confidence_use_transition_count=None,
    ):
        if len(test_patterns) < self.order:
            return [0] * (len(test_patterns) - 1), []

        if decision_mode not in {"probability", "negative_log", "avg_negative_log"}:
            raise ValueError("decision_mode 'probability', 'negative_log' veya 'avg_negative_log' olmalıdır.")

        if score_window < 1:
            raise ValueError("score_window en az 1 olmalıdır.")

        if max_mapping_distance is not None and max_mapping_distance < 0:
            raise ValueError("max_mapping_distance negatif olamaz.")

        original_confidence_settings = self._temporarily_override_transition_confidence(
            transition_confidence_enabled,
            transition_confidence_k,
            transition_confidence_weight,
            transition_confidence_mode,
            transition_confidence_use_transition_count,
        )

        try:
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

                raw_prob = self.get_transition_probability(current_state, mapped_to)
                raw_negative_log_score = float(-np.log(raw_prob + eps))

                adjusted_prob, adjusted_negative_log_score, confidence_info = self._apply_transition_confidence_to_scores(
                    raw_prob,
                    raw_negative_log_score,
                    current_state,
                    mapped_to,
                )

                cumulative_path_prob *= adjusted_prob
                path_probability = float(cumulative_path_prob)

                recent_negative_log_scores.append(adjusted_negative_log_score)
                if len(recent_negative_log_scores) > score_window:
                    recent_negative_log_scores.pop(0)
                avg_negative_log_score = float(np.mean(recent_negative_log_scores))

                if decision_mode == "negative_log":
                    decision = "anomaly" if adjusted_negative_log_score > score_threshold else "normal"
                    confidence_score = adjusted_negative_log_score
                elif decision_mode == "avg_negative_log":
                    decision = "anomaly" if avg_negative_log_score > score_threshold else "normal"
                    confidence_score = avg_negative_log_score
                else:
                    decision = "anomaly" if adjusted_prob < anomaly_threshold else "normal"
                    confidence_score = float(adjusted_prob)

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
                            alt_raw_score = float(-np.log(alt_prob + eps))
                            alt_adjusted_prob, alt_adjusted_score, _ = self._apply_transition_confidence_to_scores(
                                alt_prob,
                                alt_raw_score,
                                current_state,
                                alt_pattern,
                            )

                            if decision_mode in {"negative_log", "avg_negative_log"}:
                                alt_decision = "anomaly" if alt_adjusted_score > score_threshold else "normal"
                            else:
                                alt_decision = "normal" if alt_adjusted_prob >= anomaly_threshold else "anomaly"

                            counterfactuals.append({
                                "pattern": alt_pattern,
                                "probability": float(alt_adjusted_prob),
                                "raw_probability": float(alt_prob),
                                "would_be_anomaly": alt_decision == "anomaly",
                            })

                transition_history.append(mapped_to)

                state_str = "->".join(current_state)

                log_entry = AutomataExplainer.generate_log(
                    t, state_str, incoming_pattern, status, mapped_to, distance,
                    adjusted_prob, path_probability, decision, confidence_score,
                    transition_history.copy(), total_exits_for_state,
                    counterfactuals, similarity_report
                )

                # Explainer dict döndürüyorsa transition-confidence detaylarını bozmadan ekle.
                if isinstance(log_entry, dict):
                    log_entry.update({
                        "raw_probability": float(raw_prob),
                        "adjusted_probability": float(adjusted_prob),
                        "raw_negative_log_score": float(raw_negative_log_score),
                        "adjusted_negative_log_score": float(adjusted_negative_log_score),
                        "transition_confidence_enabled": bool(self.transition_confidence_enabled),
                        "transition_confidence_mode": self.transition_confidence_mode,
                        "transition_confidence_k": float(self.transition_confidence_k),
                        "transition_confidence_weight": float(self.transition_confidence_weight),
                        "transition_confidence_use_transition_count": bool(self.transition_confidence_use_transition_count),
                        "transition_confidence": float(confidence_info["confidence"]),
                        "transition_uncertainty": float(confidence_info["uncertainty"]),
                        "state_confidence": float(confidence_info["state_confidence"]),
                        "specific_transition_confidence": float(confidence_info["transition_confidence"]),
                        "confidence_penalty": float(confidence_info["confidence_penalty"]),
                        "resolved_state": "->".join(confidence_info["resolved_state"]),
                        "resolved_order": int(confidence_info["resolved_order"]),
                        "resolved_total_exits": float(confidence_info["total_exits"]),
                        "resolved_transition_count": float(confidence_info["transition_count"]),
                        "used_uniform_fallback": bool(confidence_info["used_uniform_fallback"]),
                    })

                explainability_logs.append(log_entry)
                predictions.append(1 if decision == "anomaly" else 0)

                current_state = current_state[1:] + (mapped_to,)

            padding_count = self.order - 1
            predictions = [0] * padding_count + predictions

            return predictions, explainability_logs

        finally:
            self._restore_transition_confidence_settings(original_confidence_settings)
