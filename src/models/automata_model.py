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
        entropy_threshold=None,
        entropy_mode="normalized",
        entropy_min_total_exits=1,
    ):
        self.smoothing = smoothing
        self.order = order
        self.learning_rate = learning_rate
        self.smoothing_alpha = smoothing_alpha

        # Entropy-threshold feature ayarları
        # entropy_threshold=None ise feature kapalıdır ve model baseline gibi davranır.
        # entropy_mode="normalized" için threshold aralığı genelde 0.0-1.0 olur.
        # entropy_mode="raw" için threshold aralığı 0.0-log(vocab_size) olur.
        self.entropy_threshold = entropy_threshold
        self.entropy_mode = entropy_mode
        self.entropy_min_total_exits = entropy_min_total_exits

        self.transitions = defaultdict(lambda: defaultdict(float))
        self.total_exits = defaultdict(float)
        self.trained_patterns = set()

        self._validate_entropy_config()

    def _validate_entropy_config(self):
        if self.entropy_mode not in {"normalized", "raw"}:
            raise ValueError("entropy_mode 'normalized' veya 'raw' olmalıdır.")

        if self.entropy_threshold is not None and self.entropy_threshold < 0:
            raise ValueError("entropy_threshold negatif olamaz.")

        if self.entropy_min_total_exits < 0:
            raise ValueError("entropy_min_total_exits negatif olamaz.")

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

    # entropy hesabında hangi back-off state'in kullanıldığını bulur
    def _resolve_backoff_state(self, current_state):
        state_to_check = current_state

        while len(state_to_check) > 0:
            total_output = self.total_exits[state_to_check]
            if total_output > 0:
                return state_to_check, float(total_output)
            state_to_check = state_to_check[1:]

        return tuple(), 0.0

    # mevcut state için olası çıkışların normalize edilmiş olasılık dağılımını döndürür
    def get_transition_distribution(self, current_state):
        if len(self.trained_patterns) == 0:
            return {}

        distribution = {}
        total_probability = 0.0

        for next_pattern in self.trained_patterns:
            prob = float(self.get_transition_probability(current_state, next_pattern))
            if prob > 0.0:
                distribution[next_pattern] = prob
                total_probability += prob

        if total_probability <= 0.0:
            return {}

        # Smoothing/back-off kaynaklı küçük sayısal sapmaları temizlemek için normalize ediyoruz.
        return {
            pattern: prob / total_probability
            for pattern, prob in distribution.items()
        }

    # mevcut state'in transition dağılım entropisini hesaplar
    def get_state_entropy(self, current_state):
        distribution = self.get_transition_distribution(current_state)
        resolved_state, resolved_total_exits = self._resolve_backoff_state(current_state)

        if not distribution:
            return {
                "entropy": 0.0,
                "normalized_entropy": 0.0,
                "resolved_state": resolved_state,
                "resolved_total_exits": resolved_total_exits,
                "support_size": 0,
            }

        eps = 1e-12
        probabilities = np.array(list(distribution.values()), dtype=float)
        entropy = float(-np.sum(probabilities * np.log(probabilities + eps)))

        vocab_size = max(len(self.trained_patterns), 2)
        max_entropy = float(np.log(vocab_size))
        normalized_entropy = float(entropy / max_entropy) if max_entropy > 0 else 0.0

        # Olası floating point taşmalarına karşı 0-1 aralığına sıkıştırıyoruz.
        normalized_entropy = min(max(normalized_entropy, 0.0), 1.0)

        return {
            "entropy": entropy,
            "normalized_entropy": normalized_entropy,
            "resolved_state": resolved_state,
            "resolved_total_exits": resolved_total_exits,
            "support_size": len(distribution),
        }

    def _get_entropy_score(self, entropy_info, entropy_mode=None):
        mode = entropy_mode if entropy_mode is not None else self.entropy_mode
        if mode == "normalized":
            return float(entropy_info["normalized_entropy"])
        if mode == "raw":
            return float(entropy_info["entropy"])
        raise ValueError("entropy_mode 'normalized' veya 'raw' olmalıdır.")

    def _is_entropy_threshold_exceeded(
        self,
        current_state,
        entropy_threshold=None,
        entropy_mode=None,
        entropy_min_total_exits=None,
    ):
        threshold = self.entropy_threshold if entropy_threshold is None else entropy_threshold
        min_total_exits = (
            self.entropy_min_total_exits
            if entropy_min_total_exits is None
            else entropy_min_total_exits
        )

        entropy_info = self.get_state_entropy(current_state)
        entropy_score = self._get_entropy_score(entropy_info, entropy_mode)

        entropy_gate_active = threshold is not None
        enough_evidence = entropy_info["resolved_total_exits"] >= min_total_exits
        entropy_exceeded = bool(
            entropy_gate_active
            and enough_evidence
            and entropy_score > threshold
        )

        entropy_info.update({
            "entropy_score": float(entropy_score),
            "entropy_threshold": threshold,
            "entropy_mode": entropy_mode if entropy_mode is not None else self.entropy_mode,
            "entropy_min_total_exits": min_total_exits,
            "entropy_gate_active": entropy_gate_active,
            "entropy_enough_evidence": bool(enough_evidence),
            "entropy_exceeded": entropy_exceeded,
        })
        return entropy_exceeded, entropy_info

    def _attach_entropy_to_log(self, log_entry, entropy_info, decision_reason):
        entropy_metadata = {
            "state_entropy": float(entropy_info["entropy"]),
            "normalized_state_entropy": float(entropy_info["normalized_entropy"]),
            "entropy_score": float(entropy_info["entropy_score"]),
            "entropy_threshold": entropy_info["entropy_threshold"],
            "entropy_mode": entropy_info["entropy_mode"],
            "entropy_gate_active": bool(entropy_info["entropy_gate_active"]),
            "entropy_enough_evidence": bool(entropy_info["entropy_enough_evidence"]),
            "entropy_exceeded": bool(entropy_info["entropy_exceeded"]),
            "entropy_resolved_state": "->".join(entropy_info["resolved_state"]),
            "entropy_resolved_total_exits": float(entropy_info["resolved_total_exits"]),
            "entropy_support_size": int(entropy_info["support_size"]),
            "decision_reason": decision_reason,
        }

        # AutomataExplainer.generate_log mevcut projede dict dönüyorsa alanları ekliyoruz.
        # Farklı bir format dönüyorsa mevcut log yapısını bozmamak için aynen bırakıyoruz.
        if isinstance(log_entry, dict):
            log_entry.update(entropy_metadata)

        return log_entry

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
        entropy_mode=None,
    ):
        if len(patterns) < self.order:
            return []

        valid_decision_modes = {"probability", "negative_log", "avg_negative_log", "entropy"}
        if decision_mode not in valid_decision_modes:
            raise ValueError(
                "decision_mode 'probability', 'negative_log', 'avg_negative_log' veya 'entropy' olmalıdır."
            )

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

            entropy_info = self.get_state_entropy(current_state)
            entropy_score = self._get_entropy_score(entropy_info, entropy_mode)

            prob = self.get_transition_probability(current_state, mapped_to)
            negative_log_score = float(-np.log(prob + eps))
            recent_negative_log_scores.append(negative_log_score)
            if len(recent_negative_log_scores) > score_window:
                recent_negative_log_scores.pop(0)

            if decision_mode == "probability":
                score = float(prob)
            elif decision_mode == "negative_log":
                score = negative_log_score
            elif decision_mode == "avg_negative_log":
                score = float(np.mean(recent_negative_log_scores))
            else:
                # Entropy score yüksekse state'in sonraki geçiş dağılımı daha belirsizdir.
                score = float(entropy_score)

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
        entropy_threshold=None,
        entropy_mode=None,
        entropy_min_total_exits=None,
    ):
        if len(test_patterns) < self.order:
            return [0] * (len(test_patterns) - 1), []

        valid_decision_modes = {"probability", "negative_log", "avg_negative_log", "entropy"}
        if decision_mode not in valid_decision_modes:
            raise ValueError(
                "decision_mode 'probability', 'negative_log', 'avg_negative_log' veya 'entropy' olmalıdır."
            )

        if score_window < 1:
            raise ValueError("score_window en az 1 olmalıdır.")

        if max_mapping_distance is not None and max_mapping_distance < 0:
            raise ValueError("max_mapping_distance negatif olamaz.")

        if entropy_min_total_exits is not None and entropy_min_total_exits < 0:
            raise ValueError("entropy_min_total_exits negatif olamaz.")

        if entropy_threshold is not None and entropy_threshold < 0:
            raise ValueError("entropy_threshold negatif olamaz.")

        if entropy_mode is not None and entropy_mode not in {"normalized", "raw"}:
            raise ValueError("entropy_mode 'normalized' veya 'raw' olmalıdır.")

        eps = 1e-12
        if score_threshold is None:
            score_threshold = -np.log(anomaly_threshold + eps)

        active_entropy_threshold = (
            self.entropy_threshold if entropy_threshold is None else entropy_threshold
        )

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
            decision_reason = "transition_probability"

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

            entropy_exceeded, entropy_info = self._is_entropy_threshold_exceeded(
                current_state,
                entropy_threshold=active_entropy_threshold,
                entropy_mode=entropy_mode,
                entropy_min_total_exits=entropy_min_total_exits,
            )

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
                decision_reason = "negative_log"
            elif decision_mode == "avg_negative_log":
                decision = "anomaly" if avg_negative_log_score > score_threshold else "normal"
                confidence_score = avg_negative_log_score
                decision_reason = "avg_negative_log"
            elif decision_mode == "entropy":
                if active_entropy_threshold is None:
                    raise ValueError(
                        "decision_mode='entropy' için entropy_threshold verilmelidir. "
                        "Örnek: predict(..., decision_mode='entropy', entropy_threshold=0.85)"
                    )
                decision = "anomaly" if entropy_exceeded else "normal"
                confidence_score = float(entropy_info["entropy_score"])
                decision_reason = "entropy_threshold"
            else:
                decision = "anomaly" if prob < anomaly_threshold else "normal"
                confidence_score = float(prob)
                decision_reason = "transition_probability"

            if forced_distance_anomaly:
                decision = "anomaly"
                decision_reason = "mapping_distance"
                if decision_mode in {"negative_log", "avg_negative_log"}:
                    confidence_score = max(float(confidence_score), float(distance))

            # Hybrid kullanım: decision_mode probability/log iken entropy_threshold ayrıca güvenlik kapısı olur.
            if decision_mode != "entropy" and entropy_exceeded:
                decision = "anomaly"
                decision_reason = "entropy_threshold"
                if decision_mode in {"negative_log", "avg_negative_log"}:
                    confidence_score = max(float(confidence_score), float(entropy_info["entropy_score"]))

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
                        elif decision_mode == "entropy":
                            alt_decision = "anomaly" if entropy_exceeded else "normal"
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
            log_entry = self._attach_entropy_to_log(log_entry, entropy_info, decision_reason)

            explainability_logs.append(log_entry)
            predictions.append(1 if decision == "anomaly" else 0)

            current_state = current_state[1:] + (mapped_to,)

        padding_count = self.order - 1
        predictions = [0] * padding_count + predictions

        return predictions, explainability_logs
