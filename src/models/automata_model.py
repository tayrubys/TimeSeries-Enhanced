import numpy as np
from collections import defaultdict
from src.models.explainability import AutomataExplainer


class ProbabilisticAutomata:
    def __init__(
        self,
        smoothing=True,
        laplace_alpha=1.0,
        adaptive=True,
        learning_rate=0.20,
        update_threshold=0.05,
        decay_rate=0.0,
        allow_new_states=False,
        restore_after_predict=True,
        max_counterfactuals=3,
        verbose=False
    ):
        
        self.smoothing = smoothing
        self.laplace_alpha = laplace_alpha

        self.adaptive = adaptive
        self.learning_rate = learning_rate
        self.update_threshold = update_threshold
        self.decay_rate = decay_rate
        self.allow_new_states = allow_new_states
        self.restore_after_predict = restore_after_predict

        self.max_counterfactuals = max_counterfactuals
        self.verbose = verbose

        self.transitions = defaultdict(lambda: defaultdict(float))
        self.total_exits = defaultdict(float)
        self.trained_patterns = set()

        # Debug / analysis counters
        self.update_count = 0
        self.skipped_update_count = 0
        self.normal_count = 0
        self.anomaly_count = 0
        self.seen_count = 0
        self.unseen_count = 0

        self.last_diagnostics = {}
        self.diagnostics_history = []

    def fit(self, train_patterns):
        """
        Eğitim pattern dizisinden geçiş sayılarını öğrenir.
        """

        if len(train_patterns) < 2:
            raise ValueError("Otomata eğitimi için en az 2 pattern gereklidir.")

        # Aynı nesne tekrar fit edilirse eski bilgiler temizlenir.
        self.transitions = defaultdict(lambda: defaultdict(float))
        self.total_exits = defaultdict(float)
        self.trained_patterns = set(train_patterns)

        for i in range(len(train_patterns) - 1):
            current_state = train_patterns[i]
            next_state = train_patterns[i + 1]

            self.transitions[current_state][next_state] += 1.0
            self.total_exits[current_state] += 1.0

    def get_transition_probability(self, current_state, next_state):
        """
        P(next_state | current_state) geçiş olasılığını hesaplar.
        """

        n_states = len(self.trained_patterns)

        if n_states == 0:
            return 0.0

        total_output = self.total_exits[current_state]
        transition_count = self.transitions[current_state][next_state]

        if total_output == 0:
            if self.smoothing:
                return 1.0 / n_states
            return 0.0

        if self.smoothing:
            numerator = transition_count + self.laplace_alpha
            denominator = total_output + self.laplace_alpha * n_states
            return numerator / denominator

        return transition_count / total_output

    def _reset_prediction_counters(self):
        self.update_count = 0
        self.skipped_update_count = 0
        self.normal_count = 0
        self.anomaly_count = 0
        self.seen_count = 0
        self.unseen_count = 0

    def _snapshot_state(self):
        """
        predict öncesi model durumunu saklar.
        Böylece adaptive update, diğer senaryolara sızmaz.
        """

        transitions_copy = {
            state: dict(targets)
            for state, targets in self.transitions.items()
        }

        total_exits_copy = dict(self.total_exits)
        trained_patterns_copy = set(self.trained_patterns)

        return transitions_copy, total_exits_copy, trained_patterns_copy

    def _restore_state(self, snapshot):
        """
        predict sonrası modeli fit edilmiş ilk haline döndürür.
        """

        transitions_copy, total_exits_copy, trained_patterns_copy = snapshot

        self.transitions = defaultdict(lambda: defaultdict(float))
        for state, targets in transitions_copy.items():
            for target, value in targets.items():
                self.transitions[state][target] = float(value)

        self.total_exits = defaultdict(float)
        for state, value in total_exits_copy.items():
            self.total_exits[state] = float(value)

        self.trained_patterns = set(trained_patterns_copy)

    def _calculate_levenshtein(self, s1, s2):
        """
        İki pattern arasındaki Levenshtein distance değerini hesaplar.
        """

        if len(s1) < len(s2):
            return self._calculate_levenshtein(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = list(range(len(s2) + 1))

        for i, c1 in enumerate(s1):
            current_row = [i + 1]

            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + int(c1 != c2)

                current_row.append(
                    min(insertions, deletions, substitutions)
                )

            previous_row = current_row

        return previous_row[-1]

    def _find_nearest_pattern(self, unseen_pattern):
        """
        Unseen pattern'i en yakın eğitim pattern'ine map eder.
        """

        if len(self.trained_patterns) == 0:
            return None, float("inf")

        best_distance = float("inf")
        nearest_pattern = None

        for trained_pattern in sorted(self.trained_patterns):
            dist = self._calculate_levenshtein(unseen_pattern, trained_pattern)

            if dist < best_distance:
                best_distance = dist
                nearest_pattern = trained_pattern

        return nearest_pattern, best_distance

    def _build_similarity_report(self, incoming_pattern, top_k=3):
        """
        Explainability için en yakın pattern'leri listeler.
        """

        distances = []

        for trained_pattern in self.trained_patterns:
            dist = self._calculate_levenshtein(
                incoming_pattern,
                trained_pattern
            )
            distances.append((trained_pattern, dist))

        distances.sort(key=lambda x: x[1])

        return [
            {
                "pattern": pattern,
                "distance": distance
            }
            for pattern, distance in distances[:top_k]
        ]

    def _build_counterfactuals(self, current_state, mapped_to, anomaly_threshold):
        """
        Aynı current_state'ten gidilebilecek en olası alternatif geçişleri üretir.
        """

        counterfactuals = []

        if len(self.trained_patterns) == 0:
            return counterfactuals

        possible_transitions = []

        for candidate_next in self.trained_patterns:
            candidate_prob = self.get_transition_probability(
                current_state,
                candidate_next
            )
            possible_transitions.append((candidate_next, candidate_prob))

        possible_transitions.sort(key=lambda x: x[1], reverse=True)

        for alt_pattern, alt_prob in possible_transitions[:self.max_counterfactuals]:
            if alt_pattern == mapped_to:
                continue

            counterfactuals.append({
                "pattern": alt_pattern,
                "probability": float(alt_prob),
                "would_be_anomaly": alt_prob < anomaly_threshold
            })

        return counterfactuals

    def _apply_decay(self, current_state):
        """
        Concept drift için mevcut state'ten çıkan eski geçişleri zayıflatır.
        """

        if self.decay_rate <= 0:
            return

        if current_state not in self.transitions:
            return

        decay_factor = 1.0 - self.decay_rate

        for next_state in list(self.transitions[current_state].keys()):
            self.transitions[current_state][next_state] *= decay_factor

        self.total_exits[current_state] = sum(
            self.transitions[current_state].values()
        )

    def _update_transition(self, current_state, next_state):
        """
        Güvenilir normal geçişleri düşük katsayıyla güçlendirir.
        """

        if next_state not in self.trained_patterns:
            if self.allow_new_states:
                self.trained_patterns.add(next_state)
            else:
                return False

        self._apply_decay(current_state)

        self.transitions[current_state][next_state] += self.learning_rate
        self.total_exits[current_state] += self.learning_rate

        return True

    def _create_diagnostics(self, anomaly_threshold, effective_update_threshold):
        total_predictions = self.normal_count + self.anomaly_count

        diagnostics = {
            "adaptive": self.adaptive,
            "learning_rate": self.learning_rate,
            "anomaly_threshold": anomaly_threshold,
            "update_threshold": self.update_threshold,
            "effective_update_threshold": effective_update_threshold,
            "decay_rate": self.decay_rate,
            "allow_new_states": self.allow_new_states,
            "restore_after_predict": self.restore_after_predict,
            "total_predictions": total_predictions,
            "normal_count": self.normal_count,
            "anomaly_count": self.anomaly_count,
            "seen_count": self.seen_count,
            "unseen_count": self.unseen_count,
            "update_count": self.update_count,
            "skipped_update_count": self.skipped_update_count
        }

        if total_predictions > 0:
            diagnostics["normal_ratio"] = self.normal_count / total_predictions
            diagnostics["anomaly_ratio"] = self.anomaly_count / total_predictions
            diagnostics["update_ratio"] = self.update_count / total_predictions
        else:
            diagnostics["normal_ratio"] = 0.0
            diagnostics["anomaly_ratio"] = 0.0
            diagnostics["update_ratio"] = 0.0

        return diagnostics

    def get_diagnostics(self):
        """
        Son predict çağrısına ait debug bilgilerini döndürür.
        """

        return self.last_diagnostics

    def predict(self, test_patterns, anomaly_threshold=0.05):
        """
        Test pattern dizisi için anomaly tahmini üretir.

        Returns
        -------
        predictions : list[int]
            1 = anomaly
            0 = normal

        explainability_logs : list[dict]
            Her geçiş için açıklanabilirlik logları.
        """

    
        effective_update_threshold = self.update_threshold

        if len(test_patterns) < 2:
            self.last_diagnostics = self._create_diagnostics(
                anomaly_threshold=anomaly_threshold,
                effective_update_threshold=effective_update_threshold
            )
            return [], []

        if len(self.trained_patterns) == 0:
            raise ValueError("Modelde eğitim pattern'i yok. Önce fit() çağrılmalı.")

        snapshot = None
        if self.restore_after_predict:
            snapshot = self._snapshot_state()

        self._reset_prediction_counters()

        predictions = []
        explainability_logs = []

        first_pattern = test_patterns[0]

        if first_pattern in self.trained_patterns:
            current_state = first_pattern
        else:
            current_state, _ = self._find_nearest_pattern(first_pattern)

        if current_state is None:
            raise ValueError("Başlangıç pattern'i için uygun state bulunamadı.")

        transition_history = [current_state]

        log_path_probability = 0.0

        for t in range(1, len(test_patterns)):
            incoming_pattern = test_patterns[t]

            status = "seen"
            mapped_to = incoming_pattern
            distance = 0
            similarity_report = []

            if incoming_pattern in self.trained_patterns:
                self.seen_count += 1
            else:
                self.unseen_count += 1
                status = "unseen"

                if self.allow_new_states:
                    mapped_to = incoming_pattern
                    distance = 0
                else:
                    mapped_to, distance = self._find_nearest_pattern(incoming_pattern)
                    similarity_report = self._build_similarity_report(
                        incoming_pattern,
                        top_k=3
                    )

            if mapped_to is None:
                prob = 0.0
            else:
                prob = self.get_transition_probability(current_state, mapped_to)

            decision = "anomaly" if prob < anomaly_threshold else "normal"
            confidence_score = float(prob)

            if decision == "normal":
                self.normal_count += 1
            else:
                self.anomaly_count += 1

            safe_prob = max(prob, 1e-12)
            log_path_probability += np.log(safe_prob)

            # exp(-745) yaklaşık olarak float alt sınırıdır.
            path_probability = float(np.exp(max(log_path_probability, -745)))

            
            counterfactuals = self._build_counterfactuals(
                current_state,
                mapped_to,
                anomaly_threshold
            )

            total_exit_before_update = self.total_exits[current_state]

            transition_history.append(mapped_to)

            adaptive_update_applied = False
            adaptive_update_reason = "not_adaptive"

            if self.adaptive and decision == "normal":
                if prob >= effective_update_threshold:
                    adaptive_update_applied = self._update_transition(
                        current_state,
                        mapped_to
                    )

                    if adaptive_update_applied:
                        self.update_count += 1
                        adaptive_update_reason = "updated"
                    else:
                        self.skipped_update_count += 1
                        adaptive_update_reason = "update_failed"
                else:
                    self.skipped_update_count += 1
                    adaptive_update_reason = "below_update_threshold"

            elif self.adaptive and decision == "anomaly":
                adaptive_update_reason = "anomaly_not_updated"

            log_entry = AutomataExplainer.generate_log(
                t,
                current_state,
                incoming_pattern,
                status,
                mapped_to,
                distance,
                prob,
                path_probability,
                decision,
                confidence_score,
                transition_history,
                total_exit_before_update,
                counterfactuals,
                similarity_report
            )

            # Explainer dict döndürüyorsa adaptive bilgileri log'a eklenir.
            if isinstance(log_entry, dict):
                log_entry["adaptive_enabled"] = self.adaptive
                log_entry["adaptive_update_applied"] = adaptive_update_applied
                log_entry["adaptive_update_reason"] = adaptive_update_reason
                log_entry["effective_update_threshold"] = float(
                    effective_update_threshold
                )
                log_entry["learning_rate"] = float(self.learning_rate)
                log_entry["decay_rate"] = float(self.decay_rate)

            explainability_logs.append(log_entry)
            predictions.append(1 if decision == "anomaly" else 0)

            current_state = mapped_to

        self.last_diagnostics = self._create_diagnostics(
            anomaly_threshold=anomaly_threshold,
            effective_update_threshold=effective_update_threshold
        )

        self.diagnostics_history.append(self.last_diagnostics)

        if self.verbose:
            print(
                "[Adaptive Automata] "
                f"adaptive={self.adaptive} | "
                f"lr={self.learning_rate} | "
                f"update_thr={self.update_threshold} | "
                f"anomaly_thr={anomaly_threshold} | "
                f"updates={self.update_count} | "
                f"skipped={self.skipped_update_count} | "
                f"normal={self.normal_count} | "
                f"anomaly={self.anomaly_count} | "
                f"seen={self.seen_count} | "
                f"unseen={self.unseen_count}"
            )

        if self.restore_after_predict and snapshot is not None:
            self._restore_state(snapshot)

        return predictions, explainability_logs