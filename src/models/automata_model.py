import numpy as np
from collections import defaultdict
from src.models.explainability import AutomataExplainer


class ProbabilisticAutomata:

    def __init__(
        self,
        smoothing=True,
        max_order=3,
        min_context_count=2,
        smoothing_alpha=1.0,
    ):
        self.smoothing = smoothing
        self.max_order = max_order
        self.min_context_count = min_context_count
        self.smoothing_alpha = smoothing_alpha

        # context -> next_state -> count
        # context tuple olarak tutulur:
        # ("A",) veya ("A", "B") veya ("A", "B", "C")
        self.transitions = defaultdict(lambda: defaultdict(float))

        # context -> toplam çıkış sayısı
        self.total_exits = defaultdict(float)

        # eğitimde görülen pattern/state kümesi
        self.trained_patterns = set()

        # root context, hiçbir suffix güvenilir değilse fallback için kullanılır
        self.root_context = tuple()

    def fit(self, train_patterns):
        """
        Probabilistic suffix tree context'lerini öğrenir.

        train_patterns:
            Pattern/state dizisi.
            Örnek:
                ["A", "B", "C", "D", ...]
        """

        if len(train_patterns) < 2:
            raise ValueError("Otomata eğitimi için en az 2 pattern gereklidir.")

        self.trained_patterns = set(train_patterns)

        for i in range(1, len(train_patterns)):
            next_state = train_patterns[i]

            # Root context: genel next_state dağılımı
            self.transitions[self.root_context][next_state] += 1.0
            self.total_exits[self.root_context] += 1.0

            # Geçmiş context: train_patterns[:i]
            start_index = max(0, i - self.max_order)
            history = train_patterns[start_index:i]

            # Tüm suffix uzunluklarını öğren:
            for order in range(1, len(history) + 1):
                context = tuple(history[-order:])
                self.transitions[context][next_state] += 1.0
                self.total_exits[context] += 1.0

    def get_transition_probability(self, current_state, next_state):
        
        context = (current_state,)
        return self._get_context_probability(context, next_state)

    def _get_context_probability(self, context, next_state):

        vocab_size = max(len(self.trained_patterns), 1)
        total_output = self.total_exits[context]
        transition_count = self.transitions[context][next_state]

        if total_output == 0:
            if self.smoothing:
                return 1.0 / vocab_size
            return 0.0

        if self.smoothing:
            alpha = self.smoothing_alpha
            return (transition_count + alpha) / (total_output + alpha * vocab_size)

        return transition_count / total_output

    def _select_best_suffix_context(self, transition_history):

        max_available_order = min(self.max_order, len(transition_history))

        for order in range(max_available_order, 0, -1):
            context = tuple(transition_history[-order:])
            context_count = self.total_exits[context]

            if context_count >= self.min_context_count:
                return context, order, context_count

        return self.root_context, 0, self.total_exits[self.root_context]

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

    def _find_nearest_pattern(self, unseen_pattern):
        best_distance = float("inf")
        nearest_pattern = None

        for trained_pattern in sorted(self.trained_patterns):
            dist = self._calculate_levenshtein(unseen_pattern, trained_pattern)

            if dist < best_distance:
                best_distance = dist
                nearest_pattern = trained_pattern

        return nearest_pattern, best_distance

    def _context_to_string(self, context):
        if context == self.root_context:
            return "<root>"

        return " -> ".join(map(str, context))

    def predict(self, test_patterns, anomaly_threshold=0.05):

        if len(test_patterns) < 2:
            return [], []

        predictions = []
        explainability_logs = []

        first_state = test_patterns[0]

        if first_state not in self.trained_patterns:
            first_state, _ = self._find_nearest_pattern(first_state)

        transition_history = [first_state]
        cumulative_path_prob = 1.0

        for t in range(1, len(test_patterns)):
            incoming_pattern = test_patterns[t]

            status = "seen"
            mapped_to = incoming_pattern
            distance = 0
            similarity_report = []

            if incoming_pattern not in self.trained_patterns:
                status = "unseen"
                mapped_to, distance = self._find_nearest_pattern(incoming_pattern)

                distances = [
                    (tp, self._calculate_levenshtein(incoming_pattern, tp))
                    for tp in self.trained_patterns
                ]
                distances.sort(key=lambda x: x[1])

                similarity_report = [
                    {
                        "pattern": p,
                        "distance": d,
                    }
                    for p, d in distances[:3]
                ]

            selected_context, selected_order, selected_context_count = (
                self._select_best_suffix_context(transition_history)
            )

            prob = self._get_context_probability(selected_context, mapped_to)

            cumulative_path_prob *= prob
            path_probability = float(cumulative_path_prob)

            decision = "anomaly" if prob < anomaly_threshold else "normal"
            confidence_score = float(prob)

            counterfactuals = []

            if self.total_exits[selected_context] > 0 or self.smoothing:
                possible_transitions = [
                    (
                        p_next,
                        self._get_context_probability(selected_context, p_next),
                    )
                    for p_next in self.trained_patterns
                ]

                possible_transitions.sort(key=lambda x: x[1], reverse=True)

                for alt_pattern, alt_prob in possible_transitions[:3]:
                    if alt_pattern != mapped_to:
                        alt_decision = (
                            "normal"
                            if alt_prob >= anomaly_threshold
                            else "anomaly"
                        )

                        counterfactuals.append(
                            {
                                "pattern": alt_pattern,
                                "probability": float(alt_prob),
                                "would_be_anomaly": alt_decision == "anomaly",
                            }
                        )

            context_as_state = self._context_to_string(selected_context)

            transition_history.append(mapped_to)

            log_entry = AutomataExplainer.generate_log(
                t,
                context_as_state,
                incoming_pattern,
                status,
                mapped_to,
                distance,
                prob,
                path_probability,
                decision,
                confidence_score,
                transition_history,
                selected_context_count,
                counterfactuals,
                similarity_report,
            )

            # Ek suffix-tree açıklamaları
            if isinstance(log_entry, dict):
                log_entry["selected_context"] = list(selected_context)
                log_entry["selected_order"] = int(selected_order)
                log_entry["selected_context_count"] = float(selected_context_count)
                log_entry["model_variant"] = "probabilistic_suffix_tree"

            explainability_logs.append(log_entry)
            predictions.append(1 if decision == "anomaly" else 0)

        return predictions, explainability_logs