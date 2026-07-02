import numpy as np
from collections import defaultdict
from src.models.explainability import AutomataExplainer

class ProbabilisticAutomata:
    def __init__(self, smoothing=True,weight_sharpness=1.0):
        self.smoothing = smoothing
        self.weight_sharpness = weight_sharpness
        self.transitions = defaultdict(lambda: defaultdict(int))
        self.total_exits = defaultdict(int)
        self.trained_patterns = set()
        self.weighted_probabilities = defaultdict(lambda: defaultdict(float))

    def fit(self, train_patterns):
        if len(train_patterns) < 2:
            raise ValueError("Otomata eğitimi için en az 2 pattern gereklidir.")

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
                base_prob = count / total_output
                weight = np.log1p(self.weight_sharpness * base_prob)
                raw_score = base_prob * weight

                raw_scores[next_state] = raw_score
                total_raw_score += raw_score

            if total_raw_score > 0:
               for next_state, raw_score in raw_scores.items():
                   #normalize yaptık
                   self.weighted_probabilities[current_state][next_state] = raw_score / total_raw_score

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
            
            counterfactuals = []
            if self.total_exits[current_state] > 0 or self.smoothing:
                possible_transitions = [(p_next, self.get_transition_probability(current_state, p_next)) for p_next in self.trained_patterns]
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

            log_entry = AutomataExplainer.generate_log(
                t, current_state, incoming_pattern, status, mapped_to, distance,
                prob, path_probability, decision, confidence_score,
                transition_history, self.total_exits[current_state],
                counterfactuals, similarity_report
            )
            explainability_logs.append(log_entry)
            predictions.append(1 if decision == "anomaly" else 0)
            current_state = mapped_to

        return predictions, explainability_logs