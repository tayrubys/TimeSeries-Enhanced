import math
from collections import defaultdict

import numpy as np


class AlergiaStateMergingAutomata:
    """
    ilk aşamada her sax pattern ayrı bir state olarak kabul edilir.
    gelecekte benzer sembol dağılımlarına sahip state'ler, hoeffding tabanlı istatistiksel uyumluluk kontrolüyle birleştirilir.
    """

    def __init__(
        self,
        merge_alpha=0.05,
        min_state_count=2,
        smoothing_alpha=1.0,
        max_pattern_distance=None
    ):
        if not 0 < merge_alpha < 1:
            raise ValueError("merge_alpha 0 ile 1 arasında olmalıdır.")

        if min_state_count < 1:
            raise ValueError("min_state_count en az 1 olmalıdır.")

        if smoothing_alpha <= 0:
            raise ValueError("smoothing_alpha sıfırdan büyük olmalıdır.")

        self.merge_alpha = merge_alpha
        self.min_state_count = min_state_count
        self.smoothing_alpha = smoothing_alpha
        self.max_pattern_distance = max_pattern_distance

        #birleştirme öncesindeki geçişler
        self.raw_transitions = defaultdict(
            lambda: defaultdict(int)
        )
        self.raw_total_exits = defaultdict(int)

        #state'in ardından gelen son sax sembollerinin sayıları
        self.next_symbol_counts = defaultdict(
            lambda: defaultdict(int)
        )

        #birleştirme sonrası geçişler
        self.transitions = defaultdict(
            lambda: defaultdict(int)
        )
        self.total_exits = defaultdict(int)

        self.trained_patterns = set()
        self.state_mapping = {}
        self.merged_members = defaultdict(list)
        self.merged_states = set()

    def fit(self, train_patterns):
        if len(train_patterns) < 2:
            raise ValueError(
                "State-merging eğitimi için en az 2 pattern gereklidir."
            )

        self._reset()

        self.trained_patterns = set(train_patterns)

        #önce orijinal pattern-state otomatasını oluştur
        for index in range(len(train_patterns) - 1):
            current_state = train_patterns[index]
            next_state = train_patterns[index + 1]

            self.raw_transitions[current_state][next_state] += 1
            self.raw_total_exits[current_state] += 1

            #sliding-window pattern'ın yeni eklenen sembolü
            next_symbol = next_state[-1]
            self.next_symbol_counts[current_state][next_symbol] += 1

        self._merge_compatible_states()
        self._build_merged_transitions()

        return self

    def _reset(self):
        self.raw_transitions = defaultdict(
            lambda: defaultdict(int)
        )
        self.raw_total_exits = defaultdict(int)
        self.next_symbol_counts = defaultdict(
            lambda: defaultdict(int)
        )

        self.transitions = defaultdict(
            lambda: defaultdict(int)
        )
        self.total_exits = defaultdict(int)

        self.trained_patterns = set()
        self.state_mapping = {}
        self.merged_members = defaultdict(list)
        self.merged_states = set()

    def _merge_compatible_states(self):
        #cok görülen state'leri önce temsilci olarak kullan.
        ordered_states = sorted(
            self.trained_patterns,
            key=lambda state: (
                -self.raw_total_exits[state],
                state
            )
        )

        representatives = []

        for state in ordered_states:
            selected_representative = None
            best_distance = float("inf")

            for representative in representatives:
                if not self._passes_pattern_distance(
                    state,
                    representative
                ):
                    continue

                if not self._are_compatible(
                    [state],
                    self.merged_members[representative]
                ):
                    continue

                distribution_distance = (
                    self._distribution_distance(
                        [state],
                        self.merged_members[representative]
                    )
                )

                if distribution_distance < best_distance:
                    best_distance = distribution_distance
                    selected_representative = representative

            if selected_representative is None:
                representatives.append(state)
                self.state_mapping[state] = state
                self.merged_members[state].append(state)
            else:
                self.state_mapping[state] = selected_representative
                self.merged_members[selected_representative].append(state)

        self.merged_states = set(representatives)

    def _passes_pattern_distance(self, state_a, state_b):
        if self.max_pattern_distance is None:
            return True

        distance = self._calculate_levenshtein(
            state_a,
            state_b
        )

        return distance <= self.max_pattern_distance

    def _aggregate_symbol_counts(self, members):
        aggregated = defaultdict(int)

        for state in members:
            for symbol, count in self.next_symbol_counts[state].items():
                aggregated[symbol] += count

        return aggregated

    def _are_compatible(self, members_a, members_b):
        counts_a = self._aggregate_symbol_counts(members_a)
        counts_b = self._aggregate_symbol_counts(members_b)

        total_a = sum(counts_a.values())
        total_b = sum(counts_b.values())

        #yetersiz gözleme sahip state'leri birleştirmiyoruz
        if (
            total_a < self.min_state_count
            or total_b < self.min_state_count
        ):
            return False

        symbols = set(counts_a) | set(counts_b)

        #hoeffding tabanlı ALERGIA uyumluluk sınırı
        confidence_term = math.sqrt(
            0.5 * math.log(2.0 / self.merge_alpha)
        )

        bound = confidence_term * (
            (1.0 / math.sqrt(total_a))
            + (1.0 / math.sqrt(total_b))
        )

        for symbol in symbols:
            probability_a = counts_a[symbol] / total_a
            probability_b = counts_b[symbol] / total_b

            if abs(probability_a - probability_b) > bound:
                return False

        return True

    def _distribution_distance(self, members_a, members_b):
        counts_a = self._aggregate_symbol_counts(members_a)
        counts_b = self._aggregate_symbol_counts(members_b)

        total_a = sum(counts_a.values())
        total_b = sum(counts_b.values())

        if total_a == 0 or total_b == 0:
            return float("inf")

        symbols = set(counts_a) | set(counts_b)

        #total variation distance
        distance = 0.0

        for symbol in symbols:
            probability_a = counts_a[symbol] / total_a
            probability_b = counts_b[symbol] / total_b

            distance += abs(probability_a - probability_b)

        return 0.5 * distance

    def _build_merged_transitions(self):
        for current_state, targets in self.raw_transitions.items():
            merged_current = self.state_mapping[current_state]

            for next_state, count in targets.items():
                merged_next = self.state_mapping[next_state]

                self.transitions[merged_current][merged_next] += count
                self.total_exits[merged_current] += count

    def get_transition_probability(
        self,
        current_state,
        next_state
    ):
        merged_current = self._to_merged_state(current_state)
        merged_next = self._to_merged_state(next_state)

        number_of_states = max(len(self.merged_states), 1)
        total_output = self.total_exits[merged_current]
        transition_count = self.transitions[
            merged_current
        ][merged_next]

        numerator = (
            transition_count
            + self.smoothing_alpha
        )

        denominator = (
            total_output
            + self.smoothing_alpha * number_of_states
        )

        return numerator / denominator

    def _to_merged_state(self, pattern):
        if pattern in self.state_mapping:
            return self.state_mapping[pattern]

        #metot doğrudan merged-state ile çağrılırsa
        if pattern in self.merged_states:
            return pattern

        raise KeyError(
            f"Pattern eğitim sözlüğünde bulunamadı: {pattern}"
        )
    #her gecis icin surprise skoru uretir(yuksek skor daha supheli geçiş anlamına gelir)
    def score_patterns(self, patterns):
        if len(patterns) < 2:
            return np.array([], dtype=float), []

        scores = []
        logs = []

        current_pattern = patterns[0]
        current_status = "seen"
        current_distance = 0

        if current_pattern not in self.trained_patterns:
            current_status = "unseen"
            current_pattern, current_distance = (
                self._find_nearest_pattern(current_pattern)
            )

        cumulative_log_probability = 0.0

        for time_step in range(1, len(patterns)):
            incoming_pattern = patterns[time_step]

            status = "seen"
            mapped_pattern = incoming_pattern
            distance = 0

            if incoming_pattern not in self.trained_patterns:
                status = "unseen"
                mapped_pattern, distance = (
                    self._find_nearest_pattern(
                        incoming_pattern
                    )
                )

            transition_probability = (
                self.get_transition_probability(
                    current_pattern,
                    mapped_pattern
                )
            )

            transition_probability = max(
                transition_probability,
                1e-300
            )

            surprise_score = -math.log(
                transition_probability
            )

            cumulative_log_probability += math.log(
                transition_probability
            )

            #cok küçük olasılıklarda sıfıra taşmayı sınırla
            path_probability = math.exp(
                max(cumulative_log_probability, -745.0)
            )

            merged_current = self._to_merged_state(
                current_pattern
            )
            merged_next = self._to_merged_state(
                mapped_pattern
            )

            scores.append(surprise_score)

            logs.append({
                "time_step": time_step,
                "previous_pattern": current_pattern,
                "incoming_pattern": incoming_pattern,
                "status": status,
                "mapped_to": mapped_pattern,
                "levenshtein_distance": distance,
                "previous_merged_state": merged_current,
                "next_merged_state": merged_next,
                "transition_probability": float(
                    transition_probability
                ),
                "surprise_score": float(surprise_score),
                "path_probability": float(path_probability),
                "current_pattern_status": current_status,
                "current_pattern_distance": current_distance
            })

            current_pattern = mapped_pattern
            current_status = status
            current_distance = distance

        return np.asarray(scores), logs

    def predict(self, patterns, score_threshold):
        scores, logs = self.score_patterns(patterns)

        predictions = (
            scores >= score_threshold
        ).astype(int)

        for prediction, log_entry in zip(
            predictions,
            logs
        ):
            log_entry["threshold"] = float(score_threshold)
            log_entry["decision"] = (
                "anomaly"
                if prediction == 1
                else "normal"
            )

        return predictions, logs

    def _find_nearest_pattern(self, unseen_pattern):
        if not self.trained_patterns:
            raise RuntimeError(
                "Model eğitilmeden unseen eşleştirme yapılamaz."
            )

        nearest_pattern = None
        best_distance = float("inf")

        for trained_pattern in sorted(
            self.trained_patterns
        ):
            distance = self._calculate_levenshtein(
                unseen_pattern,
                trained_pattern
            )

            if distance < best_distance:
                best_distance = distance
                nearest_pattern = trained_pattern

        return nearest_pattern, best_distance

    def _calculate_levenshtein(self, first, second):
        if len(first) < len(second):
            return self._calculate_levenshtein(
                second,
                first
            )

        if len(second) == 0:
            return len(first)

        previous_row = list(
            range(len(second) + 1)
        )

        for row_index, first_character in enumerate(first):
            current_row = [row_index + 1]

            for column_index, second_character in enumerate(
                second
            ):
                insertion = (
                    previous_row[column_index + 1] + 1
                )
                deletion = current_row[column_index] + 1
                substitution = (
                    previous_row[column_index]
                    + (
                        first_character
                        != second_character
                    )
                )

                current_row.append(
                    min(
                        insertion,
                        deletion,
                        substitution
                    )
                )

            previous_row = current_row

        return previous_row[-1]

    def get_model_statistics(self):
        number_of_original_states = len(
            self.trained_patterns
        )
        number_of_merged_states = len(
            self.merged_states
        )

        number_of_transitions = sum(
            len(targets)
            for targets in self.transitions.values()
        )

        if number_of_merged_states > 0:
            transition_density = (
                number_of_transitions
                / (
                    number_of_merged_states
                    * number_of_merged_states
                )
            )
        else:
            transition_density = 0.0

        return {
            "original_state_count": (
                number_of_original_states
            ),
            "merged_state_count": (
                number_of_merged_states
            ),
            "merged_state_reduction": (
                number_of_original_states
                - number_of_merged_states
            ),
            "transition_count": number_of_transitions,
            "transition_density": transition_density
        }

    def get_state_mapping(self):
        return dict(self.state_mapping)

    def get_merged_members(self):
        return {
            representative: sorted(members)
            for representative, members
            in self.merged_members.items()
        }