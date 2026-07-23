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
        max_pattern_distance=None,
        distance_penalty=0.0,
        mapping_top_k=1,
        mapping_gamma=1.0,
    ):
        if not 0 < merge_alpha < 1:
            raise ValueError("merge_alpha 0 ile 1 arasında olmalıdır.")

        if min_state_count < 1:
            raise ValueError("min_state_count en az 1 olmalıdır.")

        if smoothing_alpha <= 0:
            raise ValueError("smoothing_alpha sıfırdan büyük olmalıdır.")
        if distance_penalty < 0:
            raise ValueError("distance_penalty negatif olamaz.")

        if (
            isinstance(mapping_top_k, bool)
            or int(mapping_top_k) != mapping_top_k
            or int(mapping_top_k) < 1
        ):
            raise ValueError(
                "mapping_top_k pozitif bir tam sayı olmalıdır."
            )

        if mapping_gamma <= 0:
            raise ValueError(
                "mapping_gamma sıfırdan büyük olmalıdır."
            )

        #hiperparametrelerini kaydeder
        self.merge_alpha = merge_alpha
        self.min_state_count = min_state_count
        self.smoothing_alpha = smoothing_alpha
        self.max_pattern_distance = max_pattern_distance
        self.distance_penalty = float(distance_penalty)
        self.mapping_top_k = int(mapping_top_k)
        self.mapping_gamma = float(mapping_gamma)

        #birleştirme öncesindeki geçişler
        self.raw_transitions = defaultdict(
            lambda: defaultdict(int)
        )
        self.raw_total_exits = defaultdict(int)

        #state'ten sonra hangi tam pattern'ların geldiğini tutuyoruz
        self.next_pattern_counts = defaultdict(lambda: defaultdict(int))

        #birleştirme sonrası geçişler matrisi
        self.transitions = defaultdict(
            lambda: defaultdict(int)
        )
        self.total_exits = defaultdict(int)
        #ogrenilen patternleri ve state eşleşmelerini tutan değişkenler
        self.trained_patterns = set()
        self.pattern_counts = defaultdict(int)
        self.state_mapping = {}
        self.merged_members = defaultdict(list)
        self.merged_states = set()
    #eski kullanım bozulmasın diye tek pattern listesini bir tane sequence olarak gonderiyoz
    def fit(self, train_patterns):
        return self.fit_sequences([train_patterns])

    #birden fazla bagımısız pattern dizisiyle modeli eğitir
    def fit_sequences(self, pattern_sequences):
        """
        Sequence'ler arasında geçiş oluşturmuyoruz.
        Böylece gerçekte yan yana olmayan pattern'lar arasındayanlış geçiş öğrenilmemiş oluyor.
        """

        valid_sequences = []

        for sequence in pattern_sequences:
            sequence = list(sequence)

            #geçiş öğrenmek için en az iki pattern lazım
            if len(sequence) >= 2:
                valid_sequences.append(sequence)

        if len(valid_sequences) == 0:
            raise ValueError(
                "Model eğitimi için en az bir geçerli "
                "pattern sequence gereklidir."
            )

        self._reset()#her eğitimden sonra eski verileri temizler
        #eğitim verisini gezip patternleri ve frekansları kaydeder
        for sequence in valid_sequences:
            self.trained_patterns.update(sequence)
            for pattern in sequence:
                self.pattern_counts[pattern] += 1

            for index in range(len(sequence) - 1):
                current_state = sequence[index]
                next_state = sequence[index + 1]

                self.raw_transitions[
                    current_state
                ][next_state] += 1

                self.raw_total_exits[
                    current_state
                ] += 1

                #sadece son sac sembolünü değil, sonraki tam patternı kullanıyoruz
                self.next_pattern_counts[
                current_state
                ][next_state] += 1
        #istatistiksel olarak uyumlu olan stateleri mergeleyip grafı sadeleştirir
        self._merge_compatible_states()
        self._build_merged_transitions()

        return self
    
    #tüm dictionaryleri ve setleri sifirlar
    def _reset(self):
        self.raw_transitions = defaultdict(
            lambda: defaultdict(int)
        )
        self.raw_total_exits = defaultdict(int)
        
        self.next_pattern_counts = defaultdict(lambda: defaultdict(int))

        self.transitions = defaultdict(
            lambda: defaultdict(int)
        )
        self.total_exits = defaultdict(int)

        self.trained_patterns = set()
        self.pattern_counts = defaultdict(int)
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
                #aralarındaki mesafe belirlenen eşikten buyukse atla
                if not self._passes_pattern_distance(
                    state,
                    representative
                ):
                    continue
                #gelecekteki geçiş dağılımları benzemiyorsa atla
                if not self._are_compatible(
                    [state],
                    self.merged_members[representative]
                ):
                    continue
                #dağılım farkını hesaplar
                distribution_distance = (
                    self._distribution_distance(
                        [state],
                        self.merged_members[representative]
                    )
                )

                is_better_distance = (
                    distribution_distance
                    < best_distance - 1e-12
                )

                is_equal_distance = math.isclose(
                    distribution_distance,
                    best_distance,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                #mesafeler esitse alfabetik sıraya bakarki her çalısmada aynı sonucu versın
                is_better_tie = (
                    is_equal_distance
                    and (
                        selected_representative is None
                        or representative < selected_representative
                    )
                )

                if is_better_distance or is_better_tie:
                    best_distance = distribution_distance
                    selected_representative = representative
            #uygun temsilci bulunamadıysa kendı grubunu kurar
            if selected_representative is None:
                representatives.append(state)
                self.state_mapping[state] = state
                self.merged_members[state].append(state)
            else:
                #temsilci bulunduysa mevcut gruba bağlanır
                self.state_mapping[state] = selected_representative
                self.merged_members[selected_representative].append(state)

        self.merged_states = set(representatives)

    def _passes_pattern_distance(self, state_a, state_b):
        #max levenshtein mesafesi sınırı
        if self.max_pattern_distance is None:
            return True

        distance = self._calculate_levenshtein(
            state_a,
            state_b
        )

        return distance <= self.max_pattern_distance
    #birleştirme grubundaki statelerin sonraki pattern sayılarını tek dağılımda topla
    def _aggregate_next_pattern_counts(self, members):

        aggregated = defaultdict(int)

        for state in members:
            for next_pattern, count in self.next_pattern_counts[
                state
            ].items():
                aggregated[next_pattern] += count

        return aggregated

    def _are_compatible(self, members_a, members_b):
        #hoeffding sınırını kullanarak iki state grubunun birleşmeye uygun olup olmadığını test eder
        counts_a = self._aggregate_next_pattern_counts(members_a)
        counts_b = self._aggregate_next_pattern_counts(members_b)

        total_a = sum(counts_a.values())
        total_b = sum(counts_b.values())
        #istatistiksel karar vermek için yeterli veri yoksa birleştirme
        if (
            total_a < self.min_state_count
            or total_b < self.min_state_count
        ):
            return False

        next_patterns = sorted(set(counts_a) | set(counts_b))
        #guven sınırı hesaplama
        confidence_term = math.sqrt(
            0.5 * math.log(2.0 / self.merge_alpha)
        )

        bound = confidence_term * (
            (1.0 / math.sqrt(total_a))+ (1.0 / math.sqrt(total_b)))
        #olasılıklar arası fark sınır değerinden (bound) büyükse uyumsuzdur
        for next_pattern in next_patterns:
            probability_a = (
                counts_a.get(next_pattern, 0) / total_a
            )
            probability_b = (
                counts_b.get(next_pattern, 0) / total_b
            )

            if abs(probability_a - probability_b) > bound:
                return False

        return True
    
    #iki grubun geçiş olasılıkları arasındaki matematiksel farkı hesaplar
    def _distribution_distance(self,members_a,members_b,):
        counts_a = self._aggregate_next_pattern_counts(
            members_a
        )
        counts_b = self._aggregate_next_pattern_counts(
            members_b
        )

        total_a = sum(counts_a.values())
        total_b = sum(counts_b.values())

        if total_a == 0 or total_b == 0:
            return float("inf")

        # Set doğrudan dolaşılmıyor.
        # Her çalıştırmada aynı sıra kullanılıyor.
        next_patterns = sorted(
            set(counts_a) | set(counts_b)
        )

        differences = []

        for next_pattern in next_patterns:
            probability_a = (
                counts_a.get(next_pattern, 0) / total_a
            )
            probability_b = (
                counts_b.get(next_pattern, 0) / total_b
            )

            differences.append(
                abs(probability_a - probability_b)
            )

        #math.fsum, kayan noktalı sayıların daha kararlı biçimde toplanmasını sağlar
        return 0.5 * math.fsum(differences)
   
    #birleştirilmiş durumlara göre yeni graf yapısını ve bağlantıları gunceller
    def _build_merged_transitions(self):
        for current_state in sorted(
            self.raw_transitions
        ):
            targets = self.raw_transitions[current_state]
            merged_current = self.state_mapping[
                current_state
            ]

            for next_state in sorted(targets):
                count = targets[next_state]
                merged_next = self.state_mapping[
                    next_state
                ]

                self.transitions[
                    merged_current
                ][merged_next] += count

                self.total_exits[
                    merged_current
                ] += count

    def get_transition_probability(
        self,
        current_state,
        next_state
    ):
        #laplace smoothing
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
        #verilen bir pattern'in atandığı merged state i getirir
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

        current_input_pattern = patterns[0]
        current_status = (
            "seen"
            if current_input_pattern in self.trained_patterns
            else "unseen"
        )
        current_candidates = self._get_mapping_candidates(
            current_input_pattern
        )

        cumulative_log_probability = 0.0

        for time_step in range(1, len(patterns)):
            incoming_pattern = patterns[time_step]
            status = (
                "seen"
                if incoming_pattern in self.trained_patterns
                else "unseen"
            )
            incoming_candidates = self._get_mapping_candidates(
                incoming_pattern
            )

            # Current ve incoming adaylarının bütün geçişlerini
            # Levenshtein ağırlıklarıyla birlikte kullanır.
            transition_probability = (
                self._get_soft_transition_probability(
                    current_candidates,
                    incoming_candidates,
                )
            )
            transition_probability = max(
                transition_probability,
                1e-300,
            )

            base_surprise = -math.log(
                transition_probability
            )

            # Önceki sert eşlemedeki distance penalty korunur.
            # Tek uzaklık yerine adayların ağırlıklı beklenen
            # normalize uzaklıkları kullanılır.
            expected_current_distance = math.fsum(
                candidate["weight"]
                * (
                    candidate["distance"]
                    / max(
                        len(str(candidate["pattern"])),
                        1,
                    )
                )
                for candidate in current_candidates
            )
            expected_incoming_distance = math.fsum(
                candidate["weight"]
                * (
                    candidate["distance"]
                    / max(len(str(incoming_pattern)), 1)
                )
                for candidate in incoming_candidates
            )

            mapping_distance = (
                expected_current_distance
                + expected_incoming_distance
            )
            mapping_penalty = (
                self.distance_penalty
                * mapping_distance
            )
            surprise_score = (
                base_surprise
                + mapping_penalty
            )

            cumulative_log_probability += math.log(
                transition_probability
            )
            path_probability = math.exp(
                max(cumulative_log_probability, -745.0)
            )

            # Eski log alanları korunur. mapped_to alanında
            # en yüksek ağırlıklı ilk aday gösterilir.
            current_primary = current_candidates[0]
            incoming_primary = incoming_candidates[0]

            merged_current = self._to_merged_state(
                current_primary["pattern"]
            )
            merged_next = self._to_merged_state(
                incoming_primary["pattern"]
            )

            scores.append(surprise_score)

            logs.append(
                {
                    "time_step": time_step,
                    "previous_pattern": current_primary[
                        "pattern"
                    ],
                    "incoming_pattern": incoming_pattern,
                    "status": status,
                    "mapped_to": incoming_primary[
                        "pattern"
                    ],
                    "levenshtein_distance": int(
                        incoming_primary["distance"]
                    ),
                    "previous_merged_state": (
                        merged_current
                    ),
                    "next_merged_state": merged_next,
                    "transition_probability": float(
                        transition_probability
                    ),
                    "base_surprise": float(
                        base_surprise
                    ),
                    "mapping_distance": float(
                        mapping_distance
                    ),
                    "mapping_penalty": float(
                        mapping_penalty
                    ),
                    "distance_penalty_weight": float(
                        self.distance_penalty
                    ),
                    "surprise_score": float(
                        surprise_score
                    ),
                    "path_probability": float(
                        path_probability
                    ),
                    "current_pattern_status": (
                        current_status
                    ),
                    "current_pattern_distance": int(
                        current_primary["distance"]
                    ),
                    "mapping_top_k": int(
                        self.mapping_top_k
                    ),
                    "mapping_gamma": float(
                        self.mapping_gamma
                    ),
                    "soft_mapping_used": bool(
                        len(current_candidates) > 1
                        or len(incoming_candidates) > 1
                    ),
                    "current_expected_distance": float(
                        expected_current_distance
                    ),
                    "incoming_expected_distance": float(
                        expected_incoming_distance
                    ),
                    "current_candidates": [
                        {
                            "pattern": candidate[
                                "pattern"
                            ],
                            "distance": int(
                                candidate["distance"]
                            ),
                            "normalized_distance": (
                                float(
                                    candidate[
                                        "normalized_distance"
                                    ]
                                )
                            ),
                            "weight": float(
                                candidate["weight"]
                            ),
                        }
                        for candidate in current_candidates
                    ],
                    "incoming_candidates": [
                        {
                            "pattern": candidate[
                                "pattern"
                            ],
                            "distance": int(
                                candidate["distance"]
                            ),
                            "normalized_distance": (
                                float(
                                    candidate[
                                        "normalized_distance"
                                    ]
                                )
                            ),
                            "weight": float(
                                candidate["weight"]
                            ),
                        }
                        for candidate in incoming_candidates
                    ],
                }
            )

            current_input_pattern = incoming_pattern
            current_candidates = incoming_candidates
            current_status = status

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
    #seen pattern için kendısını dondurur, unseen icin en yakın mapping top k train patternının levenshtein tabanlı ağrlıklarıyla dondurur
    def _get_mapping_candidates(self, pattern):
        if pattern in self.trained_patterns:
            return [
                {
                    "pattern": pattern,
                    "distance": 0,
                    "normalized_distance": 0.0,
                    "weight": 1.0,
                }
            ]

        return self._find_nearest_patterns(
            pattern,
            top_k=self.mapping_top_k,
        )
    #current ve next adaylarının geçiş olasılıklarını aday ağırlıklarının çarpımıyla birleştirir
    def _get_soft_transition_probability(
        self,
        current_candidates,
        next_candidates,
    ):
        weighted_probabilities = []

        for current_candidate in current_candidates:
            for next_candidate in next_candidates:
                pair_weight = (
                    current_candidate["weight"]
                    * next_candidate["weight"]
                )
                pair_probability = (
                    self.get_transition_probability(
                        current_candidate["pattern"],
                        next_candidate["pattern"],
                    )
                )
                weighted_probabilities.append(
                    pair_weight * pair_probability
                )

        return math.fsum(weighted_probabilities)
    #unseen pattern a en yakın k train pattern ını bulur
    def _find_nearest_patterns(
        self,
        unseen_pattern,
        top_k=None,
    ):
        """
        Ağırlık:exp(-mapping_gamma * normalized_distance)
        """
        if not self.trained_patterns:
            raise RuntimeError(
                "Model eğitilmeden unseen eşleştirme yapılamaz."
            )

        candidate_count = (
            self.mapping_top_k
            if top_k is None
            else int(top_k)
        )
        candidate_count = max(
            1,
            min(
                candidate_count,
                len(self.trained_patterns),
            ),
        )

        ranked_candidates = []

        for trained_pattern in self.trained_patterns:
            distance = self._calculate_levenshtein(
                unseen_pattern,
                trained_pattern,
            )
            ranked_candidates.append(
                (
                    distance,
                    -self.pattern_counts[
                        trained_pattern
                    ],
                    trained_pattern,
                )
            )

        ranked_candidates.sort()
        selected = ranked_candidates[
            :candidate_count
        ]

        unseen_length = max(
            len(str(unseen_pattern)),
            1,
        )

        raw_weights = [
            math.exp(
                -self.mapping_gamma
                * (distance / unseen_length)
            )
            for distance, _, _ in selected
        ]
        weight_sum = math.fsum(raw_weights)

        if weight_sum <= 0.0:
            normalized_weights = [
                1.0 / len(selected)
                for _ in selected
            ]
        else:
            normalized_weights = [
                weight / weight_sum
                for weight in raw_weights
            ]

        return [
            {
                "pattern": trained_pattern,
                "distance": int(distance),
                "normalized_distance": float(
                    distance / unseen_length
                ),
                "weight": float(weight),
            }
            for (
                distance,
                _,
                trained_pattern,
            ), weight in zip(
                selected,
                normalized_weights,
            )
        ]
    #tek komsu api sini korur
    def _find_nearest_pattern(self, unseen_pattern):
        candidate = self._find_nearest_patterns(
            unseen_pattern,
            top_k=1,
        )[0]

        return (
            candidate["pattern"],
            candidate["distance"],
        )

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
            "transition_density": transition_density,
            "mapping_top_k": int(self.mapping_top_k),
            "mapping_gamma": float(self.mapping_gamma),
        }
    #hangi state in hangi merged state e atandıgını verir
    def get_state_mapping(self):
        return dict(self.state_mapping)
    #herbir temsilcinin altında hangi state lerin toplandıgını gruplar halınde verır
    def get_merged_members(self):
        return {
            representative: sorted(members)
            for representative, members
            in self.merged_members.items()
        }