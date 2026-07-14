import math
from collections import defaultdict


class LikelihoodRatioAutomata:

    def __init__(self, smoothing_alpha=1.0):
        if smoothing_alpha <= 0:
            raise ValueError(
                "smoothing_alpha sıfırdan büyük olmalıdır."
            )

        self.smoothing_alpha = float(smoothing_alpha)

        #normal etiketli geçişler
        self.normal_transitions = defaultdict(
            lambda: defaultdict(int)
        )
        self.normal_total_exits = defaultdict(int)

        #anomali etiketli geçişler
        self.anomaly_transitions = defaultdict(
            lambda: defaultdict(int)
        )
        self.anomaly_total_exits = defaultdict(int)

        self.trained_patterns = set()

        self.normal_transition_count = 0
        self.anomaly_transition_count = 0

        self.is_fitted = False

    #sınıfa özel gecis tabloları olusturur
    def fit(self, train_patterns, transition_labels):

        train_patterns = list(train_patterns)
        transition_labels = list(transition_labels)

        if len(train_patterns) < 2:
            raise ValueError(
                "Otomata eğitimi için en az 2 pattern gereklidir."
            )

        expected_label_count = len(train_patterns) - 1

        if len(transition_labels) != expected_label_count:
            raise ValueError(
                "Geçiş etiketi sayısı pattern sayısından bir eksik "
                "olmalıdır. "
                f"Pattern={len(train_patterns)}, "
                f"etiket={len(transition_labels)}, "
                f"beklenen={expected_label_count}"
            )

        # Model tekrar fit edilirse eski sayımlar kalmasın.
        self.normal_transitions.clear()
        self.normal_total_exits.clear()
        self.anomaly_transitions.clear()
        self.anomaly_total_exits.clear()

        self.normal_transition_count = 0
        self.anomaly_transition_count = 0

        self.trained_patterns = set(train_patterns)

        for index in range(len(train_patterns) - 1):
            current_state = train_patterns[index]
            next_state = train_patterns[index + 1]

            label = 1 if transition_labels[index] > 0 else 0

            if label == 1:
                self.anomaly_transitions[current_state][next_state] += 1
                self.anomaly_total_exits[current_state] += 1
                self.anomaly_transition_count += 1

            else:
                self.normal_transitions[current_state][next_state] += 1
                self.normal_total_exits[current_state] += 1
                self.normal_transition_count += 1

        if self.normal_transition_count == 0:
            raise ValueError(
                "Normal etiketli eğitim geçişi bulunamadı."
            )

        if self.anomaly_transition_count == 0:
            raise ValueError(
                "Anomali etiketli eğitim geçişi bulunamadı."
            )

        self.is_fitted = True
    
    #belirtilen sinif icin llaplace smoothing uygulanmıs gecis olasılıgını hesaplar
    def get_transition_probability(
        self,
        current_state,
        next_state,
        class_label
    ):

        if not self.is_fitted:
            raise RuntimeError(
                "Olasılık hesaplamadan önce model eğitilmelidir."
            )

        if class_label == 0:
            transitions = self.normal_transitions
            total_exits = self.normal_total_exits

        elif class_label == 1:
            transitions = self.anomaly_transitions
            total_exits = self.anomaly_total_exits

        else:
            raise ValueError(
                "class_label yalnızca 0 veya 1 olabilir."
            )

        transition_count = (
            transitions
            .get(current_state, {})
            .get(next_state, 0)
        )

        total_output = total_exits.get(current_state, 0)

        #artı bir unseen pattern'lar için ek destek elemanı
        vocabulary_size = max(
            1,
            len(self.trained_patterns) + 1
        )

        numerator = transition_count + self.smoothing_alpha

        denominator = (
            total_output
            + self.smoothing_alpha * vocabulary_size
        )

        return numerator / denominator

    def score_transition(self, current_state, next_state):
        """
        Tek bir geçişin normal ve anomali olasılıklarını
        karşılaştırır.
        """

        p_normal = self.get_transition_probability(
            current_state,
            next_state,
            class_label=0
        )

        p_anomaly = self.get_transition_probability(
            current_state,
            next_state,
            class_label=1
        )

        likelihood_ratio_score = math.log(
            p_anomaly / p_normal
        )

        return {
            "current_state": current_state,
            "next_state": next_state,
            "p_normal": float(p_normal),
            "p_anomaly": float(p_anomaly),
            "likelihood_ratio_score": float(
                likelihood_ratio_score
            ),
            "current_seen": (
                current_state in self.trained_patterns
            ),
            "next_seen": (
                next_state in self.trained_patterns
            )
        }
    #test pattern dizsindeki butun gecisler ıcın likelihood-ratio skoru uretir
    def score(self, test_patterns):

        test_patterns = list(test_patterns)

        if len(test_patterns) < 2:
            return [], []

        scores = []
        details = []

        for index in range(len(test_patterns) - 1):
            current_state = test_patterns[index]
            next_state = test_patterns[index + 1]

            transition_result = self.score_transition(
                current_state,
                next_state
            )

            scores.append(
                transition_result["likelihood_ratio_score"]
            )
            details.append(transition_result)

        return scores, details
    #likelihood-ratio skoru threshold değerine eşit veya buykse anomly=1 değeri uret
    def predict(self, test_patterns, score_threshold=0.0):

        scores, details = self.score(test_patterns)

        predictions = [
            1 if score >= score_threshold else 0
            for score in scores
        ]

        return predictions, scores, details