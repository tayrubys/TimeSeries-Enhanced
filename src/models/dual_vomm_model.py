import numpy as np

from src.models.pst_model import ProbabilisticSuffixTree


class DualVariableOrderMarkovModel:
    def __init__(
        self,
        max_depth=3,
        min_count=2,
        smoothing=True,
        smoothing_alpha=1.0
    ):
        self.max_depth = max_depth
        self.min_count = min_count
        self.smoothing = smoothing
        self.smoothing_alpha = smoothing_alpha

        #normal geçişleri öğrenen pst
        self.normal_pst = ProbabilisticSuffixTree(
            max_depth=max_depth,
            min_count=min_count,
            smoothing=smoothing,
            smoothing_alpha=smoothing_alpha
        )

        #anomalili geçişleri öğrenen pst
        self.anomaly_pst = ProbabilisticSuffixTree(
            max_depth=max_depth,
            min_count=min_count,
            smoothing=smoothing,
            smoothing_alpha=smoothing_alpha
        )

        self._trained_patterns = set()
        self.normal_transition_count = 0
        self.anomaly_transition_count = 0

    def fit(self, train_patterns, train_pattern_labels):
        if len(train_patterns) < 2:
            raise ValueError("Dual VOMM egitimi icin en az 2 pattern gereklidir.")

        self._trained_patterns = set(train_patterns)

        #iki pst nin de aynı vocabulary bilgisini kullanması gerekiyor cunkunormal ve anomaly olasılıkları daha adil karşılaştırmak için
        shared_vocabulary = set(train_patterns)
        self.normal_pst.vocabulary = shared_vocabulary.copy()
        self.anomaly_pst.vocabulary = shared_vocabulary.copy()

        # train_pattern_labels uzunluğu genelde len(train_patterns)-1 cunku ilk pattern için geçmiş geçiş yoktur.
        max_len = min(len(train_pattern_labels), len(train_patterns) - 1)

        for i in range(1, max_len + 1):
            start_idx = max(0, i - self.max_depth)
            history = train_patterns[start_idx:i]
            target = train_patterns[i]

            label = train_pattern_labels[i - 1]

            #eğer hedef pattern normal ise normal pst ye ekle
            if label == 0:
                self.normal_pst.add_observation(history, target)
                self.normal_transition_count += 1

            #eğer hedef pattern anomalili ise anomaly pst ye ekle
            else:
                self.anomaly_pst.add_observation(history, target)
                self.anomaly_transition_count += 1

    def predict(self, test_patterns, score_threshold=0.0):
        predictions = []
        explainability_logs = []

        eps = 1e-12

        for i in range(1, len(test_patterns)):
            start_idx = max(0, i - self.max_depth)
            context = test_patterns[start_idx:i]
            target = test_patterns[i]

            #aynı geçişin normal ve anomaly pst altında olasılıklarını hesapla
            p_normal = self.normal_pst.predict_probability(context, target)
            p_anomaly = self.anomaly_pst.predict_probability(context, target)

            #pozitifse geçiş anomaly modele daha yakın,
            #negatifse normal modele daha yakın kabul et
            score = np.log(p_anomaly + eps) - np.log(p_normal + eps)

            decision = 1 if score > score_threshold else 0
            predictions.append(decision)

            explainability_logs.append({
                "index": i,
                "context": list(context),
                "target": target,
                "p_normal": float(p_normal),
                "p_anomaly": float(p_anomaly),
                "score": float(score),
                "score_threshold": float(score_threshold),
                "prediction": decision
            })

        return predictions, explainability_logs

    @property
    def trained_patterns(self):
        return self._trained_patterns