from src.models.pst_model import ProbabilisticSuffixTree


class VariableOrderMarkovModel:
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

        self.pst = ProbabilisticSuffixTree(
            max_depth=max_depth,
            min_count=min_count,
            smoothing=smoothing,
            smoothing_alpha=smoothing_alpha
        )

        self._trained_patterns = set()#egittimiz verilerin seti

    def fit(self, train_patterns):
        if len(train_patterns) < 2:
            raise ValueError("VOMM egitimi icin en az 2 pattern gereklidir.")

        self._trained_patterns = set(train_patterns)
        self.pst.fit(train_patterns)

    def predict(self, test_patterns, anomaly_threshold=0.05,min_context_depth=0):
        predictions = []
        explainability_logs = []

        for i in range(1, len(test_patterns)):
            start_idx = max(0, i - self.max_depth)
            context = test_patterns[start_idx:i] #gecmis adimlar
            target = test_patterns[i] #gerceklesen
            # Olasılıkla birlikte context güvenilirlik bilgisini de alıyoruz.
            prob, context_info = self.pst.predict_probability_with_context_info(
                context,
                target
            )

            context_length = context_info["context_length"]
            context_count = context_info["context_count"]

            #context reliability filter cunku sadece dusuk olasılık yeterlı değil 
            decision = 1 if (
                prob < anomaly_threshold and
                context_length >= min_context_depth
            ) else 0

            predictions.append(decision)

            explainability_logs.append({
                "index": i,
                "context": list(context),
                "target": target,
                "probability": float(prob),
                "threshold": float(anomaly_threshold),
                "context_length": context_length,
                "context_count": context_count,
                "min_context_depth": min_context_depth,
                "prediction": decision
            })

        return predictions, explainability_logs

    @property
    def trained_patterns(self):
        return self._trained_patterns
    
    @property
    def model_stats(self):
        #pst içindeki gerçek node ve geçiş sayılarını döndürür
        num_nodes, num_transitions = self.pst.count_nodes_and_transitions()

        return {
            "num_nodes": num_nodes,
            "num_transitions": num_transitions
        }

    @property
    def transitions(self):
        return {
            pattern: {}
            for pattern in self._trained_patterns
        }