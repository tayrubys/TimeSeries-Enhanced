from collections import Counter


class SAXStateAggregator:
    """
    Benzer SAX pattern'lerini ortak prototip durumlarda birleştirir.

    Prototipler yalnızca training pattern'lerinden öğrenilir.
    Validation ve test pattern'leri bu training prototiplerine eşlenir.
    """

    def __init__(self, merge_distance=1):
        if merge_distance < 0:
            raise ValueError(
                "merge_distance değeri 0 veya daha büyük olmalıdır."
            )

        self.merge_distance = int(merge_distance)

        self.prototypes = []
        self.pattern_to_prototype = {}

        self.original_state_count = 0
        self.prototype_state_count = 0

        self._is_fitted = False

    @staticmethod
    def _hamming_distance(pattern_1, pattern_2):
        """
        Aynı uzunluktaki iki SAX pattern'i arasındaki
        farklı karakter sayısını hesaplar.
        """

        if len(pattern_1) != len(pattern_2):
            raise ValueError(
                "Hamming uzaklığı için pattern uzunlukları "
                "aynı olmalıdır: "
                f"{pattern_1} ({len(pattern_1)}) ve "
                f"{pattern_2} ({len(pattern_2)})"
            )

        return sum(
            char_1 != char_2
            for char_1, char_2 in zip(
                pattern_1,
                pattern_2
            )
        )

    def _find_nearest_prototype(self, pattern):
        """
        Verilen pattern'e en yakın training prototipini bulur.
        """

        if not self.prototypes:
            return None, float("inf")

        nearest_prototype = None
        nearest_distance = float("inf")

        for prototype in self.prototypes:
            distance = self._hamming_distance(
                pattern,
                prototype
            )

            # Daha yakın prototip bulunduysa güncelle.
            # Eşit uzaklıkta alfabetik olarak küçük olan seçilir.
            if (
                distance < nearest_distance
                or (
                    distance == nearest_distance
                    and (
                        nearest_prototype is None
                        or prototype < nearest_prototype
                    )
                )
            ):
                nearest_prototype = prototype
                nearest_distance = distance

        return nearest_prototype, nearest_distance

    def fit(self, train_patterns):
        """
        Training pattern'lerinden prototip durumları öğrenir.

        Sık görülen pattern'ler önce işlendiği için,
        prototiplerin yaygın training davranışlarını temsil etmesi
        amaçlanır.
        """

        if len(train_patterns) == 0:
            raise ValueError(
                "State aggregation eğitimi için pattern gereklidir."
            )

        pattern_counts = Counter(train_patterns)

        # En sık görülen pattern önce işlenir.
        # Eşit frekansta alfabetik sıra kullanılır.
        ordered_patterns = sorted(
            pattern_counts.items(),
            key=lambda item: (-item[1], item[0])
        )

        self.prototypes = []
        self.pattern_to_prototype = {}

        for pattern, _ in ordered_patterns:
            if not self.prototypes:
                self.prototypes.append(pattern)
                self.pattern_to_prototype[pattern] = pattern
                continue

            nearest_prototype, nearest_distance = (
                self._find_nearest_prototype(pattern)
            )

            if nearest_distance <= self.merge_distance:
                # Yeterince benzer pattern mevcut prototipe bağlanır.
                self.pattern_to_prototype[
                    pattern
                ] = nearest_prototype
            else:
                # Yeterince yakın prototip yoksa yeni durum oluşturulur.
                self.prototypes.append(pattern)
                self.pattern_to_prototype[pattern] = pattern

        self.original_state_count = len(
            pattern_counts
        )

        self.prototype_state_count = len(
            self.prototypes
        )

        self._is_fitted = True

        return self

    def transform(self, patterns):
        """
        Pattern dizisini öğrenilen prototip durumlara dönüştürür.

        Training sırasında görülmeyen bir pattern:
        - Yakın prototip varsa o prototipe eşlenir.
        - Yakın prototip yoksa değiştirilmeden bırakılır.

        Uzak pattern'in korunması, güçlü anomali sinyalinin
        normal bir prototip altında kaybolmasını önler.
        """

        if not self._is_fitted:
            raise ValueError(
                "transform çağrısından önce fit çalıştırılmalıdır."
            )

        transformed_patterns = []
        mapping_logs = []

        for pattern in patterns:
            # Training sırasında görülen pattern'in eşlemesi hazırdır.
            if pattern in self.pattern_to_prototype:
                mapped_pattern = self.pattern_to_prototype[
                    pattern
                ]

                distance = self._hamming_distance(
                    pattern,
                    mapped_pattern
                )

                was_mapped = mapped_pattern != pattern
                status = (
                    "train_aggregated"
                    if was_mapped
                    else "train_prototype"
                )

            else:
                nearest_prototype, distance = (
                    self._find_nearest_prototype(pattern)
                )

                if (
                    nearest_prototype is not None
                    and distance <= self.merge_distance
                ):
                    mapped_pattern = nearest_prototype
                    was_mapped = True
                    status = "unseen_mapped"
                else:
                    # Fazla uzak pattern değiştirilmeden korunur.
                    mapped_pattern = pattern
                    was_mapped = False
                    status = "unseen_preserved"

            transformed_patterns.append(
                mapped_pattern
            )

            mapping_logs.append({
                "original_pattern": pattern,
                "mapped_pattern": mapped_pattern,
                "distance": (
                    int(distance)
                    if distance != float("inf")
                    else None
                ),
                "was_mapped": was_mapped,
                "status": status
            })

        return transformed_patterns, mapping_logs

    def fit_transform(self, train_patterns):
        """
        Training pattern'lerinden prototipleri öğrenir ve
        training dizisini prototip durumlara dönüştürür.
        """

        self.fit(train_patterns)
        return self.transform(train_patterns)

    @property
    def state_reduction_ratio(self):
        """
        Durum sayısındaki oransal azalmayı döndürür.
        """

        if self.original_state_count == 0:
            return 0.0

        return 1.0 - (
            self.prototype_state_count
            / self.original_state_count
        )

    @property
    def model_stats(self):
        return {
            "merge_distance": self.merge_distance,
            "original_state_count": self.original_state_count,
            "prototype_state_count": self.prototype_state_count,
            "state_reduction_ratio": self.state_reduction_ratio
        }