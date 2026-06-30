import unittest
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.automata_model import ProbabilisticAutomata

class TestAutomataCore(unittest.TestCase):
    
    def test_fit_and_states(self):
        """1. DURUM: Modelin doğru eğitildiğini ve benzersiz durum (state) sayısını doğrular."""
        model = ProbabilisticAutomata(smoothing=False)
        model.fit(["abc", "bcc", "abc"])
        self.assertIn("abc", model.trained_patterns)
        self.assertEqual(len(model.trained_patterns), 2)

    def test_transition_probability_without_smoothing(self):
        """2. DURUM: Yumuşatma kapalıyken (smoothing=False) saf frekans tabanlı olasılık hesabı."""
        model = ProbabilisticAutomata(smoothing=False)
        model.fit(["S0", "S1", "S0", "S2"])
        
        self.assertEqual(model.get_transition_probability("S0", "S1"), 0.5)
        self.assertEqual(model.get_transition_probability("S0", "S2"), 0.5)
        self.assertEqual(model.get_transition_probability("S0", "S3"), 0.0)

    def test_laplace_smoothing_behavior(self):
        """3. DURUM: Yumuşatma açıkken (smoothing=True) görünmeyen geçişlerin olasılık hesabı (Laplace)."""
        model = ProbabilisticAutomata(smoothing=True)
        model.fit(["S0", "S1", "S0"]) 
        
        prob_seen = model.get_transition_probability("S0", "S1")
        self.assertAlmostEqual(prob_seen, 2.0 / 3.0)
        
        prob_unseen = model.get_transition_probability("S0", "S0")
        self.assertAlmostEqual(prob_unseen, 1.0 / 3.0)

    def test_anomaly_detection_and_predictions(self):
        """4. DURUM: Eşik değerine göre anomali kararlarının doğruluğunu test eder."""
        model = ProbabilisticAutomata(smoothing=False)
        model.fit(["S0", "S1", "S0", "S1", "S0", "S1", "S0", "S1", "S0"])   
        test_sequence = ["S0", "S2"]
        predictions, logs = model.predict(test_sequence, anomaly_threshold=0.05)
        
        self.assertEqual(predictions[0], 1) 
        self.assertEqual(logs[0]["decision"], "anomaly")

    def test_probability_bounds(self):
        """5. DURUM: Hesaplanan tüm olasılıkların [0.0, 1.0] sınırları içinde kaldığını güvenceye alır."""
        model = ProbabilisticAutomata(smoothing=True)
        model.fit(["abc", "bcc", "abc", "xyz"])
        
        for s1 in model.trained_patterns:
            for s2 in model.trained_patterns:
                prob = model.get_transition_probability(s1, s2)
                self.assertTrue(0.0 <= prob <= 1.0)

    def test_unseen_pattern_mapped_to_nearest(self):
        """6. DURUM: Unseen pattern geldiğinde Levenshtein ile en yakın eğitim pattern'ına
        doğru şekilde eşleştirildiğini ve predict akışının çökmeden devam ettiğini doğrular."""
        model = ProbabilisticAutomata(smoothing=False)
        model.fit(["abc", "bcc", "abc", "bcc"])

        nearest, distance = model._find_nearest_pattern("axc")
        self.assertEqual(nearest, "abc",
            msg="Unseen 'axc' pattern'ı en yakın eğitim pattern'ı olan 'abc'ye eşleşmeli.")
        self.assertEqual(distance, 1,
            msg="'axc' ile 'abc' arasındaki Levenshtein mesafesi 1 olmalı.")

        predictions, logs = model.predict(["abc", "axc"], anomaly_threshold=0.05)

        self.assertEqual(len(predictions), 1,
            msg="Bir adım için bir tahmin üretilmeli.")
        self.assertEqual(logs[0]["status"], "unseen",
            msg="Log'da pattern 'unseen' olarak işaretlenmeli.")
        self.assertEqual(logs[0]["mapped_to"], "abc",
            msg="Unseen pattern log'da doğru en yakın pattern'a eşlenmiş olmalı.")

    def test_unseen_pattern_does_not_crash_on_full_sequence(self):
        """7. DURUM: Ardışık birden fazla unseen pattern içeren dizide model
        hiç exception fırlatmadan tüm tahminleri tamamlamalıdır.
        """
        model = ProbabilisticAutomata(smoothing=True)
        model.fit(["aaa", "bbb", "aaa", "bbb", "aaa"])

        test_sequence = ["aaa", "zzz", "qqq", "bbb"]

        try:
            predictions, logs = model.predict(test_sequence, anomaly_threshold=0.05)
        except Exception as e:
            self.fail(f"Unseen pattern içeren dizide model exception fırlattı: {e}")

        self.assertEqual(len(predictions), 3,
            msg="3 adım için 3 tahmin üretilmeli.")

        unseen_logs = [log for log in logs if log["status"] == "unseen"]
        self.assertEqual(len(unseen_logs), 2,
            msg="'zzz' ve 'qqq' olmak üzere 2 unseen adım loglanmalı.")

        for log in unseen_logs:
            self.assertIn(log["mapped_to"], model.trained_patterns,
                msg="Unseen pattern her zaman bilinen bir eğitim pattern'ına eşleşmeli.")

if __name__ == '__main__':
    unittest.main()