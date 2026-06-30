import unittest
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.explainability import AutomataExplainer


class TestAutomataExplainer(unittest.TestCase):

    def _make_log(self, status="seen", decision="normal", distance=0,
                  mapped_to="abc", counterfactuals=None, similarity_report=None):
        """Yardımcı: standart parametrelerle log üretir."""
        return AutomataExplainer.generate_log(
            t=1,
            current_state="aab",
            incoming_pattern="abc",
            status=status,
            mapped_to=mapped_to,
            distance=distance,
            prob=0.5,
            cumulative_path_prob=0.5,
            decision=decision,
            confidence_score=0.5,
            transition_history=["aab", "abc"],
            total_exits=10,
            counterfactuals=counterfactuals,
            similarity_report=similarity_report,
        )

    def test_log_contains_required_keys(self):
        """1. DURUM: Log çıktısı zorunlu tüm alanları içermeli."""
        log = self._make_log()
        required_keys = [
            "time_step", "status", "mapped_to", "distance",
            "transition_probability", "cumulative_path_probability",
            "decision", "confidence_score", "decision_reason",
            "transition_history", "counterfactual_analysis", "similarity_analysis"
        ]
        for key in required_keys:
            self.assertIn(key, log)

    def test_decision_reason_normal(self):
        """2. DURUM: Normal kararda decision_reason 'normal' ifadesini içermeli."""
        log = self._make_log(decision="normal")
        self.assertIn("normal", log["decision_reason"])

    def test_decision_reason_anomaly(self):
        """3. DURUM: Anomali kararda decision_reason 'anomali' ifadesini içermeli."""
        log = self._make_log(decision="anomaly")
        self.assertIn("anomali", log["decision_reason"])

    def test_unseen_status_adds_levenshtein_info(self):
        """4. DURUM: Unseen pattern durumunda decision_reason Levenshtein eşleşmesini belirtmeli."""
        log = self._make_log(status="unseen", mapped_to="abc", distance=1)
        self.assertIn("Levenshtein", log["decision_reason"])
        self.assertIn("abc", log["decision_reason"])

    def test_seen_status_no_levenshtein_info(self):
        """5. DURUM: Seen pattern durumunda decision_reason Levenshtein içermemeli."""
        log = self._make_log(status="seen")
        self.assertNotIn("Levenshtein", log["decision_reason"])

    def test_counterfactual_none_becomes_empty_list(self):
        """6. DURUM: counterfactuals=None geçildiğinde alan boş liste olmalı."""
        log = self._make_log(counterfactuals=None)
        self.assertEqual(log["counterfactual_analysis"], [])

    def test_similarity_none_becomes_empty_list(self):
        """7. DURUM: similarity_report=None geçildiğinde alan boş liste olmalı."""
        log = self._make_log(similarity_report=None)
        self.assertEqual(log["similarity_analysis"], [])

    def test_counterfactual_passed_correctly(self):
        """8. DURUM: Geçilen counterfactual listesi log'a aynen yansımalı."""
        cf = [{"pattern": "aab", "probability": 0.7, "would_be_anomaly": False}]
        log = self._make_log(counterfactuals=cf)
        self.assertEqual(log["counterfactual_analysis"], cf)

    def test_similarity_passed_correctly(self):
        """9. DURUM: Geçilen similarity listesi log'a aynen yansımalı."""
        sim = [{"pattern": "abc", "distance": 1}]
        log = self._make_log(similarity_report=sim)
        self.assertEqual(log["similarity_analysis"], sim)

    def test_transition_history_is_list(self):
        """10. DURUM: transition_history alanı liste tipinde olmalı."""
        log = self._make_log()
        self.assertIsInstance(log["transition_history"], list)

    def test_time_step_recorded_correctly(self):
        """11. DURUM: time_step alanı generate_log'a geçilen t değerine eşit olmalı."""
        log = self._make_log()
        self.assertEqual(log["time_step"], 1)

    def test_probability_values_are_float(self):
        """12. DURUM: transition_probability ve confidence_score float tipinde olmalı."""
        log = self._make_log()
        self.assertIsInstance(log["transition_probability"], float)
        self.assertIsInstance(log["confidence_score"], float)


if __name__ == '__main__':
    unittest.main()