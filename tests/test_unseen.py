import unittest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.automata_model import ProbabilisticAutomata

class TestUnseenPatternManagement(unittest.TestCase):
    """
    Olasılıksal Otomata modelinin Levenshtein algoritmasını ve 
    Unseen Pattern yönetimini doğrulayan birim testler.
    """
    def setUp(self):
        self.automata = ProbabilisticAutomata(smoothing=True)
        self.train_patterns = ["aab", "abc", "bcc", "ccd"]
        self.automata.fit(self.train_patterns)

    def test_levenshtein_distance_exact_match(self):
        dist = self.automata._calculate_levenshtein("abc", "abc")
        self.assertEqual(dist, 0)

    def test_levenshtein_distance_substitution(self):
        dist = self.automata._calculate_levenshtein("abc", "adc")
        self.assertEqual(dist, 1)

    def test_levenshtein_distance_insertion_deletion(self):
        dist = self.automata._calculate_levenshtein("ab", "abc")
        self.assertEqual(dist, 1)

    def test_unseen_pattern_mapping(self):
        unseen_pattern = "ade"
        nearest_pattern, distance = self.automata._find_nearest_pattern(unseen_pattern)
        
        # "ade"ye hem "aab" hem "abc" 2 birim uzaklıktadır. Alfabetik olarak "aab" seçilir.
        self.assertEqual(nearest_pattern, "aab")
        self.assertEqual(distance, 2)

    def test_predict_status_and_distance_output(self):
        test_patterns = ["aab", "ade"]
        _, explainability_logs = self.automata.predict(test_patterns)
        target_log = explainability_logs[0]
        
        self.assertEqual(target_log["status"], "unseen")
        self.assertEqual(target_log["mapped_to"], "aab")
        self.assertEqual(target_log["distance"], 2)

if __name__ == '__main__':
    unittest.main()