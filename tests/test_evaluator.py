import unittest
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.experiments.evaluator import evaluate_binary_classification, calculate_metrics


class TestEvaluator(unittest.TestCase):

    # --- evaluate_binary_classification ---

    def test_perfect_predictions(self):
        """1. DURUM: Tüm tahminler doğruysa tüm metrikler 1.0 olmalı."""
        y_true = [0, 0, 1, 1]
        y_pred = [0, 0, 1, 1]
        metrics = evaluate_binary_classification(y_true, y_pred)
        self.assertAlmostEqual(metrics['accuracy'], 1.0)
        self.assertAlmostEqual(metrics['precision'], 1.0)
        self.assertAlmostEqual(metrics['recall'], 1.0)
        self.assertAlmostEqual(metrics['f1'], 1.0)

    def test_all_wrong_predictions(self):
        """2. DURUM: Tüm tahminler yanlışsa accuracy 0.0 olmalı."""
        y_true = [0, 0, 1, 1]
        y_pred = [1, 1, 0, 0]
        metrics = evaluate_binary_classification(y_true, y_pred)
        self.assertAlmostEqual(metrics['accuracy'], 0.0)

    def test_no_anomaly_predicted_zero_division(self):
        """3. DURUM: Hiç anomali tahmin edilmezse precision ve recall sıfır bölmeyle çökmemeli."""
        y_true = [0, 0, 1, 1]
        y_pred = [0, 0, 0, 0]
        metrics = evaluate_binary_classification(y_true, y_pred)
        self.assertAlmostEqual(metrics['precision'], 0.0)
        self.assertAlmostEqual(metrics['recall'], 0.0)

    def test_confusion_matrix_present(self):
        """4. DURUM: Çıktıda confusion_matrix alanı bulunmalı."""
        metrics = evaluate_binary_classification([0, 1], [0, 1])
        self.assertIn('confusion_matrix', metrics)

    def test_partial_correct_f1(self):
        """5. DURUM: Kısmi doğru tahminlerde F1 0 ile 1 arasında olmalı."""
        y_true = [0, 0, 1, 1, 0, 1]
        y_pred = [0, 1, 1, 1, 0, 0]
        metrics = evaluate_binary_classification(y_true, y_pred)
        self.assertGreater(metrics['f1'], 0.0)
        self.assertLess(metrics['f1'], 1.0)

    # --- calculate_metrics ---

    def test_calculate_metrics_keys(self):
        """6. DURUM: calculate_metrics dört anahtar içermeli: accuracy, precision, recall, f1_score."""
        metrics = calculate_metrics([0, 1, 1], [0, 1, 0])
        for key in ['accuracy', 'precision', 'recall', 'f1_score']:
            self.assertIn(key, metrics)

    def test_calculate_metrics_perfect(self):
        """7. DURUM: Mükemmel tahminlerde tüm metrikler 1.0 olmalı."""
        metrics = calculate_metrics([0, 0, 1, 1], [0, 0, 1, 1])
        self.assertAlmostEqual(metrics['f1_score'], 1.0)
        self.assertAlmostEqual(metrics['accuracy'], 1.0)

    def test_calculate_metrics_zero_division(self):
        """8. DURUM: Hiç pozitif tahmin yoksa sıfır bölme hatası fırlatılmamalı."""
        try:
            metrics = calculate_metrics([1, 1, 1], [0, 0, 0])
            self.assertAlmostEqual(metrics['precision'], 0.0)
        except ZeroDivisionError:
            self.fail("calculate_metrics sıfır bölme hatası fırlattı.")

    def test_calculate_metrics_values_in_range(self):
        """9. DURUM: Tüm metrik değerleri [0.0, 1.0] aralığında olmalı."""
        metrics = calculate_metrics([0, 1, 0, 1, 1], [1, 1, 0, 0, 1])
        for key in ['accuracy', 'precision', 'recall', 'f1_score']:
            self.assertGreaterEqual(metrics[key], 0.0)
            self.assertLessEqual(metrics[key], 1.0)


if __name__ == '__main__':
    unittest.main()