import unittest
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.automata_model import ProbabilisticAutomata

class TestLevenshteinDistance(unittest.TestCase):
    
    def test_exact_match(self):
        """1. DURUM: Birebir aynı iki sembolik durum arasındaki mesafe 0 olmalı."""
        model = ProbabilisticAutomata()
        self.assertEqual(model._calculate_levenshtein("abc", "abc"), 0)
        self.assertEqual(model._calculate_levenshtein("aaaa", "aaaa"), 0)

    def test_substitution(self):
        """2. DURUM: Sadece tek bir harfin değiştiği (yer değiştirme) senaryo."""
        model = ProbabilisticAutomata()
        self.assertEqual(model._calculate_levenshtein("abc", "adc"), 1)
        
    def test_empty_strings(self):
        """3. DURUM: Sınır Durum (Edge Case) - Boş dizgilerin yönetimi."""
        model = ProbabilisticAutomata()
        # Biri boşsa, mesafe diğer kelimenin uzunluğu kadar (silme/ekleme) olmalıdır
        self.assertEqual(model._calculate_levenshtein("", "abc"), 3)
        self.assertEqual(model._calculate_levenshtein("abc", ""), 3)
        self.assertEqual(model._calculate_levenshtein("", ""), 0)
        
    def test_different_lengths_and_insertion(self):
        """4. DURUM: Farklı uzunluktaki durumlar ve harf ekleme/silme performansı."""
        model = ProbabilisticAutomata()
        self.assertEqual(model._calculate_levenshtein("abc", "abcdef"), 3)
        self.assertEqual(model._calculate_levenshtein("abc", "zabc"), 1)

    def test_complex_transformation(self):
        """5. DURUM: Karmaşık harf değişimleri ve dinamik programlama doğrulaması."""
        model = ProbabilisticAutomata()
        self.assertEqual(model._calculate_levenshtein("kitten", "sitting"), 3)

if __name__ == '__main__':
    unittest.main()