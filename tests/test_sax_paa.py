import unittest
import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_pipeline.sax_paa import SaxPaaTransformer


class TestSaxPaaTransformer(unittest.TestCase):

    def setUp(self):
        self.transformer = SaxPaaTransformer(alphabet_size=3)

    # --- apply_paa ---

    def test_paa_output_length(self):
        """1. DURUM: PAA çıktısı, serinin pencere boyutuna bölümü kadar segment üretmeli."""
        series = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        result = self.transformer.apply_paa(series, window_size=2)
        self.assertEqual(len(result), 3)

    def test_paa_segment_averages(self):
        """2. DURUM: Her PAA segmenti kendi penceresinin ortalamasına eşit olmalı."""
        series = np.array([1.0, 3.0, 5.0, 7.0])
        result = self.transformer.apply_paa(series, window_size=2)
        np.testing.assert_array_almost_equal(result, [2.0, 6.0])

    def test_paa_trims_remainder(self):
        """3. DURUM: Seri uzunluğu pencereye tam bölünmüyorsa kalan kırpılmalı."""
        series = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # 5 eleman, window=2 → 4 kullanılır
        result = self.transformer.apply_paa(series, window_size=2)
        self.assertEqual(len(result), 2)

    # --- convert_to_sax ---

    def test_sax_output_length_matches_paa(self):
        """4. DURUM: SAX çıktısının uzunluğu PAA çıktısının uzunluğuna eşit olmalı."""
        paa_signal = np.array([-1.5, 0.0, 1.5])
        result = self.transformer.convert_to_sax(paa_signal)
        self.assertEqual(len(result), 3)

    def test_sax_output_only_valid_letters(self):
        """5. DURUM: SAX çıktısındaki tüm harfler geçerli alfabe harfleri olmalı."""
        paa_signal = np.array([-2.0, -0.5, 0.5, 2.0])
        result = self.transformer.convert_to_sax(paa_signal)
        valid_letters = set(self.transformer.alphabet)
        for letter in result:
            self.assertIn(letter, valid_letters)

    def test_sax_low_value_maps_to_first_letter(self):
        """6. DURUM: Çok düşük değer alfabenin ilk harfine ('a') eşleşmeli."""
        result = self.transformer.convert_to_sax(np.array([-10.0]))
        self.assertEqual(result[0], 'a')

    def test_sax_high_value_maps_to_last_letter(self):
        """7. DURUM: Çok yüksek değer alfabenin son harfine eşleşmeli."""
        result = self.transformer.convert_to_sax(np.array([10.0]))
        last_letter = self.transformer.alphabet[-1]
        self.assertEqual(result[0], last_letter)

    # --- get_sliding_windows ---

    def test_sliding_window_count(self):
        """8. DURUM: Kayan pencere sayısı len(dizi) - window_size + 1 olmalı."""
        sax_seq = ['a', 'b', 'c', 'd', 'e']
        result = self.transformer.get_sliding_windows(sax_seq, window_size=3)
        self.assertEqual(len(result), 3)  # 5 - 3 + 1 = 3

    def test_sliding_window_pattern_content(self):
        """9. DURUM: İlk ve son pattern'ların içeriği doğru olmalı."""
        sax_seq = ['a', 'b', 'c', 'd']
        result = self.transformer.get_sliding_windows(sax_seq, window_size=2)
        self.assertEqual(result[0], 'ab')
        self.assertEqual(result[-1], 'cd')

    def test_sliding_window_single_window(self):
        """10. DURUM: Dizi uzunluğu = window_size ise tek bir pattern üretilmeli."""
        sax_seq = ['a', 'b', 'c']
        result = self.transformer.get_sliding_windows(sax_seq, window_size=3)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], 'abc')

    # --- transform (pipeline) ---

    def test_transform_returns_list_of_strings(self):
        """11. DURUM: transform() string listesi döndürmeli."""
        series = np.random.randn(100)
        result = self.transformer.transform(series, window_size=4)
        self.assertIsInstance(result, list)
        for pattern in result:
            self.assertIsInstance(pattern, str)

    def test_transform_pattern_length_equals_window_size(self):
        """12. DURUM: Her pattern'ın uzunluğu window_size'a eşit olmalı."""
        series = np.random.randn(100)
        window_size = 4
        result = self.transformer.transform(series, window_size=window_size)
        for pattern in result:
            self.assertEqual(len(pattern), window_size)

    def test_transform_constant_series_single_letter(self):
        """13. DURUM: Sabit seriden üretilen tüm pattern'lar aynı harften oluşmalı."""
        series = np.ones(100)
        result = self.transformer.transform(series, window_size=4)
        for pattern in result:
            self.assertEqual(len(set(pattern)), 1)


if __name__ == '__main__':
    unittest.main()