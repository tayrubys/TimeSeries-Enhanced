import numpy as np
import pandas as pd
from scipy.stats import norm

class SaxPaaTransformer:
    """
    Zaman serisi verilerini PAA (Piecewise Aggregate Approximation) ve 
    SAX (Symbolic Aggregate Approximation) yöntemleriyle sembolik dizilere dönüştürür.
    """
    def __init__(self, alphabet_size=3):
        self.alphabet_size = alphabet_size
        # Gauss dağılımına göre kesim noktalarını (breakpoints) hesapla
        self.breakpoints = norm.ppf(np.linspace(0, 1, alphabet_size + 1)[1:-1])
        # Alfabeyi oluştur (a, b, c, d, e, f...)
        self.alphabet = [chr(97 + i) for i in range(alphabet_size)]

    def apply_paa(self, series, window_size):
        """
        Zaman serisine PAA uygular. Her pencerenin ortalamasını alır.
        """
        # Veri uzunluğu pencere boyutuna tam bölünmeli veya kırpılmalıdır
        n = len(series)
        remainder = n % window_size
        if remainder != 0:
            series = series[:-remainder]
            
        # Pencerelere böl ve her pencerenin ortalamasını hesapla
        reshaped = series.reshape(-1, window_size)
        return np.mean(reshaped, axis=1)

    def convert_to_sax(self, paa_signal):
        """
        PAA çıktısı olan sürekli değerleri harf dizisine (SAX) dönüştürür.
        """
        sax_sequence = []
        for value in paa_signal:
            # Değerin hangi Gauss kesim aralığına düştüğünü bul
            idx = np.digitize(value, self.breakpoints)
            sax_sequence.append(self.alphabet[idx])
        return sax_sequence

    def get_sliding_windows(self, sax_sequence, window_size):
        """
        SAX harf dizisi üzerinde kayan pencere işleterek pattern'ları çıkarır.
        """
        patterns = []
        for i in range(len(sax_sequence) - window_size + 1):
            pattern = "".join(sax_sequence[i:i + window_size])
            patterns.append(pattern)
        return patterns

    def transform(self, series, window_size):
        """
        Tüm boru hattını (Pipeline) ardışık çalıştırır.
        """
        # Veriyi normalize et (Z-score normalizasyonu SAX için şarttır)
        mean = np.mean(series)
        std = np.std(series) if np.std(series) > 0 else 1
        normalized_series = (series - mean) / std
        
        paa_signal = self.apply_paa(normalized_series, window_size)
        sax_sequence = self.convert_to_sax(paa_signal)
        patterns = self.get_sliding_windows(sax_sequence, window_size)
        return patterns