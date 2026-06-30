import unittest
import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_pipeline.preprocess import (
    split_features_target,
    check_missing_values,
    scale_train_val_test,
    apply_pca_train_val_test,
    split_batadal_time_ordered,
)


class TestPreprocess(unittest.TestCase):

    def _make_df(self, n=100, n_features=4, anomaly_rate=0.1):
        """Yardımcı: sentetik DataFrame üretir."""
        np.random.seed(42)
        data = {f"feature_{i}": np.random.randn(n) for i in range(n_features)}
        data["anomaly"] = (np.random.rand(n) < anomaly_rate).astype(int)
        return pd.DataFrame(data)

    # --- split_features_target ---

    def test_split_features_target_shapes(self):
        """1. DURUM: X ve y doğru boyutlarda ayrılmalı."""
        df = self._make_df()
        feature_cols = [c for c in df.columns if c != "anomaly"]
        X, y = split_features_target(df, feature_cols, "anomaly")
        self.assertEqual(X.shape, (100, 4))
        self.assertEqual(y.shape, (100,))

    def test_split_target_not_in_features(self):
        """2. DURUM: Hedef sütun X'te bulunmamalı."""
        df = self._make_df()
        feature_cols = [c for c in df.columns if c != "anomaly"]
        X, _ = split_features_target(df, feature_cols, "anomaly")
        self.assertNotIn("anomaly", X.columns)

    def test_split_does_not_modify_original(self):
        """3. DURUM: split_features_target orijinal DataFrame'i değiştirmemeli."""
        df = self._make_df()
        original_cols = list(df.columns)
        feature_cols = [c for c in df.columns if c != "anomaly"]
        split_features_target(df, feature_cols, "anomaly")
        self.assertEqual(list(df.columns), original_cols)

    # --- check_missing_values ---

    def test_no_missing_values(self):
        """4. DURUM: Eksik değer yoksa boş Series dönmeli."""
        df = self._make_df()
        result = check_missing_values(df, "test_dataset")
        self.assertEqual(len(result), 0)

    def test_missing_values_detected(self):
        """5. DURUM: Eksik değer varsa ilgili sütun raporlanmalı."""
        df = self._make_df()
        df.loc[0, "feature_0"] = np.nan
        result = check_missing_values(df, "test_dataset")
        self.assertIn("feature_0", result.index)
        self.assertEqual(result["feature_0"], 1)

    def test_missing_count_correct(self):
        """6. DURUM: Birden fazla eksik değer doğru sayılmalı."""
        df = self._make_df()
        df.loc[[0, 1, 2], "feature_1"] = np.nan
        result = check_missing_values(df, "test_dataset")
        self.assertEqual(result["feature_1"], 3)

    # --- scale_train_val_test ---

    def test_scale_train_mean_near_zero(self):
        """7. DURUM: Ölçeklendirme sonrası train verisinin ortalaması ~0 olmalı."""
        df = self._make_df(n=150)
        feature_cols = [c for c in df.columns if c != "anomaly"]
        X = df[feature_cols]
        result = scale_train_val_test(X.iloc[:90], X.iloc[90:120], X.iloc[120:])
        train_means = result["X_train_scaled"].mean()
        for col in train_means.index:
            self.assertAlmostEqual(train_means[col], 0.0, places=5)

    def test_scale_output_keys(self):
        """8. DURUM: Çıktı dict'i gerekli anahtarları içermeli."""
        df = self._make_df(n=150)
        feature_cols = [c for c in df.columns if c != "anomaly"]
        X = df[feature_cols]
        result = scale_train_val_test(X.iloc[:90], X.iloc[90:120], X.iloc[120:])
        for key in ["X_train_scaled", "X_val_scaled", "X_test_scaled", "scaler"]:
            self.assertIn(key, result)

    def test_scale_no_data_leakage(self):
        """9. DURUM: Test verisinin mean/std'si train verisinden farklı olabilir (leakage yok)."""
        df = self._make_df(n=150)
        feature_cols = [c for c in df.columns if c != "anomaly"]
        X = df[feature_cols]
        result = scale_train_val_test(X.iloc[:90], X.iloc[90:120], X.iloc[120:])
        # Test seti train ile aynı scaler'dan geçiyor, kendi mean'i 0 olmak zorunda değil
        test_means = result["X_test_scaled"].mean()
        # En az bir sütunun ortalaması 0'dan farklı olmalı (leakage yoksa)
        self.assertFalse(all(abs(test_means) < 1e-10))

    # --- apply_pca_train_val_test ---

    def test_pca_output_shape(self):
        """10. DURUM: n_components=1 ile PCA çıktısı tek sütunlu DataFrame olmalı."""
        df = self._make_df(n=150)
        feature_cols = [c for c in df.columns if c != "anomaly"]
        X = df[feature_cols]
        result = apply_pca_train_val_test(X.iloc[:90], X.iloc[90:120], X.iloc[120:], n_components=1)
        self.assertEqual(result["X_train_pca"].shape[1], 1)
        self.assertEqual(result["X_test_pca"].shape[1], 1)

    def test_pca_column_named_pc1(self):
        """11. DURUM: PCA çıktı sütunu 'PC1' olarak adlandırılmalı."""
        df = self._make_df(n=150)
        feature_cols = [c for c in df.columns if c != "anomaly"]
        X = df[feature_cols]
        result = apply_pca_train_val_test(X.iloc[:90], X.iloc[90:120], X.iloc[120:], n_components=1)
        self.assertIn("PC1", result["X_train_pca"].columns)

    def test_pca_output_keys(self):
        """12. DURUM: Çıktı dict'i gerekli anahtarları içermeli."""
        df = self._make_df(n=150)
        feature_cols = [c for c in df.columns if c != "anomaly"]
        X = df[feature_cols]
        result = apply_pca_train_val_test(X.iloc[:90], X.iloc[90:120], X.iloc[120:])
        for key in ["X_train_pca", "X_val_pca", "X_test_pca", "pca"]:
            self.assertIn(key, result)

    # --- split_batadal_time_ordered ---

    def test_batadal_split_sizes(self):
        """13. DURUM: %60/%20/%20 bölünmesinde toplam satır sayısı korunmalı."""
        df = self._make_df(n=100)
        batadal_columns = {
            "feature_cols": [c for c in df.columns if c != "anomaly"],
            "target_col": "anomaly"
        }
        result = split_batadal_time_ordered(df, batadal_columns)
        total = len(result["X_train"]) + len(result["X_val"]) + len(result["X_test"])
        self.assertEqual(total, 100)

    def test_batadal_split_no_overlap(self):
        """14. DURUM: Train, val ve test indeksleri çakışmamalı."""
        df = self._make_df(n=100)
        batadal_columns = {
            "feature_cols": [c for c in df.columns if c != "anomaly"],
            "target_col": "anomaly"
        }
        result = split_batadal_time_ordered(df, batadal_columns)
        train_idx = set(result["X_train"].index)
        val_idx = set(result["X_val"].index)
        test_idx = set(result["X_test"].index)
        self.assertEqual(len(train_idx & val_idx), 0)
        self.assertEqual(len(train_idx & test_idx), 0)
        self.assertEqual(len(val_idx & test_idx), 0)

    def test_batadal_split_output_keys(self):
        """15. DURUM: Çıktı dict'i X_train, y_train, X_val, y_val, X_test, y_test içermeli."""
        df = self._make_df(n=100)
        batadal_columns = {
            "feature_cols": [c for c in df.columns if c != "anomaly"],
            "target_col": "anomaly"
        }
        result = split_batadal_time_ordered(df, batadal_columns)
        for key in ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"]:
            self.assertIn(key, result)


if __name__ == '__main__':
    unittest.main()