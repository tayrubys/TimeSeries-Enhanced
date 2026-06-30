import unittest
import numpy as np
from src.models.deep_model import build_lstm_model, build_gru_model
from src.config import get_dl_config


class TestDeepModelArchitecture(unittest.TestCase):

    def setUp(self):
        self.batadal_input_shape = (20, 43)
        self.skab_input_shape = (20, 8)
        self.cfg = get_dl_config()

    def test_lstm_output_shape_batadal(self):
        model = build_lstm_model(input_shape=self.batadal_input_shape)
        self.assertEqual(model.output_shape, (None, 1))

    def test_gru_output_shape_batadal(self):
        model = build_gru_model(input_shape=self.batadal_input_shape)
        self.assertEqual(model.output_shape, (None, 1))

    def test_lstm_output_shape_skab(self):
        model = build_lstm_model(input_shape=self.skab_input_shape)
        self.assertEqual(model.output_shape, (None, 1))

    def test_gru_output_shape_skab(self):
        model = build_gru_model(input_shape=self.skab_input_shape)
        self.assertEqual(model.output_shape, (None, 1))

    def test_lstm_prediction_range(self):
        model = build_lstm_model(input_shape=self.skab_input_shape)
        X_dummy = np.random.rand(10, 20, 8).astype("float32")
        preds = model.predict(X_dummy)
        self.assertTrue(np.all(preds >= 0.0) and np.all(preds <= 1.0))

    def test_gru_prediction_range(self):
        model = build_gru_model(input_shape=self.skab_input_shape)
        X_dummy = np.random.rand(10, 20, 8).astype("float32")
        preds = model.predict(X_dummy)
        self.assertTrue(np.all(preds >= 0.0) and np.all(preds <= 1.0))

    def test_lstm_units_from_config(self):
        model = build_lstm_model(input_shape=self.skab_input_shape)
        self.assertEqual(model.layers[0].units, self.cfg["lstm_units"])

    def test_gru_units_from_config(self):
        model = build_gru_model(input_shape=self.skab_input_shape)
        self.assertEqual(model.layers[0].units, self.cfg["gru_units"])


if __name__ == "__main__":
    unittest.main()