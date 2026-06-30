from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from src.config import get_dl_config


def build_lstm_model(input_shape, learning_rate=None):
    cfg = get_dl_config()
    lr = learning_rate if learning_rate is not None else cfg["learning_rate"]

    model = Sequential()
    model.add(LSTM(units=cfg["lstm_units"], input_shape=input_shape, return_sequences=False))
    model.add(Dropout(cfg["dropout_rate_1"]))
    model.add(Dense(cfg["dense_units"], activation="relu"))
    model.add(Dropout(cfg["dropout_rate_2"]))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


def build_gru_model(input_shape, learning_rate=None):
    cfg = get_dl_config()
    lr = learning_rate if learning_rate is not None else cfg["learning_rate"]

    model = Sequential()
    model.add(GRU(units=cfg["gru_units"], input_shape=input_shape, return_sequences=False))
    model.add(Dropout(cfg["dropout_rate_1"]))
    model.add(Dense(cfg["dense_units"], activation="relu"))
    model.add(Dropout(cfg["dropout_rate_2"]))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model