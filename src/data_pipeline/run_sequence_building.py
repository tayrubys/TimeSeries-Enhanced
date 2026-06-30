import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.config import load_config
from src.data_pipeline.sequence_builder import (
    build_and_save_batadal_sequences,
    build_and_save_skab_sequences,
)


config = load_config()
DL_CONFIG = config["deep_learning"]

WINDOW_SIZE = DL_CONFIG["sequence_window_size"]
N_FOLDS = DL_CONFIG["n_folds"]


build_and_save_batadal_sequences(
    processed_dir="data/processed",
    window_size=WINDOW_SIZE
)

build_and_save_skab_sequences(
    processed_dir="data/processed",
    window_size=WINDOW_SIZE,
    n_folds=N_FOLDS
)