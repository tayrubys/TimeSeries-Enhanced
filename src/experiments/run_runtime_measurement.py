import time
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path

from src.models.deep_model import build_lstm_model, build_gru_model
from src.data_pipeline.sax_paa import SaxPaaTransformer
from src.models.automata_model import ProbabilisticAutomata
from src.config import get_dl_config
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight


def measure_deep_learning_runtime(model_type, X_train, y_train, X_val, y_val, X_test):
    cfg = get_dl_config()

    np.random.seed(42)
    tf.random.set_seed(42)

    if model_type == "LSTM":
        model = build_lstm_model(input_shape=X_train.shape[1:])
    else:
        model = build_gru_model(input_shape=X_train.shape[1:])

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=cfg["early_stopping_patience"],
        restore_best_weights=True
    )

    class_weights_array = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=y_train.astype(int)
    )
    class_weights = {0: class_weights_array[0], 1: class_weights_array[1]}

    # Training süresi
    t_start = time.time()
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        callbacks=[early_stopping],
        class_weight=class_weights,
        verbose=0
    )
    training_time = time.time() - t_start

    # Inference süresi
    t_start = time.time()
    model.predict(X_test, verbose=0)
    inference_time = time.time() - t_start

    return round(training_time, 2), round(inference_time, 4)


def measure_automata_runtime(X_train_pc1, X_test_pc1):
    from src.config import get_automata_config
    cfg = get_automata_config()

    transformer = SaxPaaTransformer(alphabet_size=cfg["alphabet_size"])

    # Training süresi
    t_start = time.time()
    train_patterns = transformer.transform(X_train_pc1, window_size=cfg["window_size"])
    model = ProbabilisticAutomata(smoothing=True)
    model.fit(train_patterns)
    training_time = time.time() - t_start

    # Inference süresi
    test_patterns = transformer.transform(X_test_pc1, window_size=cfg["window_size"])
    t_start = time.time()
    model.predict(test_patterns, anomaly_threshold=cfg["batadal_anomaly_threshold"])
    inference_time = time.time() - t_start

    return round(training_time, 4), round(inference_time, 4)


def main():
    processed_dir = "data/processed"
    results = []

    # --- BATADAL üzerinde ölç ---
    batadal_seq = Path(processed_dir) / "batadal_X_train_seq.npy"
    batadal_pc1 = Path(processed_dir) / "batadal_X_train_pc1.csv"

    if batadal_seq.exists():
        X_train = np.load(f"{processed_dir}/batadal_X_train_seq.npy").astype("float32")
        y_train = np.load(f"{processed_dir}/batadal_y_train_seq.npy").astype("float32")
        X_val   = np.load(f"{processed_dir}/batadal_X_val_seq.npy").astype("float32")
        y_val   = np.load(f"{processed_dir}/batadal_y_val_seq.npy").astype("float32")
        X_test  = np.load(f"{processed_dir}/batadal_X_test_seq.npy").astype("float32")

        for model_type in ["LSTM", "GRU"]:
            print(f"Ölçülüyor: BATADAL {model_type}...")
            train_t, infer_t = measure_deep_learning_runtime(
                model_type, X_train, y_train, X_val, y_val, X_test
            )
            results.append({
                "dataset": "BATADAL", "model": model_type,
                "training_time_sec": train_t, "inference_time_sec": infer_t
            })
            print(f"  Training: {train_t}s | Inference: {infer_t}s")

    if batadal_pc1.exists():
        print("Ölçülüyor: BATADAL Automata...")
        X_train_pc1 = pd.read_csv(f"{processed_dir}/batadal_X_train_pc1.csv").values.flatten()
        X_test_pc1  = pd.read_csv(f"{processed_dir}/batadal_X_test_pc1.csv").values.flatten()
        train_t, infer_t = measure_automata_runtime(X_train_pc1, X_test_pc1)
        results.append({
            "dataset": "BATADAL", "model": "Automata",
            "training_time_sec": train_t, "inference_time_sec": infer_t
        })
        print(f"  Training: {train_t}s | Inference: {infer_t}s")

    # --- SKAB fold 1 üzerinde ölç (temsili) ---
    skab_seq = Path(processed_dir) / "skab_fold1_X_train_seq.npy"
    skab_pc1 = Path(processed_dir) / "skab_fold1_X_train_pc1.csv"

    if skab_seq.exists():
        X_train = np.load(f"{processed_dir}/skab_fold1_X_train_seq.npy").astype("float32")
        y_train = np.load(f"{processed_dir}/skab_fold1_y_train_seq.npy").astype("float32")
        X_test  = np.load(f"{processed_dir}/skab_fold1_X_test_seq.npy").astype("float32")
        y_test  = np.load(f"{processed_dir}/skab_fold1_y_test_seq.npy").astype("float32")

        val_start = int(len(X_train) * 0.8)
        X_val = X_train[val_start:]
        y_val = y_train[val_start:]
        X_tr  = X_train[:val_start]
        y_tr  = y_train[:val_start]

        for model_type in ["LSTM", "GRU"]:
            print(f"Ölçülüyor: SKAB {model_type} (fold 1)...")
            train_t, infer_t = measure_deep_learning_runtime(
                model_type, X_tr, y_tr, X_val, y_val, X_test
            )
            results.append({
                "dataset": "SKAB (fold 1)", "model": model_type,
                "training_time_sec": train_t, "inference_time_sec": infer_t
            })
            print(f"  Training: {train_t}s | Inference: {infer_t}s")

    if skab_pc1.exists():
        print("Ölçülüyor: SKAB Automata (fold 1)...")
        X_train_pc1 = pd.read_csv(f"{processed_dir}/skab_fold1_X_train_pc1.csv").values.flatten()
        X_test_pc1  = pd.read_csv(f"{processed_dir}/skab_fold1_X_test_pc1.csv").values.flatten()
        train_t, infer_t = measure_automata_runtime(X_train_pc1, X_test_pc1)
        results.append({
            "dataset": "SKAB (fold 1)", "model": "Automata",
            "training_time_sec": train_t, "inference_time_sec": infer_t
        })
        print(f"  Training: {train_t}s | Inference: {infer_t}s")

    # Kaydet
    output_dir = Path("results/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)
    df.to_csv(output_dir / "runtime_results.csv", index=False)

    print("\n=== Runtime Özet ===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()