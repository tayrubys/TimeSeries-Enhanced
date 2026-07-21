import numpy as np

X_train = np.load(
"data2/processed/robust_adasyn/batadal_X_train_seq.npy",
    mmap_mode="r"
)

print("X_train shape:", X_train.shape)
print("Sequence uzunluğu:", X_train.shape[1])
print("Özellik sayısı:", X_train.shape[2])