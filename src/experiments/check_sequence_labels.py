import pandas as pd

WINDOW_SIZE = 20

# Orijinal validation label dosyası
y_val = pd.read_csv(
    "data/processed/batadal_y_val.csv"
).values.ravel()

# -999 -> 0
y_val = [0 if x == -999 else x for x in y_val]

# İncelenecek sequence
sequence_index = 671

start = sequence_index - WINDOW_SIZE + 1
end = sequence_index + 1

print("="*50)
print(f"Sequence index : {sequence_index}")
print(f"Satırlar       : {start} - {end-1}")
print("="*50)

labels = y_val[start:end]

for i, label in enumerate(labels):
    print(f"Satır {start+i:4d}  Label = {label}")

print("\nToplam attack sayısı :", sum(labels))