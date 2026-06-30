from pathlib import Path
import pandas as pd

#deney sonuclarını csv dosyasına kaydeder
def save_results_to_csv(results, output_path="results/outputs/deep_learning_results.csv"):
    """
    results:
        Liste içinde dictionary formatında sonuçlar bekler.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)

    print(f"Sonuçlar kaydedildi: {output_path}")