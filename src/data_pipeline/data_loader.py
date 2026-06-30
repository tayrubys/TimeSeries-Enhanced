from pathlib import Path
import json
from typing import Any, Dict, List

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "src" / "config" / "settings.json"

# setting.json dosyasını okuma
def load_config(config_path: Path = CONFIG_PATH) -> Dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config dosyası bulunamadı: {config_path}")

    with open(config_path, "r", encoding="utf-8") as file:
        return json.load(file)


def resolve_path(relative_path: str) -> Path:
    return PROJECT_ROOT / relative_path

# skab valve1 ve valve2 klasorlerını bırlestırp kaynak ve grup ekleme
def load_skab(config: Dict[str, Any]) -> pd.DataFrame:
    skab_root = resolve_path(config["paths"]["skab_root"])
    groups = config["skab"]["groups"]

    all_dataframes: List[pd.DataFrame] = []

    for group in groups:
        group_path = skab_root / group

        if not group_path.exists():
            raise FileNotFoundError(f"SKAB klasörü bulunamadı: {group_path}")

        csv_files = sorted(group_path.glob("*.csv"))

        if not csv_files:
            raise FileNotFoundError(f"CSV dosyası bulunamadı: {group_path}")

        for csv_file in csv_files:
            df = pd.read_csv(csv_file, sep=";") #sutunlari ; gore ayırma
            df.columns = df.columns.str.strip()

            df["source_group"] = group
            df["source_file"] = csv_file.name

            all_dataframes.append(df)

    return pd.concat(all_dataframes, ignore_index=True)

# hangi sutunların modele girecegini belirler
def get_skab_columns(df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    target_col = config["skab"]["target_col"]

    if target_col not in df.columns:
        raise ValueError(f"SKAB hedef sütunu bulunamadı: {target_col}")

    exclude_cols = set(config["skab"]["exclude_cols"])

    feature_cols = [
        col for col in df.columns
        if col not in exclude_cols
    ]

    return {
        "target_col": target_col,
        "feature_cols": feature_cols,
        "exclude_cols": sorted(exclude_cols),
    }

# batadal dosyasını okuma
def load_batadal_training_2(config: Dict[str, Any]) -> pd.DataFrame:
    batadal_path = resolve_path(config["paths"]["batadal_training_2"])

    if not batadal_path.exists():
        raise FileNotFoundError(f"BATADAL dosyası bulunamadı: {batadal_path}")

    df = pd.read_csv(batadal_path)
    df.columns = df.columns.str.strip()

    return df

#batadal daki hedef sütunu bulmaya çalışır
def guess_batadal_target_column(df: pd.DataFrame, config: Dict[str, Any]) -> str:
    candidates = config["batadal"]["target_candidates"]

    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    raise ValueError(
        "BATADAL hedef sütunu bulunamadı. "
        f"Mevcut sütunlar: {df.columns.tolist()}"
    )

#feature ve target ayrımı yapar
def get_batadal_columns(df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    target_col = guess_batadal_target_column(df, config)
    time_keywords = config["batadal"]["time_keywords"]

    time_cols = []

    for col in df.columns:
        col_lower = col.lower()

        if any(keyword in col_lower for keyword in time_keywords):
            time_cols.append(col)

    exclude_cols = set(time_cols + [target_col])

    feature_cols = [
        col for col in df.columns
        if col not in exclude_cols
    ]

    return {
        "target_col": target_col,
        "feature_cols": feature_cols,
        "exclude_cols": sorted(exclude_cols),
    }


def print_summary(dataset_name: str, df: pd.DataFrame, columns_info: Dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print(dataset_name)
    print("=" * 60)

    print("Shape:", df.shape)
    print("Target column:", columns_info["target_col"])
    print("Feature count:", len(columns_info["feature_cols"]))
    print("Feature columns:", columns_info["feature_cols"])
    print("Excluded columns:", columns_info["exclude_cols"])


def main() -> None:
    config = load_config()

    skab_df = load_skab(config)
    skab_columns = get_skab_columns(skab_df, config)
    print_summary("SKAB", skab_df, skab_columns)

    batadal_df = load_batadal_training_2(config)
    batadal_columns = get_batadal_columns(batadal_df, config)
    print_summary("BATADAL Training Dataset 2", batadal_df, batadal_columns)


if __name__ == "__main__":
    main()