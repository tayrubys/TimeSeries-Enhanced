import json
import os
from collections import Counter

import numpy as np
import pandas as pd


OUTPUT_DIR = "results/outputs"
BEST_CONFIG_PATH = os.path.join(
    OUTPUT_DIR,
    "skab_class_specific_dual_alergia_best_configs.csv",
)
METRICS_PATH = os.path.join(
    OUTPUT_DIR,
    "skab_class_specific_dual_alergia_fold_seed_metrics.csv",
)

PARAMETER_COLUMNS = [
    "normal_merge_alpha",
    "normal_min_state_count",
    "normal_smoothing_alpha",
    "anomaly_merge_alpha",
    "anomaly_min_state_count",
    "anomaly_smoothing_alpha",
    "normal_distance_penalty",
    "anomaly_distance_penalty",
    "min_anomaly_run",
]

NORMAL_CONFIG_COLUMNS = [
    "normal_merge_alpha",
    "normal_min_state_count",
    "normal_smoothing_alpha",
]

ANOMALY_CONFIG_COLUMNS = [
    "anomaly_merge_alpha",
    "anomaly_min_state_count",
    "anomaly_smoothing_alpha",
]

FULL_CONFIG_COLUMNS = PARAMETER_COLUMNS.copy()


def require_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Gerekli dosya bulunamadı: {path}\n"
            "Önce SKAB Class-Specific Dual ALERGIA deneyini çalıştırmalısın."
        )


def normalize_bool(series):
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({
            "true": True,
            "false": False,
            "1": True,
            "0": False,
            "yes": True,
            "no": False,
        })
        .fillna(False)
        .astype(bool)
    )


def tuple_key(row, columns):
    return tuple(row[column] for column in columns)


def format_config(values, columns):
    return " | ".join(
        f"{column}={value}"
        for column, value in zip(columns, values)
    )


def mode_and_agreement(series):
    values = series.dropna().tolist()
    if not values:
        return np.nan, 0, 0.0, 0

    counts = Counter(values)
    # Eşitlikte tekrarlanabilir sonuç için metinsel sırayı kullan.
    mode_value, mode_count = sorted(
        counts.items(),
        key=lambda item: (-item[1], str(item[0])),
    )[0]

    return (
        mode_value,
        int(mode_count),
        float(mode_count / len(values)),
        int(len(counts)),
    )


def safe_correlation(first, second):
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    valid = np.isfinite(first) & np.isfinite(second)
    first = first[valid]
    second = second[valid]

    if len(first) < 2:
        return 0.0
    if np.std(first) == 0.0 or np.std(second) == 0.0:
        return 0.0

    return float(np.corrcoef(first, second)[0, 1])


def build_parameter_frequency(best_df):
    rows = []

    for parameter in PARAMETER_COLUMNS:
        counts = (
            best_df[parameter]
            .value_counts(dropna=False)
            .rename_axis("value")
            .reset_index(name="selection_count")
        )
        counts["parameter"] = parameter
        counts["selection_rate"] = (
            counts["selection_count"] / len(best_df)
        )
        rows.extend(counts.to_dict("records"))

    return pd.DataFrame(rows)[
        [
            "parameter",
            "value",
            "selection_count",
            "selection_rate",
        ]
    ]


def build_combination_frequency(best_df, columns, combination_name):
    keys = best_df.apply(
        lambda row: tuple_key(row, columns),
        axis=1,
    )

    counts = keys.value_counts().reset_index()
    counts.columns = ["configuration", "selection_count"]
    counts["selection_rate"] = counts["selection_count"] / len(best_df)
    counts["combination_type"] = combination_name
    counts["configuration_text"] = counts["configuration"].apply(
        lambda values: format_config(values, columns)
    )

    return counts[
        [
            "combination_type",
            "configuration_text",
            "selection_count",
            "selection_rate",
        ]
    ]


def build_group_consistency(best_df, group_column):
    rows = []

    for group_value, group in best_df.groupby(group_column, sort=True):
        row = {
            group_column: group_value,
            "run_count": int(len(group)),
            "selected_shared_pair_count": int(
                group["selected_pair_shared_parameters"].sum()
            ),
            "selected_shared_pair_rate": float(
                group["selected_pair_shared_parameters"].mean()
            ),
            "final_validation_f1_mean": float(
                group["final_validation_f1"].mean()
            ),
            "final_validation_f1_std": float(
                group["final_validation_f1"].std(ddof=0)
            ),
        }

        if "original_test_f1" in group.columns:
            row.update(
                {
                    "original_test_f1_mean": float(
                        group["original_test_f1"].mean()
                    ),
                    "original_test_f1_std": float(
                        group["original_test_f1"].std(ddof=0)
                    ),
                    "generalization_gap_mean": float(
                        group["generalization_gap"].mean()
                    ),
                }
            )

        normal_keys = group.apply(
            lambda item: tuple_key(item, NORMAL_CONFIG_COLUMNS),
            axis=1,
        )
        anomaly_keys = group.apply(
            lambda item: tuple_key(item, ANOMALY_CONFIG_COLUMNS),
            axis=1,
        )
        full_keys = group.apply(
            lambda item: tuple_key(item, FULL_CONFIG_COLUMNS),
            axis=1,
        )

        for prefix, values, columns in [
            ("normal_config", normal_keys, NORMAL_CONFIG_COLUMNS),
            ("anomaly_config", anomaly_keys, ANOMALY_CONFIG_COLUMNS),
            ("full_config", full_keys, FULL_CONFIG_COLUMNS),
        ]:
            mode_value, mode_count, agreement, unique_count = (
                mode_and_agreement(values)
            )
            row[f"{prefix}_unique_count"] = unique_count
            row[f"{prefix}_mode_count"] = mode_count
            row[f"{prefix}_agreement_rate"] = agreement
            row[f"{prefix}_mode"] = (
                format_config(mode_value, columns)
                if isinstance(mode_value, tuple)
                else ""
            )

        for parameter in PARAMETER_COLUMNS:
            mode_value, mode_count, agreement, unique_count = (
                mode_and_agreement(group[parameter])
            )
            row[f"{parameter}_mode"] = mode_value
            row[f"{parameter}_agreement_rate"] = agreement
            row[f"{parameter}_unique_count"] = unique_count

        rows.append(row)

    return pd.DataFrame(rows)


def add_original_test_metrics(best_df):
    if not os.path.exists(METRICS_PATH):
        print(
            "[UYARI] Fold/seed metrics dosyası bulunamadı. "
            "Validation-test genelleşme analizi atlanacak."
        )
        return best_df

    metrics_df = pd.read_csv(METRICS_PATH)
    required = {"fold", "seed", "scenario", "f1_score"}
    missing = sorted(required - set(metrics_df.columns))
    if missing:
        print(
            "[UYARI] Metrics dosyasında gerekli kolonlar eksik: "
            f"{missing}. Genelleşme analizi atlanacak."
        )
        return best_df

    original = metrics_df.loc[
        metrics_df["scenario"].astype(str).str.lower() == "original",
        ["fold", "seed", "f1_score", "precision", "recall"],
    ].copy()

    original = original.rename(
        columns={
            "f1_score": "original_test_f1",
            "precision": "original_test_precision",
            "recall": "original_test_recall",
        }
    )

    # Aynı fold/seed için beklenmedik tekrarlar varsa ortalamasını al.
    original = (
        original.groupby(["fold", "seed"], as_index=False)
        .mean(numeric_only=True)
    )

    merged = best_df.merge(
        original,
        on=["fold", "seed"],
        how="left",
        validate="one_to_one",
    )

    merged["generalization_gap"] = (
        merged["original_test_f1"]
        - merged["final_validation_f1"]
    )
    merged["class_specific_validation_gain"] = (
        merged["best_any_pair_validation_f1"]
        - merged["best_shared_pair_validation_f1"]
    )

    return merged


def build_summary(best_df, combination_df):
    normal_keys = best_df.apply(
        lambda row: tuple_key(row, NORMAL_CONFIG_COLUMNS),
        axis=1,
    )
    anomaly_keys = best_df.apply(
        lambda row: tuple_key(row, ANOMALY_CONFIG_COLUMNS),
        axis=1,
    )
    full_keys = best_df.apply(
        lambda row: tuple_key(row, FULL_CONFIG_COLUMNS),
        axis=1,
    )

    normal_mode = mode_and_agreement(normal_keys)
    anomaly_mode = mode_and_agreement(anomaly_keys)
    full_mode = mode_and_agreement(full_keys)

    summary = {
        "run_count": int(len(best_df)),
        "fold_count": int(best_df["fold"].nunique()),
        "seed_count": int(best_df["seed"].nunique()),
        "selected_shared_pair_count": int(
            best_df["selected_pair_shared_parameters"].sum()
        ),
        "selected_shared_pair_rate": float(
            best_df["selected_pair_shared_parameters"].mean()
        ),
        "unique_normal_configuration_count": int(normal_mode[3]),
        "most_common_normal_configuration_count": int(normal_mode[1]),
        "most_common_normal_configuration_rate": float(normal_mode[2]),
        "most_common_normal_configuration": format_config(
            normal_mode[0], NORMAL_CONFIG_COLUMNS
        ),
        "unique_anomaly_configuration_count": int(anomaly_mode[3]),
        "most_common_anomaly_configuration_count": int(anomaly_mode[1]),
        "most_common_anomaly_configuration_rate": float(anomaly_mode[2]),
        "most_common_anomaly_configuration": format_config(
            anomaly_mode[0], ANOMALY_CONFIG_COLUMNS
        ),
        "unique_full_configuration_count": int(full_mode[3]),
        "most_common_full_configuration_count": int(full_mode[1]),
        "most_common_full_configuration_rate": float(full_mode[2]),
        "final_validation_f1_mean": float(
            best_df["final_validation_f1"].mean()
        ),
        "final_validation_f1_std": float(
            best_df["final_validation_f1"].std(ddof=0)
        ),
        "class_specific_validation_gain_mean": float(
            (
                best_df["best_any_pair_validation_f1"]
                - best_df["best_shared_pair_validation_f1"]
            ).mean()
        ),
        "class_specific_validation_gain_median": float(
            (
                best_df["best_any_pair_validation_f1"]
                - best_df["best_shared_pair_validation_f1"]
            ).median()
        ),
        "class_specific_pair_better_count": int(
            (
                best_df["best_any_pair_validation_f1"]
                > best_df["best_shared_pair_validation_f1"] + 1e-12
            ).sum()
        ),
    }

    if "original_test_f1" in best_df.columns:
        valid = best_df["original_test_f1"].notna()
        valid_df = best_df.loc[valid].copy()

        summary.update(
            {
                "original_test_f1_mean": float(
                    valid_df["original_test_f1"].mean()
                ),
                "original_test_f1_std": float(
                    valid_df["original_test_f1"].std(ddof=0)
                ),
                "generalization_gap_mean": float(
                    valid_df["generalization_gap"].mean()
                ),
                "generalization_gap_median": float(
                    valid_df["generalization_gap"].median()
                ),
                "validation_test_f1_correlation": safe_correlation(
                    valid_df["final_validation_f1"],
                    valid_df["original_test_f1"],
                ),
                "validation_gain_test_f1_correlation": safe_correlation(
                    valid_df["class_specific_validation_gain"],
                    valid_df["original_test_f1"],
                ),
            }
        )

    summary_df = pd.DataFrame(
        [{"metric": key, "value": value} for key, value in summary.items()]
    )

    return summary, summary_df


def print_top_parameter_frequencies(frequency_df):
    print("\n--- EN SIK SEÇİLEN PARAMETRELER ---")

    for parameter in PARAMETER_COLUMNS:
        subset = frequency_df.loc[
            frequency_df["parameter"] == parameter
        ].sort_values(
            ["selection_count", "value"],
            ascending=[False, True],
        )
        top = subset.iloc[0]
        print(
            f"{parameter}: {top['value']} | "
            f"{int(top['selection_count'])}/{int(subset['selection_count'].sum())} "
            f"({float(top['selection_rate']):.2%})"
        )


def main():
    print("\n--- SKAB CLASS-SPECIFIC PARAMETRE TUTARLILIK ANALİZİ ---")

    require_file(BEST_CONFIG_PATH)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    best_df = pd.read_csv(BEST_CONFIG_PATH)

    required_columns = {
        "fold",
        "seed",
        "selected_pair_shared_parameters",
        "best_shared_pair_validation_f1",
        "best_any_pair_validation_f1",
        "final_validation_f1",
        *PARAMETER_COLUMNS,
    }
    missing = sorted(required_columns - set(best_df.columns))
    if missing:
        raise ValueError(
            "Best-config dosyasında gerekli kolonlar eksik: "
            f"{missing}"
        )

    if best_df.duplicated(["fold", "seed"]).any():
        duplicates = best_df.loc[
            best_df.duplicated(["fold", "seed"], keep=False),
            ["fold", "seed"],
        ]
        raise ValueError(
            "Best-config dosyasında tekrarlanan fold/seed satırları var:\n"
            f"{duplicates.to_string(index=False)}"
        )

    best_df["selected_pair_shared_parameters"] = normalize_bool(
        best_df["selected_pair_shared_parameters"]
    )

    best_df = add_original_test_metrics(best_df)

    frequency_df = build_parameter_frequency(best_df)
    combination_df = pd.concat(
        [
            build_combination_frequency(
                best_df,
                NORMAL_CONFIG_COLUMNS,
                "normal_model",
            ),
            build_combination_frequency(
                best_df,
                ANOMALY_CONFIG_COLUMNS,
                "anomaly_model",
            ),
            build_combination_frequency(
                best_df,
                FULL_CONFIG_COLUMNS,
                "full_selection",
            ),
        ],
        ignore_index=True,
    )

    fold_df = build_group_consistency(best_df, "fold")
    seed_df = build_group_consistency(best_df, "seed")
    summary, summary_df = build_summary(best_df, combination_df)

    detailed_path = os.path.join(
        OUTPUT_DIR,
        "skab_class_specific_parameter_selection_details.csv",
    )
    frequency_path = os.path.join(
        OUTPUT_DIR,
        "skab_class_specific_parameter_frequency.csv",
    )
    combination_path = os.path.join(
        OUTPUT_DIR,
        "skab_class_specific_configuration_frequency.csv",
    )
    fold_path = os.path.join(
        OUTPUT_DIR,
        "skab_class_specific_fold_parameter_consistency.csv",
    )
    seed_path = os.path.join(
        OUTPUT_DIR,
        "skab_class_specific_seed_parameter_consistency.csv",
    )
    summary_path = os.path.join(
        OUTPUT_DIR,
        "skab_class_specific_parameter_consistency_summary.csv",
    )
    report_path = os.path.join(
        OUTPUT_DIR,
        "skab_class_specific_parameter_consistency_report.json",
    )

    best_df.to_csv(detailed_path, index=False)
    frequency_df.to_csv(frequency_path, index=False)
    combination_df.to_csv(combination_path, index=False)
    fold_df.to_csv(fold_path, index=False)
    seed_df.to_csv(seed_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print(
        f"Okunan fold/seed seçimi={len(best_df)} | "
        f"fold={best_df['fold'].nunique()} | seed={best_df['seed'].nunique()}"
    )
    print(
        "Aynı parametreli normal/anomaly çift seçimi: "
        f"{summary['selected_shared_pair_count']}/{summary['run_count']} "
        f"({summary['selected_shared_pair_rate']:.2%})"
    )
    print(
        "Normal config çeşitliliği: "
        f"{summary['unique_normal_configuration_count']} farklı | "
        "en sık config oranı="
        f"{summary['most_common_normal_configuration_rate']:.2%}"
    )
    print(
        "Anomaly config çeşitliliği: "
        f"{summary['unique_anomaly_configuration_count']} farklı | "
        "en sık config oranı="
        f"{summary['most_common_anomaly_configuration_rate']:.2%}"
    )
    print(
        "Tüm seçim çeşitliliği: "
        f"{summary['unique_full_configuration_count']} farklı tam ayar | "
        "en sık tam ayar oranı="
        f"{summary['most_common_full_configuration_rate']:.2%}"
    )
    print(
        "Class-specific validation kazancı: "
        f"ortalama={summary['class_specific_validation_gain_mean']:+.6f} | "
        f"medyan={summary['class_specific_validation_gain_median']:+.6f} | "
        "shared çiftten iyi çalışma="
        f"{summary['class_specific_pair_better_count']}/{summary['run_count']}"
    )

    if "original_test_f1_mean" in summary:
        print(
            "Validation/Test ilişkisi: "
            f"validation F1 ort={summary['final_validation_f1_mean']:.4f} | "
            f"test F1 ort={summary['original_test_f1_mean']:.4f} | "
            f"gap={summary['generalization_gap_mean']:+.4f} | "
            "corr="
            f"{summary['validation_test_f1_correlation']:+.4f}"
        )
        print(
            "Class-specific validation kazancı ile test F1 korelasyonu: "
            f"{summary['validation_gain_test_f1_correlation']:+.4f}"
        )

    print_top_parameter_frequencies(frequency_df)

    print("\n--- FOLD BAZINDA TAM AYAR UZLAŞMASI ---")
    for _, row in fold_df.iterrows():
        print(
            f"Fold {int(row['fold'])}: "
            f"farklı tam ayar={int(row['full_config_unique_count'])} | "
            f"mode agreement={row['full_config_agreement_rate']:.2%} | "
            f"val F1 std={row['final_validation_f1_std']:.4f}"
            + (
                f" | test F1 std={row['original_test_f1_std']:.4f}"
                if "original_test_f1_std" in row.index
                else ""
            )
        )

    print("\nKaydedilen analiz dosyaları:")
    for path in [
        detailed_path,
        frequency_path,
        combination_path,
        fold_path,
        seed_path,
        summary_path,
        report_path,
    ]:
        print(f"- {path}")


if __name__ == "__main__":
    main()