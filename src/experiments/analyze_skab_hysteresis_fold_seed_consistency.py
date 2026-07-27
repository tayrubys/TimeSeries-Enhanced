import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.stats import binomtest, wilcoxon
except ImportError:  # SciPy yoksa analiz temel özetlerle devam eder.
    binomtest = None
    wilcoxon = None


INPUT_PATH = Path(
    "results/outputs/skab_hysteresis_fold_seed_metrics.csv"
)
OUTPUT_DIR = Path("results/outputs")

BASELINE_METHOD = "baseline_single_threshold"
HYSTERESIS_METHOD = "dual_threshold_hysteresis"
ANALYZED_SCENARIOS = ["original", "gaussian_noise"]

PAIR_KEYS = ["fold", "seed", "scenario"]
METRIC_COLUMNS = ["accuracy", "precision", "recall", "f1_score"]
TIE_TOLERANCE = 1e-12
BOOTSTRAP_REPETITIONS = 10_000
BOOTSTRAP_SEED = 42


PAIR_OUTPUT_PATH = OUTPUT_DIR / (
    "skab_hysteresis_paired_fold_seed_comparison.csv"
)
SUMMARY_OUTPUT_PATH = OUTPUT_DIR / (
    "skab_hysteresis_consistency_summary.csv"
)
FOLD_OUTPUT_PATH = OUTPUT_DIR / (
    "skab_hysteresis_fold_consistency_summary.csv"
)
SEED_OUTPUT_PATH = OUTPUT_DIR / (
    "skab_hysteresis_seed_consistency_summary.csv"
)
REPORT_OUTPUT_PATH = OUTPUT_DIR / (
    "skab_hysteresis_consistency_report.json"
)


def require_columns(dataframe, columns):
    missing = sorted(set(columns) - set(dataframe.columns))
    if missing:
        raise ValueError(
            "Girdi CSV dosyasında gerekli sütunlar eksik: "
            + ", ".join(missing)
        )


def coerce_bool(series):
    """CSV'den bool veya metin olarak gelen değerleri güvenli biçimde çevirir."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)

    normalized = series.astype(str).str.strip().str.lower()
    true_values = {"true", "1", "yes", "evet"}
    false_values = {"false", "0", "no", "hayir", "hayır", "nan", "none", ""}

    unknown = sorted(
        set(normalized.unique()) - true_values - false_values
    )
    if unknown:
        raise ValueError(
            "baseline_selected sütununda tanınmayan değerler var: "
            + ", ".join(unknown)
        )

    return normalized.isin(true_values)


def validate_unique_pairs(dataframe):
    counts = (
        dataframe.groupby(PAIR_KEYS + ["method"])
        .size()
        .reset_index(name="row_count")
    )
    duplicates = counts[counts["row_count"] != 1]

    if not duplicates.empty:
        raise ValueError(
            "Her fold + seed + scenario + method için tam bir satır "
            "bekleniyordu. Uygun olmayan gruplar:\n"
            + duplicates.to_string(index=False)
        )


def classify_difference(value):
    if value > TIE_TOLERANCE:
        return "hysteresis"
    if value < -TIE_TOLERANCE:
        return "baseline"
    return "tie"


def bootstrap_mean_confidence_interval(values):
    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), float(values[0])

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    sample_indices = rng.integers(
        0,
        len(values),
        size=(BOOTSTRAP_REPETITIONS, len(values)),
    )
    sampled_means = values[sample_indices].mean(axis=1)

    return (
        float(np.quantile(sampled_means, 0.025)),
        float(np.quantile(sampled_means, 0.975)),
    )


def calculate_rank_biserial(differences):
    differences = np.asarray(differences, dtype=float)
    nonzero = differences[np.abs(differences) > TIE_TOLERANCE]

    if len(nonzero) == 0:
        return 0.0

    absolute_ranks = pd.Series(np.abs(nonzero)).rank(
        method="average"
    ).to_numpy(dtype=float)
    positive_rank_sum = float(absolute_ranks[nonzero > 0].sum())
    negative_rank_sum = float(absolute_ranks[nonzero < 0].sum())
    total_rank_sum = positive_rank_sum + negative_rank_sum

    if total_rank_sum == 0:
        return 0.0

    return (
        positive_rank_sum - negative_rank_sum
    ) / total_rank_sum


def calculate_paired_tests(differences):
    differences = np.asarray(differences, dtype=float)
    nonzero = differences[np.abs(differences) > TIE_TOLERANCE]

    wins = int((nonzero > 0).sum())
    losses = int((nonzero < 0).sum())

    if len(nonzero) == 0:
        wilcoxon_statistic = 0.0
        wilcoxon_p_value = 1.0
        sign_test_p_value = 1.0
    else:
        if wilcoxon is None:
            wilcoxon_statistic = float("nan")
            wilcoxon_p_value = float("nan")
        else:
            result = wilcoxon(
                nonzero,
                zero_method="wilcox",
                alternative="two-sided",
                method="auto",
            )
            wilcoxon_statistic = float(result.statistic)
            wilcoxon_p_value = float(result.pvalue)

        if binomtest is None:
            sign_test_p_value = float("nan")
        else:
            sign_test_p_value = float(
                binomtest(
                    wins,
                    n=wins + losses,
                    p=0.5,
                    alternative="two-sided",
                ).pvalue
            )

    return {
        "wilcoxon_statistic": wilcoxon_statistic,
        "wilcoxon_p_value": wilcoxon_p_value,
        "sign_test_p_value": sign_test_p_value,
        "rank_biserial_correlation": float(
            calculate_rank_biserial(differences)
        ),
    }


def create_paired_table(metrics_df):
    filtered = metrics_df[
        metrics_df["scenario"].isin(ANALYZED_SCENARIOS)
        & metrics_df["method"].isin(
            [BASELINE_METHOD, HYSTERESIS_METHOD]
        )
    ].copy()

    validate_unique_pairs(filtered)

    baseline = filtered[
        filtered["method"] == BASELINE_METHOD
    ].copy()
    hysteresis = filtered[
        filtered["method"] == HYSTERESIS_METHOD
    ].copy()

    baseline_columns = PAIR_KEYS + METRIC_COLUMNS + [
        "predicted_anomaly_count",
        "raw_predicted_anomaly_count",
        "validation_f1_score",
        "validation_precision",
        "validation_recall",
    ]
    hysteresis_columns = PAIR_KEYS + METRIC_COLUMNS + [
        "predicted_anomaly_count",
        "raw_predicted_anomaly_count",
        "validation_f1_score",
        "validation_precision",
        "validation_recall",
        "baseline_selected",
        "high_threshold",
        "low_threshold",
        "hysteresis_width",
        "min_anomaly_run",
    ]

    require_columns(baseline, baseline_columns)
    require_columns(hysteresis, hysteresis_columns)

    baseline = baseline[baseline_columns].rename(
        columns={
            column: f"baseline_{column}"
            for column in baseline_columns
            if column not in PAIR_KEYS
        }
    )
    hysteresis = hysteresis[hysteresis_columns].rename(
        columns={
            column: f"hysteresis_{column}"
            for column in hysteresis_columns
            if column not in PAIR_KEYS
        }
    )

    paired = baseline.merge(
        hysteresis,
        on=PAIR_KEYS,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )

    unmatched = paired[paired["_merge"] != "both"]
    if not unmatched.empty:
        raise ValueError(
            "Baseline ve hysteresis satırları tam eşleşmedi:\n"
            + unmatched[PAIR_KEYS + ["_merge"]].to_string(index=False)
        )

    paired = paired.drop(columns="_merge")
    paired["hysteresis_baseline_selected"] = coerce_bool(
        paired["hysteresis_baseline_selected"]
    )
    paired["actual_hysteresis_selected"] = ~paired[
        "hysteresis_baseline_selected"
    ]

    for metric in METRIC_COLUMNS:
        paired[f"{metric}_difference"] = (
            paired[f"hysteresis_{metric}"]
            - paired[f"baseline_{metric}"]
        )

    paired["predicted_anomaly_count_difference"] = (
        paired["hysteresis_predicted_anomaly_count"]
        - paired["baseline_predicted_anomaly_count"]
    )
    paired["f1_winner"] = paired["f1_score_difference"].apply(
        classify_difference
    )

    return paired.sort_values(PAIR_KEYS).reset_index(drop=True)


def summarize_scenario(scenario_df):
    f1_differences = scenario_df["f1_score_difference"].to_numpy(
        dtype=float
    )
    wins = int((f1_differences > TIE_TOLERANCE).sum())
    losses = int((f1_differences < -TIE_TOLERANCE).sum())
    ties = int(len(f1_differences) - wins - losses)
    non_tie_count = wins + losses

    ci_low, ci_high = bootstrap_mean_confidence_interval(
        f1_differences
    )
    paired_tests = calculate_paired_tests(f1_differences)

    standard_deviation = float(
        np.std(f1_differences, ddof=1)
    ) if len(f1_differences) > 1 else 0.0
    cohen_dz = (
        float(np.mean(f1_differences) / standard_deviation)
        if standard_deviation > 0
        else 0.0
    )

    actual_selected = scenario_df[
        scenario_df["actual_hysteresis_selected"]
    ]
    selected_wins = int(
        (actual_selected["f1_score_difference"] > TIE_TOLERANCE).sum()
    )
    selected_losses = int(
        (actual_selected["f1_score_difference"] < -TIE_TOLERANCE).sum()
    )
    selected_ties = int(
        len(actual_selected) - selected_wins - selected_losses
    )

    summary = {
        "scenario": str(scenario_df["scenario"].iloc[0]),
        "pair_count": int(len(scenario_df)),
        "hysteresis_win_count": wins,
        "baseline_win_count": losses,
        "tie_count": ties,
        "hysteresis_win_rate_all": float(wins / len(scenario_df)),
        "hysteresis_win_rate_non_tie": (
            float(wins / non_tie_count) if non_tie_count else 0.0
        ),
        "f1_difference_mean": float(np.mean(f1_differences)),
        "f1_difference_median": float(np.median(f1_differences)),
        "f1_difference_std": standard_deviation,
        "f1_difference_min": float(np.min(f1_differences)),
        "f1_difference_max": float(np.max(f1_differences)),
        "f1_difference_bootstrap_ci_low": ci_low,
        "f1_difference_bootstrap_ci_high": ci_high,
        "precision_difference_mean": float(
            scenario_df["precision_difference"].mean()
        ),
        "recall_difference_mean": float(
            scenario_df["recall_difference"].mean()
        ),
        "accuracy_difference_mean": float(
            scenario_df["accuracy_difference"].mean()
        ),
        "predicted_anomaly_count_difference_mean": float(
            scenario_df["predicted_anomaly_count_difference"].mean()
        ),
        "actual_hysteresis_selected_count": int(len(actual_selected)),
        "baseline_selected_count": int(
            scenario_df["hysteresis_baseline_selected"].sum()
        ),
        "selected_hysteresis_test_win_count": selected_wins,
        "selected_hysteresis_test_loss_count": selected_losses,
        "selected_hysteresis_test_tie_count": selected_ties,
        "cohen_dz": cohen_dz,
        **paired_tests,
    }

    return summary


def create_group_summary(paired_df, group_column):
    rows = []

    for (scenario, group_value), group in paired_df.groupby(
        ["scenario", group_column], sort=True
    ):
        differences = group["f1_score_difference"]
        wins = int((differences > TIE_TOLERANCE).sum())
        losses = int((differences < -TIE_TOLERANCE).sum())
        ties = int(len(group) - wins - losses)

        rows.append(
            {
                "scenario": scenario,
                group_column: group_value,
                "pair_count": int(len(group)),
                "hysteresis_win_count": wins,
                "baseline_win_count": losses,
                "tie_count": ties,
                "f1_difference_mean": float(differences.mean()),
                "f1_difference_median": float(differences.median()),
                "f1_difference_std": float(differences.std(ddof=1))
                if len(group) > 1
                else 0.0,
                "precision_difference_mean": float(
                    group["precision_difference"].mean()
                ),
                "recall_difference_mean": float(
                    group["recall_difference"].mean()
                ),
                "accuracy_difference_mean": float(
                    group["accuracy_difference"].mean()
                ),
                "actual_hysteresis_selected_count": int(
                    group["actual_hysteresis_selected"].sum()
                ),
            }
        )

    return pd.DataFrame(rows)


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if math.isnan(float(value)):
            return None
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def print_scenario_summary(summary):
    print("\n" + "-" * 72)
    print(f"SENARYO: {summary['scenario']}")
    print("-" * 72)
    print(
        "Eşleştirilmiş çalışma: "
        f"{summary['pair_count']} | "
        f"Hysteresis kazandı={summary['hysteresis_win_count']} | "
        f"Baseline kazandı={summary['baseline_win_count']} | "
        f"Berabere={summary['tie_count']}"
    )
    print(
        "F1 farkı (hysteresis - baseline): "
        f"ortalama={summary['f1_difference_mean']:+.6f} | "
        f"medyan={summary['f1_difference_median']:+.6f} | "
        f"min={summary['f1_difference_min']:+.6f} | "
        f"max={summary['f1_difference_max']:+.6f}"
    )
    print(
        "F1 ortalama farkı bootstrap %95 GA: "
        f"[{summary['f1_difference_bootstrap_ci_low']:+.6f}, "
        f"{summary['f1_difference_bootstrap_ci_high']:+.6f}]"
    )
    print(
        "Ortalama metrik farkları: "
        f"Precision={summary['precision_difference_mean']:+.6f} | "
        f"Recall={summary['recall_difference_mean']:+.6f} | "
        f"Accuracy={summary['accuracy_difference_mean']:+.6f}"
    )
    print(
        "Inner validation gerçek hysteresis seçimi: "
        f"{summary['actual_hysteresis_selected_count']}/"
        f"{summary['pair_count']} | "
        "Outer test sonucu: "
        f"kazanç={summary['selected_hysteresis_test_win_count']} | "
        f"kayıp={summary['selected_hysteresis_test_loss_count']} | "
        f"berabere={summary['selected_hysteresis_test_tie_count']}"
    )
    print(
        "Wilcoxon p="
        f"{summary['wilcoxon_p_value']:.6f} | "
        "Sign test p="
        f"{summary['sign_test_p_value']:.6f} | "
        "Rank-biserial="
        f"{summary['rank_biserial_correlation']:+.4f}"
    )


def main():
    print("\n--- SKAB HYSTERESIS FOLD/SEED TUTARLILIK ANALİZİ ---")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Sonuç dosyası bulunamadı: {INPUT_PATH}\n"
            "Önce run_skab_dual_threshold_hysteresis_experiment.py "
            "dosyasını çalıştırmalısın."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.read_csv(INPUT_PATH)

    require_columns(
        metrics_df,
        PAIR_KEYS
        + ["method"]
        + METRIC_COLUMNS
        + [
            "predicted_anomaly_count",
            "raw_predicted_anomaly_count",
            "validation_f1_score",
            "validation_precision",
            "validation_recall",
            "baseline_selected",
            "high_threshold",
            "low_threshold",
            "hysteresis_width",
            "min_anomaly_run",
        ],
    )

    paired_df = create_paired_table(metrics_df)

    expected_pair_count = (
        paired_df[["fold", "seed"]].drop_duplicates().shape[0]
    )
    print(
        f"Okunan satır={len(metrics_df)} | "
        f"benzersiz fold/seed çifti={expected_pair_count} | "
        f"analiz senaryoları={ANALYZED_SCENARIOS}"
    )

    summary_rows = []
    scenario_reports = {}

    for scenario in ANALYZED_SCENARIOS:
        scenario_df = paired_df[
            paired_df["scenario"] == scenario
        ].copy()

        if scenario_df.empty:
            print(f"UYARI: {scenario} senaryosu bulunamadı, atlandı.")
            continue

        summary = summarize_scenario(scenario_df)
        summary_rows.append(summary)
        scenario_reports[scenario] = summary
        print_scenario_summary(summary)

    summary_df = pd.DataFrame(summary_rows)
    fold_summary_df = create_group_summary(paired_df, "fold")
    seed_summary_df = create_group_summary(paired_df, "seed")

    paired_df.to_csv(PAIR_OUTPUT_PATH, index=False)
    summary_df.to_csv(SUMMARY_OUTPUT_PATH, index=False)
    fold_summary_df.to_csv(FOLD_OUTPUT_PATH, index=False)
    seed_summary_df.to_csv(SEED_OUTPUT_PATH, index=False)

    report = {
        "input_path": str(INPUT_PATH),
        "analyzed_scenarios": ANALYZED_SCENARIOS,
        "difference_definition": "hysteresis - baseline",
        "tie_tolerance": TIE_TOLERANCE,
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "scenario_summaries": scenario_reports,
        "fold_summaries": fold_summary_df.to_dict(orient="records"),
        "seed_summaries": seed_summary_df.to_dict(orient="records"),
    }

    with open(REPORT_OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(
            json_safe(report),
            file,
            indent=4,
            ensure_ascii=False,
        )

    print("\nKaydedilen analiz dosyaları:")
    print(f"- {PAIR_OUTPUT_PATH}")
    print(f"- {SUMMARY_OUTPUT_PATH}")
    print(f"- {FOLD_OUTPUT_PATH}")
    print(f"- {SEED_OUTPUT_PATH}")
    print(f"- {REPORT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()