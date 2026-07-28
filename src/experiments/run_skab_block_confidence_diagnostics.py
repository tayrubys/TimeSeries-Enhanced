import json
import math
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..")
    )
)

from src.data_pipeline.sax_paa import SaxPaaTransformer
from src.experiments.run_skab_alergia_state_merging_experiment import (
    choose_inner_split,
    fit_dual_models,
    load_config,
    load_source_files,
    load_vector,
    score_dual,
    to_binary,
)


OUTPUT_DIR = "results/outputs"
BEST_CONFIG_PATH = os.path.join(
    OUTPUT_DIR,
    "skab_dual_alergia_best_configs.csv",
)

BLOCK_OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "skab_block_confidence_validation_blocks.csv",
)
FEATURE_OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "skab_block_confidence_feature_summary.csv",
)
FOLD_SEED_OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "skab_block_confidence_fold_seed_summary.csv",
)
REPORT_OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "skab_block_confidence_diagnostics.json",
)

FEATURE_COLUMNS = [
    "block_length",
    "mean_margin",
    "median_margin",
    "max_margin",
    "min_margin",
    "sum_margin",
    "margin_std",
    "top_quartile_margin_mean",
    "strong_point_ratio",
]


def load_best_configs(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            "Mevcut SKAB en iyi ayar dosyası bulunamadı:\n"
            f"{path}\n\n"
            "Önce şu deneyi çalıştırmalısın:\n"
            "python src/experiments/"
            "run_skab_alergia_state_merging_experiment.py"
        )

    frame = pd.read_csv(path)

    required = {
        "fold",
        "seed",
        "merge_alpha",
        "min_state_count",
        "smoothing_alpha",
        "distance_penalty",
        "selected_threshold",
    }
    missing = required.difference(frame.columns)

    if missing:
        raise ValueError(
            "Best-config CSV dosyasında eksik sütunlar var: "
            f"{sorted(missing)}"
        )

    duplicated = frame.duplicated(["fold", "seed"], keep=False)

    if duplicated.any():
        duplicate_rows = frame.loc[
            duplicated,
            ["fold", "seed"],
        ].drop_duplicates()

        raise ValueError(
            "Aynı fold/seed için birden fazla best-config satırı var:\n"
            f"{duplicate_rows.to_string(index=False)}"
        )

    return frame


def find_binary_blocks(values):
    values = np.asarray(values, dtype=int).flatten()
    blocks = []
    start = None

    for index, value in enumerate(values):
        if value == 1 and start is None:
            start = index
        elif value == 0 and start is not None:
            blocks.append((start, index))
            start = None

    if start is not None:
        blocks.append((start, len(values)))

    return blocks


def safe_top_quartile_mean(values):
    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        return 0.0

    threshold = float(np.quantile(values, 0.75))
    selected = values[values >= threshold]

    if len(selected) == 0:
        return float(np.max(values))

    return float(np.mean(selected))


def calculate_block_features(
    source_file,
    fold_id,
    seed,
    scores,
    labels,
    threshold,
):
    scores = np.asarray(scores, dtype=float).flatten()
    labels = np.asarray(labels, dtype=int).flatten()

    if len(scores) != len(labels):
        raise ValueError(
            f"{source_file} skor/etiket uzunlukları farklı: "
            f"{len(scores)}/{len(labels)}"
        )

    predictions = (scores >= threshold).astype(int)
    margins = scores - float(threshold)
    predicted_margins = margins[predictions == 1]

    if len(predicted_margins):
        strong_margin_threshold = float(
            np.quantile(predicted_margins, 0.75)
        )
    else:
        strong_margin_threshold = 0.0

    rows = []

    for block_id, (start, end) in enumerate(
        find_binary_blocks(predictions),
        start=1,
    ):
        block_scores = scores[start:end]
        block_labels = labels[start:end]
        block_margins = margins[start:end]

        true_positive_points = int(block_labels.sum())
        false_positive_points = int(
            len(block_labels) - true_positive_points
        )
        overlaps_true_anomaly = true_positive_points > 0

        rows.append(
            {
                "fold": int(fold_id),
                "seed": int(seed),
                "source_file": str(source_file),
                "block_id": int(block_id),
                "start_index": int(start),
                "end_index": int(end - 1),
                "block_length": int(end - start),
                "block_type": (
                    "true_positive_block"
                    if overlaps_true_anomaly
                    else "false_positive_block"
                ),
                "target": int(overlaps_true_anomaly),
                "true_positive_points": true_positive_points,
                "false_positive_points": false_positive_points,
                "true_point_ratio": float(
                    true_positive_points / max(end - start, 1)
                ),
                "threshold": float(threshold),
                "mean_score": float(np.mean(block_scores)),
                "max_score": float(np.max(block_scores)),
                "min_score": float(np.min(block_scores)),
                "mean_margin": float(np.mean(block_margins)),
                "median_margin": float(np.median(block_margins)),
                "max_margin": float(np.max(block_margins)),
                "min_margin": float(np.min(block_margins)),
                "sum_margin": float(np.sum(block_margins)),
                "margin_std": float(np.std(block_margins)),
                "top_quartile_margin_mean": (
                    safe_top_quartile_mean(block_margins)
                ),
                "strong_margin_threshold": (
                    strong_margin_threshold
                ),
                "strong_point_ratio": float(
                    np.mean(
                        block_margins
                        >= strong_margin_threshold
                    )
                ),
            }
        )

    return rows


def calculate_auc(targets, values):
    targets = np.asarray(targets, dtype=int)
    values = np.asarray(values, dtype=float)

    if len(np.unique(targets)) < 2:
        return math.nan

    try:
        return float(roc_auc_score(targets, values))
    except ValueError:
        return math.nan


def create_feature_summary(block_frame):
    rows = []

    for feature in FEATURE_COLUMNS:
        true_values = block_frame.loc[
            block_frame["target"] == 1,
            feature,
        ].astype(float)

        false_values = block_frame.loc[
            block_frame["target"] == 0,
            feature,
        ].astype(float)

        auc = calculate_auc(
            block_frame["target"],
            block_frame[feature],
        )

        rows.append(
            {
                "feature": feature,
                "true_block_count": int(len(true_values)),
                "false_block_count": int(len(false_values)),
                "true_mean": (
                    float(true_values.mean())
                    if len(true_values)
                    else math.nan
                ),
                "false_mean": (
                    float(false_values.mean())
                    if len(false_values)
                    else math.nan
                ),
                "mean_difference": (
                    float(
                        true_values.mean()
                        - false_values.mean()
                    )
                    if len(true_values) and len(false_values)
                    else math.nan
                ),
                "true_median": (
                    float(true_values.median())
                    if len(true_values)
                    else math.nan
                ),
                "false_median": (
                    float(false_values.median())
                    if len(false_values)
                    else math.nan
                ),
                "median_difference": (
                    float(
                        true_values.median()
                        - false_values.median()
                    )
                    if len(true_values) and len(false_values)
                    else math.nan
                ),
                "roc_auc": auc,
                "promising_direction": bool(
                    not math.isnan(auc)
                    and auc >= 0.65
                    and len(true_values)
                    and len(false_values)
                    and true_values.median()
                    > false_values.median()
                ),
            }
        )

    return pd.DataFrame(rows)


def create_fold_seed_summary(block_frame):
    rows = []

    for (fold_id, seed), group in block_frame.groupby(
        ["fold", "seed"],
        sort=True,
    ):
        true_group = group[group["target"] == 1]
        false_group = group[group["target"] == 0]

        rows.append(
            {
                "fold": int(fold_id),
                "seed": int(seed),
                "predicted_block_count": int(len(group)),
                "true_positive_block_count": int(len(true_group)),
                "false_positive_block_count": int(len(false_group)),
                "true_block_ratio": float(
                    len(true_group) / max(len(group), 1)
                ),
                "true_mean_length": (
                    float(true_group["block_length"].mean())
                    if len(true_group)
                    else math.nan
                ),
                "false_mean_length": (
                    float(false_group["block_length"].mean())
                    if len(false_group)
                    else math.nan
                ),
                "true_mean_margin": (
                    float(true_group["mean_margin"].mean())
                    if len(true_group)
                    else math.nan
                ),
                "false_mean_margin": (
                    float(false_group["mean_margin"].mean())
                    if len(false_group)
                    else math.nan
                ),
                "true_max_margin_mean": (
                    float(true_group["max_margin"].mean())
                    if len(true_group)
                    else math.nan
                ),
                "false_max_margin_mean": (
                    float(false_group["max_margin"].mean())
                    if len(false_group)
                    else math.nan
                ),
            }
        )

    return pd.DataFrame(rows)


def get_best_row(best_configs, fold_id, seed):
    selected = best_configs[
        (best_configs["fold"].astype(int) == int(fold_id))
        & (best_configs["seed"].astype(int) == int(seed))
    ]

    if len(selected) != 1:
        raise ValueError(
            f"Fold={fold_id}, seed={seed} için tam bir "
            "best-config satırı bulunamadı."
        )

    return selected.iloc[0]


def analyze_fold_seed(
    fold_id,
    seed,
    config,
    transformer,
    best_row,
):
    paths = {
        "X_train": (
            f"data/processed/"
            f"skab_fold{fold_id}_X_train_pc1.csv"
        ),
        "y_train": (
            f"data/processed/"
            f"skab_fold{fold_id}_y_train.csv"
        ),
        "train_source": (
            f"data/processed/"
            f"skab_fold{fold_id}_train_source_file.csv"
        ),
    }

    X_train = np.asarray(
        load_vector(paths["X_train"]),
        dtype=float,
    )
    y_train = to_binary(
        load_vector(paths["y_train"])
    )
    train_source = load_source_files(
        paths["train_source"]
    )

    split_info = choose_inner_split(
        X_train,
        y_train,
        train_source,
        transformer,
        config,
        fold_id,
        seed,
    )

    kwargs = {
        "merge_alpha": float(
            best_row["merge_alpha"]
        ),
        "min_state_count": int(
            best_row["min_state_count"]
        ),
        "smoothing_alpha": float(
            best_row["smoothing_alpha"]
        ),
        "max_pattern_distance": config[
            "max_pattern_distance"
        ],
        "distance_penalty": float(
            best_row["distance_penalty"]
        ),
    }

    normal_model, anomaly_model = fit_dual_models(
        split_info["normal_sequences"],
        split_info["anomaly_sequences"],
        kwargs,
    )

    threshold = float(
        best_row["selected_threshold"]
    )

    block_rows = []
    validation_transition_count = 0
    validation_anomaly_count = 0

    for item in split_info["grouped_val"]:
        scores, _ = score_dual(
            normal_model,
            anomaly_model,
            item["patterns"],
        )
        labels = np.asarray(
            item["transition_labels"],
            dtype=int,
        )

        validation_transition_count += len(labels)
        validation_anomaly_count += int(labels.sum())

        block_rows.extend(
            calculate_block_features(
                source_file=item["source_file"],
                fold_id=fold_id,
                seed=seed,
                scores=scores,
                labels=labels,
                threshold=threshold,
            )
        )

    return {
        "block_rows": block_rows,
        "validation_transition_count": (
            validation_transition_count
        ),
        "validation_anomaly_count": (
            validation_anomaly_count
        ),
        "validation_file_count": len(
            split_info["grouped_val"]
        ),
    }


def main():
    print(
        "\n--- SKAB BLOCK CONFIDENCE "
        "VALIDATION TANI ANALİZİ ---"
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    config = load_config()
    best_configs = load_best_configs(
        BEST_CONFIG_PATH
    )
    transformer = SaxPaaTransformer(
        alphabet_size=config["alphabet_size"]
    )

    all_block_rows = []
    run_summaries = []

    expected_pairs = {
        (int(row.fold), int(row.seed))
        for row in best_configs.itertuples()
    }

    for fold_id in range(
        1,
        config["n_folds"] + 1,
    ):
        for seed in config["seeds"]:
            pair = (int(fold_id), int(seed))

            if pair not in expected_pairs:
                raise ValueError(
                    "Best-config dosyasında eksik "
                    f"fold/seed çifti var: {pair}"
                )

            best_row = get_best_row(
                best_configs,
                fold_id,
                seed,
            )

            result = analyze_fold_seed(
                fold_id,
                seed,
                config,
                transformer,
                best_row,
            )

            all_block_rows.extend(
                result["block_rows"]
            )

            run_summaries.append(
                {
                    "fold": int(fold_id),
                    "seed": int(seed),
                    "validation_file_count": (
                        result[
                            "validation_file_count"
                        ]
                    ),
                    "validation_transition_count": (
                        result[
                            "validation_transition_count"
                        ]
                    ),
                    "validation_anomaly_count": (
                        result[
                            "validation_anomaly_count"
                        ]
                    ),
                    "predicted_block_count": len(
                        result["block_rows"]
                    ),
                }
            )

            true_count = sum(
                row["target"] == 1
                for row in result["block_rows"]
            )
            false_count = sum(
                row["target"] == 0
                for row in result["block_rows"]
            )

            print(
                f"Fold={fold_id} | seed={seed} | "
                f"validation file="
                f"{result['validation_file_count']} | "
                f"predicted block="
                f"{len(result['block_rows'])} | "
                f"true={true_count} | false={false_count}"
            )

    block_frame = pd.DataFrame(all_block_rows)

    if block_frame.empty:
        raise RuntimeError(
            "Validation üzerinde tahmin edilen anomaly "
            "bloğu bulunamadı."
        )

    if block_frame["target"].nunique() < 2:
        raise RuntimeError(
            "Tanı için hem true-positive hem de "
            "false-positive blok bulunması gerekir."
        )

    feature_summary = create_feature_summary(
        block_frame
    )
    fold_seed_summary = create_fold_seed_summary(
        block_frame
    )

    run_summary_frame = pd.DataFrame(
        run_summaries
    )
    fold_seed_summary = run_summary_frame.merge(
        fold_seed_summary,
        on=["fold", "seed"],
        how="left",
        suffixes=("", "_feature"),
    )

    true_blocks = block_frame[
        block_frame["target"] == 1
    ]
    false_blocks = block_frame[
        block_frame["target"] == 0
    ]

    promising = feature_summary[
        feature_summary["promising_direction"]
    ].sort_values(
        ["roc_auc", "median_difference"],
        ascending=False,
    )

    report = {
        "analysis_scope": (
            "inner_validation_only"
        ),
        "outer_test_used": False,
        "fold_seed_count": int(
            len(fold_seed_summary)
        ),
        "predicted_block_count": int(
            len(block_frame)
        ),
        "true_positive_block_count": int(
            len(true_blocks)
        ),
        "false_positive_block_count": int(
            len(false_blocks)
        ),
        "true_positive_block_ratio": float(
            len(true_blocks)
            / max(len(block_frame), 1)
        ),
        "promising_features": (
            promising["feature"].tolist()
        ),
        "feature_summary": (
            feature_summary.replace(
                {np.nan: None}
            ).to_dict(orient="records")
        ),
        "decision_rule": (
            "Bir özellik ROC-AUC >= 0.65 ve "
            "true-block medyanı false-block "
            "medyanından yüksekse ilk deney için "
            "umut verici kabul edilir."
        ),
    }

    block_frame.to_csv(
        BLOCK_OUTPUT_PATH,
        index=False,
    )
    feature_summary.to_csv(
        FEATURE_OUTPUT_PATH,
        index=False,
    )
    fold_seed_summary.to_csv(
        FOLD_SEED_OUTPUT_PATH,
        index=False,
    )

    with open(
        REPORT_OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False,
            allow_nan=False,
        )

    print("\n--- GENEL BLOK DAĞILIMI ---")
    print(
        f"Tahmin edilen blok={len(block_frame)} | "
        f"true-positive block={len(true_blocks)} | "
        f"false-positive block={len(false_blocks)}"
    )

    print("\n--- ÖZELLİK AYRIŞMA ÖZETİ ---")
    display_columns = [
        "feature",
        "true_median",
        "false_median",
        "median_difference",
        "roc_auc",
        "promising_direction",
    ]
    print(
        feature_summary[
            display_columns
        ].to_string(index=False)
    )

    print("\n--- TANI KARARI ---")
    if len(promising):
        print(
            "Blok filtreleme deneyi için umut veren "
            "özellikler: "
            + ", ".join(
                promising["feature"].tolist()
            )
        )
        print(
            "Sonraki adımda yalnızca bu özellikleri "
            "kullanan küçük bir validation grid'i "
            "denenebilir."
        )
    else:
        print(
            "True ve false blokları güvenilir biçimde "
            "ayıran bir özellik bulunamadı."
        )
        print(
            "Bu durumda Block-Level Confidence Filter "
            "parametre taramasına geçmek önerilmez."
        )

    print("\nKaydedilen tanı dosyaları:")
    print(f"- {BLOCK_OUTPUT_PATH}")
    print(f"- {FEATURE_OUTPUT_PATH}")
    print(f"- {FOLD_SEED_OUTPUT_PATH}")
    print(f"- {REPORT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()