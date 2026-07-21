from src.data_pipeline.data_loader import (
    load_config,
    load_batadal_training_2
)

from src.data_pipeline.v1_preprocessing import (
    process_batadal_data2
)


def main():

    config = load_config()

    batadal_df = load_batadal_training_2(config)

    feature_cols = [
        c for c in batadal_df.columns
        if c not in ["DATETIME", "ATT_FLAG"]
    ]

    process_batadal_data2(
        batadal_df=batadal_df,
        feature_cols=feature_cols,
        target_col="ATT_FLAG",

        experiment_name="robust_adasyn",

    )


if __name__ == "__main__":
    main()