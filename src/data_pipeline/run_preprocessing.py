from pathlib import Path

from src.data_pipeline.data_loader import (
    load_config,
    load_skab,
    get_skab_columns,
    load_batadal_training_2,
    get_batadal_columns,
)

from src.data_pipeline.preprocess import (
    check_missing_values,
    split_batadal_time_ordered,
    scale_train_val_test,
    apply_pca_train_val_test,
    save_processed_data,
    prepare_skab_xy,
    create_skab_group_folds,
    save_all_skab_folds_processed_data,
)


def process_batadal(config: dict, output_dir: Path) -> None:
    batadal_df = load_batadal_training_2(config)
    check_missing_values(batadal_df, "BATADAL")
    batadal_columns = get_batadal_columns(batadal_df, config)

    split_data = split_batadal_time_ordered(
        batadal_df=batadal_df,
        batadal_columns=batadal_columns,
    )

    scaled_data = scale_train_val_test(
        X_train=split_data["X_train"],
        X_val=split_data["X_val"],
        X_test=split_data["X_test"],
    )

    pca_data = apply_pca_train_val_test(
        X_train=scaled_data["X_train_scaled"],
        X_val=scaled_data["X_val_scaled"],
        X_test=scaled_data["X_test_scaled"],
    )

    save_processed_data(
        output_dir=output_dir,
        data={
            "batadal_X_train_scaled.csv": scaled_data["X_train_scaled"],
            "batadal_y_train.csv": split_data["y_train"],
            "batadal_X_val_scaled.csv": scaled_data["X_val_scaled"],
            "batadal_y_val.csv": split_data["y_val"],
            "batadal_X_test_scaled.csv": scaled_data["X_test_scaled"],
            "batadal_y_test.csv": split_data["y_test"],
            "batadal_X_train_pc1.csv": pca_data["X_train_pca"],
            "batadal_X_val_pc1.csv": pca_data["X_val_pca"],
            "batadal_X_test_pc1.csv": pca_data["X_test_pca"],
        },
    )

    print("BATADAL preprocessing completed.")


def process_skab(config: dict, output_dir: Path) -> None:
    skab_df = load_skab(config)
    check_missing_values(skab_df, "SKAB")
    skab_columns = get_skab_columns(skab_df, config)

    X, y, groups = prepare_skab_xy(
        skab_df=skab_df,
        skab_columns=skab_columns,
    )

    folds = create_skab_group_folds(
        X=X,
        y=y,
        groups=groups,
        n_splits=5,
    )

    save_all_skab_folds_processed_data(
        output_dir=output_dir,
        X=X,
        y=y,
        groups=groups,
        folds=folds,
    )

    print("SKAB preprocessing completed.")


def main() -> None:
    config = load_config()
    output_dir = Path(config["paths"]["processed_dir"])

    process_batadal(config, output_dir)
    process_skab(config, output_dir)

    print("All preprocessing steps completed.")


if __name__ == "__main__":
    main()