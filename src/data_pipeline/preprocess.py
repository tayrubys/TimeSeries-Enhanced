from pathlib import Path
from typing import Any, Dict, Tuple
import pandas as pd
from sklearn.preprocessing import StandardScaler #Her feature sütununu ortalaması 0,standart sapması 1 olacak şekilde dönüştürür
from sklearn.model_selection import GroupKFold
from sklearn.decomposition import PCA

# x ve y diye ayırma kısmı
def split_features_target(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    return X, y

#her sütunda kaç eksik değer olduğunu sayma
def check_missing_values(df: pd.DataFrame, dataset_name: str) -> pd.Series:
    missing_counts = df.isna().sum()
    missing_counts = missing_counts[missing_counts > 0]

    if missing_counts.empty:
        print(f"{dataset_name}: Eksik veri yok.")
    else:
        print(f"{dataset_name}: Eksik veri bulundu.")
        print(missing_counts)

    return missing_counts

#aynı CSV dosyasındaki satırlar hem train hem test içine karışmasın
def prepare_skab_xy(
    skab_df: pd.DataFrame,
    skab_columns: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    feature_cols = skab_columns["feature_cols"]
    target_col = skab_columns["target_col"]

    X, y = split_features_target(
        df=skab_df,
        feature_cols=feature_cols,
        target_col=target_col,
    )

    groups = skab_df["source_file"].copy()

    return X, y, groups

#zaman sıralı bolme data leakage onlemek icin
def split_batadal_time_ordered(
    batadal_df: pd.DataFrame,
    batadal_columns: Dict[str, Any],
    train_ratio: float = 0.60,
    val_ratio: float = 0.20,
) -> Dict[str, pd.DataFrame | pd.Series]:
    feature_cols = batadal_columns["feature_cols"]
    target_col = batadal_columns["target_col"]

    X, y = split_features_target(
        df=batadal_df,
        feature_cols=feature_cols,
        target_col=target_col,
    )

    n_rows = len(batadal_df)

    train_end = int(n_rows * train_ratio)
    val_end = int(n_rows * (train_ratio + val_ratio))

    return {
        "X_train": X.iloc[:train_end].copy(),
        "y_train": y.iloc[:train_end].copy(),
        "X_val": X.iloc[train_end:val_end].copy(),
        "y_val": y.iloc[train_end:val_end].copy(),
        "X_test": X.iloc[val_end:].copy(),
        "y_test": y.iloc[val_end:].copy(),
    }

#
def scale_train_val_test(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Dict[str, pd.DataFrame | StandardScaler]:
    scaler = StandardScaler()
   #fit-train verisinin ortalama ve standart sapmasını öğren transform-train verisini bu bilgilere göre dönüştür
    X_train_scaled_array = scaler.fit_transform(X_train)
    X_val_scaled_array = scaler.transform(X_val)
    X_test_scaled_array = scaler.transform(X_test)

    #sutun ısımlerını kaybetmemek ıcın
    X_train_scaled = pd.DataFrame(
        X_train_scaled_array,
        columns=X_train.columns,
        index=X_train.index,
    )

    X_val_scaled = pd.DataFrame(
        X_val_scaled_array,
        columns=X_val.columns,
        index=X_val.index,
    )

    X_test_scaled = pd.DataFrame(
        X_test_scaled_array,
        columns=X_test.columns,
        index=X_test.index,
    )

    return {
        "X_train_scaled": X_train_scaled,
        "X_val_scaled": X_val_scaled,
        "X_test_scaled": X_test_scaled,
        "scaler": scaler,
    }

#PCA
def apply_pca_train_val_test(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    n_components: int = 1,
) -> Dict[str, pd.DataFrame | PCA]:
    pca = PCA(n_components=n_components)
    #PCA’yı sadece train verisinden öğreniyor data data leakage onlemek ici
    X_train_pca_array = pca.fit_transform(X_train)
    #Validation ve test’e aynı PCA dönüşümünü uyguluyor.
    X_val_pca_array = pca.transform(X_val)
    X_test_pca_array = pca.transform(X_test)

    columns = [f"PC{i + 1}" for i in range(n_components)]

    X_train_pca = pd.DataFrame(
        X_train_pca_array,
        columns=columns,
        index=X_train.index,
    )

    X_val_pca = pd.DataFrame(
        X_val_pca_array,
        columns=columns,
        index=X_val.index,
    )

    X_test_pca = pd.DataFrame(
        X_test_pca_array,
        columns=columns,
        index=X_test.index,
    )

    return {
        "X_train_pca": X_train_pca,
        "X_val_pca": X_val_pca,
        "X_test_pca": X_test_pca,
        "pca": pca,
    }

#data kaydetme
def save_processed_data(
    output_dir: Path,
    data: Dict[str, pd.DataFrame | pd.Series],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    #verileri gezer
    for file_name, value in data.items():
        file_path = output_dir / file_name
        #y_train gibi hedef verileri Series tipinde olur
        if isinstance(value, pd.Series):
            value.to_frame().to_csv(file_path, index=False)
        else:
            value.to_csv(file_path, index=False)

#skab ıcın bes fold olusturma
def create_skab_group_folds(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    n_splits: int = 5,
) -> list[Dict[str, Any]]:
    group_kfold = GroupKFold(n_splits=n_splits)

    folds = []

    for fold_id, (train_idx, test_idx) in enumerate(
        #x ve y verisini bölerken groups bilgisine dikkat et
        group_kfold.split(X, y, groups),
        start=1,
    ):
        #train dosyaları ile test dosyaları çakışıyor mu
        train_groups = set(groups.iloc[train_idx])
        test_groups = set(groups.iloc[test_idx])

        common_groups = train_groups.intersection(test_groups)

        if common_groups:
            raise ValueError(
                f"Fold {fold_id} içinde train ve test dosyaları çakışıyor: "
                f"{common_groups}"
            )

        folds.append(
            {
                "fold_id": fold_id,
                "train_idx": train_idx, #model egitilirken kullanılacak
                "test_idx": test_idx, #model egitilirken kullanılacak
                "train_files": sorted(train_groups),
                "test_files": sorted(test_groups),
            }
        )

    return folds

#her flod ıcın test/train ayrımı
def scale_skab_fold(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    fold: Dict[str, Any],
) -> Dict[str, Any]:
    #GroupKFold’dan gelen satır indekslerini alır
    train_idx = fold["train_idx"]
    test_idx = fold["test_idx"]

    X_train = X.iloc[train_idx].copy()
    y_train = y.iloc[train_idx].copy()

    X_test = X.iloc[test_idx].copy()
    y_test = y.iloc[test_idx].copy()
    train_source_file = groups.iloc[train_idx].copy()
    test_source_file = groups.iloc[test_idx].copy()
    #fit_transform sadece train üzerinde
    #transform test üzerinde data leakage engellemek icin 
    scaler = StandardScaler()

    X_train_scaled_array = scaler.fit_transform(X_train)
    X_test_scaled_array = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(
        X_train_scaled_array,
        columns=X_train.columns,
        index=X_train.index,
    )

    X_test_scaled = pd.DataFrame(
        X_test_scaled_array,
        columns=X_test.columns,
        index=X_test.index,
    )

    return {
        "X_train_scaled": X_train_scaled,
        "y_train": y_train,
        "X_test_scaled": X_test_scaled,
        "y_test": y_test,
        "scaler": scaler,
        "train_files": fold["train_files"],
        "test_files": fold["test_files"],
        "train_source_file": train_source_file,
        "test_source_file": test_source_file,
    }

#skab ıcın PCA
def apply_pca_skab_fold(
    X_train_scaled: pd.DataFrame,
    X_test_scaled: pd.DataFrame,
    n_components: int = 1,
) -> Dict[str, pd.DataFrame | PCA]:
    pca = PCA(n_components=n_components)
    #PCA yı sadece train verisinden öğren
    X_train_pca_array = pca.fit_transform(X_train_scaled)
    #test verisine aynı PCA dönüşümünü uygula
    X_test_pca_array = pca.transform(X_test_scaled)

    columns = [f"PC{i + 1}" for i in range(n_components)]

    X_train_pca = pd.DataFrame(
        X_train_pca_array,
        columns=columns,
        index=X_train_scaled.index,
    )

    X_test_pca = pd.DataFrame(
        X_test_pca_array,
        columns=columns,
        index=X_test_scaled.index,
    )

    return {
        "X_train_pca": X_train_pca,
        "X_test_pca": X_test_pca,
        "pca": pca,
    }
#skab ıcın kaydetme save_processed_data kullanıyoruz
def save_skab_fold_processed_data(
    output_dir: Path,
    fold_id: int,
    fold_data: Dict[str, Any],
    pca_data: Dict[str, Any],
) -> None:
    save_processed_data(
        output_dir=output_dir,
        data={
            f"skab_fold{fold_id}_X_train_scaled.csv": fold_data["X_train_scaled"],
            f"skab_fold{fold_id}_y_train.csv": fold_data["y_train"],
            f"skab_fold{fold_id}_X_test_scaled.csv": fold_data["X_test_scaled"],
            f"skab_fold{fold_id}_y_test.csv": fold_data["y_test"],
            f"skab_fold{fold_id}_X_train_pc1.csv": pca_data["X_train_pca"],
            f"skab_fold{fold_id}_X_test_pc1.csv": pca_data["X_test_pca"],
            # SKAB sequence/window üretirken dosya sınırlarını korumak için
            f"skab_fold{fold_id}_train_source_file.csv": fold_data["train_source_file"],
            f"skab_fold{fold_id}_test_source_file.csv": fold_data["test_source_file"],
        },
    )

#bes flodun hepsını kaydetme
def save_all_skab_folds_processed_data(
    output_dir: Path,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    folds: list[Dict[str, Any]],
) -> None:
    for fold in folds:
        #her flod ıcın normolızasyon yapıyor
        fold_data = scale_skab_fold(
            X=X,
            y=y,
            groups=groups,
            fold=fold,
        )
        #sonra pca
        pca_data = apply_pca_skab_fold(
            X_train_scaled=fold_data["X_train_scaled"],
            X_test_scaled=fold_data["X_test_scaled"],
        )
        #en sonda kaydteme
        save_skab_fold_processed_data(
            output_dir=output_dir,
            fold_id=fold["fold_id"],
            fold_data=fold_data,
            pca_data=pca_data,
        )
