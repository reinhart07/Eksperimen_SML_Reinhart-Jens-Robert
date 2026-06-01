import pandas as pd
import numpy as np
import os
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_data(filepath="winequality_raw.csv"):
    logger.info("Loading dataset Wine Quality...")
    df = pd.read_csv(filepath, sep=';')
    logger.info(f"Dataset berhasil dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")
    return df


def check_missing_values(df):
    logger.info("Memeriksa missing values...")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        logger.warning(f"Terdapat missing values:\n{missing[missing > 0]}")
        df = df.dropna()
        logger.info("Missing values telah dihapus.")
    else:
        logger.info("Tidak ada missing values.")
    return df


def check_duplicates(df):
    logger.info("Memeriksa duplikasi data...")
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        logger.warning(f"Terdapat {duplicates} baris duplikat. Menghapus duplikasi...")
        df = df.drop_duplicates()
        logger.info("Duplikasi berhasil dihapus.")
    else:
        logger.info("Tidak ada duplikasi data.")
    return df


def remove_outliers(df, columns):
    logger.info("Menghapus outlier menggunakan IQR...")
    initial_rows = len(df)
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    removed = initial_rows - len(df)
    logger.info(f"Outlier dihapus: {removed} baris. Sisa: {len(df)} baris.")
    return df


def binarize_target(df):
    logger.info("Mengubah target menjadi biner (0=buruk, 1=baik)...")
    df['quality'] = (df['quality'] >= 6).astype(int)
    logger.info(f"Distribusi kelas:\n{df['quality'].value_counts()}")
    return df


def split_data(df, feature_cols, target_col, test_size=0.2, random_state=42):
    logger.info(f"Membagi data: test_size={test_size}, random_state={random_state}")
    X = df[feature_cols]
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(f"Train: {X_train.shape[0]} baris | Test: {X_test.shape[0]} baris")
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    logger.info("Melakukan feature scaling (StandardScaler)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    logger.info("Feature scaling selesai.")
    return X_train_scaled, X_test_scaled, scaler


def save_preprocessed_data(X_train, X_test, y_train, y_test, feature_cols, output_dir="winequality_preprocessing"):
    os.makedirs(output_dir, exist_ok=True)

    train_df = pd.DataFrame(X_train, columns=feature_cols)
    train_df['quality'] = y_train.values
    test_df = pd.DataFrame(X_test, columns=feature_cols)
    test_df['quality'] = y_test.values

    train_path = os.path.join(output_dir, "winequality_train.csv")
    test_path = os.path.join(output_dir, "winequality_test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info(f"Data train disimpan ke: {train_path}")
    logger.info(f"Data test disimpan ke: {test_path}")
    return train_path, test_path


def run_preprocessing(raw_filepath="winequality_raw.csv", output_dir="winequality_preprocessing"):
    logger.info("=" * 50)
    logger.info("MULAI PREPROCESSING OTOMATIS")
    logger.info("=" * 50)

    df = load_data(raw_filepath)
    df = check_missing_values(df)
    df = check_duplicates(df)

    feature_cols = [col for col in df.columns if col != 'quality']
    df = remove_outliers(df, feature_cols)
    df = binarize_target(df)

    X_train, X_test, y_train, y_test = split_data(df, feature_cols, 'quality')
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    train_path, test_path = save_preprocessed_data(
        X_train_scaled, X_test_scaled, y_train, y_test, feature_cols, output_dir
    )

    logger.info("=" * 50)
    logger.info("PREPROCESSING SELESAI")
    logger.info("=" * 50)
    return train_path, test_path


if __name__ == "__main__":
    run_preprocessing(
        raw_filepath="winequality_raw.csv",
        output_dir="winequality_preprocessing"
    )