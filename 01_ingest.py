"""
01_ingest.py - Data Ingestion & Cleaning (PySpark)
Spotify Recommendation System

Downloads the Spotify Tracks dataset, ingests it with PySpark,
performs cleaning, feature engineering, and exports to Parquet
for the scikit-learn training stage.
"""

import os
import urllib.request
import zipfile
import sys

# ── PySpark setup ────────────────────────────────────────────────────────────
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml import Pipeline as SparkPipeline

# ── Constants ────────────────────────────────────────────────────────────────
DATA_URL = "https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset/resolve/main/dataset.csv"
# Fallback: if URL is blocked, script generates a realistic synthetic dataset
RAW_PATH = "data/raw/spotify_tracks.csv"
PARQUET_PATH = "data/parquet/spotify_features.parquet"

AUDIO_FEATURES = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo"
]

POPULARITY_THRESHOLD = 60


def download_data():
    """Download Spotify tracks dataset, with synthetic fallback."""
    if os.path.exists(RAW_PATH):
        print(f"  [SKIP] {RAW_PATH} already exists.")
        return
    os.makedirs("data/raw", exist_ok=True)
    try:
        print(f"  Downloading dataset from HuggingFace...")
        urllib.request.urlretrieve(DATA_URL, RAW_PATH)
        print(f"  Saved -> {RAW_PATH}")
    except Exception as e:
        print(f"  Download failed ({e}). Generating realistic synthetic dataset...")
        _generate_synthetic()


def _generate_synthetic():
    """Generate a realistic 50K-track synthetic Spotify dataset."""
    import pandas as pd
    import numpy as np
    np.random.seed(42)
    n = 50000
    popularity = np.clip(np.random.normal(40, 22, n), 0, 100).astype(int)
    danceability = np.clip(np.random.beta(5, 3, n), 0, 1)
    energy = np.clip(np.random.beta(4, 3, n), 0, 1)
    # Popular tracks are slightly more danceable and energetic
    popular_mask = popularity > 60
    danceability[popular_mask] = np.clip(
        danceability[popular_mask] + np.random.normal(0.08, 0.02, popular_mask.sum()), 0, 1)
    energy[popular_mask] = np.clip(
        energy[popular_mask] + np.random.normal(0.06, 0.02, popular_mask.sum()), 0, 1)
    df = pd.DataFrame({
        'track_id': [f'track_{i:06d}' for i in range(n)],
        'track_name': [f'Track {i}' for i in range(n)],
        'artists': [f'Artist {np.random.randint(1, 2000)}' for _ in range(n)],
        'track_genre': np.random.choice(
            ['pop','rock','hip-hop','electronic','jazz','classical','country','r&b','indie','metal'], n),
        'popularity': popularity,
        'danceability': danceability,
        'energy': energy,
        'loudness': np.random.normal(-8, 5, n).clip(-60, 0),
        'speechiness': np.clip(np.random.exponential(0.1, n), 0, 1),
        'acousticness': np.clip(np.random.beta(2, 4, n), 0, 1),
        'instrumentalness': np.clip(np.random.exponential(0.15, n), 0, 1),
        'liveness': np.clip(np.random.beta(2, 8, n), 0, 1),
        'valence': np.clip(np.random.beta(4, 4, n), 0, 1),
        'tempo': np.random.normal(120, 30, n).clip(40, 220),
    })
    df.to_csv(RAW_PATH, index=False)
    print(f"  Generated {len(df):,} synthetic tracks -> {RAW_PATH}")


def build_spark():
    """Initialize Spark session."""
    return (
        SparkSession.builder
        .appName("SpotifyRecommendation")
        .config("spark.driver.memory", "2g")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )


def ingest_and_clean(spark):
    """Load CSV, enforce types, handle nulls and outliers."""
    print("\n[STEP 1] Ingesting raw CSV...")
    df = spark.read.csv(RAW_PATH, header=True, inferSchema=True)
    raw_count = df.count()
    print(f"  Raw rows: {raw_count:,}")

    # Drop rows missing key columns
    df = df.dropna(subset=["popularity", "track_name"] + AUDIO_FEATURES)

    # Cast audio features to double
    for col in AUDIO_FEATURES:
        df = df.withColumn(col, df[col].cast(DoubleType()))

    # Cast popularity to integer
    df = df.withColumn("popularity", df["popularity"].cast("integer"))

    # Remove duplicates on track_id
    if "track_id" in df.columns:
        df = df.dropDuplicates(["track_id"])

    # Filter outliers: tempo must be > 0, loudness in reasonable range
    df = df.filter(
        (F.col("tempo") > 0) &
        (F.col("loudness") >= -60) &
        (F.col("loudness") <= 0) &
        (F.col("popularity") >= 0) &
        (F.col("popularity") <= 100)
    )

    clean_count = df.count()
    print(f"  Clean rows: {clean_count:,}  (dropped {raw_count - clean_count:,})")
    return df


def engineer_features(df):
    """
    Feature engineering:
    1. Binary target: liked = popularity > threshold
    2. Engagement score: composite of popularity + energy + danceability
    3. Derived features: energy_danceability interaction, tempo bucket
    """
    print("\n[STEP 2] Engineering features...")

    # Stage 1 target: liked
    df = df.withColumn(
        "liked",
        F.when(F.col("popularity") > POPULARITY_THRESHOLD, 1).otherwise(0)
    )

    # Normalize popularity, energy, danceability to [0, 1]
    df = df.withColumn("popularity_norm", F.col("popularity") / 100.0)
    df = df.withColumn("energy_norm", F.col("energy"))         # already 0-1
    df = df.withColumn("danceability_norm", F.col("danceability"))  # already 0-1

    # Stage 2 target: engagement score (composite ranking signal)
    df = df.withColumn(
        "engagement_score",
        (F.col("popularity_norm") * 0.5) +
        (F.col("energy_norm") * 0.25) +
        (F.col("danceability_norm") * 0.25)
    )

    # Derived feature: energy x danceability interaction
    df = df.withColumn(
        "energy_dance_interaction",
        F.col("energy") * F.col("danceability")
    )

    # Derived feature: tempo bucket (slow/medium/fast)
    df = df.withColumn(
        "tempo_bucket",
        F.when(F.col("tempo") < 90, 0.0)
         .when(F.col("tempo") < 130, 1.0)
         .otherwise(2.0)
    )

    liked_count = df.filter(F.col("liked") == 1).count()
    total = df.count()
    print(f"  Liked tracks: {liked_count:,} / {total:,}  ({liked_count/total*100:.1f}%)")
    print(f"  Engagement score range: computed successfully")

    return df


def normalize_features(df):
    """
    Use PySpark ML Pipeline to standardize audio features.
    Returns df with a 'features_scaled' vector column.
    """
    print("\n[STEP 3] Normalizing audio features with PySpark ML...")

    feature_cols = AUDIO_FEATURES + ["energy_dance_interaction", "tempo_bucket"]

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_raw")
    scaler = StandardScaler(
        inputCol="features_raw",
        outputCol="features_scaled",
        withMean=True,
        withStd=True
    )

    pipeline = SparkPipeline(stages=[assembler, scaler])
    model = pipeline.fit(df)
    df = model.transform(df)

    print(f"  Feature vector dimensions: {len(feature_cols)}")
    return df, feature_cols


def export_parquet(df, feature_cols):
    """
    Export selected columns to Parquet for scikit-learn stage.
    Converts Spark vector to individual float columns.
    """
    print("\n[STEP 4] Exporting to Parquet for scikit-learn...")

    # Extract vector elements as individual columns
    for i, col_name in enumerate(feature_cols):
        df = df.withColumn(
            f"feat_{col_name}",
            F.udf(lambda v, idx=i: float(v[idx]) if v is not None else None,
                  returnType=DoubleType())(F.col("features_scaled"))
        )

    export_cols = (
        ["liked", "engagement_score", "popularity", "track_name"] +
        [f"feat_{c}" for c in feature_cols]
    )

    # Add track_id and artists if available
    for opt_col in ["track_id", "artists", "track_genre"]:
        if opt_col in df.columns:
            export_cols.append(opt_col)

    os.makedirs("data/parquet", exist_ok=True)
    df.select(export_cols).write.mode("overwrite").parquet(PARQUET_PATH)

    exported = df.count()
    print(f"  Exported {exported:,} rows -> {PARQUET_PATH}")
    return exported


def audit(df):
    """Print key data quality stats."""
    print("\n[AUDIT] Data quality summary:")
    total = df.count()
    nulls = {c: df.filter(F.col(c).isNull()).count() for c in ["liked", "energy", "danceability"]}
    liked = df.filter(F.col("liked") == 1).count()
    print(f"  Total rows     : {total:,}")
    print(f"  Liked (1)      : {liked:,}  ({liked/total*100:.1f}%)")
    print(f"  Not liked (0)  : {total-liked:,}  ({(total-liked)/total*100:.1f}%)")
    for col, n in nulls.items():
        print(f"  Null {col:15s}: {n}")


def main():
    print("=" * 60)
    print("  Spotify Recommendation System — Ingestion (PySpark)")
    print("=" * 60)

    download_data()

    spark = build_spark()
    spark.sparkContext.setLogLevel("ERROR")

    try:
        df = ingest_and_clean(spark)
        df = engineer_features(df)
        df, feature_cols = normalize_features(df)
        audit(df)
        export_parquet(df, feature_cols)
        print("\n[01_ingest.py] Complete. Run 02_train.py next.")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
