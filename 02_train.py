"""
02_train.py - Model Training & Evaluation (scikit-learn)
Spotify Recommendation System

Reads Parquet output from PySpark ingestion stage.
Stage 1: Binary classifier (liked/not liked) — Random Forest + XGBoost
Stage 2: Engagement scorer (ranks liked tracks by composite signal)
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve
)
from sklearn.inspection import permutation_importance
from scipy import stats
import xgboost as xgb

warnings.filterwarnings("ignore")

PARQUET_PATH = "data/parquet/spotify_features.parquet"
OUTPUTS_DIR = "outputs"
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_PCA_COMPONENTS = 8
CV_FOLDS = 5


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_data():
    """Load Parquet output from PySpark stage using DuckDB."""
    print("[STEP 1] Loading Parquet data...")
    import duckdb
    con = duckdb.connect()
    df = con.execute(f"SELECT * FROM read_parquet('{PARQUET_PATH}/*.parquet')").df()
    con.close()
    feat_cols = [c for c in df.columns if c.startswith("feat_")]
    print(f"  Rows: {len(df):,}  |  Feature columns: {len(feat_cols)}")
    print(f"  Class balance — Liked: {df['liked'].mean()*100:.1f}%")
    return df, feat_cols


def hypothesis_testing(df, feat_cols):
    """
    Statistical validation: t-tests comparing liked vs not liked
    for each audio feature. Reports significant features.
    """
    print("\n[STEP 2] Hypothesis testing (t-test per feature)...")
    results = []
    liked = df[df["liked"] == 1]
    not_liked = df[df["liked"] == 0]

    for col in feat_cols:
        t_stat, p_val = stats.ttest_ind(
            liked[col].dropna(),
            not_liked[col].dropna(),
            equal_var=False
        )
        results.append({
            "feature": col.replace("feat_", ""),
            "t_statistic": round(t_stat, 4),
            "p_value": round(p_val, 6),
            "significant": p_val < 0.05,
            "liked_mean": round(liked[col].mean(), 4),
            "not_liked_mean": round(not_liked[col].mean(), 4)
        })

    results_df = pd.DataFrame(results).sort_values("p_value")
    sig = results_df[results_df["significant"]]
    print(f"  Significant features (p < 0.05): {len(sig)} / {len(results_df)}")
    for _, row in sig.head(5).iterrows():
        print(f"    {row['feature']:30s}  p={row['p_value']:.6f}  "
              f"liked_mean={row['liked_mean']:.3f}  not_liked_mean={row['not_liked_mean']:.3f}")

    results_df.to_csv(f"{OUTPUTS_DIR}/hypothesis_tests.csv", index=False)
    return results_df


def run_pca(X_train, X_test):
    """Apply PCA for dimensionality reduction and variance analysis."""
    print(f"\n[STEP 3] PCA — reducing to {N_PCA_COMPONENTS} components...")
    pca = PCA(n_components=N_PCA_COMPONENTS, random_state=RANDOM_STATE)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    explained = pca.explained_variance_ratio_.cumsum()
    print(f"  Cumulative variance explained by {N_PCA_COMPONENTS} components: "
          f"{explained[-1]*100:.1f}%")

    # Plot scree
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(1, N_PCA_COMPONENTS + 1),
           pca.explained_variance_ratio_ * 100, color="steelblue")
    ax.plot(range(1, N_PCA_COMPONENTS + 1),
            explained * 100, "ro-", label="Cumulative")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance Explained (%)")
    ax.set_title("PCA Scree Plot — Spotify Audio Features")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUTS_DIR}/pca_scree.png", dpi=150)
    plt.close()

    return X_train_pca, X_test_pca, pca


def train_random_forest(X_train, y_train, X_test, y_test):
    """Train Random Forest with cross-validation."""
    print("\n[STEP 4a] Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=10,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(rf, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)
    print(f"  CV F1 scores: {cv_scores.round(3)}  |  Mean: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
    print(f"  Test F1: {f1:.3f}  |  AUC-ROC: {auc:.3f}")
    return rf, f1, auc


def train_xgboost(X_train, y_train, X_test, y_test):
    """Train XGBoost with cross-validation."""
    print("\n[STEP 4b] Training XGBoost...")

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0
    )

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)
    print(f"  CV F1 scores: {cv_scores.round(3)}  |  Mean: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1])
    print(f"  Test F1: {f1:.3f}  |  AUC-ROC: {auc:.3f}")
    return xgb_model, f1, auc


def plot_results(rf_model, xgb_model, X_test, y_test, feat_cols):
    """Generate evaluation plots."""
    print("\n[STEP 5] Generating evaluation plots...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Confusion matrix — XGBoost
    y_pred_xgb = xgb_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_xgb)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Not Liked", "Liked"],
                yticklabels=["Not Liked", "Liked"], ax=axes[0])
    axes[0].set_title("XGBoost Confusion Matrix")
    axes[0].set_ylabel("Actual")
    axes[0].set_xlabel("Predicted")

    # ROC curves
    for model, label, color in [
        (rf_model, "Random Forest", "steelblue"),
        (xgb_model, "XGBoost", "darkorange")
    ]:
        fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        axes[1].plot(fpr, tpr, label=f"{label} (AUC={auc:.3f})", color=color, lw=2)
    axes[1].plot([0, 1], [0, 1], "k--", lw=1)
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("ROC Curves")
    axes[1].legend()

    # Feature importance — XGBoost
    importance = xgb_model.feature_importances_
    feat_names = [c.replace("feat_", "") for c in feat_cols[:len(importance)]]
    idx = np.argsort(importance)[-10:]
    axes[2].barh([feat_names[i] for i in idx],
                 importance[idx], color="darkorange")
    axes[2].set_title("XGBoost Feature Importance (Top 10)")
    axes[2].set_xlabel("Importance Score")

    plt.tight_layout()
    plt.savefig(f"{OUTPUTS_DIR}/model_evaluation.png", dpi=150)
    plt.close()
    print(f"  Saved -> {OUTPUTS_DIR}/model_evaluation.png")


def stage2_engagement_ranking(df, xgb_model, feat_cols, X_test_pca, y_test, test_idx):
    """
    Stage 2: Among predicted-liked tracks, rank by engagement score.
    Simulates a two-tower recommendation: filter → rank.
    """
    print("\n[STEP 6] Stage 2 — Engagement ranking on predicted-liked tracks...")

    test_df = df.iloc[test_idx].copy()
    test_df["predicted_liked"] = xgb_model.predict(X_test_pca)
    test_df["like_probability"] = xgb_model.predict_proba(X_test_pca)[:, 1]

    # Filter to predicted liked
    liked_df = test_df[test_df["predicted_liked"] == 1].copy()

    # Sort by engagement score (Stage 2 ranking signal)
    liked_df = liked_df.sort_values("engagement_score", ascending=False)

    print(f"  Predicted liked: {len(liked_df):,} tracks")
    print(f"\n  Top 10 recommended tracks (filter → rank):")
    print(f"  {'Track':<45} {'Engagement':>10} {'Like Prob':>10} {'Popularity':>10}")
    print(f"  {'-'*80}")

    display_col = "track_name" if "track_name" in liked_df.columns else liked_df.columns[0]
    for _, row in liked_df.head(10).iterrows():
        name = str(row.get("track_name", "Unknown"))[:43]
        print(f"  {name:<45} {row['engagement_score']:>10.3f} "
              f"{row['like_probability']:>10.3f} {int(row['popularity']):>10}")

    # Save top recommendations
    output_cols = [c for c in ["track_name", "artists", "track_genre",
                                "engagement_score", "like_probability",
                                "popularity", "energy", "danceability"]
                   if c in liked_df.columns]
    liked_df[output_cols].head(100).to_csv(
        f"{OUTPUTS_DIR}/top_recommendations.csv", index=False
    )
    print(f"\n  Saved top 100 recommendations -> {OUTPUTS_DIR}/top_recommendations.csv")
    return liked_df


def save_results(rf_f1, rf_auc, xgb_f1, xgb_auc, hypothesis_df):
    """Save run manifest with key metrics."""
    results = {
        "random_forest": {"f1": round(rf_f1, 4), "auc_roc": round(rf_auc, 4)},
        "xgboost": {"f1": round(xgb_f1, 4), "auc_roc": round(xgb_auc, 4)},
        "best_model": "xgboost" if xgb_f1 >= rf_f1 else "random_forest",
        "significant_features": hypothesis_df[hypothesis_df["significant"]]["feature"].tolist(),
        "pca_components": N_PCA_COMPONENTS,
        "test_size": TEST_SIZE,
        "cv_folds": CV_FOLDS
    }
    with open(f"{OUTPUTS_DIR}/run_manifest.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved run manifest -> {OUTPUTS_DIR}/run_manifest.json")
    return results


def main():
    print("=" * 60)
    print("  Spotify Recommendation System — Training (scikit-learn)")
    print("=" * 60)

    df, feat_cols = load_data()

    hypothesis_df = hypothesis_testing(df, feat_cols)

    X = df[feat_cols].values
    y = df["liked"].values

    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        X, y, np.arange(len(df)),
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    X_train_pca, X_test_pca, pca = run_pca(X_train, X_test)

    rf_model, rf_f1, rf_auc = train_random_forest(
        X_train_pca, y_train, X_test_pca, y_test
    )
    xgb_model, xgb_f1, xgb_auc = train_xgboost(
        X_train_pca, y_train, X_test_pca, y_test
    )

    print(f"\n[RESULTS SUMMARY]")
    print(f"  Random Forest : F1={rf_f1:.3f}  AUC={rf_auc:.3f}")
    print(f"  XGBoost       : F1={xgb_f1:.3f}  AUC={xgb_auc:.3f}")

    best_model = xgb_model if xgb_f1 >= rf_f1 else rf_model
    best_name = "XGBoost" if xgb_f1 >= rf_f1 else "Random Forest"
    print(f"  Best model    : {best_name}")

    print(f"\n[CLASSIFICATION REPORT — {best_name}]")
    print(classification_report(y_test, best_model.predict(X_test_pca),
                                target_names=["Not Liked", "Liked"]))

    plot_results(rf_model, xgb_model, X_test_pca, y_test, feat_cols)

    stage2_engagement_ranking(df, xgb_model, feat_cols, X_test_pca, y_test, test_idx)

    save_results(rf_f1, rf_auc, xgb_f1, xgb_auc, hypothesis_df)

    print("\n[02_train.py] Complete.")
    print(f"  Outputs saved to: {OUTPUTS_DIR}/")


if __name__ == "__main__":
    main()
