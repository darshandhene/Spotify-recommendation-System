Spotify Recommendation System
An end-to-end ML pipeline that predicts whether a user will like a track based on audio features, using a two-stage approach: binary classification to filter likely-liked tracks, followed by engagement scoring to rank them by predicted appeal.

Built on a real-world dataset of 114,000 Spotify tracks with a clean PySpark → scikit-learn handoff that mirrors production data architecture.

The Problem
Recommending music is harder than it looks. A model that just returns "popular" tracks is useless — popularity is the target, not a feature. A model that optimizes purely for audio similarity surfaces obscure tracks with no real-world traction.

This pipeline solves it in two stages:

Stage 1 — Binary classifier: Does the user like this track? (popularity > 60 as proxy)

Stage 2 — Engagement ranker: Among liked tracks, which ones will they engage with most? Composite score:

engagement_score = (popularity_norm × 0.5) + (energy_norm × 0.25) + (danceability_norm × 0.25)
This mirrors a two-tower recommendation architecture: filter → rank.

Architecture
Raw CSV (114K tracks)
       │
       ▼
┌─────────────────────────────┐
│  PySpark (01_ingest.py)     │
│  - Null/outlier removal     │
│  - Feature engineering      │
│  - StandardScaler (ML Pipeline) │
│  - Parquet export           │
└────────────┬────────────────┘
             │
             ▼ Parquet
┌─────────────────────────────┐
│  scikit-learn (02_train.py) │
│  - Hypothesis testing       │
│  - PCA (8 components)       │
│  - Random Forest + XGBoost  │
│  - Stage 2 engagement rank  │
└─────────────────────────────┘
             │
             ▼
     outputs/ (plots, CSV, manifest)
Tech Stack
Layer	Tools
Distributed processing	PySpark 3.4+, PySpark ML Pipeline
Feature engineering	StandardScaler, VectorAssembler
Statistical analysis	scipy (t-test), NumPy
Dimensionality reduction	scikit-learn PCA
Models	Random Forest, XGBoost
Visualization	Matplotlib, Seaborn
Data export	DuckDB (Parquet reader)
Dataset
Source: Spotify Tracks Dataset — Kaggle

Stat	Value
Raw tracks	114,000
Clean tracks	89,395
Liked tracks (popularity > 60)	8,827 (9.9%)
Audio features used	11
Audio features: danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo

Engineered features: energy × danceability interaction, tempo bucket (slow / medium / fast)

Results
Stage 1 — Binary Classification
Model	Weighted F1	AUC-ROC	CV F1 (mean ± std)
Random Forest	0.72	0.694	0.256 ± 0.004
XGBoost	0.72	0.691	0.256 ± 0.003
PCA: 8 components explain 96.8% of variance.

Stage 2 — Engagement Ranking (Sample Output)
Track	Engagement Score	Like Probability	Popularity
Me Porto Bonito	0.891	0.642	97
One Kiss (with Dua Lipa)	0.858	0.655	89
I Ain't Worried	0.855	0.608	96
Quevedo: Bzrp Music Sessions Vol. 52	0.846	0.647	99
Moscow Mule	0.839	0.806	94
Statistical Findings
Hypothesis testing (Welch's t-test, p < 0.05) across all 11 features revealed significant differences between liked and not-liked tracks:

Positive predictors of popularity:

danceability — strongest positive signal (liked mean: 0.201 vs not-liked: -0.022)
loudness — louder tracks correlate with higher popularity
energy — energetic tracks are more likely to be liked
Negative predictors:

acousticness — acoustic tracks less likely to go mainstream
instrumentalness — instrumental tracks significantly less popular
Key insight: Danceability and loudness are the strongest predictors of whether a track will be liked, outperforming genre tags and tempo. Acoustic and instrumental content negatively predicts mainstream popularity — which aligns with how streaming charts actually work.

How to Run
Prerequisites
Python 3.9+
Java 17+ (required for PySpark)
On Mac: brew install openjdk@17 libomp
# Clone and install
git clone https://github.com/[your-username]/spotify-recommendation
cd spotify-recommendation
pip install -r requirements.txt

# Stage 1: PySpark ingestion and feature engineering
python3 01_ingest.py

# Stage 2: scikit-learn training and evaluation
python3 02_train.py
Outputs
outputs/
├── model_evaluation.png      # Confusion matrix, ROC curves, feature importance
├── pca_scree.png             # PCA variance explained plot
├── top_recommendations.csv  # Top 100 ranked tracks from Stage 2
└── run_manifest.json        # Model metrics and feature summary
Project Structure
spotify-recommendation/
├── 01_ingest.py          # PySpark: ingest, clean, engineer, export Parquet
├── 02_train.py           # scikit-learn: hypothesis testing, PCA, train, rank
├── requirements.txt
├── data/
│   ├── raw/              # Downloaded CSV
│   └── parquet/          # PySpark → scikit-learn handoff
└── outputs/              # Plots, recommendations, manifest
Design Decisions
Why PySpark for ingestion? PySpark handles the full 114K dataset in a distributed-ready way. The StandardScaler is applied via a PySpark ML Pipeline — the same pattern used in production Spark environments — before handing off a clean Parquet file to scikit-learn.

Why PCA before modeling? 11 correlated audio features contain redundancy. PCA reduces to 8 components retaining 96.8% of variance, improving model generalization and reducing training time.

Why two-stage prediction? Binary classification alone answers "will they like it?" but not "which liked track should surface first?" The engagement score adds a ranking layer that mirrors real recommendation systems — filter then rank.

Why popularity > 60 as the target? Spotify's popularity score is computed from recent play counts and skip rates — it is the closest public proxy for "a user liked this track enough to replay it." Threshold at 60 produces a 90/10 class split, realistic for music recommendation.
