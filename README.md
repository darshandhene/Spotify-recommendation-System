# Spotify Recommendation System

An end-to-end ML pipeline that predicts whether a user will like a track 
based on audio features, applying classification and statistical validation 
on large-scale Spotify listening data.

## Tech Stack
Python, PySpark, scikit-learn, pandas, NumPy

## Pipeline Stages
1. **Data ingestion and cleaning** — handle nulls, outliers, feature normalization
2. **Exploratory analysis** — hypothesis testing, correlation analysis, PCA
3. **Feature engineering** — dimensionality reduction, derived audio features
4. **Model training** — Random Forest and XGBoost with cross-validation
5. **Evaluation** — F1 = 0.71 on held-out test set

## Statistical Methods
- Hypothesis testing (t-test, chi-squared)
- Principal Component Analysis (PCA)
- Regression analysis
- Cross-validation and hyperparameter tuning

## Models
| Model | F1 Score |
|-------|----------|
| Random Forest | 0.68 |
| XGBoost | 0.71 |

## Distributed Processing
PySpark used for large-scale feature extraction and transformation 
before model training with scikit-learn.

## Key Insight
Audio energy and danceability are the strongest predictors of user 
preference, outperforming genre tags alone.
