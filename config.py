"""
Configuration file for Kaggle S6E1 Competition
Contains all hyperparameters, paths, and settings
"""

import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR
MODELS_DIR = BASE_DIR / "saved_models"
SUBMISSIONS_DIR = BASE_DIR / "submissions"
OOF_DIR = BASE_DIR / "oof_predictions"

# Create directories if they don't exist
MODELS_DIR.mkdir(exist_ok=True)
SUBMISSIONS_DIR.mkdir(exist_ok=True)
OOF_DIR.mkdir(exist_ok=True)

# Data files
TRAIN_FILE = DATA_DIR / "train.csv"
TEST_FILE = DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_FILE = DATA_DIR / "sample_submission.csv"

# Random seeds for reproducibility
RANDOM_SEEDS = [42, 123, 456, 789, 2024]
MAIN_SEED = 42

# Cross-validation (increased for better stability)
N_FOLDS = 10

# Feature engineering
TARGET_COL = 'exam_score'
ID_COL = 'id'

CATEGORICAL_FEATURES = [
    'gender', 'course', 'internet_access', 
    'sleep_quality', 'study_method', 
    'facility_rating', 'exam_difficulty'
]

NUMERICAL_FEATURES = [
    'age', 'study_hours', 'class_attendance', 'sleep_hours'
]

# XGBoost hyperparameters (optimized)
XGBOOST_PARAMS = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'tree_method': 'hist',
    'device': 'cuda',  # GPU acceleration
    'learning_rate': 0.02,
    'max_depth': 6,
    'min_child_weight': 1,
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'gamma': 0.05,
    'reg_alpha': 0.5,
    'reg_lambda': 2.0,
    'n_estimators': 5000,
    'random_state': MAIN_SEED,
    'n_jobs': -1,
    'verbosity': 1
}

# LightGBM hyperparameters (optimized - our best performer)
LIGHTGBM_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'device': 'gpu',
    'gpu_platform_id': 0,
    'gpu_device_id': 0,
    'learning_rate': 0.02,
    'num_leaves': 48,
    'max_depth': 8,
    'min_child_samples': 25,
    'min_data_in_leaf': 20,
    'subsample': 0.85,
    'subsample_freq': 1,
    'colsample_bytree': 0.9,
    'reg_alpha': 0.3,
    'reg_lambda': 0.8,
    'min_split_gain': 0.01,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.85,
    'bagging_freq': 1,
    'n_estimators': 5000,
    'random_state': MAIN_SEED,
    'n_jobs': -1,
    'verbose': -1
}

# CatBoost hyperparameters (CPU mode - GPU not detected by CatBoost)
CATBOOST_PARAMS = {
    'loss_function': 'RMSE',
    'task_type': 'CPU',  # Changed to CPU (CatBoost GPU issue)
    'thread_count': -1,  # Use all CPU cores
    'learning_rate': 0.03,
    'depth': 8,
    'l2_leaf_reg': 5,
    'bootstrap_type': 'Bernoulli',
    'subsample': 0.85,
    'random_strength': 1.5,
    'colsample_bylevel': 0.9,
    'n_estimators': 5000,
    'random_state': MAIN_SEED,
    'verbose': 100,
    'early_stopping_rounds': 100
}

# Ridge regression
RIDGE_ALPHA = 1.0

# Ensemble weights (adjusted based on initial results)
ENSEMBLE_WEIGHTS = {
    'xgboost': 0.25,
    'lightgbm': 0.45,  # Best performer
    'catboost': 0.20,
    'ridge': 0.10
}

# Optuna settings
OPTUNA_N_TRIALS = 50
OPTUNA_TIMEOUT = 3600  # 1 hour

# Training settings
EARLY_STOPPING_ROUNDS = 100
VERBOSE_EVAL = 200
