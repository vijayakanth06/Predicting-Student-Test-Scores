# Kaggle S6E1 Competition - Training Guide

## Quick Start

**IMPORTANT**: Always activate gpu-env before running Python scripts:

```powershell
conda activate gpu-env
python train_pipeline.py
```

Or run directly:
```powershell
conda activate gpu-env
python run_optimized_training.py
```

## What We've Built

### ✅ Complete Pipeline Components

1. **Configuration** (`config.py`)
   - Optimized hyperparameters for all models
   - Fixed CatBoost parameters (added `bootstrap_type='Bernoulli'`)
   - GPU acceleration enabled for all models

2. **Feature Engineering** (`feature_engineering.py`)
   - 39 engineered features including:
     - Interaction features (study_effectiveness, sleep_score, etc.)
     - Polynomial features (squared, cubed terms)
     - Binning (age_group, study_intensity, attendance_level)
     - Target encoding (K-Fold to prevent leakage)

3. **Models** (`models/`)
   - Ridge Regression: Baseline model  
   - XGBoost: GPU-accelerated gradient boosting
   - LightGBM: Best performer so far (CV: 8.765)
   - CatBoost: Now fixed and ready to train

4. **Main Pipeline** (`train_pipeline.py`)
   - Trains all models with 5-Fold CV
   - Generates OOF predictions
   - Optimizes ensemble weights
   - Creates submission file

## Current Performance

Based on initial run (with old parameters):
- **Ridge**: 8.899 RMSE
- **XGBoost**: 8.896 RMSE
- **LightGBM**: 8.765 RMSE ⭐ (best)
- **CatBoost**: Failed → Now Fixed

## Optimizations Made

### XGBoost
- ↓ Learning rate: 0.05 → 0.02 (more conservative)
- ↓ Max depth: 7 → 6 (less complex trees)
- ↑ Regularization: alpha 0.1 → 0.5, lambda 1.0 → 2.0
- ↑ n_estimators: 1000 → 2000

### LightGBM (Our Best Model)
- ↓ Learning rate: 0.05 → 0.02
- ↑ Max depth: 7 → 8
- ↓ Num leaves: 63 → 48 (better balance)
- Added feature_fraction, bagging_fraction for regularization
- ↑ n_estimators: 1000 → 3000

### CatBoost (Fixed!)
- ✅ Added `bootstrap_type='Bernoulli'` to support subsampling
- ↓ Learning rate: 0.05 → 0.03
- ↑ Depth: 7 → 8
- ↑ n_estimators: 1000 → 2000

### Ensemble
- Adjusted weights to favor LightGBM (45%)
- Will optimize further based on final OOF predictions

## Expected Improvements

With these optimizations, we expect:
- LightGBM: **8.765** → **~8.55-8.65** (target: < 8.54 for Top 3)
- XGBoost: **8.896** → **~8.70-8.80**
- CatBoost: Should perform around **~8.60-8.75**
- **Ensemble**: Target **< 8.54** for Top 3 placement

## Next Steps

1. ✅ Run optimized training
2. Monitor individual model scores
3. If still above 8.54:
   - Add more advanced features
   - Implement pseudo-labeling
   - Try residual boosting
   - Multi-seed ensembling

## Files Overview

```
Predicting Student Test Scores/
├── config.py                  # All hyperparameters (OPTIMIZED)
├── feature_engineering.py     # Feature creation pipeline
├── train_pipeline.py          # Main training orchestrator
├── run_optimized_training.py  # Quick runner script
├── models/
│   ├── xgboost_model.py      # XGBoost trainer
│   ├── lightgbm_model.py     # LightGBM trainer  
│   ├── catboost_model.py     # CatBoost trainer
│   └── ridge_model.py        # Ridge regression
├── utils/
│   ├── metrics.py            # RMSE calculation, logging
│   └── data_loader.py        # Data loading utilities
├── saved_models/             # Trained models saved here
├── oof_predictions/          # Out-of-fold predictions
└── submissions/              # Submission files

```

## Troubleshooting

**Problem**: `ModuleNotFoundError: No module named 'sklearn'`
**Solution**: You're in base environment. Run:
```powershell
conda activate gpu-env
```

**Problem**: CatBoost error about 'subsample'
**Solution**: ✅ Already fixed in config.py

**Problem**: Training too slow
**Solution**: All models use GPU. Make sure gpu-env has CUDA enabled.
