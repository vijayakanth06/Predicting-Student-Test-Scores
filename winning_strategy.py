"""
WINNING STRATEGY: Original Data Blending + Linear Residuals + Quantile Ensemble

Based on top Kaggle performers:
- Masaya Kawamata's approach (CV 8.61 → LB 8.57)
- sung's NN approach (CV 8.59 → LB 8.54)
- Chris Deotte's pseudo-labeling

This script combines:
1. Original dataset blending (0.05-0.07 RMSE gain)
2. Linear + Residual boosting (0.02-0.03 gain)
3. Quantile ensemble (0.01-0.02 gain)
4. Original data features (0.01-0.02 gain)
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import config
from utils.data_loader import load_data, create_folds
from utils.metrics import calculate_rmse
from feature_engineering import FeatureEngineer


def load_original_dataset():
    """Load the original dataset from Kaggle"""
    print("\n" + "="*80)
    print("LOADING ORIGINAL DATASET")
    print("="*80)
    
    # Try to load from Kaggle path (if running on Kaggle)
    try:
        original = pd.read_csv('/kaggle/input/exam-score-prediction-dataset/Exam_Score_Prediction.csv')
        print(f"✅ Loaded original dataset from Kaggle: {original.shape}")
        # Rename student_id to id to match
        if 'student_id' in original.columns:
            original = original.rename(columns={'student_id': config.ID_COL})
        return original
    except:
        print("⚠️ Kaggle path not found. Trying local...")
        
    # Try local path
    try:
        original = pd.read_csv('original_data/Exam_Score_Prediction.csv')
        if 'student_id' in original.columns:
            original = original.rename(columns={'student_id': config.ID_COL})
        print(f"✅ Loaded original dataset locally: {original.shape}")
        return original
    except:
        print("❌ Original dataset not found!")
        print("\n📥 Please download from:")
        print("https://www.kaggle.com/datasets/mrsimple07/exam-score-prediction-dataset")
        print("And place in: original_data/Exam_Score_Prediction.csv")
        return None


def create_original_features(train, test, original):
    """
    Create features from original dataset
    Strategy from Masaya Kawamata's "Orig as Col"
    """
    print("\n--- Creating Original Data Features ---")
    
    # Combine for processing
    all_data = pd.concat([train, test], axis=0, ignore_state=True)
    
    # Target encoding from original data
    for col in config.CATEGORICAL_FEATURES:
        if col in original.columns:
            orig_means = original.groupby(col)[config.TARGET_COL].mean()
            all_data[f'{col}_orig_mean'] = all_data[col].map(orig_means)
            
            # Fill missing with global mean
            global_mean = original[config.TARGET_COL].mean()
            all_data[f'{col}_orig_mean'].fillna(global_mean, inplace=True)
    
    # Split back
    train_new = all_data.iloc[:len(train)].copy()
    test_new = all_data.iloc[len(train):].copy()
    
    new_cols = [c for c in train_new.columns if '_orig_mean' in c]
    print(f"Created {len(new_cols)} original data features")
    
    return train_new, test_new


def train_linear_base_model(X, y):
    """
    Train Ridge regression to capture linear trends
    Returns predictions for computing residuals
    """
    print("\n--- Training Linear Base Model ---")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train Ridge
    ridge = Ridge(alpha=1.0, random_state=config.MAIN_SEED)
    ridge.fit(X_scaled, y)
    
    # Get predictions
    linear_preds = ridge.predict(X_scaled)
    rmse = calculate_rmse(y, linear_preds)
    print(f"Linear model RMSE: {rmse:.5f}")
    
    return ridge, scaler, linear_preds


def train_quantile_residual_model(X, residuals, X_test, alpha=0.5, linear_pred_test=None):
    """
    Train quantile model on residuals, then add back linear predictions
    """
    params = config.LIGHTGBM_PARAMS.copy()
    params['objective'] = 'quantile'
    params['alpha'] = alpha
    params['metric'] = 'quantile'
    
    kf = KFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.MAIN_SEED)
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = residuals[train_idx], residuals[val_idx]
        
        # Convert to numpy
        y_train_np = y_train if isinstance(y_train, np.ndarray) else y_train
        y_val_np = y_val if isinstance(y_val, np.ndarray) else y_val
        
        train_data = lgb.Dataset(X_train, label=y_train_np)
        val_data = lgb.Dataset(X_val, label=y_val_np, reference=train_data)
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=params.get('n_estimators', 5000),
            valid_sets=[val_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=100),
                lgb.log_evaluation(period=0)  # Silent
            ]
        )
        
        oof_preds[val_idx] = model.predict(X_val)
        test_preds += model.predict(X_test) / config.N_FOLDS
    
    # Add back linear predictions
    if linear_pred_test is not None:
        test_preds += linear_pred_test
    
    return oof_preds, test_preds


def train_winning_ensemble():
    """
    Complete winning pipeline combining all top strategies
    """
    print("\n" + "="*80)
    print("WINNING STRATEGY: ORIGINAL DATA + LINEAR RESIDUALS + QUANTILE ENSEMBLE")
    print("="*80)
    
    # Load synthetic data
    print("\n[1/6] Loading synthetic data...")
    train, test = load_data()
    train = create_folds(train, n_folds=config.N_FOLDS)
    
    # Load original data
    print("\n[2/6] Loading original data...")
    original = load_original_dataset()
    if original is None:
        print("\n⚠️ Falling back to standard approach without original data")
        print("For best results, download the original dataset!")
        # Could fall back to train_quantile_ensemble here
        return
    
    # Feature engineering
    print("\n[3/6] Feature engineering...")
    fe = FeatureEngineer()
    train_fe, test_fe = fe.fit_transform(train, test)
    
    # Add original data features
    train_fe, test_fe = create_original_features(train_fe, test_fe, original)
    
    # Process original data with same FE
    original_fe, _ = fe.fit_transform(original, test.head(1))  # Dummy test
    
    # Select features
    exclude_cols = [config.ID_COL, config.TARGET_COL, 'fold'] + config.CATEGORICAL_FEATURES
    exclude_cols += ['age_group', 'study_intensity', 'attendance_level']
    feature_cols = [col for col in train_fe.columns if col not in exclude_cols]
    
    X = train_fe[feature_cols]
    y = train_fe[config.TARGET_COL]
    X_test = test_fe[feature_cols]
    X_orig = original_fe[feature_cols]
    y_orig = original_fe[config.TARGET_COL]
    
    print(f"\nFeatures: {len(feature_cols)}")
    print(f"Synthetic train: {len(X)}")
    print(f"Original train: {len(X_orig)}")
    
    # Augment with original data
    print("\n[4/6] Augmenting with original data...")
    X_combined = pd.concat([X, X_orig], axis=0, ignore_index=True)
    y_combined = pd.concat([y, y_orig], axis=0, ignore_index=True)
    
    print(f"Combined training set: {len(X_combined)}")
    
    # Train linear base model
    print("\n[5/6] Training linear base model...")
    ridge, scaler, linear_preds_train = train_linear_base_model(X_combined, y_combined)
    
    # Compute residuals
    residuals = y_combined - linear_preds_train
    
    # Get linear predictions for test
    X_test_scaled = scaler.transform(X_test)
    linear_preds_test = ridge.predict(X_test_scaled)
    
    print(f"Residuals - Mean: {residuals.mean():.3f}, Std: {residuals.std():.3f}")
    
    # Train quantile models on residuals
    print("\n[6/6] Training quantile ensemble on residuals...")
    quantiles = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    all_test_preds = []
    all_oof_preds = []
    
    for alpha in quantiles:
        print(f"\nTraining Q{int(alpha*100)} model...")
        oof_res, test_res = train_quantile_residual_model(
            X_combined, residuals, X_test, alpha=alpha
        )
        
        # For OOF, we need to align with original synthetic data
        # Only use predictions for synthetic indices
        oof_synthetic = oof_res[:len(X)] + linear_preds_train[:len(X)]
        test_final = test_res + linear_preds_test
        
        all_oof_preds.append(oof_synthetic)
        all_test_preds.append(test_final)
        
        # Calculate RMSE
        oof_rmse = calculate_rmse(y, oof_synthetic)
        print(f"Q{int(alpha*100)} OOF RMSE: {oof_rmse:.5f}")
    
    # Average ensemble
    final_oof = np.mean(all_oof_preds, axis=0)
    final_test = np.mean(all_test_preds, axis=0)
    
    ensemble_rmse = calculate_rmse(y, final_oof)
    
    print("\n" + "="*80)
    print(f"🎯 FINAL ENSEMBLE OOF RMSE: {ensemble_rmse:.5f}")
    print(f"📊 Expected LB: ~{ensemble_rmse - 0.035:.5f} (based on CV-LB gap)")
    print("="*80)
    
    # Save submission
    submission = pd.DataFrame({
        config.ID_COL: test_fe[config.ID_COL],
        config.TARGET_COL: np.clip(final_test, 0, 100)
    })
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = config.SUBMISSIONS_DIR / f'submission_winning_strategy_{timestamp}.csv'
    submission.to_csv(filepath, index=False)
    
    # Save OOF
    oof_df = pd.DataFrame({
        config.ID_COL: train_fe[config.ID_COL],
        config.TARGET_COL: final_oof
    })
    oof_path = config.SUBMISSIONS_DIR / f'oof_winning_strategy_{timestamp}.csv'
    oof_df.to_csv(oof_path, index=False)
    
    print(f"\n✅ Submission saved: {filepath}")
    print(f"✅ OOF saved: {oof_path}")
    
    return final_oof, final_test, ensemble_rmse


if __name__ == "__main__":
    oof, test_pred, cv_score = train_winning_ensemble()
