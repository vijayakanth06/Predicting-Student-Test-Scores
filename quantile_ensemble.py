"""
Strategy 2: Quantile Regression Ensemble
Train multiple quantile models and average their predictions
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
import config
from utils.data_loader import load_data, create_folds
from utils.metrics import calculate_rmse
from feature_engineering import FeatureEngineer, create_target_encoding_features


def train_quantile_model(X, y, X_test, alpha=0.5, params=None):
    """
    Train a single quantile regression model
    
    Args:
        X: Training features
        y: Training target
        X_test: Test features
        alpha: Quantile to predict (0.5 = median)
        params: LightGBM parameters
        
    Returns:
        oof_preds, test_preds, cv_score
    """
    if params is None:
        params = config.LIGHTGBM_PARAMS.copy()
    
    # Set quantile objective
    params['objective'] = 'quantile'
    params['alpha'] = alpha
    params['metric'] = 'quantile'
    
    print(f"\n--- Training Q{int(alpha*100)} Model ---")
    
    kf = KFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.MAIN_SEED)
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Convert to numpy
        y_train_np = y_train.values if hasattr(y_train, 'values') else y_train
        y_val_np = y_val.values if hasattr(y_val, 'values') else y_val
        
        train_data = lgb.Dataset(X_train, label=y_train_np)
        val_data = lgb.Dataset(X_val, label=y_val_np, reference=train_data)
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=params.get('n_estimators', 5000),
            valid_sets=[val_data],
            valid_names=['val'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=100),
                lgb.log_evaluation(period=0)  # Silent
            ]
        )
        
        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)
        
        oof_preds[val_idx] = val_pred
        test_preds += test_pred / config.N_FOLDS
        
        fold_rmse = calculate_rmse(y_val, val_pred)
        fold_scores.append(fold_rmse)
    
    cv_score = np.mean(fold_scores)
    print(f"Q{int(alpha*100)} CV RMSE: {cv_score:.5f}")
    
    return oof_preds, test_preds, cv_score


def train_quantile_ensemble(quantiles=[0.1, 0.3, 0.5, 0.7, 0.9]):
    """
    Train multiple quantile models and ensemble them
    
    Args:
        quantiles: List of quantiles to predict
        
    Returns:
        Final predictions and CV score
    """
    print("\n" + "="*80)
    print("QUANTILE REGRESSION ENSEMBLE")
    print(f"Training {len(quantiles)} quantile models: {quantiles}")
    print("="*80)
    
    # Load data
    print("\n[1/3] Loading data...")
    train, test = load_data()
    train = create_folds(train, n_folds=config.N_FOLDS)
    
    # Feature engineering
    print("\n[2/3] Feature engineering...")
    fe = FeatureEngineer()
    train_fe, test_fe = fe.fit_transform(train, test)
    train_fe, test_fe = create_target_encoding_features(train_fe, test_fe, n_folds=config.N_FOLDS)
    
    # Select features
    exclude_cols = [config.ID_COL, config.TARGET_COL, 'fold'] + config.CATEGORICAL_FEATURES
    exclude_cols += ['age_group', 'study_intensity', 'attendance_level']
    feature_cols = [col for col in train_fe.columns if col not in exclude_cols]
    
    X = train_fe[feature_cols]
    y = train_fe[config.TARGET_COL]
    X_test = test_fe[feature_cols]
    
    # Train quantile models
    print(f"\n[3/3] Training {len(quantiles)} quantile models...")
    
    all_oof_preds = []
    all_test_preds = []
    quantile_scores = {}
    
    for alpha in quantiles:
        oof, test_pred, cv_score = train_quantile_model(X, y, X_test, alpha=alpha)
        all_oof_preds.append(oof)
        all_test_preds.append(test_pred)
        quantile_scores[f'Q{int(alpha*100)}'] = cv_score
    
    # Average predictions
    final_oof = np.mean(all_oof_preds, axis=0)
    final_test = np.mean(all_test_preds, axis=0)
    
    # Calculate ensemble CV
    ensemble_cv = calculate_rmse(y, final_oof)
    
    print("\n" + "="*80)
    print("QUANTILE ENSEMBLE RESULTS")
    print("="*80)
    print("\nIndividual Quantile Scores:")
    for name, score in sorted(quantile_scores.items(), key=lambda x: x[1]):
        print(f"  {name}: {score:.5f}")
    
    print(f"\n{'='*80}")
    print(f"⭐ ENSEMBLE CV RMSE: {ensemble_cv:.5f}")
    print(f"{'='*80}")
    
    # Calculate improvement over median
    median_idx = quantiles.index(0.5) if 0.5 in quantiles else len(quantiles) // 2
    median_cv = quantile_scores[f'Q{int(quantiles[median_idx]*100)}']
    improvement = median_cv - ensemble_cv
    print(f"Improvement over median: {improvement:+.5f}")
    
    # Save submission
    submission = pd.DataFrame({
        config.ID_COL: test_fe[config.ID_COL],
        config.TARGET_COL: final_test
    })
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = config.SUBMISSIONS_DIR / f'submission_quantile_ensemble_{timestamp}.csv'
    submission.to_csv(filepath, index=False)
    
    print(f"\nSubmission saved: {filepath}")
    
    return final_oof, final_test, ensemble_cv, filepath


if __name__ == "__main__":
    # Test different quantile combinations
    
    # Strategy 1: 3 quantiles (fast)
    print("\n" + "="*80)
    print("STRATEGY 1: 3 Quantiles (10th, 50th, 90th)")
    print("="*80)
    oof1, test1, cv1, file1 = train_quantile_ensemble(quantiles=[0.1, 0.5, 0.9])
    
    # Strategy 2: 5 quantiles (balanced)
    print("\n" + "="*80)
    print("STRATEGY 2: 5 Quantiles (10th, 30th, 50th, 70th, 90th)")
    print("="*80)
    oof2, test2, cv2, file2 = train_quantile_ensemble(quantiles=[0.1, 0.3, 0.5, 0.7, 0.9])
    
    # Best
    if cv1 < cv2:
        print(f"\n⭐ Best: 3 Quantiles (CV: {cv1:.5f})")
        print(f"Use submission: {file1}")
    else:
        print(f"\n⭐ Best: 5 Quantiles (CV: {cv2:.5f})")
        print(f"Use submission: {file2}")
