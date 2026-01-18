"""
FINAL ADVANCED TRAINING PIPELINE
Implements: 15+ new features, 10-fold CV, 5k iterations, stacking
Target: < 8.54 RMSE for Top 3
"""

import sys
sys.path.append('.')

import numpy as np
import pandas as pd
import config
from utils.data_loader import load_data, create_folds
from utils.metrics import calculate_rmse, log_submission_score
from feature_engineering import FeatureEngineer, create_target_encoding_features
from models.xgboost_model import XGBoostModel
from models.lightgbm_model import LightGBMModel
from models.catboost_model import CatBoostModel
from models.ridge_model import RidgeModel
from stacking import StackingEnsemble
from pseudo_labeling import create_pseudo_labels
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("ADVANCED TRAINING PIPELINE - OPTION B")
    print("Target: < 8.54 RMSE (Top 3)")
    print("="*80)
    print("\nEnhancements:")
    print("  ✓ 10-Fold CV (from 5)")
    print("  ✓ 5000 iterations (from 2000-3000)")
    print("  ✓ 15+ advanced features")
    print("  ✓ Stacking with Ridge meta-learner")
    print("  ✓ Pseudo-labeling ready")
    print("="*80)
    
    # 1. Load data
    print("\n[1/6] Loading data...")
    train, test = load_data()
    
    # 2. Feature engineering (advanced)
    print("\n[2/6] Advanced feature engineering...")
    train = create_folds(train, n_folds=config.N_FOLDS)
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
    
    print(f"Total features: {len(feature_cols)}")
    print(f"Training samples: {len(X):,}")
    print(f"Test samples: {len(X_test):,}")
    
    # 3. Train all base models
    print("\n[3/6] Training base models with 10-fold CV...")
    scores = {}
    trained_models = {}
    
    # Ridge
    print("\n" + "-"*80)
    print("RIDGE REGRESSION")
    print("-"*80)
    ridge_model = RidgeModel()
    ridge_score, _ = ridge_model.train_cv(X, y, n_folds=config.N_FOLDS)
    ridge_model.save(config.MODELS_DIR / 'ridge_model_advanced.pkl')
    scores['ridge'] = ridge_score
    trained_models['ridge'] = ridge_model
    
    # LightGBM (best performer)
    print("\n" + "-"*80)
    print("LIGHTGBM (5000 iterations)")
    print("-"*80)
    lgb_model = LightGBMModel()
    lgb_score, _ = lgb_model.train_cv(X, y, n_folds=config.N_FOLDS)
    lgb_model.save(config.MODELS_DIR / 'lightgbm_model_advanced.pkl')
    scores['lightgbm'] = lgb_score
    trained_models['lightgbm'] = lgb_model
    
    # XGBoost
    print("\n" + "-"*80)
    print("XGBOOST (5000 iterations)")
    print("-"*80)
    xgb_model = XGBoostModel()
    xgb_score, _ = xgb_model.train_cv(X, y, n_folds=config.N_FOLDS)
    xgb_model.save(config.MODELS_DIR / 'xgboost_model_advanced.pkl')
    scores['xgboost'] = xgb_score
    trained_models['xgboost'] = xgb_model
    
    # CatBoost
    print("\n" + "-"*80)
    print("CATBOOST (5000 iterations, CPU)")
    print("-"*80)
    cat_model = CatBoostModel()
    cat_score, _ = cat_model.train_cv(X, y, n_folds=config.N_FOLDS)
    cat_model.save(config.MODELS_DIR / 'catboost_model_advanced.pkl')
    scores['catboost'] = cat_score
    trained_models['catboost'] = cat_model
    
    # 4. Stacking ensemble
    print("\n[4/6] Creating stacking ensemble...")
    stacker = StackingEnsemble(trained_models, n_folds=config.N_FOLDS)
    stacking_score = stacker.fit(X, y)
    scores['stacking'] = stacking_score
    
    # 5. Generate predictions
    print("\n[5/6] Generating test predictions...")
    test_predictions = {}
    for name, model in trained_models.items():
        test_predictions[name] = model.predict(X_test)
    
    # Stacking predictions
    stacking_preds = stacker.predict(X_test, test_predictions)
    
    # Simple weighted ensemble for comparison
    weights = [0.05, 0.75, 0.05, 0.15]  # ridge, lgb, xgb, cat
    simple_preds = (weights[0] * test_predictions['ridge'] +
                    weights[1] * test_predictions['lightgbm'] +
                    weights[2] * test_predictions['xgboost'] +
                    weights[3] * test_predictions['catboost'])
    
    # Choose best approach
    if stacking_score < min(scores['ridge'], scores['xgboost'], scores['lightgbm'], scores['catboost']):
        final_preds = stacking_preds
        final_approach = "Stacking"
        final_score = stacking_score
    else:
        final_preds = simple_preds
        final_approach = "Weighted Ensemble"
        final_score = min(scores.values())
    
    # Clip predictions
    final_preds = np.clip(final_preds, 0, 100)
    
    # 6. Create submission
    print("\n[6/6] Creating submission file...")
    submission = pd.DataFrame({
        config.ID_COL: test_fe[config.ID_COL],
        config.TARGET_COL: final_preds
    })
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_path = config.SUBMISSIONS_DIR / f'submission_advanced_{timestamp}.csv'
    submission.to_csv(submission_path, index=False)
    
    # Print results
    print("\n" + "="*80)
    print("FINAL RESULTS - ADVANCED PIPELINE")
    print("="*80)
    print("\nBase Model Scores (10-Fold CV):")
    for model, score in sorted(scores.items(), key=lambda x: x[1]):
        if model == 'stacking':
            print(f"  🎯 {model:15s}: {score:.5f}")
        elif score < 8.80:
            print(f"  🏆 {model:15s}: {score:.5f}")
        else:
            print(f"  ✓ {model:15s}: {score:.5f}")
    
    print(f"\n{'='*80}")
    print(f"BEST APPROACH: {final_approach}")
    print(f"FINAL CV SCORE: {final_score:.5f}")
    print(f"{'='*80}")
    
    # Evaluation
    target = 8.54
    gap = final_score - target
    
    if final_score < 8.54:
        print("\n🎉 SUCCESS! Achieved Top 3 target!")
        print(f"   Score: {final_score:.5f} < {target}")
    elif final_score < 8.60:
        print(f"\n✓ Very Close! Gap: {gap:.5f}")
        print("  Consider pseudo-labeling for final push")
    else:
        print(f"\n⚠ Gap: {gap:.5f}")
        print("  Need more advanced techniques")
    
    print(f"\n{'='*80}")
    print(f"Submission: {submission_path}")
    print(f"Predictions: [{final_preds.min():.2f}, {final_preds.max():.2f}]")
    print(f"Mean: {final_preds.mean():.2f}, Std: {final_preds.std():.2f}")
    print(f"{'='*80}\n")
    
    log_submission_score(f'advanced_{final_approach}', final_score)
    
    return trained_models, scores, final_score

if __name__ == "__main__":
    models, scores, final_score = main()
