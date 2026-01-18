"""
FINAL COMPREHENSIVE TRAINING - TOP 3 PUSH
Includes: Pseudo-labeling + 6 models + Hill climbing optimization
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
from models.extratrees_model import ExtraTreesModel
from models.histgb_model import HistGBModel
from pseudo_labeling_trainer import train_with_pseudo_labels
from hill_climbing import hill_climb_weights
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

def main(use_pseudo_labeling=True, use_hill_climbing=True):
    """
    Final comprehensive training pipeline
    
    Args:
        use_pseudo_labeling: Use Chris Deotte's pseudo-labeling technique
        use_hill_climbing: Use hill climbing for ensemble optimization
    """
    print("="*80)
    print("FINAL TRAINING - TOP 3 PUSH")
    print("Target: < 8.54 RMSE (Top 3)")
    print("="*80)
    print("\nTechniques:")
    print(f"  ✓ 6 diverse models (XGB, LGB, Cat, Ridge, ExtraTrees, HistGB)")
    print(f"  ✓ 10-Fold CV")
    print(f"  ✓ 56 advanced features")
    print(f"  ✓ Pseudo-labeling: {'YES' if use_pseudo_labeling else 'NO'}")
    print(f"  ✓ Hill climbing: {'YES' if use_hill_climbing else 'NO'}")
    print("="*80)
    
    # 1. Load data
    print("\n[1/5] Loading data...")
    train, test = load_data()
    
    # 2. Feature engineering
    print("\n[2/5] Feature engineering...")
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
    
    print(f"Features: {len(feature_cols)}, Samples: {len(X):,}, Test: {len(X_test):,}")
    
    # 3. Train all models
    print("\n[3/5] Training 6 models...")
    model_configs = [
        ("LightGBM", LightGBMModel),
        ("CatBoost", CatBoostModel),
        ("XGBoost", XGBoostModel),
        ("ExtraTrees", ExtraTreesModel),
        ("HistGB", HistGBModel),
        ("Ridge", RidgeModel),
    ]
    
    oof_predictions = {}
    test_predictions = {}
    cv_scores = {}
    
    for model_name, ModelClass in model_configs:
        print(f"\n{'-'*80}")
        print(f"TRAINING: {model_name}")
        print(f"{'-'*80}")
        
        if use_pseudo_labeling:
            # Use pseudo-labeling
            oof, test_pred, models, cv_score = train_with_pseudo_labels(
                ModelClass, X, y, X_test, n_folds=config.N_FOLDS, model_name=model_name
            )
        else:
            # Standard training
            model = ModelClass()
            cv_score, _ = model.train_cv(X, y, n_folds=config.N_FOLDS)
            oof = model.oof_predictions
            test_pred = model.predict(X_test)
        
        oof_predictions[model_name.lower()] = oof
        test_predictions[model_name.lower()] = test_pred
        cv_scores[model_name.lower()] = cv_score
    
    # 4. Optimize ensemble
    print("\n[4/5] Optimizing ensemble...")
    
    if use_hill_climbing:
        # Hill climbing optimization
        initial_weights = {
            'lightgbm': 0.35,
            'catboost': 0.25,
            'xgboost': 0.20,
            'extratrees': 0.10,
            'histgb': 0.07,
            'ridge': 0.03
        }
        
        best_weights, ensemble_score = hill_climb_weights(
            oof_predictions, y, initial_weights=initial_weights, 
            max_iterations=1000, step_size=0.01
        )
    else:
        # Simple weighted average
        best_weights = {
            'lightgbm': 0.35,
            'catboost': 0.25,
            'xgboost': 0.20,
            'extratrees': 0.10,
            'histgb': 0.07,
            'ridge': 0.03
        }
        
        ensemble_pred = np.zeros(len(y))
        for name, weight in best_weights.items():
            ensemble_pred += weight * oof_predictions[name]
        ensemble_score = calculate_rmse(y, ensemble_pred)
    
    # 5. Generate final predictions
    print("\n[5/5] Generating submission...")
    
    final_test_pred = np.zeros(len(X_test))
    for name, weight in best_weights.items():
        final_test_pred += weight * test_predictions[name]
    
    final_test_pred = np.clip(final_test_pred, 0, 100)
    
    # Create submission
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission = pd.DataFrame({
        config.ID_COL: test_fe[config.ID_COL],
        config.TARGET_COL: final_test_pred
    })
    
    submission_path = config.SUBMISSIONS_DIR / f'submission_final_top3_{timestamp}.csv'
    submission.to_csv(submission_path, index=False)
    
    # Print results
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print("\nIndividual Model CV Scores:")
    for name, score in sorted(cv_scores.items(), key=lambda x: x[1]):
        print(f"  {name:15s}: {score:.5f}")
    
    print(f"\n{'='*80}")
    print(f"⭐ FINAL ENSEMBLE CV: {ensemble_score:.5f}")
    print(f"{'='*80}")
    
    # Evaluation
    target = 8.54
    gap = ensemble_score - target
    
    if ensemble_score < 8.54:
        print(f"\n🎉 SUCCESS! Top 3 target achieved!")
        print(f"   CV Score: {ensemble_score:.5f} < {target}")
    elif ensemble_score < 8.60:
        print(f"\n✓ Very close! Gap: {gap:.5f}")
        print(f"   Expected LB: ~{ensemble_score - 0.035:.3f} (CV - 0.035)")
        if ensemble_score - 0.035 < 8.54:
            print(f"   🎯 Should achieve Top 3 on LB!")
    else:
        print(f"\n⚠ Gap: {gap:.5f}")
        print(f"   Need more optimization")
    
    print(f"\n{'='*80}")
    print(f"Submission: {submission_path}")
    print(f"Predictions: [{final_test_pred.min():.2f}, {final_test_pred.max():.2f}]")
    print(f"Mean: {final_test_pred.mean():.2f}, Std: {final_test_pred.std():.2f}")
    print(f"{'='*80}\n")
    
    log_submission_score('final_top3', ensemble_score)
    
    print("\n🎯 READY TO SUBMIT TO KAGGLE!")
    print(f"File: {submission_path}")
    
    return cv_scores, ensemble_score, best_weights

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Final training for Top 3')
    parser.add_argument('--no-pseudo', action='store_true', help='Disable pseudo-labeling')
    parser.add_argument('--no-hill-climb', action='store_true', help='Disable hill climbing')
    
    args = parser.parse_args()
    
    cv_scores, ensemble_score, weights = main(
        use_pseudo_labeling=not args.no_pseudo,
        use_hill_climbing=not args.no_hill_climb
    )
    
    print("\n✅ TRAINING COMPLETE!")
    print(f"Best CV Score: {ensemble_score:.5f}")
    print(f"Expected LB: ~{ensemble_score - 0.035:.3f}")
