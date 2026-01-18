"""
Main Training Pipeline for Kaggle S6E1
Orchestrates all models, feature engineering, and ensemble creation
"""

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
import warnings
import argparse
from datetime import datetime
warnings.filterwarnings('ignore')

def prepare_features(train, test, use_target_encoding=True):
    """Prepare features with engineering"""
    print("\n" + "="*80)
    print("FEATURE ENGINEERING")
    print("="*80)
    
    # Create folds for target encoding
    train = create_folds(train, n_folds=config.N_FOLDS)
    
    # Basic feature engineering
    fe = FeatureEngineer()
    train_fe, test_fe = fe.fit_transform(train, test)
    
    # Target encoding
    if use_target_encoding:
        train_fe, test_fe = create_target_encoding_features(train_fe, test_fe)
    
    # Select feature columns
    exclude_cols = [config.ID_COL, config.TARGET_COL, 'fold'] + config.CATEGORICAL_FEATURES
    # Also exclude non-encoded categorical columns like age_group, study_intensity, etc
    exclude_cols += ['age_group', 'study_intensity', 'attendance_level']
    
    feature_cols = [col for col in train_fe.columns if col not in exclude_cols]
    
    print(f"\nTotal features created: {len(feature_cols)}")
    print(f"Feature columns: {feature_cols[:10]}... (showing first 10)")
    
    return train_fe, test_fe, feature_cols

def train_all_models(train_fe, feature_cols):
    """Train all base models"""
    X = train_fe[feature_cols]
    y = train_fe[config.TARGET_COL]
    
    scores = {}
    oof_predictions = {}
    trained_models = {}
    
    # 1. Ridge Regression (fast baseline)
    print("\n" + "="*80)
    print("TRAINING RIDGE REGRESSION")
    print("="*80)
    ridge_model = RidgeModel()
    ridge_score, _ = ridge_model.train_cv(X, y)
    ridge_model.save(config.MODELS_DIR / 'ridge_model.pkl')
    scores['ridge'] = ridge_score
    oof_predictions['ridge'] = ridge_model.oof_predictions
    trained_models['ridge'] = ridge_model
    
    # 2. XGBoost
    print("\n" + "="*80)
    print("TRAINING XGBOOST")
    print("="*80)
    xgb_model = XGBoostModel()
    xgb_score, _ = xgb_model.train_cv(X, y)
    xgb_model.save(config.MODELS_DIR / 'xgboost_model.pkl')
    scores['xgboost'] = xgb_score
    oof_predictions['xgboost'] = xgb_model.oof_predictions
    trained_models['xgboost'] = xgb_model
    
    # 3. LightGBM
    print("\n" + "="*80)
    print("TRAINING LIGHTGBM")
    print("="*80)
    lgb_model = LightGBMModel()
    lgb_score, _ = lgb_model.train_cv(X, y)
    lgb_model.save(config.MODELS_DIR / 'lightgbm_model.pkl')
    scores['lightgbm'] = lgb_score
    oof_predictions['lightgbm'] = lgb_model.oof_predictions
    trained_models['lightgbm'] = lgb_model
    
    # 4. CatBoost
    print("\n" + "="*80)
    print("TRAINING CATBOOST")
    print("="*80)
    cat_model = CatBoostModel()
    cat_score, _ = cat_model.train_cv(X, y)
    cat_model.save(config.MODELS_DIR / 'catboost_model.pkl')
    scores['catboost'] = cat_score
    oof_predictions['catboost'] = cat_model.oof_predictions
    trained_models['catboost'] = cat_model
    
    return trained_models, scores, oof_predictions

def create_ensemble_predictions(oof_predictions, y_true, weights=None):
    """Create weighted ensemble from OOF predictions"""
    if weights is None:
        weights = config.ENSEMBLE_WEIGHTS
    
    print("\n" + "="*80)
    print("CREATING ENSEMBLE")
    print("="*80)
    
    ensemble_oof = np.zeros(len(y_true))
    
    for model_name, weight in weights.items():
        if model_name in oof_predictions:
            ensemble_oof += weight * oof_predictions[model_name]
            print(f"{model_name}: {weight:.3f}")
    
    ensemble_score = calculate_rmse(y_true, ensemble_oof)
    print(f"\nEnsemble CV RMSE: {ensemble_score:.5f}")
    
    return ensemble_oof, ensemble_score

def optimize_ensemble_weights(oof_predictions, y_true):
    """Find optimal ensemble weights using scipy.optimize"""
    from scipy.optimize import minimize
    
    model_names = list(oof_predictions.keys())
    n_models = len(model_names)
    
    def objective(weights):
        """Minimization objective"""
        ensemble = np.zeros(len(y_true))
        for i, name in enumerate(model_names):
            ensemble += weights[i] * oof_predictions[name]
        return calculate_rmse(y_true, ensemble)
    
    # Constraints: weights sum to 1 and are non-negative
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = [(0, 1) for _ in range(n_models)]
    
    # Initial weights (equal)
    initial_weights = np.array([1/n_models] * n_models)
    
    # Optimize
    result = minimize(objective, initial_weights, bounds=bounds, constraints=constraints, method='SLSQP')
    
    optimal_weights = {name: weight for name, weight in zip(model_names, result.x)}
    print("\nOptimized Weights:")
    for name, weight in optimal_weights.items():
        print(f"  {name}: {weight:.4f}")
    print(f"Optimized Ensemble RMSE: {result.fun:.5f}")
    
    return optimal_weights

def generate_submission(trained_models, test_fe, feature_cols, weights=None, submission_name=None):
    """Generate submission file"""
    if weights is None:
        weights = config.ENSEMBLE_WEIGHTS
    
    print("\n" + "="*80)
    print("GENERATING PREDICTIONS FOR TEST SET")
    print("="*80)
    
    X_test = test_fe[feature_cols]
    test_predictions = np.zeros(len(X_test))
    
    for model_name, weight in weights.items():
        if model_name in trained_models:
            model = trained_models[model_name]
            preds = model.predict(X_test)
            test_predictions += weight * preds
            print(f"{model_name} predictions: mean={preds.mean():.2f}, std={preds.std():.2f}")
    
    # Clip predictions to valid range
    test_predictions = np.clip(test_predictions, 0, 100)
    
    # Create submission
    if submission_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        submission_name = f"submission_{timestamp}.csv"
    
    submission = pd.DataFrame({
        config.ID_COL: test_fe[config.ID_COL],
        config.TARGET_COL: test_predictions
    })
    
    submission_path = config.SUBMISSIONS_DIR / submission_name
    submission.to_csv(submission_path, index=False)
    
    print(f"\nSubmission saved to: {submission_path}")
    print(f"Predictions: mean={test_predictions.mean():.2f}, std={test_predictions.std():.2f}")
    print(f"Predictions range: [{test_predictions.min():.2f}, {test_predictions.max():.2f}]")
    
    return submission, submission_path

def main(optimize_weights=True):
    """Main training pipeline"""
    print("="*80)
    print("KAGGLE S6E1 - PREDICTING STUDENT TEST SCORES")
    print("="*80)
    
    # 1. Load data
    train, test = load_data()
    
    # 2. Feature engineering
    train_fe, test_fe, feature_cols = prepare_features(train, test)
    
    # 3. Train all models
    trained_models, scores, oof_predictions = train_all_models(train_fe, feature_cols)
    
    # 4. Print individual model scores
    print("\n" + "="*80)
    print("INDIVIDUAL MODEL SCORES")
    print("="*80)
    for model_name, score in sorted(scores.items(), key=lambda x: x[1]):
        print(f"{model_name:15s}: {score:.5f}")
    
    # 5. Optimize ensemble weights
    y_true = train_fe[config.TARGET_COL]
    
    if optimize_weights:
        optimal_weights = optimize_ensemble_weights(oof_predictions, y_true)
        ensemble_oof, ensemble_score = create_ensemble_predictions(oof_predictions, y_true, optimal_weights)
    else:
        optimal_weights = config.ENSEMBLE_WEIGHTS
        ensemble_oof, ensemble_score = create_ensemble_predictions(oof_predictions, y_true)
    
    # 6. Generate submission
    submission, submission_path = generate_submission(
        trained_models, test_fe, feature_cols, 
        weights=optimal_weights,
        submission_name='submission_ensemble.csv'
    )
    
    # 7. Log results
    log_submission_score('ensemble', ensemble_score)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"Best Model: {min(scores, key=scores.get)} ({min(scores.values()):.5f})")
    print(f"Ensemble CV Score: {ensemble_score:.5f}")
    print(f"Submission file: {submission_path}")
    print("="*80)
    
    return trained_models, scores, ensemble_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train models for Kaggle S6E1')
    parser.add_argument('--optimize-weights', action='store_true', help='Optimize ensemble weights')
    parser.add_argument('--no-optimize-weights', dest='optimize_weights', action='store_false')
    parser.set_defaults(optimize_weights=True)
    
    args = parser.parse_args()
    
    main(optimize_weights=args.optimize_weights)
