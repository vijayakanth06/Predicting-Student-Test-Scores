"""
Strategy 1: Target Transformation Training
Implements log, sqrt, and rank transformations to find best approach
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
import config
from utils.data_loader import load_data, create_folds
from utils.metrics import calculate_rmse
from feature_engineering import FeatureEngineer, create_target_encoding_features

class TargetTransformer:
    """Handle target transformations for better predictions"""
    
    def __init__(self, method='log'):
        """
        Args:
            method: 'log', 'sqrt', 'rank', or 'none'
        """
        self.method = method
        self.rank_values = None
        
    def transform(self, y):
        """Transform target"""
        if self.method == 'log':
            return np.log1p(y)
        elif self.method == 'sqrt':
            return np.sqrt(y)
        elif self.method == 'rank':
            # Convert to ranks (percentiles)
            self.rank_values = np.sort(y)
            ranks = np.searchsorted(self.rank_values, y) / len(y)
            return ranks
        else:
            return y
    
    def inverse_transform(self, y_pred):
        """Inverse transform predictions"""
        if self.method == 'log':
            return np.expm1(y_pred)
        elif self.method == 'sqrt':
            return np.maximum(0, y_pred ** 2)  # Ensure non-negative
        elif self.method == 'rank':
            # Convert ranks back to values
            indices = np.clip((y_pred * len(self.rank_values)).astype(int), 
                            0, len(self.rank_values) - 1)
            return self.rank_values[indices]
        else:
            return y_pred


def train_with_target_transform(X, y, X_test, method='log', params=None):
    """
    Train LightGBM with target transformation
    
    Args:
        X: Training features
        y: Training target
        X_test: Test features
        method: Transformation method
        params: LightGBM parameters
        
    Returns:
        oof_preds, test_preds, cv_score
    """
    if params is None:
        params = config.LIGHTGBM_PARAMS.copy()
    
    print(f"\n{'='*80}")
    print(f"Training with {method.upper()} Target Transformation")
    print(f"{'='*80}")
    
    transformer = TargetTransformer(method=method)
    
    # Transform target for training
    y_transformed = transformer.transform(y.values)
    
    # K-Fold CV
    kf = KFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.MAIN_SEED)
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\n--- Fold {fold + 1}/{config.N_FOLDS} ---")
        
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y_transformed[train_idx], y_transformed[val_idx]
        
        # Create transformer for this fold (for rank method)
        fold_transformer = TargetTransformer(method=method)
        if method == 'rank':
            fold_transformer.rank_values = transformer.rank_values
        
        # Convert to numpy
        y_train_np = y_train if isinstance(y_train, np.ndarray) else y_train.values
        y_val_np = y_val if isinstance(y_val, np.ndarray) else y_val.values
        
        # Train
        train_data = lgb.Dataset(X_train, label=y_train_np)
        val_data = lgb.Dataset(X_val, label=y_val_np, reference=train_data)
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=params.get('n_estimators', 5000),
            valid_sets=[train_data, val_data],
            valid_names=['train', 'val'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=100),
                lgb.log_evaluation(period=200)
            ]
        )
        
        # Predict in transformed space
        val_pred_transformed = model.predict(X_val)
        test_pred_transformed = model.predict(X_test)
        
        # Inverse transform  
        val_pred = fold_transformer.inverse_transform(val_pred_transformed)
        test_pred = fold_transformer.inverse_transform(test_pred_transformed)
        
        oof_preds[val_idx] = val_pred
        test_preds += test_pred / config.N_FOLDS
        
        # Calculate RMSE on original scale
        fold_rmse = calculate_rmse(y.iloc[val_idx], val_pred)
        fold_scores.append(fold_rmse)
        print(f"Fold {fold + 1} RMSE (original scale): {fold_rmse:.5f}")
    
    cv_score = np.mean(fold_scores)
    print(f"\n{'='*80}")
    print(f"{method.upper()} Transformation CV RMSE: {cv_score:.5f} (+/- {np.std(fold_scores):.5f})")
    print(f"{'='*80}")
    
    return oof_preds, test_preds, cv_score


def find_best_transformation():
    """Test all transformations and find the best one"""
    print("\n" + "="*80)
    print("TESTING ALL TARGET TRANSFORMATIONS")
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
    
    # Test all methods
    print("\n[3/3] Testing transformations...")
    methods = ['none', 'log', 'sqrt']
    results = {}
    predictions = {}
    
    for method in methods:
        oof, test_pred, cv_score = train_with_target_transform(X, y, X_test, method=method)
        results[method] = cv_score
        predictions[method] = test_pred
    
    # Find best
    best_method = min(results, key=results.get)
    best_score = results[best_method]
    
    print("\n" + "="*80)
    print("TRANSFORMATION COMPARISON")
    print("="*80)
    for method, score in sorted(results.items(), key=lambda x: x[1]):
        improvement = results['none'] - score
        marker = " ⭐ BEST" if method == best_method else ""
        print(f"{method.upper():10s}: {score:.5f} (improvement: {improvement:+.5f}){marker}")
    
    print(f"\n{'='*80}")
    print(f"Best Method: {best_method.upper()}")
    print(f"Best CV Score: {best_score:.5f}")
    print(f"Improvement over baseline: {results['none'] - best_score:.5f}")
    print(f"{'='*80}")
    
    # Save best predictions
    submission = pd.DataFrame({
        config.ID_COL: test_fe[config.ID_COL],
        config.TARGET_COL: predictions[best_method]
    })
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = config.SUBMISSIONS_DIR / f'submission_transform_{best_method}_{timestamp}.csv'
    submission.to_csv(filepath, index=False)
    
    print(f"\nSubmission saved: {filepath}")
    return best_method, best_score, filepath


if __name__ == "__main__":
    best_method, best_score, filepath = find_best_transformation()
