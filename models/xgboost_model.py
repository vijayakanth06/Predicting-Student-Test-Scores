"""
XGBoost Model Training with Cross-Validation
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold
import optuna
from optuna.samplers import TPESampler
import config
from utils.metrics import calculate_rmse, log_cv_score
import pickle
import warnings
warnings.filterwarnings('ignore')

class XGBoostModel:
    def __init__(self, params=None):
        self.params = params if params is not None else config.XGBOOST_PARAMS.copy()
        self.models = []
        self.oof_predictions = None
        self.feature_importance = None
        
    def train_fold(self, X_train, y_train, X_val, y_val, fold=None):
        """Train a single fold"""
        # Convert to numpy for NumPy 2.0 compatibility
        y_train_np = y_train.values if hasattr(y_train, 'values') else y_train
        y_val_np = y_val.values if hasattr(y_val, 'values') else y_val
        
        dtrain = xgb.DMatrix(X_train, label=y_train_np)
        dval = xgb.DMatrix(X_val, label=y_val_np)
        
        model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=self.params.get('n_estimators', 1000),
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=config.EARLY_STOPPING_ROUNDS,
            verbose_eval=config.VERBOSE_EVAL
        )
        
        self.models.append(model)
        return model
    
    def train_cv(self, X, y, n_folds=config.N_FOLDS):
        """Train with K-Fold cross-validation"""
        print(f"\n{'='*60}")
        print("Training XGBoost with {n_folds}-Fold CV")
        print(f"{'='*60}\n")
        
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=config.MAIN_SEED)
        oof_preds = np.zeros(len(X))
        fold_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            print(f"\n--- Fold {fold + 1}/{n_folds} ---")
            
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model = self.train_fold(X_train, y_train, X_val, y_val, fold)
            
            # Predictions
            dval = xgb.DMatrix(X_val)
            val_preds = model.predict(dval)
            oof_preds[val_idx] = val_preds
            
            fold_rmse = calculate_rmse(y_val, val_preds)
            fold_scores.append(fold_rmse)
            print(f"Fold {fold + 1} RMSE: {fold_rmse:.5f}")
        
        self.oof_predictions = oof_preds
        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        
        log_cv_score('XGBoost', fold_scores, mean_score, std_score)
        
        return mean_score, fold_scores
    
    def fit(self, X, y):
        """Fit single model for pseudo-labeling (uses native API for consistency)"""
        split_idx = int(len(X) * 0.9)
        X_train = X.iloc[:split_idx]
        y_train = y[:split_idx]  # Works for Series or array
        X_val = X.iloc[split_idx:]
        y_val = y[split_idx:]
        
        # Convert to numpy for NumPy 2.0 compatibility
        y_train_np = y_train.values if hasattr(y_train, 'values') else y_train
        y_val_np = y_val.values if hasattr(y_val, 'values') else y_val
        
        # Use native API for consistency with train_fold
        dtrain = xgb.DMatrix(X_train, label=y_train_np)
        dval = xgb.DMatrix(X_val, label=y_val_np)
        
        model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=self.params.get('n_estimators', 1000),
            evals=[(dval, 'val')],
            verbose_eval=False
        )
        
        self.models = [model]
        return self
    
    def predict(self, X):
        """Predict using ensemble of fold models"""
        if not self.models:
            raise ValueError("No models trained yet!")
        
        dtest = xgb.DMatrix(X)
        predictions = np.zeros(len(X))
        
        for model in self.models:
            predictions += model.predict(dtest)
        
        predictions /= len(self.models)
        return predictions
    
    def save(self, filepath):
        """Save model"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'models': self.models,
                'params': self.params,
                'oof_predictions': self.oof_predictions
            }, f)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.models = data['models']
            self.params = data['params']
            self.oof_predictions = data['oof_predictions']
        print(f"Model loaded from {filepath}")

def optimize_xgboost(X, y, n_trials=50):
    """Optimize XGBoost hyperparameters using Optuna"""
    
    def objective(trial):
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'tree_method': 'hist',
            'device': 'cuda',
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'max_depth': trial.suggest_int('max_depth', 4, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0.0, 0.5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0),
            'random_state': config.MAIN_SEED
        }
        
        # Quick 3-fold CV for optimization
        kf = KFold(n_splits=3, shuffle=True, random_state=config.MAIN_SEED)
        scores = []
        
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dval = xgb.DMatrix(X_val, label=y_val)
            
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=300,
                evals=[(dval, 'val')],
                early_stopping_rounds=20,
                verbose_eval=False
            )
            
            preds = model.predict(dval)
            rmse = calculate_rmse(y_val, preds)
            scores.append(rmse)
        
        return np.mean(scores)
    
    print("Starting Optuna optimization for XGBoost...")
    study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=config.MAIN_SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"\nBest parameters: {study.best_params}")
    print(f"Best CV score: {study.best_value:.5f}")
    
    return study.best_params

if __name__ == "__main__":
    # Test the model
    from utils.data_loader import load_data, create_folds
    from feature_engineering import FeatureEngineer, create_target_encoding_features
    
    print("Loading data...")
    train, test = load_data()
    train = create_folds(train, n_folds=config.N_FOLDS)
    
    # Feature engineering
    fe = FeatureEngineer()
    train_fe, test_fe = fe.fit_transform(train, test)
    train_fe, test_fe = create_target_encoding_features(train_fe, test_fe)
    
    # Prepare features
    feature_cols = [col for col in train_fe.columns if col not in 
                    [config.ID_COL, config.TARGET_COL, 'fold'] + config.CATEGORICAL_FEATURES]
    
    X = train_fe[feature_cols]
    y = train_fe[config.TARGET_COL]
    
    # Train model
    model = XGBoostModel()
    mean_score, fold_scores = model.train_cv(X, y)
    
    # Save model
    model.save(config.MODELS_DIR / 'xgboost_model.pkl')
    
    # Save OOF predictions
    pd.DataFrame({
        config.ID_COL: train_fe[config.ID_COL],
        'oof_predictions': model.oof_predictions
    }).to_csv(config.OOF_DIR / 'xgboost_oof.csv', index=False)
