"""
LightGBM Model Training with Cross-Validation
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
import config
from utils.metrics import calculate_rmse, log_cv_score
import pickle
import warnings
warnings.filterwarnings('ignore')

class LightGBMModel:
    def __init__(self, params=None):
        self.params = params if params is not None else config.LIGHTGBM_PARAMS.copy()
        self.models = []
        self.oof_predictions = None
        
    def train_fold(self, X_train, y_train, X_val, y_val, categorical_features=None):
        """Train a single fold"""
        train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_features)
        val_data = lgb.Dataset(X_val, label=y_val, categorical_feature=categorical_features, reference=train_data)
        
        model = lgb.train(
            self.params,
            train_data,
            num_boost_round=self.params.get('n_estimators', 1000),
            valid_sets=[train_data, val_data],
            valid_names=['train', 'val'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=config.EARLY_STOPPING_ROUNDS),
                lgb.log_evaluation(period=config.VERBOSE_EVAL)
            ]
        )
        
        self.models.append(model)
        return model
    
    def train_cv(self, X, y, categorical_features=None, n_folds=config.N_FOLDS):
        """Train with K-Fold cross-validation"""
        print(f"\n{'='*60}")
        print(f"Training LightGBM with {n_folds}-Fold CV")
        print(f"{'='*60}\n")
        
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=config.MAIN_SEED)
        oof_preds = np.zeros(len(X))
        fold_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            print(f"\n--- Fold {fold + 1}/{n_folds} ---")
            
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model = self.train_fold(X_train, y_train, X_val, y_val, categorical_features)
            
            val_preds = model.predict(X_val)
            oof_preds[val_idx] = val_preds
            
            fold_rmse = calculate_rmse(y_val, val_preds)
            fold_scores.append(fold_rmse)
            print(f"Fold {fold + 1} RMSE: {fold_rmse:.5f}")
        
        self.oof_predictions = oof_preds
        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        
        log_cv_score('LightGBM', fold_scores, mean_score, std_score)
        
        return mean_score, fold_scores
    
    def predict(self, X):
        """Predict using ensemble of fold models"""
        if not self.models:
            raise ValueError("No models trained yet!")
        
        predictions = np.zeros(len(X))
        for model in self.models:
            predictions += model.predict(X)
        
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
