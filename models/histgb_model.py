"""
HistGradientBoosting Model - Fast sklearn gradient boosting
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
import config
from utils.metrics import calculate_rmse, log_cv_score
import pickle

class HistGBModel:
    def __init__(self, max_iter=1000, max_depth=10, learning_rate=0.05):
        self.max_iter = max_iter
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.models = []
        self.oof_predictions = None
        
    def train_fold(self, X_train, y_train, X_val, y_val):
        """Train a single fold"""
        model = HistGradientBoostingRegressor(
            max_iter=self.max_iter,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=config.MAIN_SEED,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=50,
            verbose=0
        )
        
        model.fit(X_train, y_train)
        self.models.append(model)
        return model
    
    def train_cv(self, X, y, n_folds=config.N_FOLDS):
        """Train with K-Fold cross-validation"""
        print(f"\n{'='*60}")
        print(f"Training HistGradientBoosting with {n_folds}-Fold CV")
        print(f"{'='*60}\n")
        
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=config.MAIN_SEED)
        oof_preds = np.zeros(len(X))
        fold_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            print(f"--- Fold {fold + 1}/{n_folds} ---")
            
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model = self.train_fold(X_train, y_train, X_val, y_val)
            
            val_preds = model.predict(X_val)
            oof_preds[val_idx] = val_preds
            
            fold_rmse = calculate_rmse(y_val, val_preds)
            fold_scores.append(fold_rmse)
            print(f"Fold {fold + 1} RMSE: {fold_rmse:.5f}")
        
        self.oof_predictions = oof_preds
        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        
        log_cv_score('HistGB', fold_scores, mean_score, std_score)
        
        return mean_score, fold_scores
    
    def fit(self, X, y):
        """Fit for pseudo-labeling (single model)"""
        model = HistGradientBoostingRegressor(
            max_iter=self.max_iter,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=config.MAIN_SEED,
            verbose=0
        )
        model.fit(X, y)
        self.models = [model]
        return self
    
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
                'oof_predictions': self.oof_predictions,
                'params': {
                    'max_iter': self.max_iter,
                    'max_depth': self.max_depth,
                    'learning_rate': self.learning_rate
                }
            }, f)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.models = data['models']
            self.oof_predictions = data['oof_predictions']
        print(f"Model loaded from {filepath}")
