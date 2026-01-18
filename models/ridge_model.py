"""
Ridge Regression Model Training
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import config
from utils.metrics import calculate_rmse, log_cv_score
import pickle

class RidgeModel:
    def __init__(self, alpha=config.RIDGE_ALPHA):
        self.alpha = alpha
        self.models = []
        self.scalers = []
        self.oof_predictions = None
        
    def train_fold(self, X_train, y_train, X_val, y_val):
        """Train a single fold"""
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Train model
        model = Ridge(alpha=self.alpha, random_state=config.MAIN_SEED)
        model.fit(X_train_scaled, y_train)
        
        self.models.append(model)
        self.scalers.append(scaler)
        
        return model, scaler
    
    def train_cv(self, X, y, n_folds=config.N_FOLDS):
        """Train with K-Fold cross-validation"""
        print(f"\n{'='*60}")
        print(f"Training Ridge Regression with {n_folds}-Fold CV")
        print(f"{'='*60}\n")
        
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=config.MAIN_SEED)
        oof_preds = np.zeros(len(X))
        fold_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            print(f"\n--- Fold {fold + 1}/{n_folds} ---")
            
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model, scaler = self.train_fold(X_train, y_train, X_val, y_val)
            
            # Predictions
            X_val_scaled = scaler.transform(X_val)
            val_preds = model.predict(X_val_scaled)
            oof_preds[val_idx] = val_preds
            
            fold_rmse = calculate_rmse(y_val, val_preds)
            fold_scores.append(fold_rmse)
            print(f"Fold {fold + 1} RMSE: {fold_rmse:.5f}")
        
        self.oof_predictions = oof_preds
        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        
        log_cv_score('Ridge', fold_scores, mean_score, std_score)
        
        return mean_score, fold_scores
    
    def fit(self, X, y):
        """Fit single model for pseudo-labeling"""
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train model
        model = Ridge(alpha=self.alpha, random_state=config.MAIN_SEED)
        model.fit(X_scaled, y)
        
        self.models = [model]
        self.scalers = [scaler]
        return self
    
    def predict(self, X):
        """Predict using ensemble of fold models"""
        if not self.models:
            raise ValueError("No models trained yet!")
        
        predictions = np.zeros(len(X))
        
        for model, scaler in zip(self.models, self.scalers):
            X_scaled = scaler.transform(X)
            predictions += model.predict(X_scaled)
        
        predictions /= len(self.models)
        return predictions
    
    def save(self, filepath):
        """Save model"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'models': self.models,
                'scalers': self.scalers,
                'alpha': self.alpha,
                'oof_predictions': self.oof_predictions
            }, f)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.models = data['models']
            self.scalers = data['scalers']
            self.alpha = data['alpha']
            self.oof_predictions = data['oof_predictions']
        print(f"Model loaded from {filepath}")
