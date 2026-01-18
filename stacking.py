"""
Stacking implementation with Ridge meta-learner
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from utils.metrics import calculate_rmse
import config

class StackingEnsemble:
    def __init__(self, base_models, meta_model=None, n_folds=5):
        """
        Stacking ensemble with Ridge meta-learner
        
        Args:
            base_models: Dict of base models
            meta_model: Meta-learner (default: Ridge)
            n_folds: Folds for generating meta-features
        """
        self.base_models = base_models
        self.meta_model = meta_model if meta_model is not None else Ridge(alpha=10.0)
        self.n_folds = n_folds
        
    def fit(self, X, y):
        """
        Fit stacking ensemble
        
        Uses base model OOF predictions as meta-features
        """
        print("\n" + "="*80)
        print("TRAINING STACKING ENSEMBLE")
        print("="*80)
        
        # Create meta-features from base model OOF predictions
        meta_features = np.zeros((len(X), len(self.base_models)))
        
        for i, (name, model) in enumerate(self.base_models.items()):
            if hasattr(model, 'oof_predictions') and model.oof_predictions is not None:
                meta_features[:, i] = model.oof_predictions
                print(f"  {name}: Using OOF predictions")
            else:
                print(f"  {name}: Warning - no OOF predictions, using zeros")
        
        # Train meta-learner on meta-features
        print(f"\nTraining meta-learner (Ridge)...")
        self.meta_model.fit(meta_features, y)
        
        # Calculate stacking CV score
        stacking_pred = self.meta_model.predict(meta_features)
        stacking_score = calculate_rmse(y, stacking_pred)
        
        print(f"Stacking CV RMSE: {stacking_score:.5f}")
        print("="*80)
        
        return stacking_score
    
    def predict(self, X_test, test_predictions):
        """
        Predict using stacking ensemble
        
        Args:
            X_test: Test features (not used, kept for API consistency)
            test_predictions: Dict of base model predictions on test set
        
        Returns:
            Stacking predictions
        """
        # Create meta-features from base model test predictions
        meta_features = np.zeros((len(next(iter(test_predictions.values()))), len(self.base_models)))
        
        for i, name in enumerate(self.base_models.keys()):
            if name in test_predictions:
                meta_features[:, i] = test_predictions[name]
        
        # Meta-learner prediction
        stacking_pred = self.meta_model.predict(meta_features)
        
        return stacking_pred
