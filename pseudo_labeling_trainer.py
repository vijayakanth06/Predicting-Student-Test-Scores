"""
Pseudo-Labeling Trainer (Chris Deotte's Approach)
Trains models twice: first to generate pseudo-labels, then on augmented data
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import config
from utils.metrics import calculate_rmse

def train_with_pseudo_labels(model_class, X, y, X_test, n_folds=10, model_name="Model"):
    """
    Train model with pseudo-labeling
    
    Chris Deotte's approach:
    1. For each fold, train model on (X_train, y_train)
    2. Generate OOF predictions on X_val
    3. Generate test predictions
    4. Combine all data with pseudo-labels
    5. Retrain on augmented dataset
    6. Use retrained model for final predictions
    
    Args:
        model_class: Model class to instantiate
        X: Training features
        y: Training targets
        X_test: Test features
        n_folds: Number of CV folds
        model_name: Name for logging
        
    Returns:
        oof_preds_final: OOF predictions from pseudo-labeled model
        test_preds_final: Test predictions from pseudo-labeled model
        models: List of trained models
        cv_score: CV RMSE
    """
    print(f"\n{'='*80}")
    print(f"PSEUDO-LABELING: {model_name}")
    print(f"{'='*80}")
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=config.MAIN_SEED)
    
    oof_preds_first = np.zeros(len(X))
    oof_preds_final = np.zeros(len(X))
    test_preds_final = np.zeros(len(X_test))
    
    models_first = []
    models_final = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\n--- Fold {fold + 1}/{n_folds} ---")
        
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # STEP 1: Train first model
        print("  [1/3] Training base model...")
        model_first = model_class()
        
        # Train on training set only
        if hasattr(model_first, 'train_fold'):
            model_first.train_fold(X_train, y_train, X_val, y_val)
            
            # Check model type for proper prediction
            first_model = model_first.models[0]
            if 'xgboost' in str(type(first_model).__module__):
                # XGBoost Booster needs DMatrix
                import xgboost as xgb
                oof_preds = first_model.predict(xgb.DMatrix(X_val))
                test_preds = first_model.predict(xgb.DMatrix(X_test))
            else:
                # LightGBM, CatBoost, others
                oof_preds = first_model.predict(X_val)
                test_preds = first_model.predict(X_test)
        else:
            # Sklearn-style models (Ridge, ExtraTrees, HistGB)
            model_first.fit(X_train, y_train)
            oof_preds = model_first.predict(X_val)
            test_preds = model_first.predict(X_test)
        
        oof_preds_first[val_idx] = oof_preds
        models_first.append(model_first)
        
        # STEP 2: Create pseudo-labeled dataset
        print("  [2/3] Creating pseudo-labels...")
        X_train2 = pd.concat([X_train, X_val, X_test], axis=0, ignore_index=True)
        
        # Convert to numpy for compatibility
        y_train_np = y_train.values if hasattr(y_train, 'values') else y_train
        y_train2 = np.concatenate([y_train_np, oof_preds, test_preds], axis=0)
        
        print(f"  Original training size: {len(X_train):,}")
        print(f"  Augmented training size: {len(X_train2):,} (+{len(X_val) + len(X_test):,})")
        
        # STEP 3: Train second model on augmented data
        print("  [3/3] Training with pseudo-labels...")
        model_final = model_class()
        
        # Train on augmented dataset
        if hasattr(model_final, 'train_fold'):
            # Custom models - create dummy validation set (just use last 10% of augmented data)
            split_idx = int(len(X_train2) * 0.9)
            X_train_aug = X_train2.iloc[:split_idx]
            y_train_aug = y_train2[:split_idx]
            X_val_dummy = X_train2.iloc[split_idx:]
            y_val_dummy = y_train2[split_idx:]
            
            model_final.train_fold(X_train_aug, y_train_aug, X_val_dummy, y_val_dummy)
            
            # Check model type for proper prediction
            final_model = model_final.models[0]
            if 'xgboost' in str(type(final_model).__module__):
                # XGBoost Booster needs DMatrix
                import xgboost as xgb
                final_oof = final_model.predict(xgb.DMatrix(X_val))
                final_test = final_model.predict(xgb.DMatrix(X_test))
            else:
                # LightGBM, CatBoost, others
                final_oof = final_model.predict(X_val)
                final_test = final_model.predict(X_test)
        else:
            # Sklearn-style models (Ridge, ExtraTrees, HistGB)
            model_final.fit(X_train2, y_train2)
            final_oof = model_final.predict(X_val)
            final_test = model_final.predict(X_test)
        
        oof_preds_final[val_idx] = final_oof
        test_preds_final += final_test / n_folds
        
        models_final.append(model_final)
        
        # Calculate improvement
        rmse_first = calculate_rmse(y_val, oof_preds)
        rmse_final = calculate_rmse(y_val, final_oof)
        improvement = rmse_first - rmse_final
        
        print(f"  First model RMSE: {rmse_first:.5f}")
        print(f"  Pseudo-labeled RMSE: {rmse_final:.5f}")
        print(f"  Improvement: {improvement:+.5f}")
    
    # Calculate overall CV scores
    cv_first = calculate_rmse(y, oof_preds_first)
    cv_final = calculate_rmse(y, oof_preds_final)
    
    print(f"\n{'='*80}")
    print(f"{model_name} - PSEUDO-LABELING RESULTS")
    print(f"{'='*80}")
    print(f"CV RMSE (base model): {cv_first:.5f}")
    print(f"CV RMSE (pseudo-labeled): {cv_final:.5f}")
    print(f"Improvement: {cv_first - cv_final:+.5f}")
    print(f"{'='*80}")
    
    return oof_preds_final, test_preds_final, models_final, cv_final
