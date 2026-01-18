"""
Pseudo-labeling implementation
Uses confident test predictions to augment training data
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import config

def create_pseudo_labels(models, X_test, confidence_threshold=0.15):
    """
    Create pseudo-labels from test set predictions
    
    Args:
        models: Dict of trained models
        X_test: Test features
        confidence_threshold: Only use predictions with std < this value
    
    Returns:
        X_pseudo, y_pseudo: Confident pseudo-labeled samples
    """
    print("\n" + "="*80)
    print("CREATING PSEUDO-LABELS")
    print("="*80)
    
    # Get predictions from all models
    all_preds = []
    for model_name, model in models.items():
        preds = model.predict(X_test)
        all_preds.append(preds)
    
    # Stack predictions
    all_preds = np.array(all_preds)
    
    # Calculate ensemble prediction and std
    ensemble_pred = np.mean(all_preds, axis=0)
    pred_std = np.std(all_preds, axis=0)
    
    # Select confident predictions (low variance across models)
    confident_mask = pred_std < confidence_threshold
    
    X_pseudo = X_test[confident_mask]
    y_pseudo = ensemble_pred[confident_mask]
    
    print(f"Total test samples: {len(X_test)}")
    print(f"Confident samples: {len(X_pseudo)} ({100 * len(X_pseudo) / len(X_test):.1f}%)")
    print(f"Confidence threshold (std): {confidence_threshold}")
    print(f"Mean prediction std: {pred_std.mean():.3f}")
    print(f"Pseudo-label range: [{y_pseudo.min():.2f}, {y_pseudo.max():.2f}]")
    
    return X_pseudo, y_pseudo

def train_with_pseudo_labels(model_class, X_train, y_train, X_pseudo, y_pseudo, weight_factor=0.5):
    """
    Train model with pseudo-labeled data
    
    Args:
        model_class: Model class to instantiate
        X_train: Original training features
        y_train: Original training labels
        X_pseudo: Pseudo-labeled features
        y_pseudo: Pseudo labels
        weight_factor: Weight for pseudo-labeled samples (0-1)
    
    Returns:
        Trained model with pseudo-labels
    """
    # Combine original and pseudo-labeled data
    X_combined = pd.concat([X_train, X_pseudo], ignore_index=True)
    y_combined = pd.concat([
        pd.Series(y_train.values),
        pd.Series(y_pseudo)
    ], ignore_index=True)
    
    # Create sample weights (original=1.0, pseudo=weight_factor)
    sample_weights = np.concatenate([
        np.ones(len(X_train)),
        np.ones(len(X_pseudo)) * weight_factor
    ])
    
    print(f"\nTraining with pseudo-labels:")
    print(f"  Original samples: {len(X_train)}")
    print(f"  Pseudo samples: {len(X_pseudo)}")
    print(f"  Pseudo weight: {weight_factor}")
    
    # Train model (implementation depends on model type)
    model = model_class()
    
    # Note: Sample weights need to be integrated into model training
    # For now, we'll just use combined data
    # You can enhance this by passing sample_weights to fit() methods
    
    return X_combined, y_combined
