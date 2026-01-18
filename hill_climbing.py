"""
Hill Climbing Ensemble Optimizer
Iteratively optimize ensemble weights for best CV score
"""

import numpy as np
from utils.metrics import calculate_rmse

def hill_climb_weights(oof_predictions, y_true, initial_weights=None, max_iterations=1000, step_size=0.01):
    """
    Optimize ensemble weights using hill climbing
    
    Args:
        oof_predictions: Dict of {model_name: oof_predictions_array}
        y_true: True target values
        initial_weights: Starting weights (dict), if None uses equal weights
        max_iterations: Maximum optimization iterations
        step_size: Size of weight adjustments
        
    Returns:
        best_weights: Optimized weights dict
        best_score: Best RMSE achieved
    """
    model_names = list(oof_predictions.keys())
    n_models = len(model_names)
    
    # Initialize weights
    if initial_weights is None:
        weights = {name: 1.0 / n_models for name in model_names}
    else:
        weights = initial_weights.copy()
    
    def calculate_ensemble_score(w):
        """Calculate RMSE for given weights"""
        ensemble_pred = np.zeros(len(y_true))
        for name, weight in w.items():
            ensemble_pred += weight * oof_predictions[name]
        return calculate_rmse(y_true, ensemble_pred)
    
    def normalize_weights(w):
        """Normalize weights to sum to 1"""
        total = sum(w.values())
        return {name: val / total for name, val in w.items()}
    
    # Initial score
    best_weights = weights.copy()
    best_score = calculate_ensemble_score(weights)
    
    print(f"\n{'='*80}")
    print("HILL CLIMBING ENSEMBLE OPTIMIZATION")
    print(f"{'='*80}")
    print(f"Initial weights: {weights}")
    print(f"Initial RMSE: {best_score:.5f}")
    print(f"\nOptimizing (max {max_iterations} iterations, step={step_size})...")
    
    improvements = 0
    
    for iteration in range(max_iterations):
        improved = False
        
        # Try adjusting each weight
        for model_name in model_names:
            # Try increasing
            test_weights = best_weights.copy()
            test_weights[model_name] = min(1.0, test_weights[model_name] + step_size)
            test_weights = normalize_weights(test_weights)
            test_score = calculate_ensemble_score(test_weights)
            
            if test_score < best_score:
                best_score = test_score
                best_weights = test_weights.copy()
                improved = True
                improvements += 1
                continue
            
            # Try decreasing
            test_weights = best_weights.copy()
            test_weights[model_name] = max(0.0, test_weights[model_name] - step_size)
            test_weights = normalize_weights(test_weights)
            test_score = calculate_ensemble_score(test_weights)
            
            if test_score < best_score:
                best_score = test_score
                best_weights = test_weights.copy()
                improved = True
                improvements += 1
        
        # Print progress every 100 iterations
        if (iteration + 1) % 100 == 0:
            print(f"  Iteration {iteration + 1}: RMSE = {best_score:.5f}, Improvements = {improvements}")
        
        # Early stopping if no improvement
        if not improved:
            print(f"\nConverged at iteration {iteration + 1}")
            break
    
    print(f"\n{'='*80}")
    print("OPTIMIZATION COMPLETE")
    print(f"{'='*80}")
    print("Optimized weights:")
    for name, weight in sorted(best_weights.items(), key=lambda x: -x[1]):
        print(f"  {name:15s}: {weight:.4f}")
    print(f"\nBest RMSE: {best_score:.5f}")
    print(f"Total improvements: {improvements}")
    print(f"{'='*80}\n")
    
    return best_weights, best_score
