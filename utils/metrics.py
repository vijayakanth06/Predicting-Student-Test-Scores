"""
Metrics and evaluation utilities
"""

import numpy as np
from sklearn.metrics import mean_squared_error
import json
from datetime import datetime
from pathlib import Path

def calculate_rmse(y_true, y_pred):
    """Calculate Root Mean Squared Error"""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def log_cv_score(model_name, fold_scores, mean_score, std_score, log_file="cv_scores.json"):
    """Log cross-validation scores to file"""
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'model': model_name,
        'fold_scores': fold_scores,
        'mean_cv_score': mean_score,
        'std_cv_score': std_score
    }
    
    log_path = Path(log_file)
    
    # Load existing logs if file exists
    if log_path.exists():
        with open(log_path, 'r') as f:
            logs = json.load(f)
    else:
        logs = []
    
    logs.append(log_entry)
    
    # Save updated logs
    with open(log_path, 'w') as f:
        json.dump(logs, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"Fold Scores: {fold_scores}")
    print(f"Mean CV RMSE: {mean_score:.5f} (+/- {std_score:.5f})")
    print(f"{'='*60}\n")
    
    return mean_score

def log_submission_score(submission_name, cv_score, lb_score=None, log_file="submission_log.json"):
    """Log submission information"""
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'submission_name': submission_name,
        'cv_score': cv_score,
        'lb_score': lb_score
    }
    
    log_path = Path(log_file)
    
    if log_path.exists():
        with open(log_path, 'r') as f:
            logs = json.load(f)
    else:
        logs = []
    
    logs.append(log_entry)
    
    with open(log_path, 'w') as f:
        json.dump(logs, f, indent=2)
    
    print(f"\nSubmission logged: {submission_name}")
    print(f"CV Score: {cv_score:.5f}")
    if lb_score:
        print(f"LB Score: {lb_score:.5f}")
