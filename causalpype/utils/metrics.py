import numpy as np
from typing import Dict, List, Union

def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate comprehensive regression metrics."""
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1e-8, y_true))) * 100
    
    return {
        'mae': float(mae),
        'mse': float(mse), 
        'rmse': float(rmse),
        'r2': float(r2),
        'mape': float(mape)
    }

def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate comprehensive classification metrics."""
    accuracy = np.mean(y_true == y_pred)
    
    # For binary classification
    if len(np.unique(y_true)) == 2:
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1)
        }
    
    return {'accuracy': float(accuracy)}

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, task_type: str = 'auto') -> Dict[str, float]:
    """Calculate appropriate metrics based on task type."""
    if task_type == 'auto':
        task_type = 'classification' if len(np.unique(y_true)) <= 10 and np.all(y_true == y_true.astype(int)) else 'regression'
    
    if task_type == 'regression':
        return calculate_regression_metrics(y_true, y_pred)
    else:
        return calculate_classification_metrics(y_true, y_pred)

def aggregate_fold_metrics(fold_metrics: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Aggregate metrics across cross-validation folds."""
    if not fold_metrics:
        return {}
    
    metric_names = fold_metrics[0].keys()
    aggregated = {}
    
    for metric_name in metric_names:
        values = [fold[metric_name] for fold in fold_metrics]
        aggregated[metric_name] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'fold_scores': values
        }
    
    return aggregated