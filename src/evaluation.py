"""
Model Evaluation Module for Triangle Sports Analytics
Contains evaluation metrics and cross-validation utilities
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from sklearn.model_selection import TimeSeriesSplit, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


class ModelEvaluator:
    """Evaluate point spread prediction models"""
    
    def __init__(self, model: Any):
        """
        Args:
            model: A fitted model with predict() method
        """
        self.model = model
        self.results = {}
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate model on test data
        
        Args:
            X_test: Feature matrix
            y_test: True point spreads
            
        Returns:
            Dictionary of evaluation metrics
        """
        y_pred = self.model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        residuals = y_test - y_pred
        
        self.results = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mean_residual': np.mean(residuals),
            'std_residual': np.std(residuals),
            'median_absolute_error': np.median(np.abs(residuals)),
            'max_error': np.max(np.abs(residuals)),
            'predictions': y_pred,
            'residuals': residuals
        }
        
        return self.results
    
    def time_series_cv(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        n_splits: int = 5
    ) -> Dict[str, float]:
        """
        Perform time series cross-validation
        
        IMPORTANT: Use this for sports data, not random splits!
        
        Args:
            X: Feature matrix (should be sorted by date)
            y: Target values
            n_splits: Number of CV folds
            
        Returns:
            Dictionary with mean and std of metrics
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        mae_scores = []
        rmse_scores = []
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Clone and fit model
            model_clone = self.model.__class__(**self._get_model_params())
            model_clone.fit(X_train, y_train)
            
            y_pred = model_clone.predict(X_test)
            
            mae_scores.append(mean_absolute_error(y_test, y_pred))
            rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_pred)))
        
        return {
            'mae_mean': np.mean(mae_scores),
            'mae_std': np.std(mae_scores),
            'rmse_mean': np.mean(rmse_scores),
            'rmse_std': np.std(rmse_scores),
            'mae_scores': mae_scores,
            'rmse_scores': rmse_scores
        }
    
    def _get_model_params(self) -> Dict:
        """Get model parameters for cloning"""
        if hasattr(self.model, 'get_params'):
            return self.model.get_params()
        return {}
    
    def plot_residuals(self, figsize: Tuple[int, int] = (12, 4)) -> plt.Figure:
        """
        Plot residual diagnostics
        
        Returns:
            Matplotlib figure
        """
        if 'residuals' not in self.results:
            raise ValueError("Run evaluate() first")
        
        residuals = self.results['residuals']
        predictions = self.results['predictions']
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Residuals vs Predicted
        axes[0].scatter(predictions, residuals, alpha=0.5)
        axes[0].axhline(y=0, color='r', linestyle='--')
        axes[0].set_xlabel('Predicted Spread')
        axes[0].set_ylabel('Residual')
        axes[0].set_title('Residuals vs Predicted')
        
        # Residual histogram
        axes[1].hist(residuals, bins=20, edgecolor='black')
        axes[1].axvline(x=0, color='r', linestyle='--')
        axes[1].set_xlabel('Residual')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Residual Distribution')
        
        # QQ Plot approximation
        sorted_residuals = np.sort(residuals)
        n = len(sorted_residuals)
        theoretical_quantiles = np.linspace(0.01, 0.99, n)
        theoretical_values = np.quantile(np.random.standard_normal(10000), theoretical_quantiles)
        
        axes[2].scatter(theoretical_values, sorted_residuals, alpha=0.5)
        axes[2].plot([-3, 3], [-3 * np.std(residuals), 3 * np.std(residuals)], 'r--')
        axes[2].set_xlabel('Theoretical Quantiles')
        axes[2].set_ylabel('Sample Quantiles')
        axes[2].set_title('Q-Q Plot')
        
        plt.tight_layout()
        return fig
    
    def plot_predictions_vs_actual(self, y_test: pd.Series, figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
        """
        Plot predicted vs actual values
        
        Args:
            y_test: True point spreads
            
        Returns:
            Matplotlib figure
        """
        if 'predictions' not in self.results:
            raise ValueError("Run evaluate() first")
        
        predictions = self.results['predictions']
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.scatter(y_test, predictions, alpha=0.5)
        
        # Perfect prediction line
        min_val = min(y_test.min(), predictions.min())
        max_val = max(y_test.max(), predictions.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        
        ax.set_xlabel('Actual Point Spread')
        ax.set_ylabel('Predicted Point Spread')
        ax.set_title(f'Predicted vs Actual (MAE: {self.results["mae"]:.2f})')
        ax.legend()
        
        plt.tight_layout()
        return fig


def compare_models(
    models: Dict[str, Any], 
    X_train: pd.DataFrame, 
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> pd.DataFrame:
    """
    Compare multiple models
    
    Args:
        models: Dictionary of model_name -> model object
        X_train, y_train: Training data
        X_test, y_test: Test data
        
    Returns:
        DataFrame with comparison metrics
    """
    results = []
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        results.append({
            'model': name,
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        })
    
    return pd.DataFrame(results).sort_values('mae')


def calculate_betting_metrics(
    predictions: np.ndarray,
    actuals: np.ndarray,
    vegas_lines: np.ndarray = None
) -> Dict[str, float]:
    """
    Calculate betting-related metrics
    
    Args:
        predictions: Predicted spreads
        actuals: Actual game margins
        vegas_lines: Optional Vegas lines for comparison
        
    Returns:
        Dictionary of betting metrics
    """
    metrics = {}
    
    # Direction accuracy (did we predict correct winner?)
    pred_winner = predictions > 0
    actual_winner = actuals > 0
    metrics['direction_accuracy'] = np.mean(pred_winner == actual_winner)
    
    # If we have Vegas lines, calculate against the spread
    if vegas_lines is not None:
        # Assuming negative line means home favored
        # Our prediction "covers" if:
        # - We predict home wins by more than the line OR
        # - We predict away wins when line has home favored by less
        
        our_pick_covers = np.sign(predictions - vegas_lines) == np.sign(actuals - vegas_lines)
        metrics['ats_accuracy'] = np.mean(our_pick_covers)
    
    return metrics
