"""
Model Classes for Triangle Sports Analytics
Contains baseline and advanced models for point spread prediction
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import warnings

# Optional imports for advanced models
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


class BaselineModel:
    """
    Simple baseline model using team averages
    Predicts spread = (Home Avg Margin) - (Away Avg Margin) + Home Court Advantage
    """
    
    HOME_COURT_ADVANTAGE = 3.5
    
    def __init__(self):
        self.team_stats = {}
        self.is_fitted = False
    
    def fit(self, historical_data: pd.DataFrame) -> 'BaselineModel':
        """
        Calculate team averages from historical data
        
        Args:
            historical_data: DataFrame with columns ['team', 'points_scored', 'points_allowed']
        """
        for team in historical_data['team'].unique():
            team_data = historical_data[historical_data['team'] == team]
            self.team_stats[team] = {
                'avg_score': team_data['points_scored'].mean(),
                'avg_allowed': team_data['points_allowed'].mean(),
                'avg_margin': (team_data['points_scored'] - team_data['points_allowed']).mean(),
                'std_margin': (team_data['points_scored'] - team_data['points_allowed']).std(),
                'games_played': len(team_data)
            }
        
        self.is_fitted = True
        return self
    
    def fit_from_team_stats(self, team_stats_df: pd.DataFrame) -> 'BaselineModel':
        """
        Fit from pre-aggregated team statistics
        
        Args:
            team_stats_df: DataFrame with columns ['team', 'ppg', 'opp_ppg']
        """
        for _, row in team_stats_df.iterrows():
            team = row['team']
            self.team_stats[team] = {
                'avg_score': row.get('ppg', 75.0),
                'avg_allowed': row.get('opp_ppg', 70.0),
                'avg_margin': row.get('ppg', 75.0) - row.get('opp_ppg', 70.0),
            }
        
        self.is_fitted = True
        return self
    
    def predict_single(self, home_team: str, away_team: str) -> float:
        """
        Predict point spread for a single game
        
        Args:
            home_team: Home team name
            away_team: Away team name
            
        Returns:
            Predicted point spread (positive = home team favored)
        """
        # Default stats if team not found
        default_stats = {'avg_margin': 0.0}
        
        home_stats = self.team_stats.get(home_team, default_stats)
        away_stats = self.team_stats.get(away_team, default_stats)
        
        # Spread = difference in average margins + home court advantage
        spread = (home_stats['avg_margin'] - away_stats['avg_margin']) / 2 + self.HOME_COURT_ADVANTAGE
        
        return spread
    
    def predict(self, matchups_df: pd.DataFrame) -> np.ndarray:
        """
        Predict point spreads for multiple games
        
        Args:
            matchups_df: DataFrame with 'Home' and 'Away' columns
            
        Returns:
            Array of predicted spreads
        """
        predictions = []
        for _, row in matchups_df.iterrows():
            pred = self.predict_single(row['Home'], row['Away'])
            predictions.append(pred)
        
        return np.array(predictions)


class LinearSpreadModel:
    """Linear regression model for point spread prediction"""
    
    def __init__(self, regularization: str = 'none', alpha: float = 1.0):
        """
        Args:
            regularization: 'none', 'ridge', or 'lasso'
            alpha: Regularization strength
        """
        if regularization == 'ridge':
            self.model = Ridge(alpha=alpha)
        elif regularization == 'lasso':
            self.model = Lasso(alpha=alpha)
        else:
            self.model = LinearRegression()
        
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_fitted = False
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> 'LinearSpreadModel':
        """Train the model"""
        self.feature_names = X_train.columns.tolist()
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)
        self.is_fitted = True
        return self
    
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        X_scaled = self.scaler.transform(X_test[self.feature_names])
        return self.model.predict(X_scaled)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature coefficients"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        return pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': self.model.coef_
        }).sort_values('coefficient', key=abs, ascending=False)


class GradientBoostingSpreadModel:
    """Gradient boosting model for point spread prediction"""
    
    def __init__(
        self, 
        model_type: str = 'sklearn',
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 5,
        random_state: int = 42
    ):
        """
        Args:
            model_type: 'sklearn', 'xgboost', or 'lightgbm'
            n_estimators: Number of boosting rounds
            learning_rate: Learning rate
            max_depth: Maximum tree depth
            random_state: Random seed
        """
        self.model_type = model_type
        self.feature_names = None
        self.is_fitted = False
        
        if model_type == 'xgboost' and HAS_XGBOOST:
            self.model = xgb.XGBRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=random_state,
                verbosity=0
            )
        elif model_type == 'lightgbm' and HAS_LIGHTGBM:
            self.model = lgb.LGBMRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=random_state,
                verbose=-1
            )
        else:
            self.model = GradientBoostingRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=random_state
            )
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> 'GradientBoostingSpreadModel':
        """Train the model"""
        self.feature_names = X_train.columns.tolist()
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        return self
    
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        return self.model.predict(X_test[self.feature_names])
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)


class EnsembleSpreadModel:
    """Ensemble model combining multiple base models"""
    
    def __init__(self, models: List[Any], weights: Optional[List[float]] = None):
        """
        Args:
            models: List of fitted models
            weights: Optional weights for each model (default: equal weights)
        """
        self.models = models
        self.weights = weights if weights else [1/len(models)] * len(models)
        
        if len(self.weights) != len(self.models):
            raise ValueError("Number of weights must match number of models")
    
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """Make weighted ensemble predictions"""
        predictions = np.zeros(len(X_test))
        
        for model, weight in zip(self.models, self.weights):
            predictions += weight * model.predict(X_test)
        
        return predictions


class PredictionIntervalModel:
    """
    Quantile regression model for prediction intervals
    Uses gradient boosting with quantile loss
    """
    
    def __init__(self, coverage: float = 0.70, n_estimators: int = 100):
        """
        Args:
            coverage: Desired coverage probability (e.g., 0.70 for 70%)
            n_estimators: Number of boosting rounds
        """
        self.coverage = coverage
        alpha = 1 - coverage
        
        self.lower_model = GradientBoostingRegressor(
            loss='quantile',
            alpha=alpha/2,
            n_estimators=n_estimators,
            max_depth=4,
            random_state=42
        )
        self.upper_model = GradientBoostingRegressor(
            loss='quantile',
            alpha=1 - alpha/2,
            n_estimators=n_estimators,
            max_depth=4,
            random_state=42
        )
        self.median_model = GradientBoostingRegressor(
            loss='quantile',
            alpha=0.5,
            n_estimators=n_estimators,
            max_depth=4,
            random_state=42
        )
        
        self.feature_names = None
        self.is_fitted = False
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> 'PredictionIntervalModel':
        """Train all quantile models"""
        self.feature_names = X_train.columns.tolist()
        
        self.lower_model.fit(X_train, y_train)
        self.upper_model.fit(X_train, y_train)
        self.median_model.fit(X_train, y_train)
        
        self.is_fitted = True
        return self
    
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """Predict median (point estimate)"""
        return self.median_model.predict(X_test[self.feature_names])
    
    def predict_interval(self, X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict lower and upper bounds
        
        Returns:
            Tuple of (lower_bound, upper_bound) arrays
        """
        X_subset = X_test[self.feature_names]
        lower = self.lower_model.predict(X_subset)
        upper = self.upper_model.predict(X_subset)
        return lower, upper
    
    def evaluate_coverage(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate prediction interval coverage
        
        Returns:
            Dictionary with coverage and average width
        """
        lower, upper = self.predict_interval(X_test)
        
        # Check if actual values fall within intervals
        in_interval = (y_test >= lower) & (y_test <= upper)
        coverage = in_interval.mean()
        
        # Average interval width
        avg_width = (upper - lower).mean()
        
        return {
            'coverage': coverage,
            'avg_width': avg_width,
            'meets_requirement': coverage >= self.coverage
        }
