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
    
    def __init__(self) -> None:
        self.team_stats = {}
        self.is_fitted = False
    
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> 'BaselineModel':
        """
        Fit model using sklearn-style interface

        Args:
            X: Feature DataFrame with 'home_team' and 'away_team' columns
            y: Target array of spreads (home_margin)

        Returns:
            Self for method chaining
        """
        # Build team statistics from X, y pairs
        team_margins = {}

        for idx, row in X.iterrows():
            home = row['home_team']
            away = row['away_team']
            margin = y[idx] if isinstance(idx, int) else y[X.index.get_loc(idx)]

            # Track home team performance
            if home not in team_margins:
                team_margins[home] = []
            team_margins[home].append(margin)

            # Track away team performance (negative margin)
            if away not in team_margins:
                team_margins[away] = []
            team_margins[away].append(-margin)

        # Calculate statistics for each team
        for team, margins in team_margins.items():
            self.team_stats[team] = {
                'avg_margin': np.mean(margins),
                'std_margin': np.std(margins),
                'games_played': len(margins)
            }

        self.is_fitted = True
        return self

    def fit_from_historical_data(self, historical_data: pd.DataFrame) -> 'BaselineModel':
        """
        Calculate team averages from historical data (legacy method)

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
            matchups_df: DataFrame with 'Home'/'home_team' and 'Away'/'away_team' columns

        Returns:
            Array of predicted spreads
        """
        # Support both capitalized and lowercase column names
        home_col = 'Home' if 'Home' in matchups_df.columns else 'home_team'
        away_col = 'Away' if 'Away' in matchups_df.columns else 'away_team'

        predictions = []
        for _, row in matchups_df.iterrows():
            pred = self.predict_single(row[home_col], row[away_col])
            predictions.append(pred)

        return np.array(predictions)


class LinearSpreadModel:
    """Linear regression model for point spread prediction"""
    
    def __init__(self, model_type: str = 'none', alpha: float = 1.0) -> None:
        """
        Args:
            model_type: 'none', 'ridge', or 'lasso'
            alpha: Regularization strength
        """
        if model_type == 'ridge':
            self.model = Ridge(alpha=alpha)
        elif model_type == 'lasso':
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
    ) -> None:
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
    
    def __init__(self, models: List[Any], weights: Optional[List[float]] = None) -> None:
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


class ImprovedSpreadModel:
    """
    Improved ensemble model combining Ridge and LightGBM

    Research shows:
    - LightGBM achieves ~9.09 MAE with proper features
    - Ridge provides stable baseline
    - Ensemble typically outperforms individual models
    """

    def __init__(
        self,
        ridge_alpha: float = 1.0,
        lgbm_params: Optional[Dict] = None,
        weights: Tuple[float, float] = (0.4, 0.6),
        use_lgbm: bool = True
    ):
        """
        Initialize improved spread model

        Args:
            ridge_alpha: Regularization for Ridge
            lgbm_params: Parameters for LightGBM
            weights: (ridge_weight, lgbm_weight) for ensemble
            use_lgbm: If False, fall back to GradientBoosting
        """
        self.ridge = Ridge(alpha=ridge_alpha)
        self.scaler = StandardScaler()
        self.weights = weights
        self.use_lgbm = use_lgbm and HAS_LIGHTGBM

        # Default LightGBM parameters (tuned for spread prediction)
        default_lgbm_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'num_leaves': 31,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': 42,
            'verbose': -1,
        }

        if lgbm_params:
            default_lgbm_params.update(lgbm_params)

        if self.use_lgbm:
            self.gbm = lgb.LGBMRegressor(**default_lgbm_params)
        else:
            # Fallback to sklearn GradientBoosting
            self.gbm = GradientBoostingRegressor(
                n_estimators=default_lgbm_params['n_estimators'],
                max_depth=default_lgbm_params['max_depth'],
                learning_rate=default_lgbm_params['learning_rate'],
                random_state=42
            )

        self.feature_names = None
        self.is_fitted = False

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> 'ImprovedSpreadModel':
        """
        Train both models

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Optional validation features (for early stopping)
            y_val: Optional validation targets
        """
        self.feature_names = X_train.columns.tolist()

        # Scale features for Ridge
        X_scaled = self.scaler.fit_transform(X_train)
        self.ridge.fit(X_scaled, y_train)

        # Train GBM (unscaled - tree models don't need scaling)
        if self.use_lgbm and X_val is not None and y_val is not None:
            # Use early stopping
            self.gbm.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
            )
        else:
            self.gbm.fit(X_train, y_train)

        self.is_fitted = True
        return self

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """Make ensemble predictions"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")

        X_subset = X_test[self.feature_names]

        # Ridge prediction (needs scaling)
        X_scaled = self.scaler.transform(X_subset)
        ridge_pred = self.ridge.predict(X_scaled)

        # GBM prediction
        gbm_pred = self.gbm.predict(X_subset)

        # Weighted ensemble
        ensemble_pred = (
            self.weights[0] * ridge_pred +
            self.weights[1] * gbm_pred
        )

        return ensemble_pred

    def predict_components(self, X_test: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Return predictions from each component model"""
        X_subset = X_test[self.feature_names]

        X_scaled = self.scaler.transform(X_subset)

        return {
            'ridge': self.ridge.predict(X_scaled),
            'lgbm': self.gbm.predict(X_subset),
            'ensemble': self.predict(X_test)
        }

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from GBM model"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")

        importance = self.gbm.feature_importances_

        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

    def get_ridge_coefficients(self) -> pd.DataFrame:
        """Get Ridge regression coefficients"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")

        return pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': self.ridge.coef_
        }).sort_values('coefficient', key=abs, ascending=False)


class CalibratedSpreadModel:
    """
    Model focused on calibration rather than just accuracy

    Research shows calibration matters more than accuracy for betting:
    - ROI +34.69% with calibration-based selection
    - ROI -35.17% with accuracy-based selection
    """

    def __init__(self, base_model: Any = None):
        """
        Args:
            base_model: Base model to calibrate (default: Ridge)
        """
        if base_model is None:
            self.base_model = Ridge(alpha=1.0)
        else:
            self.base_model = base_model

        self.scaler = StandardScaler()
        self.residual_std = None
        self.feature_names = None
        self.is_fitted = False

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> 'CalibratedSpreadModel':
        """Train model and estimate prediction uncertainty"""
        self.feature_names = X_train.columns.tolist()

        X_scaled = self.scaler.fit_transform(X_train)
        self.base_model.fit(X_scaled, y_train)

        # Estimate residual standard deviation for calibration
        y_pred = self.base_model.predict(X_scaled)
        residuals = y_train - y_pred
        self.residual_std = np.std(residuals)

        self.is_fitted = True
        return self

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """Make point predictions"""
        X_scaled = self.scaler.transform(X_test[self.feature_names])
        return self.base_model.predict(X_scaled)

    def predict_with_confidence(
        self,
        X_test: pd.DataFrame,
        confidence: float = 0.80
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Make predictions with confidence intervals

        Args:
            X_test: Test features
            confidence: Confidence level (default 80%)

        Returns:
            Tuple of (predictions, lower_bound, upper_bound)
        """
        from scipy import stats

        predictions = self.predict(X_test)

        # Calculate z-score for confidence level
        alpha = 1 - confidence
        z = stats.norm.ppf(1 - alpha / 2)

        # Confidence interval
        margin = z * self.residual_std
        lower = predictions - margin
        upper = predictions + margin

        return predictions, lower, upper

    def brier_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Brier score for calibration assessment

        Lower is better (0 = perfect calibration)
        """
        # Convert to win probability (sigmoid of spread)
        prob_true = (y_true > 0).astype(float)
        prob_pred = 1 / (1 + np.exp(-y_pred / 5))  # Spread to probability

        return np.mean((prob_pred - prob_true) ** 2)


def cross_validate_model(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    time_series: bool = True
) -> Dict[str, List[float]]:
    """
    Cross-validate a spread prediction model

    Args:
        model: Model instance with fit/predict methods
        X: Features
        y: Target (margins)
        n_splits: Number of CV folds
        time_series: If True, use TimeSeriesSplit

    Returns:
        Dictionary with MAE and RMSE for each fold
    """
    if time_series:
        cv = TimeSeriesSplit(n_splits=n_splits)
    else:
        from sklearn.model_selection import KFold
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    mae_scores = []
    rmse_scores = []
    brier_scores = []

    for train_idx, val_idx in cv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Clone and fit model
        model_clone = type(model)() if hasattr(model, '__class__') else model
        model_clone.fit(X_train, y_train)

        # Predict
        y_pred = model_clone.predict(X_val)

        # Calculate metrics
        mae = np.abs(y_pred - y_val).mean()
        rmse = np.sqrt(((y_pred - y_val) ** 2).mean())

        mae_scores.append(mae)
        rmse_scores.append(rmse)

        # Brier score if available
        if hasattr(model_clone, 'brier_score'):
            brier = model_clone.brier_score(y_val.values, y_pred)
            brier_scores.append(brier)

    return {
        'mae': mae_scores,
        'rmse': rmse_scores,
        'brier': brier_scores if brier_scores else None,
        'mae_mean': np.mean(mae_scores),
        'mae_std': np.std(mae_scores),
        'rmse_mean': np.mean(rmse_scores),
        'rmse_std': np.std(rmse_scores),
    }
