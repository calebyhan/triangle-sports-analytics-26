"""
Model interpretability utilities using SHAP values
"""

from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
import warnings

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    warnings.warn("SHAP not installed. Install with: pip install shap")


class ModelExplainer:
    """Explain model predictions using SHAP values"""

    def __init__(self, model: Any, X_background: pd.DataFrame):
        """
        Initialize explainer

        Args:
            model: Trained ImprovedSpreadModel instance
            X_background: Background dataset for SHAP (typically training data sample)
        """
        if not HAS_SHAP:
            raise ImportError("SHAP is required. Install with: pip install shap")

        self.model = model
        self.X_background = X_background

        # Create explainer for the LightGBM component
        try:
            # Use TreeExplainer for tree-based models (LightGBM)
            self.lgbm_explainer = shap.TreeExplainer(model.gbm)
        except Exception as e:
            print(f"Warning: Could not create TreeExplainer: {e}")
            self.lgbm_explainer = None

        # Use KernelExplainer for Ridge (linear models)
        # Sample background to 100 rows for speed
        background_sample = shap.sample(X_background, min(100, len(X_background)))

        def ridge_predict(X):
            """Wrapper for ridge predictions"""
            X_scaled = model.scaler.transform(X)
            return model.ridge.predict(X_scaled)

        try:
            self.ridge_explainer = shap.KernelExplainer(
                ridge_predict,
                background_sample
            )
        except Exception as e:
            print(f"Warning: Could not create KernelExplainer: {e}")
            self.ridge_explainer = None

    def explain_prediction(
        self,
        X: pd.DataFrame,
        game_idx: int = 0
    ) -> Dict[str, Any]:
        """
        Explain a single game prediction

        Args:
            X: Feature dataframe
            game_idx: Index of game to explain

        Returns:
            Dictionary with SHAP values and feature contributions
        """
        X_single = X.iloc[[game_idx]]

        # Get SHAP values for both components
        lgbm_shap = None
        ridge_shap = None

        if self.lgbm_explainer:
            lgbm_shap = self.lgbm_explainer.shap_values(X_single)

        if self.ridge_explainer:
            ridge_shap = self.ridge_explainer.shap_values(X_single)

        # Weight by ensemble weights
        weights = self.model.weights

        if lgbm_shap is not None and ridge_shap is not None:
            ensemble_shap = (
                weights[0] * ridge_shap[0] +
                weights[1] * lgbm_shap[0]
            )
        elif lgbm_shap is not None:
            ensemble_shap = lgbm_shap[0]
        elif ridge_shap is not None:
            ensemble_shap = ridge_shap[0]
        else:
            ensemble_shap = np.zeros(len(X.columns))

        # Get base values
        lgbm_base = self.lgbm_explainer.expected_value if self.lgbm_explainer else 0
        ridge_base = self.ridge_explainer.expected_value if self.ridge_explainer else 0
        ensemble_base = weights[0] * ridge_base + weights[1] * lgbm_base

        # Get actual prediction
        prediction = self.model.predict(X_single)[0]

        return {
            'prediction': prediction,
            'base_value': ensemble_base,
            'shap_values': ensemble_shap,
            'feature_names': X.columns.tolist(),
            'feature_values': X_single.values[0],
            'lgbm_contribution': lgbm_shap[0] if lgbm_shap is not None else None,
            'ridge_contribution': ridge_shap[0] if ridge_shap is not None else None
        }

    def get_top_features(
        self,
        X: pd.DataFrame,
        game_idx: int = 0,
        top_n: int = 5
    ) -> pd.DataFrame:
        """
        Get top contributing features for a prediction

        Args:
            X: Feature dataframe
            game_idx: Index of game to explain
            top_n: Number of top features to return

        Returns:
            DataFrame with feature contributions sorted by absolute impact
        """
        explanation = self.explain_prediction(X, game_idx)

        df = pd.DataFrame({
            'feature': explanation['feature_names'],
            'value': explanation['feature_values'],
            'shap_value': explanation['shap_values'],
            'contribution': explanation['shap_values']  # Absolute contribution
        })

        df = df.reindex(df['shap_value'].abs().sort_values(ascending=False).index)
        return df.head(top_n)

    def summary_plot(self, X: pd.DataFrame, save_path: Optional[str] = None):
        """
        Generate SHAP summary plot

        Args:
            X: Feature dataframe
            save_path: Optional path to save plot
        """
        if not HAS_SHAP or not self.lgbm_explainer:
            print("SHAP or LightGBM explainer not available")
            return

        # Calculate SHAP values for dataset
        lgbm_shap = self.lgbm_explainer.shap_values(X)

        # Create summary plot
        shap.summary_plot(
            lgbm_shap,
            X,
            plot_type="bar",
            show=False
        )

        if save_path:
            import matplotlib.pyplot as plt
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
            print(f"   ✓ SHAP summary plot saved to {save_path}")


def create_game_explanation_report(
    explainer: ModelExplainer,
    X: pd.DataFrame,
    game_idx: int,
    home_team: str,
    away_team: str
) -> str:
    """
    Create human-readable explanation for a game prediction

    Args:
        explainer: ModelExplainer instance
        X: Feature dataframe
        game_idx: Index of game
        home_team: Home team name
        away_team: Away team name

    Returns:
        Formatted string with prediction explanation
    """
    explanation = explainer.explain_prediction(X, game_idx)
    top_features = explainer.get_top_features(X, game_idx, top_n=5)

    report = f"""
Game Prediction Explanation
{'='*60}
Matchup: {home_team} vs {away_team}
Predicted Spread: {explanation['prediction']:.2f} points

Base Prediction (average): {explanation['base_value']:.2f}

Top Contributing Features:
{'-'*60}
"""

    for idx, row in top_features.iterrows():
        impact = "increases" if row['shap_value'] > 0 else "decreases"
        report += f"{row['feature']:20s} = {row['value']:7.2f}  →  {impact} spread by {abs(row['shap_value']):.2f} pts\n"

    report += f"\n{'='*60}\n"

    return report
