"""
Model Interpretability with SHAP

Provides SHAP (SHapley Additive exPlanations) integration for understanding
model predictions. SHAP values explain how much each feature contributed to
a specific prediction.

Key use cases:
1. Understand which features drive blowout predictions
2. Validate that model is using features sensibly
3. Debug unexpected predictions
4. Generate human-readable explanations for stakeholders
"""

import pandas as pd
import numpy as np
import shap
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt
from pathlib import Path
import warnings


class ModelExplainer:
    """
    SHAP-based model explainer for point spread predictions.

    Supports both Ridge and LightGBM models in the ensemble.
    """

    def __init__(self, model, feature_names: List[str]):
        """
        Initialize explainer.

        Args:
            model: Trained model (ImprovedSpreadModel or sklearn/lightgbm model)
            feature_names: List of feature names in order used for training
        """
        self.model = model
        self.feature_names = feature_names
        self._explainer = None
        self._shap_values = None

    def create_explainer(self, X_background: pd.DataFrame, max_background: int = 100):
        """
        Create SHAP explainer using background dataset.

        Args:
            X_background: Background dataset for SHAP (typically training set sample)
            max_background: Maximum number of background samples (for speed)

        Note:
            For tree models (LightGBM), uses TreeExplainer (fast, exact)
            For linear models (Ridge), uses LinearExplainer (fast, exact)
            For ensemble, uses KernelExplainer (slower, approximation)
        """
        # Sample background if too large
        if len(X_background) > max_background:
            X_background = X_background.sample(n=max_background, random_state=42)

        # Store background as DataFrame for feature names
        self._X_background = X_background.copy()

        # Detect model type and create appropriate explainer
        if hasattr(self.model, 'ridge') and hasattr(self.model, 'gbm'):
            # ImprovedSpreadModel ensemble - use the GBM component (LightGBM or GradientBoosting)
            # (Ridge is linear so less interesting for SHAP visualization)
            print("Detected ensemble model - using GBM component for SHAP")
            # Create TreeExplainer directly on the GBM model
            # Suppress warnings during SHAP computation
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='.*does not have valid feature names.*')
                self._explainer = shap.TreeExplainer(self.model.gbm)
        elif hasattr(self.model, 'tree_'):
            # Tree-based model
            self._explainer = shap.TreeExplainer(self.model)
        elif hasattr(self.model, 'coef_'):
            # Linear model
            self._explainer = shap.LinearExplainer(self.model, X_background)
        else:
            # Fallback to KernelExplainer (slower but works for any model)
            warnings.warn("Using KernelExplainer - this may be slow for large datasets")
            self._explainer = shap.KernelExplainer(self.model.predict, X_background)

        return self._explainer

    def explain_predictions(
        self,
        X_test: pd.DataFrame,
        max_samples: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute SHAP values for test predictions.

        Args:
            X_test: Test dataset to explain
            max_samples: Maximum number of samples to explain (for speed)

        Returns:
            SHAP values array (n_samples, n_features)
        """
        if self._explainer is None:
            raise ValueError("Must call create_explainer() first")

        # Sample if needed
        if max_samples and len(X_test) > max_samples:
            X_test = X_test.sample(n=max_samples, random_state=42)

        # Store sampled X_test for later use
        self._X_test = X_test.copy()

        # Keep as DataFrame to preserve feature names for tree explainers
        # TreeExplainer can handle DataFrames and will preserve feature names
        X_test_data = X_test if isinstance(X_test, pd.DataFrame) else pd.DataFrame(X_test, columns=self.feature_names)

        # Compute SHAP values, suppressing feature name warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='.*does not have valid feature names.*')
            self._shap_values = self._explainer.shap_values(X_test_data)

        return self._shap_values

    def get_feature_importance(
        self,
        shap_values: Optional[np.ndarray] = None,
        sort: bool = True
    ) -> pd.DataFrame:
        """
        Get global feature importance from SHAP values.

        Args:
            shap_values: SHAP values array (uses cached if None)
            sort: Whether to sort by importance

        Returns:
            DataFrame with feature names and importance scores
        """
        if shap_values is None:
            shap_values = self._shap_values

        if shap_values is None:
            raise ValueError("Must call explain_predictions() first or provide shap_values")

        # Mean absolute SHAP value = feature importance
        importance = np.abs(shap_values).mean(axis=0)

        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance,
            'importance_pct': 100 * importance / importance.sum()
        })

        if sort:
            importance_df = importance_df.sort_values('importance', ascending=False)

        return importance_df

    def explain_single_prediction(
        self,
        X_single: pd.DataFrame,
        prediction: float,
        show_top_n: int = 10
    ) -> pd.DataFrame:
        """
        Explain a single prediction in human-readable format.

        Args:
            X_single: Single row DataFrame with feature values
            prediction: Model's prediction for this sample
            show_top_n: Number of top contributing features to show

        Returns:
            DataFrame explaining the prediction
        """
        if self._explainer is None:
            raise ValueError("Must call create_explainer() first")

        # Compute SHAP values for this sample
        shap_values_single = self._explainer.shap_values(X_single)

        # Create explanation DataFrame
        explanation = pd.DataFrame({
            'feature': self.feature_names,
            'value': X_single.iloc[0].values,
            'shap_value': shap_values_single[0],
            'abs_shap': np.abs(shap_values_single[0])
        })

        # Sort by absolute SHAP value
        explanation = explanation.sort_values('abs_shap', ascending=False)

        # Add cumulative contribution
        explanation['cumulative_contribution'] = explanation['shap_value'].cumsum()

        return explanation.head(show_top_n)

    def generate_summary_plot(
        self,
        X_test: pd.DataFrame,
        shap_values: Optional[np.ndarray] = None,
        save_path: Optional[Path] = None,
        max_display: int = 20
    ):
        """
        Generate SHAP summary plot showing feature importance and effects.

        Args:
            X_test: Test dataset
            shap_values: SHAP values (uses cached if None)
            save_path: Path to save plot (shows if None)
            max_display: Maximum number of features to display
        """
        if shap_values is None:
            shap_values = self._shap_values

        if shap_values is None:
            raise ValueError("Must call explain_predictions() first")

        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values,
            X_test,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False
        )

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"Summary plot saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def generate_waterfall_plot(
        self,
        X_single: pd.DataFrame,
        shap_values_single: Optional[np.ndarray] = None,
        save_path: Optional[Path] = None,
        max_display: int = 10
    ):
        """
        Generate waterfall plot for a single prediction.

        Shows how the prediction builds up from base value through each feature.

        Args:
            X_single: Single row DataFrame
            shap_values_single: SHAP values for this sample
            save_path: Path to save plot
            max_display: Maximum features to display
        """
        if shap_values_single is None and self._explainer is None:
            raise ValueError("Must provide shap_values_single or call create_explainer() first")

        if shap_values_single is None:
            shap_values_single = self._explainer.shap_values(X_single)

        # Create explanation object for waterfall plot
        explanation = shap.Explanation(
            values=shap_values_single[0],
            base_values=self._explainer.expected_value if hasattr(self._explainer, 'expected_value') else 0,
            data=X_single.iloc[0].values,
            feature_names=self.feature_names
        )

        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(explanation, max_display=max_display, show=False)

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"Waterfall plot saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def generate_text_explanation(
        self,
        X_single: pd.DataFrame,
        prediction: float,
        actual: Optional[float] = None,
        top_n: int = 5
    ) -> str:
        """
        Generate human-readable text explanation of a prediction.

        Args:
            X_single: Single row DataFrame with features
            prediction: Model prediction
            actual: Actual value (if available)
            top_n: Number of top features to explain

        Returns:
            Formatted text explanation
        """
        explanation_df = self.explain_single_prediction(X_single, prediction, top_n)

        # Build text explanation
        lines = []
        lines.append(f"Predicted Point Spread: {prediction:.2f}")
        if actual is not None:
            error = abs(prediction - actual)
            lines.append(f"Actual Point Spread: {actual:.2f}")
            lines.append(f"Prediction Error: {error:.2f}")
        lines.append("")
        lines.append(f"Top {top_n} Contributing Features:")
        lines.append("")

        for i, row in explanation_df.iterrows():
            feature = row['feature']
            value = row['value']
            shap_val = row['shap_value']
            direction = "increases" if shap_val > 0 else "decreases"

            lines.append(f"{i+1}. {feature} = {value:.2f}")
            lines.append(f"   → {direction} spread by {abs(shap_val):.2f} points")
            lines.append("")

        return "\n".join(lines)


def analyze_blowout_features(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    predictions: np.ndarray,
    feature_names: List[str],
    output_dir: Path
) -> Dict[str, Any]:
    """
    Analyze which features are most important for blowout predictions.

    Args:
        model: Trained model
        X_train: Training features (for background)
        X_test: Test features
        y_test: Actual test labels
        predictions: Model predictions
        feature_names: List of feature names
        output_dir: Directory to save outputs

    Returns:
        Dictionary with analysis results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    explainer = ModelExplainer(model, feature_names)

    print("Creating SHAP explainer...")
    explainer.create_explainer(X_train, max_background=100)

    print("Computing SHAP values...")
    shap_values = explainer.explain_predictions(X_test, max_samples=500)

    # Overall feature importance
    print("\nGlobal Feature Importance:")
    importance_df = explainer.get_feature_importance()
    print(importance_df.head(15).to_string(index=False))

    # Save importance
    importance_df.to_csv(output_dir / 'feature_importance_shap.csv', index=False)

    # Generate summary plot
    print("\nGenerating summary plot...")
    explainer.generate_summary_plot(
        X_test.head(500),
        save_path=output_dir / 'shap_summary_plot.png'
    )

    # Analyze blowout-specific predictions
    # Note: SHAP values are computed on sampled data (max 500 rows), not full test set
    n_shap_samples = len(shap_values)

    # Use only the sampled portion for blowout analysis
    y_test_sample = y_test.iloc[:n_shap_samples] if len(y_test) > n_shap_samples else y_test
    X_test_sample = X_test.iloc[:n_shap_samples] if len(X_test) > n_shap_samples else X_test
    predictions_sample = predictions[:n_shap_samples] if len(predictions) > n_shap_samples else predictions

    abs_spread = np.abs(y_test_sample)
    blowout_mask = abs_spread >= 15
    blowout_indices = np.where(blowout_mask)[0]

    if len(blowout_indices) > 0:
        print(f"\nAnalyzing {len(blowout_indices)} blowout games...")

        # Feature importance specifically for blowouts
        blowout_shap = shap_values[blowout_indices]
        blowout_importance = pd.DataFrame({
            'feature': feature_names,
            'blowout_importance': np.abs(blowout_shap).mean(axis=0),
        })
        blowout_importance = blowout_importance.sort_values('blowout_importance', ascending=False)

        print("\nFeature Importance for Blowout Games:")
        print(blowout_importance.head(10).to_string(index=False))

        blowout_importance.to_csv(output_dir / 'blowout_feature_importance.csv', index=False)

        # Generate example explanations for a few blowouts
        for i, idx in enumerate(blowout_indices[:3]):  # First 3 blowouts
            text_explanation = explainer.generate_text_explanation(
                X_test_sample.iloc[[idx]],
                predictions_sample[idx],
                y_test_sample.iloc[idx],
                top_n=5
            )

            print(f"\n{'='*60}")
            print(f"Example Blowout Prediction #{i+1}")
            print('='*60)
            print(text_explanation)

            # Save explanation
            with open(output_dir / f'blowout_example_{i+1}.txt', 'w') as f:
                f.write(text_explanation)

    return {
        'explainer': explainer,
        'shap_values': shap_values,
        'importance': importance_df,
        'blowout_importance': blowout_importance if len(blowout_indices) > 0 else None
    }
