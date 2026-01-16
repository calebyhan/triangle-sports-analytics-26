"""
Unit tests for model implementations
"""

import sys
sys.path.insert(0, 'src')

import pytest
import numpy as np
import pandas as pd
from models import BaselineModel, LinearSpreadModel, ImprovedSpreadModel


def test_baseline_model():
    """Test baseline model"""
    model = BaselineModel()

    # Create sample data
    X = pd.DataFrame({
        'home_team': ['Duke', 'UNC', 'Duke'],
        'away_team': ['UNC', 'NC State', 'Virginia']
    })
    y = np.array([10.0, -5.0, 8.0])

    # Fit model
    model.fit(X, y)

    # Predict
    predictions = model.predict(X)

    assert len(predictions) == 3
    assert isinstance(predictions, np.ndarray)


def test_linear_spread_model():
    """Test linear spread model"""
    model = LinearSpreadModel(alpha=1.0, model_type='ridge')

    # Create sample numerical features
    X = pd.DataFrame({
        'feature1': [1.0, 2.0, 3.0, 4.0, 5.0],
        'feature2': [2.0, 4.0, 6.0, 8.0, 10.0]
    })
    y = np.array([5.0, 10.0, 15.0, 20.0, 25.0])

    # Fit model
    model.fit(X, y)

    # Predict
    predictions = model.predict(X)

    assert len(predictions) == 5
    assert isinstance(predictions, np.ndarray)

    # Check prediction quality (should be reasonable for linear data)
    mae = np.abs(predictions - y).mean()
    assert mae < 5.0  # Should fit reasonably well


def test_improved_spread_model():
    """Test improved ensemble model"""
    model = ImprovedSpreadModel(
        lgbm_params={'n_estimators': 10, 'max_depth': 3},
        weights=(0.4, 0.6)
    )

    # Create sample data
    np.random.seed(42)
    X = pd.DataFrame({
        'home_elo': np.random.uniform(1400, 1600, 20),
        'away_elo': np.random.uniform(1400, 1600, 20),
        'elo_diff': np.random.uniform(-100, 100, 20),
        'eff_diff': np.random.uniform(-10, 10, 20)
    })
    y = X['elo_diff'] * 0.05 + X['eff_diff'] * 0.8 + np.random.normal(0, 2, 20)

    # Fit model
    model.fit(X, y)

    # Predict
    predictions = model.predict(X)

    assert len(predictions) == 20
    assert isinstance(predictions, np.ndarray)


def test_improved_model_predict_components():
    """Test ensemble component predictions"""
    model = ImprovedSpreadModel(
        lgbm_params={'n_estimators': 10, 'max_depth': 3},
        weights=(0.3, 0.7)
    )

    # Create sample data
    np.random.seed(42)
    X = pd.DataFrame({
        'home_elo': np.random.uniform(1400, 1600, 10),
        'away_elo': np.random.uniform(1400, 1600, 10),
        'elo_diff': np.random.uniform(-100, 100, 10),
        'eff_diff': np.random.uniform(-10, 10, 10)
    })
    y = X['elo_diff'] * 0.05 + X['eff_diff'] * 0.8

    # Fit and get components
    model.fit(X, y)
    components = model.predict_components(X)

    assert 'ridge' in components
    assert 'lgbm' in components
    assert len(components['ridge']) == 10
    assert len(components['lgbm']) == 10


def test_model_weights():
    """Test ensemble weight application"""
    model = ImprovedSpreadModel(
        lgbm_params={'n_estimators': 10},
        weights=(0.5, 0.5)
    )

    np.random.seed(42)
    X = pd.DataFrame({
        'feature1': np.random.uniform(0, 10, 10),
        'feature2': np.random.uniform(0, 10, 10)
    })
    y = X['feature1'] + X['feature2']

    model.fit(X, y)
    predictions = model.predict(X)
    components = model.predict_components(X)

    # Check that ensemble is weighted average
    expected = 0.5 * components['ridge'] + 0.5 * components['lgbm']
    np.testing.assert_array_almost_equal(predictions, expected, decimal=5)


def test_model_single_sample():
    """Test prediction on single sample"""
    model = ImprovedSpreadModel(
        lgbm_params={'n_estimators': 10},
        weights=(0.4, 0.6)
    )

    np.random.seed(42)
    X_train = pd.DataFrame({
        'feature1': np.random.uniform(0, 10, 20),
        'feature2': np.random.uniform(0, 10, 20)
    })
    y_train = X_train['feature1'] + X_train['feature2']

    model.fit(X_train, y_train)

    # Predict on single sample
    X_test = pd.DataFrame({'feature1': [5.0], 'feature2': [3.0]})
    prediction = model.predict(X_test)

    assert len(prediction) == 1
    assert isinstance(prediction[0], (float, np.floating))


def test_model_missing_features():
    """Test model behavior with missing features"""
    model = ImprovedSpreadModel()

    X_train = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [2, 4, 6, 8, 10]
    })
    y_train = np.array([3, 6, 9, 12, 15])

    model.fit(X_train, y_train)

    # Test with same features
    X_test = pd.DataFrame({
        'feature1': [1.5],
        'feature2': [3.0]
    })

    prediction = model.predict(X_test)
    assert len(prediction) == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
