"""
Test script for blowout prediction improvements.

Tests:
1. Blowout-specific features (momentum, run differential)
2. SHAP model interpretability
3. Performance comparison: baseline vs enhanced features

Run this script to evaluate whether blowout features improve model accuracy.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import config
from src.elo import EloRatingSystem
from src.models import ImprovedSpreadModel
from src.utils import fetch_barttorvik_year
from src.blowout_features import add_blowout_features_to_training_data
from src.model_explainer import analyze_blowout_features
from src.logger import setup_logger

logger = setup_logger(__name__)


def load_and_prepare_data():
    """Load historical games and compute Elo ratings."""
    logger.info("Loading historical games...")
    games = pd.read_csv(config.HISTORICAL_GAMES_FILE)
    logger.info(f"Loaded {len(games)} historical games")

    # Initialize Elo system
    logger.info("Computing Elo ratings...")
    elo = EloRatingSystem(
        k_factor=config.ELO_CONFIG['k_factor'],
        hca=config.ELO_CONFIG['home_court_advantage'],
        carryover=config.ELO_CONFIG['season_carryover']
    )
    elo.load_conference_mappings(config.CONFERENCE_MAPPINGS)

    # Process games chronologically
    games_with_elo = elo.process_games(
        games,
        date_col='date',
        home_col='home_team',
        away_col='away_team',
        home_score_col='home_score',
        away_score_col='away_score'
    )

    logger.info("Elo ratings computed successfully")
    return games_with_elo


def fetch_efficiency_data():
    """Fetch Barttorvik efficiency stats for training years."""
    logger.info("Fetching Barttorvik efficiency data...")

    all_ratings = []
    for year in config.TRAINING_YEARS:
        logger.info(f"  Fetching {year} data...")
        try:
            df = fetch_barttorvik_year(year)
            df['year'] = year
            all_ratings.append(df)
        except Exception as e:
            logger.error(f"  Failed to fetch {year}: {e}")

    ratings_df = pd.concat(all_ratings, ignore_index=True)
    logger.info(f"Fetched efficiency data for {len(ratings_df)} team-seasons")
    return ratings_df


def merge_efficiency_data(games_df, ratings_df):
    """Merge efficiency stats with games."""
    # Create team-year lookup
    ratings_lookup = {}
    for _, row in ratings_df.iterrows():
        key = (row['team'], row['year'])
        # Calculate AdjEM from AdjOE - AdjDE
        adj_oe = row.get('adjoe', 100.0)
        adj_de = row.get('adjde', 100.0)
        adj_em = adj_oe - adj_de
        ratings_lookup[key] = {
            'adj_oe': adj_oe,
            'adj_de': adj_de,
            'adj_em': adj_em,
        }

    # Add efficiency stats to games
    def get_efficiency(team, date):
        year = pd.to_datetime(date).year
        # Try current year, then fall back to previous year
        for y in [year, year - 1]:
            if (team, y) in ratings_lookup:
                return ratings_lookup[(team, y)]
        return {'adj_oe': 100.0, 'adj_de': 100.0, 'adj_em': 0.0}

    games_df['home_adj_oe'] = games_df.apply(lambda r: get_efficiency(r['home_team'], r['date'])['adj_oe'], axis=1)
    games_df['home_adj_de'] = games_df.apply(lambda r: get_efficiency(r['home_team'], r['date'])['adj_de'], axis=1)
    games_df['home_adj_em'] = games_df.apply(lambda r: get_efficiency(r['home_team'], r['date'])['adj_em'], axis=1)

    games_df['away_adj_oe'] = games_df.apply(lambda r: get_efficiency(r['away_team'], r['date'])['adj_oe'], axis=1)
    games_df['away_adj_de'] = games_df.apply(lambda r: get_efficiency(r['away_team'], r['date'])['adj_de'], axis=1)
    games_df['away_adj_em'] = games_df.apply(lambda r: get_efficiency(r['away_team'], r['date'])['adj_em'], axis=1)

    # Compute differentials
    games_df['eff_diff'] = games_df['home_adj_em'] - games_df['away_adj_em']
    games_df['elo_diff'] = games_df['home_elo_before'] - games_df['away_elo_before']

    # Add predicted_spread if not already present
    if 'predicted_spread' not in games_df.columns:
        # Use Elo difference to approximate predicted spread
        # Conversion: 100 Elo points ≈ 28 point spread (from config)
        games_df['predicted_spread'] = games_df['elo_diff'] / config.ELO_CONFIG['points_per_elo']

    return games_df


def prepare_features(games_df, include_blowout_features=False):
    """Prepare feature matrix."""
    # Start with baseline features
    feature_cols = config.BASELINE_FEATURES.copy()

    X = games_df[feature_cols].copy()
    y = games_df['home_score'] - games_df['away_score']

    if include_blowout_features:
        logger.info("Adding blowout-specific features...")
        games_enhanced = add_blowout_features_to_training_data(games_df, lookback_games=10)

        # Add blowout features to X
        blowout_feature_cols = [
            'run_diff_differential',
            'max_run_diff_differential',
            'blowout_tendency_diff',
            'consistency_ratio',
            'momentum_differential',
            'win_pct_differential',
            'trend_slope_differential',
            'hot_streak_advantage',
            'home_avg_run_diff',
            'away_avg_run_diff',
            'home_blowout_rate',
            'away_blown_out_rate',
            'home_momentum',
            'away_momentum',
        ]

        for col in blowout_feature_cols:
            if col in games_enhanced.columns:
                X[col] = games_enhanced[col]

        feature_cols.extend(blowout_feature_cols)
        logger.info(f"Total features: {len(feature_cols)}")

    return X, y, feature_cols


def evaluate_model(X, y, feature_cols, model_name):
    """Evaluate model with time-series cross-validation."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating: {model_name}")
    logger.info(f"{'='*60}")

    # Time-series cross-validation
    tscv = TimeSeriesSplit(n_splits=config.CV_CONFIG['n_splits'])

    fold_scores = []
    fold_blowout_scores = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Train model
        model = ImprovedSpreadModel(
            ridge_alpha=config.MODEL_CONFIG['ridge_alpha'],
            lgbm_params={
                'n_estimators': config.MODEL_CONFIG['n_estimators'],
                'max_depth': config.MODEL_CONFIG['max_depth'],
                'learning_rate': config.MODEL_CONFIG['learning_rate'],
                'verbosity': -1,
            },
            weights=(
                config.MODEL_CONFIG['ridge_weight'],
                config.MODEL_CONFIG['lgbm_weight']
            )
        )

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # Overall MAE
        mae = mean_absolute_error(y_test, predictions)
        fold_scores.append(mae)

        # Blowout MAE
        blowout_mask = np.abs(y_test) >= 15
        if blowout_mask.sum() > 0:
            blowout_mae = mean_absolute_error(
                y_test[blowout_mask],
                predictions[blowout_mask]
            )
            fold_blowout_scores.append(blowout_mae)

        logger.info(f"Fold {fold}: MAE = {mae:.4f}, Blowout MAE = {blowout_mae:.4f}")

    # Summary
    overall_mae = np.mean(fold_scores)
    overall_std = np.std(fold_scores)
    blowout_mae = np.mean(fold_blowout_scores)
    blowout_std = np.std(fold_blowout_scores)

    logger.info(f"\nResults Summary:")
    logger.info(f"Overall MAE: {overall_mae:.4f} ± {overall_std:.4f}")
    logger.info(f"Blowout MAE: {blowout_mae:.4f} ± {blowout_std:.4f}")

    return {
        'model_name': model_name,
        'overall_mae': overall_mae,
        'overall_std': overall_std,
        'blowout_mae': blowout_mae,
        'blowout_std': blowout_std,
        'fold_scores': fold_scores,
        'model': model,  # Last fold's model
        'X_train': X_train,
        'X_test': X_test,
        'y_test': y_test,
        'predictions': predictions,
        'feature_cols': feature_cols
    }


def main():
    """Main evaluation script."""
    logger.info("="*60)
    logger.info("BLOWOUT FEATURE EVALUATION")
    logger.info("="*60)

    # Load data
    games_with_elo = load_and_prepare_data()
    ratings_df = fetch_efficiency_data()
    games_df = merge_efficiency_data(games_with_elo, ratings_df)

    # Filter to games with efficiency data
    games_df = games_df.dropna(subset=config.BASELINE_FEATURES)
    logger.info(f"\nGames with complete features: {len(games_df)}")

    # Test 1: Baseline model
    logger.info("\n" + "="*60)
    logger.info("TEST 1: BASELINE MODEL (11 features)")
    logger.info("="*60)
    X_baseline, y, feature_cols_baseline = prepare_features(games_df, include_blowout_features=False)
    results_baseline = evaluate_model(X_baseline, y, feature_cols_baseline, "Baseline")

    # Test 2: Enhanced model with blowout features
    logger.info("\n" + "="*60)
    logger.info("TEST 2: ENHANCED MODEL (baseline + blowout features)")
    logger.info("="*60)
    X_enhanced, y, feature_cols_enhanced = prepare_features(games_df, include_blowout_features=True)
    results_enhanced = evaluate_model(X_enhanced, y, feature_cols_enhanced, "Enhanced with Blowout Features")

    # Comparison
    logger.info("\n" + "="*60)
    logger.info("COMPARISON")
    logger.info("="*60)

    improvement_overall = ((results_baseline['overall_mae'] - results_enhanced['overall_mae'])
                          / results_baseline['overall_mae'] * 100)
    improvement_blowout = ((results_baseline['blowout_mae'] - results_enhanced['blowout_mae'])
                           / results_baseline['blowout_mae'] * 100)

    logger.info(f"\nBaseline Overall MAE:     {results_baseline['overall_mae']:.4f}")
    logger.info(f"Enhanced Overall MAE:     {results_enhanced['overall_mae']:.4f}")
    logger.info(f"Improvement:              {improvement_overall:+.2f}%")
    logger.info(f"\nBaseline Blowout MAE:     {results_baseline['blowout_mae']:.4f}")
    logger.info(f"Enhanced Blowout MAE:     {results_enhanced['blowout_mae']:.4f}")
    logger.info(f"Blowout Improvement:      {improvement_blowout:+.2f}%")

    # Save comparison results
    comparison_df = pd.DataFrame([
        {
            'model': 'Baseline',
            'overall_mae': results_baseline['overall_mae'],
            'overall_std': results_baseline['overall_std'],
            'blowout_mae': results_baseline['blowout_mae'],
            'blowout_std': results_baseline['blowout_std'],
            'num_features': len(feature_cols_baseline)
        },
        {
            'model': 'Enhanced (Blowout Features)',
            'overall_mae': results_enhanced['overall_mae'],
            'overall_std': results_enhanced['overall_std'],
            'blowout_mae': results_enhanced['blowout_mae'],
            'blowout_std': results_enhanced['blowout_std'],
            'num_features': len(feature_cols_enhanced)
        }
    ])

    output_path = config.OUTPUTS_DIR / 'blowout_feature_evaluation.csv'
    comparison_df.to_csv(output_path, index=False)
    logger.info(f"\nResults saved to: {output_path}")

    # Test 3: SHAP Analysis
    logger.info("\n" + "="*60)
    logger.info("TEST 3: SHAP INTERPRETABILITY ANALYSIS")
    logger.info("="*60)

    shap_output_dir = config.OUTPUTS_DIR / 'shap_analysis'
    try:
        shap_results = analyze_blowout_features(
            results_enhanced['model'],
            results_enhanced['X_train'],
            results_enhanced['X_test'],
            results_enhanced['y_test'],
            results_enhanced['predictions'],
            results_enhanced['feature_cols'],
            shap_output_dir
        )
        logger.info(f"\nSHAP analysis complete! Results saved to: {shap_output_dir}")
    except Exception as e:
        logger.error(f"SHAP analysis failed: {e}")
        logger.error("This is non-critical - main evaluation still succeeded")

    logger.info("\n" + "="*60)
    logger.info("EVALUATION COMPLETE")
    logger.info("="*60)


if __name__ == '__main__':
    main()
