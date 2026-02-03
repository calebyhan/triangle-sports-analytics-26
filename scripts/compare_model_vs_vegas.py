"""
Compare trained model accuracy vs Vegas odds baseline strategy.

This script evaluates:
1. Trained ensemble model (Ridge + LightGBM) using historical data
2. Real Vegas closing lines from Sportsbookreviewsonline historical data

Metrics compared:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Direction Accuracy (correct winner prediction)
- Against-the-Spread (ATS) accuracy
- Head-to-head comparison

Usage:
    python scripts/compare_model_vs_vegas.py [--season 2015-16]

Data source:
    Download Excel files from https://www.sportsbookreviewsonline.com/scoresoddsarchives/
    Place in data/raw/odds/ with names like ncaa-basketball-2015-16.xlsx
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr

from src import config
from src.elo import EloRatingSystem
from src.models import ImprovedSpreadModel
from src.utils import fetch_barttorvik_year
from src.vegas_odds import load_historical_odds_from_excel, normalize_team_name
from src.logger import setup_logger

logger = setup_logger(__name__)


def setup_trained_model() -> Tuple[ImprovedSpreadModel, EloRatingSystem, pd.DataFrame]:
    """
    Set up the trained model using historical data.

    Returns:
        Tuple of (trained model, Elo system, training data)
    """
    logger.info("Loading historical games for model training...")
    games_path = config.HISTORICAL_GAMES_FILE

    if not games_path.exists():
        raise FileNotFoundError(f"Historical games file not found: {games_path}")

    games = pd.read_csv(games_path, parse_dates=['date'])

    # Initialize Elo
    logger.info("Processing games through Elo system...")
    elo = EloRatingSystem(
        k_factor=config.ELO_CONFIG['k_factor'],
        hca=config.ELO_CONFIG['home_court_advantage'],
        carryover=config.ELO_CONFIG['season_carryover']
    )
    elo.load_conference_mappings(config.CONFERENCE_MAPPINGS)

    elo_snapshots = elo.process_games(
        games,
        date_col='date',
        home_col='home_team',
        away_col='away_team',
        home_score_col='home_score',
        away_score_col='away_score',
        neutral_col='neutral_site',
        season_col='season',
        save_snapshots=True
    )

    # Load efficiency stats for training years
    logger.info("Loading team efficiency stats...")
    all_stats = []
    for year in config.TRAINING_YEARS:
        df = fetch_barttorvik_year(year)
        df['season'] = year
        all_stats.append(df[['team', 'adjoe', 'adjde', 'season']])

    team_stats = pd.concat(all_stats, ignore_index=True)
    team_stats.columns = ['team', 'adj_oe', 'adj_de', 'season']
    team_stats['adj_em'] = team_stats['adj_oe'] - team_stats['adj_de']

    # Merge to create training data
    elo_snapshots['season'] = elo_snapshots['date'].dt.year

    train_data = elo_snapshots.merge(
        team_stats,
        left_on=['home_team', 'season'],
        right_on=['team', 'season'],
        how='left'
    ).rename(columns={'adj_oe': 'home_adj_oe', 'adj_de': 'home_adj_de', 'adj_em': 'home_adj_em'})

    train_data = train_data.drop(columns=['team'], errors='ignore').merge(
        team_stats,
        left_on=['away_team', 'season'],
        right_on=['team', 'season'],
        how='left'
    ).rename(columns={'adj_oe': 'away_adj_oe', 'adj_de': 'away_adj_de', 'adj_em': 'away_adj_em'})

    train_data = train_data.drop(columns=['team'], errors='ignore')
    train_data['eff_diff'] = train_data['home_adj_em'] - train_data['away_adj_em']
    train_data['elo_diff'] = train_data['home_elo_before'] - train_data['away_elo_before']
    train_data = train_data.dropna(subset=['home_adj_oe', 'away_adj_oe'])

    # Train model
    logger.info(f"Training model on {len(train_data)} games...")
    feature_cols = config.BASELINE_FEATURES

    X = train_data[feature_cols]
    y = train_data['actual_margin']

    model = ImprovedSpreadModel(
        ridge_alpha=config.MODEL_CONFIG['ridge_alpha'],
        lgbm_params={
            'n_estimators': config.MODEL_CONFIG['n_estimators'],
            'max_depth': config.MODEL_CONFIG['max_depth'],
            'learning_rate': config.MODEL_CONFIG['learning_rate']
        },
        weights=(config.MODEL_CONFIG['ridge_weight'], config.MODEL_CONFIG['lgbm_weight']),
        use_lgbm=True
    )

    model.fit(X, y)
    logger.info("Model trained successfully")

    return model, elo, train_data


def generate_model_predictions(
    games: pd.DataFrame,
    model: ImprovedSpreadModel,
    elo: EloRatingSystem,
    team_stats: pd.DataFrame
) -> np.ndarray:
    """
    Generate model predictions for games.

    Args:
        games: DataFrame with home_team, away_team columns
        model: Trained model
        elo: Elo rating system
        team_stats: Team efficiency stats

    Returns:
        Array of predicted spreads
    """
    # Drop duplicates keeping first occurrence
    team_stats_unique = team_stats.drop_duplicates(subset=['team'], keep='first')
    team_dict = team_stats_unique.set_index('team').to_dict('index')
    features_list = []
    valid_mask = []

    for _, row in games.iterrows():
        home = row['home_team']
        away = row['away_team']

        home_stats = team_dict.get(home, {})
        away_stats = team_dict.get(away, {})

        # Check if we have stats for both teams
        if not home_stats or not away_stats:
            features_list.append(None)
            valid_mask.append(False)
            continue

        home_oe = home_stats.get('adj_oe', 100)
        home_de = home_stats.get('adj_de', 100)
        away_oe = away_stats.get('adj_oe', 100)
        away_de = away_stats.get('adj_de', 100)

        features = {
            'home_adj_oe': home_oe,
            'home_adj_de': home_de,
            'home_adj_em': home_oe - home_de,
            'away_adj_oe': away_oe,
            'away_adj_de': away_de,
            'away_adj_em': away_oe - away_de,
            'eff_diff': (home_oe - home_de) - (away_oe - away_de),
            'home_elo_before': elo.get_rating(home),
            'away_elo_before': elo.get_rating(away),
            'elo_diff': elo.get_rating(home) - elo.get_rating(away),
            'predicted_spread': elo.predict_spread(home, away),
        }

        features_list.append(features)
        valid_mask.append(True)

    # Build feature matrix for valid games only
    valid_features = [f for f in features_list if f is not None]
    if not valid_features:
        return np.array([]), np.array(valid_mask)

    X = pd.DataFrame(valid_features)
    predictions = model.predict(X)

    # Expand predictions to full array with NaN for invalid
    full_predictions = np.full(len(games), np.nan)
    pred_idx = 0
    for i, valid in enumerate(valid_mask):
        if valid:
            full_predictions[i] = predictions[pred_idx]
            pred_idx += 1

    return full_predictions, np.array(valid_mask)


def calculate_comparison_metrics(
    actual: np.ndarray,
    model_pred: np.ndarray,
    vegas_pred: np.ndarray
) -> Dict[str, Dict]:
    """
    Calculate comprehensive comparison metrics.

    Args:
        actual: Actual game margins
        model_pred: Model predictions
        vegas_pred: Vegas spreads

    Returns:
        Dictionary with metrics for model, vegas, and comparison
    """
    metrics = {
        'model': {},
        'vegas': {},
        'comparison': {}
    }

    # MAE
    metrics['model']['mae'] = mean_absolute_error(actual, model_pred)
    metrics['vegas']['mae'] = mean_absolute_error(actual, vegas_pred)

    # RMSE
    metrics['model']['rmse'] = np.sqrt(mean_squared_error(actual, model_pred))
    metrics['vegas']['rmse'] = np.sqrt(mean_squared_error(actual, vegas_pred))

    # R-squared
    metrics['model']['r2'] = r2_score(actual, model_pred)
    metrics['vegas']['r2'] = r2_score(actual, vegas_pred)

    # Direction accuracy (correct winner prediction)
    metrics['model']['direction_acc'] = np.mean((model_pred > 0) == (actual > 0))
    metrics['vegas']['direction_acc'] = np.mean((vegas_pred > 0) == (actual > 0))

    # Correlation with actual
    metrics['model']['correlation'], _ = pearsonr(model_pred, actual)
    metrics['vegas']['correlation'], _ = pearsonr(vegas_pred, actual)

    # Head-to-head: which prediction is closer to actual
    model_errors = np.abs(model_pred - actual)
    vegas_errors = np.abs(vegas_pred - actual)

    model_wins = np.sum(model_errors < vegas_errors)
    vegas_wins = np.sum(vegas_errors < model_errors)
    ties = np.sum(model_errors == vegas_errors)

    metrics['comparison']['model_wins'] = int(model_wins)
    metrics['comparison']['vegas_wins'] = int(vegas_wins)
    metrics['comparison']['ties'] = int(ties)
    metrics['comparison']['model_win_pct'] = model_wins / len(actual) * 100
    metrics['comparison']['vegas_win_pct'] = vegas_wins / len(actual) * 100

    # MAE improvement
    mae_improvement = metrics['vegas']['mae'] - metrics['model']['mae']
    mae_improvement_pct = mae_improvement / metrics['vegas']['mae'] * 100

    metrics['comparison']['mae_improvement'] = mae_improvement
    metrics['comparison']['mae_improvement_pct'] = mae_improvement_pct

    # ATS (against the spread) - model vs vegas line
    model_covers = np.sign(model_pred - vegas_pred) == np.sign(actual - vegas_pred)
    metrics['comparison']['model_ats_accuracy'] = np.mean(model_covers) * 100

    return metrics


def generate_report(
    metrics: Dict,
    n_games: int,
    season_info: str,
    output_path: Optional[Path] = None
) -> str:
    """
    Generate formatted comparison report.

    Args:
        metrics: Comparison metrics dictionary
        n_games: Number of games evaluated
        season_info: Description of seasons evaluated
        output_path: Optional path to save report

    Returns:
        Formatted report string
    """
    report_lines = [
        "=" * 70,
        "MODEL VS VEGAS ODDS COMPARISON",
        "=" * 70,
        "",
        f"Evaluation: {season_info}",
        f"Total Games: {n_games}",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "-" * 70,
        "ACCURACY METRICS",
        "-" * 70,
        "",
        f"{'Metric':<25} {'Trained Model':>15} {'Vegas Lines':>15}",
        "-" * 55,
        f"{'MAE (points)':<25} {metrics['model']['mae']:>15.3f} {metrics['vegas']['mae']:>15.3f}",
        f"{'RMSE (points)':<25} {metrics['model']['rmse']:>15.3f} {metrics['vegas']['rmse']:>15.3f}",
        f"{'R-squared':<25} {metrics['model']['r2']:>15.3f} {metrics['vegas']['r2']:>15.3f}",
        f"{'Direction Accuracy':<25} {metrics['model']['direction_acc']*100:>14.1f}% {metrics['vegas']['direction_acc']*100:>14.1f}%",
        f"{'Correlation':<25} {metrics['model']['correlation']:>15.3f} {metrics['vegas']['correlation']:>15.3f}",
        "",
        "-" * 70,
        "HEAD-TO-HEAD COMPARISON",
        "-" * 70,
        "",
        f"Model closer to actual:  {metrics['comparison']['model_wins']:>5} games ({metrics['comparison']['model_win_pct']:.1f}%)",
        f"Vegas closer to actual:  {metrics['comparison']['vegas_wins']:>5} games ({metrics['comparison']['vegas_win_pct']:.1f}%)",
        f"Ties:                    {metrics['comparison']['ties']:>5} games",
        "",
        f"Model ATS Accuracy:      {metrics['comparison']['model_ats_accuracy']:.1f}%",
        "",
        "-" * 70,
        "IMPROVEMENT SUMMARY",
        "-" * 70,
        "",
    ]

    if metrics['comparison']['mae_improvement'] > 0:
        report_lines.extend([
            f"MAE Improvement: {metrics['comparison']['mae_improvement']:.3f} points",
            f"               ({metrics['comparison']['mae_improvement_pct']:.1f}% better than Vegas)",
            "",
            "CONCLUSION: Trained model OUTPERFORMS Vegas baseline",
        ])
    else:
        report_lines.extend([
            f"MAE Difference: {-metrics['comparison']['mae_improvement']:.3f} points",
            f"              (Vegas is {-metrics['comparison']['mae_improvement_pct']:.1f}% better)",
            "",
            "CONCLUSION: Vegas baseline outperforms trained model",
        ])

    report_lines.append("=" * 70)

    report = "\n".join(report_lines)

    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
        logger.info(f"Report saved to {output_path}")

    return report


def run_comparison(seasons: List[str] = None) -> Tuple[Dict, pd.DataFrame]:
    """
    Run the full comparison using historical Vegas odds data.

    Args:
        seasons: List of season strings (e.g., ['2015-16']). If None, uses all available.

    Returns:
        Tuple of (metrics dictionary, detailed results DataFrame)
    """
    # Load historical Vegas odds
    logger.info("Loading historical Vegas odds from Excel files...")
    vegas_data = load_historical_odds_from_excel(seasons=seasons)

    if vegas_data.empty:
        raise ValueError(
            "No Vegas odds data found. Download Excel files from "
            "https://www.sportsbookreviewsonline.com/scoresoddsarchives/ "
            "and place in data/raw/odds/"
        )

    # Filter to games with valid spreads and margins
    vegas_data = vegas_data.dropna(subset=['vegas_spread', 'actual_margin'])
    logger.info(f"Found {len(vegas_data)} games with valid Vegas spreads")

    # Set up trained model
    model, elo, _ = setup_trained_model()

    # Determine the season year for efficiency stats
    # Extract from season string (e.g., "2015-16" -> 2016)
    season_years = vegas_data['season'].unique()
    logger.info(f"Processing seasons: {list(season_years)}")

    # Load efficiency stats for each season
    all_stats = []
    for season_str in season_years:
        year = int(season_str.split('-')[0]) + 1  # "2015-16" -> 2016
        try:
            df = fetch_barttorvik_year(year)
            # The CSV has rank as index, and 'rank' column contains team names
            # Reset index and use correct column mapping
            df = df.reset_index(drop=True)
            # 'rank' column has team name, 'record' has adjoe value, 'oe Rank' has adjde value
            df_clean = pd.DataFrame({
                'team': df['rank'],  # Team name is in 'rank' column
                'adjoe': df['record'],  # AdjOE is in 'record' column
                'adjde': df['oe Rank'],  # AdjDE is in 'oe Rank' column
                'season': season_str
            })
            all_stats.append(df_clean)
        except Exception as e:
            logger.warning(f"Could not load efficiency stats for {year}: {e}")

    if not all_stats:
        raise ValueError("Could not load efficiency stats for any season")

    team_stats = pd.concat(all_stats, ignore_index=True)
    team_stats.columns = ['team', 'adj_oe', 'adj_de', 'season']
    team_stats['adj_em'] = team_stats['adj_oe'] - team_stats['adj_de']

    # Normalize team names in Vegas data to match our format
    vegas_data['home_team_norm'] = vegas_data['home_team'].apply(normalize_team_name)
    vegas_data['away_team_norm'] = vegas_data['away_team'].apply(normalize_team_name)

    # Generate model predictions
    logger.info("Generating model predictions...")

    # Create a copy for prediction with normalized names
    games_for_pred = vegas_data.copy()
    games_for_pred['home_team'] = games_for_pred['home_team_norm']
    games_for_pred['away_team'] = games_for_pred['away_team_norm']

    # Get team stats for each game's season
    model_preds_list = []
    valid_indices = []

    for season_str in season_years:
        season_games = games_for_pred[games_for_pred['season'] == season_str]
        season_stats = team_stats[team_stats['season'] == season_str]

        if season_games.empty or season_stats.empty:
            continue

        preds, valid_mask = generate_model_predictions(
            season_games, model, elo, season_stats
        )

        for i, (idx, valid) in enumerate(zip(season_games.index, valid_mask)):
            if valid and not np.isnan(preds[i]):
                model_preds_list.append((idx, preds[i]))
                valid_indices.append(idx)

    if not valid_indices:
        raise ValueError("Could not generate predictions for any games")

    # Filter to games where we have both model predictions and Vegas spreads
    valid_games = vegas_data.loc[valid_indices].copy()
    model_preds = np.array([p[1] for p in model_preds_list])
    vegas_preds = valid_games['vegas_spread'].values
    actuals = valid_games['actual_margin'].values

    logger.info(f"Comparing {len(valid_games)} games with both model and Vegas predictions")

    # Calculate metrics
    metrics = calculate_comparison_metrics(actuals, model_preds, vegas_preds)

    # Create detailed results DataFrame
    results_df = valid_games.copy()
    results_df['model_prediction'] = model_preds
    results_df['model_error'] = np.abs(model_preds - actuals)
    results_df['vegas_error'] = np.abs(vegas_preds - actuals)
    results_df['model_closer'] = results_df['model_error'] < results_df['vegas_error']

    return metrics, results_df


def main():
    parser = argparse.ArgumentParser(
        description="Compare trained model accuracy vs Vegas odds baseline"
    )
    parser.add_argument(
        '--season',
        type=str,
        nargs='*',
        help="Season(s) to evaluate (e.g., 2015-16 2016-17). If not specified, uses all available."
    )
    parser.add_argument(
        '--output',
        type=str,
        help="Output file path for detailed results CSV"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("MODEL VS VEGAS ODDS COMPARISON")
    print("=" * 70)

    seasons = args.season if args.season else None
    season_info = f"Seasons: {', '.join(seasons)}" if seasons else "All available seasons"

    print(f"\nEvaluating: {season_info}")
    print("-" * 70)

    try:
        metrics, results_df = run_comparison(seasons=seasons)
    except Exception as e:
        logger.error(f"Comparison failed: {e}")
        raise

    # Generate and print report
    report = generate_report(
        metrics,
        len(results_df),
        season_info
    )
    print("\n" + report)

    # Save detailed results
    output_path = args.output or (config.OUTPUTS_DIR / "model_vs_vegas_comparison.csv")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(output_path, index=False)
    print(f"\nDetailed results saved to: {output_path}")

    # Save report
    report_path = output_path.with_suffix('.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
