"""
Hyperparameter tuning for the ensemble model
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

from elo import EloRatingSystem
from models import ImprovedSpreadModel
from sklearn.model_selection import TimeSeriesSplit
import ssl
import urllib.request
from io import StringIO
from itertools import product
from urllib.error import URLError, HTTPError
import certifi
import time

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def fetch_barttorvik_year(year: int, max_retries: int = 3, retry_delay: float = 1.0) -> pd.DataFrame:
    """
    Fetch team efficiency stats from Barttorvik with proper SSL verification and retry logic

    Args:
        year: Season year to fetch
        max_retries: Maximum number of retry attempts
        retry_delay: Initial delay between retries (doubles each retry)

    Returns:
        DataFrame with team stats
    """
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    url = f"https://barttorvik.com/{year}_team_results.csv"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})

    last_error = None
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, context=ssl_context, timeout=30) as response:
                content = response.read().decode('utf-8')
                return pd.read_csv(StringIO(content))
        except (URLError, HTTPError, ssl.SSLError) as e:
            last_error = e
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)
                print(f"   Attempt {attempt + 1}/{max_retries} failed for {year}: {e}")
                print(f"   Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"   ⚠ All attempts failed, trying without SSL verification...")
                try:
                    ssl_context_unverified = ssl.create_default_context()
                    ssl_context_unverified.check_hostname = False
                    ssl_context_unverified.verify_mode = ssl.CERT_NONE
                    with urllib.request.urlopen(req, context=ssl_context_unverified, timeout=30) as response:
                        content = response.read().decode('utf-8')
                        return pd.read_csv(StringIO(content))
                except Exception as fallback_error:
                    print(f"   ✗ Fallback failed: {fallback_error}")
                    raise

    raise last_error if last_error else RuntimeError(f"Failed to fetch data for {year}")


def load_training_data():
    """Load and prepare training data"""
    print("Loading training data...")

    # 1. Load real historical games
    games = pd.read_csv(RAW_DATA_DIR / 'games' / 'historical_games_2019_2025.csv', parse_dates=['date'])
    print(f"   ✓ Loaded {len(games)} games")

    # 2. Initialize Elo and process games
    elo = EloRatingSystem(k_factor=38, hca=4.0, carryover=0.64)

    conferences = {
        'ACC': ['Duke', 'North Carolina', 'NC State', 'Virginia', 'Virginia Tech',
               'Clemson', 'Florida State', 'Miami', 'Pitt', 'Syracuse', 'Louisville',
               'Wake Forest', 'Georgia Tech', 'Boston College', 'Notre Dame',
               'California', 'Stanford', 'SMU'],
        'SEC': ['Kentucky', 'Tennessee', 'Alabama', 'Auburn', 'Florida', 'Texas A&M'],
        'Big Ten': ['Purdue', 'Michigan', 'Michigan State', 'Ohio State', 'Illinois'],
        'Big 12': ['Houston', 'Kansas', 'Baylor', 'Iowa State', 'BYU'],
        'Big East': ['UConn', 'Creighton', 'Marquette', 'Villanova', 'Xavier'],
    }
    elo.load_conference_mappings(conferences)

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
    print(f"   ✓ Processed {len(elo_snapshots)} games")

    # 3. Load efficiency stats
    all_stats = []
    for year in [2020, 2021, 2022, 2023, 2024, 2025]:
        df = fetch_barttorvik_year(year)
        df['season'] = year
        all_stats.append(df[['team', 'adjoe', 'adjde', 'season']])

    team_stats = pd.concat(all_stats, ignore_index=True)
    team_stats.columns = ['team', 'adj_oe', 'adj_de', 'season']
    team_stats['adj_em'] = team_stats['adj_oe'] - team_stats['adj_de']
    print(f"   ✓ Loaded {len(team_stats)} team-season efficiency records")

    # 4. Merge to create training data
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

    print(f"   ✓ Created {len(train_data)} training samples\n")

    feature_cols = [
        'home_adj_oe', 'home_adj_de', 'home_adj_em',
        'away_adj_oe', 'away_adj_de', 'away_adj_em',
        'eff_diff',
        'home_elo_before', 'away_elo_before', 'elo_diff', 'predicted_spread'
    ]

    X = train_data[feature_cols]
    y = train_data['actual_margin']

    return X, y


def evaluate_params(X, y, lgbm_params, weights, n_splits=5):
    """Evaluate a parameter configuration using cross-validation"""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_scores = []

    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = ImprovedSpreadModel(
            ridge_alpha=1.0,
            lgbm_params=lgbm_params,
            weights=weights,
            use_lgbm=True
        )
        model.fit(X_train, y_train)

        preds = model.predict(X_val)
        mae = np.abs(preds - y_val).mean()
        cv_scores.append(mae)

    return np.mean(cv_scores), np.std(cv_scores)


def tune_hyperparameters():
    """Perform grid search for hyperparameter tuning"""
    print("="*60)
    print("HYPERPARAMETER TUNING")
    print("="*60)
    print()

    # Load data
    X, y = load_training_data()

    # Define parameter grid
    print("Defining parameter grid...")
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.05, 0.1, 0.15],
    }

    weight_options = [
        (0.3, 0.7),
        (0.4, 0.6),
        (0.5, 0.5),
    ]

    print(f"   LightGBM parameters: {len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['learning_rate'])} combinations")
    print(f"   Ensemble weights: {len(weight_options)} options")
    print(f"   Total: {len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['learning_rate']) * len(weight_options)} configurations\n")

    # Grid search
    print("Starting grid search...")
    print("-"*60)

    results = []
    best_mae = float('inf')
    best_config = None

    total_configs = len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['learning_rate']) * len(weight_options)
    config_num = 0

    for n_est in param_grid['n_estimators']:
        for depth in param_grid['max_depth']:
            for lr in param_grid['learning_rate']:
                for weights in weight_options:
                    config_num += 1

                    lgbm_params = {
                        'n_estimators': n_est,
                        'max_depth': depth,
                        'learning_rate': lr
                    }

                    print(f"Config {config_num}/{total_configs}: n_est={n_est}, depth={depth}, lr={lr}, weights={weights}", end=" ... ")

                    mean_mae, std_mae = evaluate_params(X, y, lgbm_params, weights)

                    print(f"MAE: {mean_mae:.3f} ± {std_mae:.3f}")

                    results.append({
                        'n_estimators': n_est,
                        'max_depth': depth,
                        'learning_rate': lr,
                        'ridge_weight': weights[0],
                        'lgbm_weight': weights[1],
                        'mean_mae': mean_mae,
                        'std_mae': std_mae
                    })

                    if mean_mae < best_mae:
                        best_mae = mean_mae
                        best_config = {
                            'lgbm_params': lgbm_params,
                            'weights': weights,
                            'mean_mae': mean_mae,
                            'std_mae': std_mae
                        }

    print("-"*60)
    print()

    # Display results
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('mean_mae')

    print("="*60)
    print("TUNING RESULTS")
    print("="*60)
    print()

    print("Top 10 Configurations:")
    print(results_df.head(10).to_string(index=False))
    print()

    print("Best Configuration:")
    print(f"   LightGBM params: {best_config['lgbm_params']}")
    print(f"   Ensemble weights: {best_config['weights']}")
    print(f"   CV MAE: {best_config['mean_mae']:.3f} ± {best_config['std_mae']:.3f}")
    print()

    # Compare to baseline
    baseline_mae = 5.459  # From current model
    improvement = baseline_mae - best_config['mean_mae']
    print(f"Baseline MAE (current): {baseline_mae:.3f}")
    print(f"Best MAE (tuned): {best_config['mean_mae']:.3f}")
    print(f"Improvement: {improvement:+.3f} points ({abs(improvement)/baseline_mae*100:.1f}%)")
    print()

    # Save results
    # Ensure outputs directory exists
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUTS_DIR / 'hyperparameter_tuning_results.csv', index=False)
    print(f"Full results saved to: outputs/hyperparameter_tuning_results.csv")
    print()

    print("="*60)

    return best_config, results_df


if __name__ == "__main__":
    best_config, results_df = tune_hyperparameters()
