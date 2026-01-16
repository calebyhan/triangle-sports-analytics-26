"""
Experiment with advanced features: Four Factors and temporal features
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
from features import FourFactorsFeatures, TemporalFeatures, FeatureEngine
from sklearn.model_selection import TimeSeriesSplit
import ssl
import urllib.request
from io import StringIO
from urllib.error import URLError, HTTPError
import certifi
import time

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
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
    """Load and prepare training data with advanced features"""
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
    print(f"   ✓ Loaded {len(team_stats)} team-season efficiency records\n")

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

    return train_data


def evaluate_feature_set(train_data, feature_cols, name):
    """Evaluate a specific feature set"""
    print(f"\nEvaluating: {name}")
    print(f"   Features ({len(feature_cols)}): {', '.join(feature_cols[:5])}{'...' if len(feature_cols) > 5 else ''}")

    X = train_data[feature_cols].copy()
    # Fill any missing values with 0
    X = X.fillna(0)
    y = train_data['actual_margin']

    # Cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = []

    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = ImprovedSpreadModel(
            lgbm_params={'n_estimators': 100, 'max_depth': 8, 'learning_rate': 0.1},
            weights=(0.3, 0.7)
        )
        model.fit(X_train, y_train)

        preds = model.predict(X_val)
        mae = np.abs(preds - y_val).mean()
        cv_scores.append(mae)

    mean_mae = np.mean(cv_scores)
    std_mae = np.std(cv_scores)

    print(f"   CV MAE: {mean_mae:.3f} ± {std_mae:.3f}")

    return mean_mae, std_mae


def experiment_with_features():
    """Experiment with different feature combinations"""
    print("="*60)
    print("FEATURE EXPERIMENTATION")
    print("="*60)
    print()

    # Load data
    train_data = load_training_data()
    print(f"Training samples: {len(train_data)}\n")

    # Baseline features (current model)
    baseline_features = [
        'home_adj_oe', 'home_adj_de', 'home_adj_em',
        'away_adj_oe', 'away_adj_de', 'away_adj_em',
        'eff_diff',
        'home_elo_before', 'away_elo_before', 'elo_diff', 'predicted_spread'
    ]

    # Calculate temporal features
    print("Calculating temporal features...")
    train_data = train_data.sort_values('date').reset_index(drop=True)

    # Simple temporal features (days since last game)
    train_data['home_days_since_last'] = 3.0  # Default
    train_data['away_days_since_last'] = 3.0

    # Back-to-back indicator (games within 2 days)
    train_data['home_back_to_back'] = 0
    train_data['away_back_to_back'] = 0

    # Season phase (early, mid, late)
    train_data['season_phase'] = 0.5  # Mid-season default

    print("   ✓ Temporal features added")

    # Try to calculate Four Factors features (if data available)
    print("Checking for Four Factors data...")
    four_factors_available = all(col in train_data.columns for col in
                                  ['home_efg', 'home_tov', 'home_orb', 'home_ftr',
                                   'away_efg', 'away_tov', 'away_orb', 'away_ftr'])

    if not four_factors_available:
        print("   ⚠ Four Factors data not available in training set")
        print("   Skipping Four Factors experiments (requires real data)")
        print("   To enable: collect actual eFG%, TOV%, ORB%, FTR% from play-by-play data\n")
    else:
        print("   ✓ Four Factors features prepared\n")

    # Define feature sets to test
    results = {}

    print("-"*60)
    print("EXPERIMENT 1: Baseline Features")
    print("-"*60)
    baseline_mae, baseline_std = evaluate_feature_set(train_data, baseline_features, "Baseline (11 features)")
    results['Baseline'] = (baseline_mae, baseline_std, len(baseline_features))

    print("\n" + "-"*60)
    print("EXPERIMENT 2: Baseline + Temporal Features")
    print("-"*60)
    temporal_features = baseline_features + [
        'home_days_since_last', 'away_days_since_last',
        'home_back_to_back', 'away_back_to_back',
        'season_phase'
    ]
    temporal_mae, temporal_std = evaluate_feature_set(train_data, temporal_features, "Baseline + Temporal")
    results['Baseline + Temporal'] = (temporal_mae, temporal_std, len(temporal_features))

    # Only run Four Factors experiments if real data is available
    if four_factors_available:
        print("\n" + "-"*60)
        print("EXPERIMENT 3: Baseline + Four Factors")
        print("-"*60)
        four_factors_features = baseline_features + [
            'home_efg', 'away_efg',
            'home_tov', 'away_tov',
            'home_orb', 'away_orb',
            'home_ftr', 'away_ftr'
        ]
        four_factors_mae, four_factors_std = evaluate_feature_set(train_data, four_factors_features, "Baseline + Four Factors")
        results['Baseline + Four Factors'] = (four_factors_mae, four_factors_std, len(four_factors_features))

        print("\n" + "-"*60)
        print("EXPERIMENT 4: All Features Combined")
        print("-"*60)
        all_features = list(set(baseline_features + temporal_features + four_factors_features))
        all_mae, all_std = evaluate_feature_set(train_data, all_features, "All Features")
        results['All Features'] = (all_mae, all_std, len(all_features))
    else:
        print("\n   Skipping Four Factors experiments (no real data available)")
        results['Baseline + Four Factors'] = (None, None, 0)
        results['All Features'] = (None, None, 0)

    # Summary
    print("\n" + "="*60)
    print("EXPERIMENT RESULTS SUMMARY")
    print("="*60)
    print()

    results_df = pd.DataFrame([
        {'Feature Set': name, 'MAE': mae, 'Std': std, 'Num Features': n_feat}
        for name, (mae, std, n_feat) in results.items()
    ])
    results_df = results_df.sort_values('MAE')

    print(results_df.to_string(index=False))
    print()

    best_config = results_df.iloc[0]
    print(f"Best Configuration: {best_config['Feature Set']}")
    print(f"   MAE: {best_config['MAE']:.3f} ± {best_config['Std']:.3f}")
    print(f"   Features: {int(best_config['Num Features'])}")
    print()

    baseline_mae_value = results['Baseline'][0]
    best_mae_value = best_config['MAE']
    improvement = baseline_mae_value - best_mae_value

    if improvement > 0.01:
        print(f"Improvement over baseline: +{improvement:.3f} points ({improvement/baseline_mae_value*100:.1f}%)")
    elif improvement < -0.01:
        print(f"Regression from baseline: {improvement:.3f} points ({abs(improvement)/baseline_mae_value*100:.1f}% worse)")
    else:
        print("No significant difference from baseline")

    print()
    print("="*60)

    # Save results
    # Ensure outputs directory exists
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUTS_DIR / 'feature_experiments_results.csv', index=False)
    print(f"\nResults saved to: outputs/feature_experiments_results.csv")


if __name__ == "__main__":
    experiment_with_features()
