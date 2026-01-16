"""
Error analysis: breakdown by team strength, temporal trends, and game characteristics
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
from utils import fetch_barttorvik_year
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import config


def load_and_predict():
    """Load data, train model, and generate predictions for error analysis"""
    print("Loading data and training model...")

    # 1. Load real historical games
    games = pd.read_csv(RAW_DATA_DIR / 'games' / 'historical_games_2019_2025.csv', parse_dates=['date'])

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

    # 3. Load efficiency stats
    all_stats = []
    for year in [2020, 2021, 2022, 2023, 2024, 2025]:
        df = fetch_barttorvik_year(year)
        df['season'] = year
        all_stats.append(df[['team', 'adjoe', 'adjde', 'season']])

    team_stats = pd.concat(all_stats, ignore_index=True)
    team_stats.columns = ['team', 'adj_oe', 'adj_de', 'season']
    team_stats['adj_em'] = team_stats['adj_oe'] - team_stats['adj_de']

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

    # 5. Train model and predict on full dataset
    feature_cols = [
        'home_adj_oe', 'home_adj_de', 'home_adj_em',
        'away_adj_oe', 'away_adj_de', 'away_adj_em',
        'eff_diff',
        'home_elo_before', 'away_elo_before', 'elo_diff', 'predicted_spread'
    ]

    X = train_data[feature_cols]
    y = train_data['actual_margin']

    model = ImprovedSpreadModel(
        lgbm_params={'n_estimators': 100, 'max_depth': 8, 'learning_rate': 0.1},
        weights=(0.3, 0.7)
    )
    model.fit(X, y)

    # Generate predictions for all data
    train_data['model_prediction'] = model.predict(X)
    train_data['error'] = train_data['model_prediction'] - train_data['actual_margin']
    train_data['abs_error'] = np.abs(train_data['error'])

    print(f"   ✓ Loaded and predicted on {len(train_data)} games\n")

    return train_data


def analyze_errors(train_data):
    """Perform comprehensive error analysis"""
    print("="*60)
    print("ERROR ANALYSIS")
    print("="*60)
    print()

    # Overall statistics
    print("Overall Performance:")
    print(f"   MAE: {train_data['abs_error'].mean():.3f}")
    print(f"   RMSE: {np.sqrt((train_data['error']**2).mean()):.3f}")
    print(f"   Mean Error (bias): {train_data['error'].mean():.3f}")
    print(f"   Median Abs Error: {train_data['abs_error'].median():.3f}")
    print()

    # 1. Error by team strength (Elo rating)
    print("-"*60)
    print("ANALYSIS 1: Error by Team Strength (Elo Rating)")
    print("-"*60)

    # Categorize games by favorite strength
    train_data['favorite_elo'] = train_data[['home_elo_before', 'away_elo_before']].max(axis=1)
    train_data['strength_category'] = pd.cut(
        train_data['favorite_elo'],
        bins=[0, 1450, 1500, 1550, 2000],
        labels=['Weak (<1450)', 'Average (1450-1500)', 'Strong (1500-1550)', 'Elite (>1550)']
    )

    strength_analysis = train_data.groupby('strength_category', observed=True).agg({
        'abs_error': ['mean', 'std', 'count']
    }).round(3)

    print(strength_analysis)
    print()

    # 2. Error by predicted spread size
    print("-"*60)
    print("ANALYSIS 2: Error by Predicted Spread Size")
    print("-"*60)

    train_data['spread_category'] = pd.cut(
        np.abs(train_data['model_prediction']),
        bins=[0, 5, 10, 15, 100],
        labels=['Close (<5)', 'Moderate (5-10)', 'Large (10-15)', 'Blowout (>15)']
    )

    spread_analysis = train_data.groupby('spread_category', observed=True).agg({
        'abs_error': ['mean', 'std', 'count']
    }).round(3)

    print(spread_analysis)
    print()

    # 3. Error by season
    print("-"*60)
    print("ANALYSIS 3: Error by Season")
    print("-"*60)

    season_analysis = train_data.groupby('season').agg({
        'abs_error': ['mean', 'std', 'count']
    }).round(3)

    print(season_analysis)
    print()

    # 4. Error by month (temporal trends)
    print("-"*60)
    print("ANALYSIS 4: Error by Month (Temporal Trends)")
    print("-"*60)

    train_data['month'] = train_data['date'].dt.month
    train_data['month_name'] = train_data['date'].dt.strftime('%B')

    month_analysis = train_data.groupby('month_name').agg({
        'abs_error': ['mean', 'std', 'count']
    }).round(3)

    # Sort by month number for logical ordering
    month_order = ['November', 'December', 'January', 'February', 'March']
    month_analysis = month_analysis.reindex([m for m in month_order if m in month_analysis.index])

    print(month_analysis)
    print()

    # 5. Error by efficiency differential
    print("-"*60)
    print("ANALYSIS 5: Error by Efficiency Differential")
    print("-"*60)

    train_data['eff_diff_category'] = pd.cut(
        np.abs(train_data['eff_diff']),
        bins=[0, 5, 10, 15, 100],
        labels=['Matched (<5)', 'Slight Edge (5-10)', 'Clear Edge (10-15)', 'Mismatch (>15)']
    )

    eff_analysis = train_data.groupby('eff_diff_category', observed=True).agg({
        'abs_error': ['mean', 'std', 'count']
    }).round(3)

    print(eff_analysis)
    print()

    # 6. Direction accuracy (did we pick the right winner?)
    print("-"*60)
    print("ANALYSIS 6: Direction Accuracy (Winner Prediction)")
    print("-"*60)

    train_data['predicted_winner'] = np.where(train_data['model_prediction'] > 0, 'home', 'away')
    train_data['actual_winner'] = np.where(train_data['actual_margin'] > 0, 'home', 'away')
    train_data['correct_winner'] = train_data['predicted_winner'] == train_data['actual_winner']

    direction_accuracy = train_data['correct_winner'].mean()
    print(f"   Overall Direction Accuracy: {direction_accuracy*100:.1f}%")

    # By confidence level
    train_data['confidence_level'] = pd.cut(
        np.abs(train_data['model_prediction']),
        bins=[0, 5, 10, 100],
        labels=['Low (<5)', 'Medium (5-10)', 'High (>10)']
    )

    confidence_analysis = train_data.groupby('confidence_level', observed=True).agg({
        'correct_winner': ['mean', 'count']
    })

    print(f"\n   Direction Accuracy by Confidence Level:")
    for level in confidence_analysis.index:
        acc = confidence_analysis.loc[level, ('correct_winner', 'mean')]
        count = int(confidence_analysis.loc[level, ('correct_winner', 'count')])
        print(f"      {level}: {acc*100:.1f}% ({count} games)")
    print()

    # 7. Large errors analysis
    print("-"*60)
    print("ANALYSIS 7: Large Errors (>15 points)")
    print("-"*60)

    large_errors = train_data[train_data['abs_error'] > 15].sort_values('abs_error', ascending=False)
    print(f"   Number of large errors: {len(large_errors)} ({len(large_errors)/len(train_data)*100:.1f}%)")

    if len(large_errors) > 0:
        print(f"\n   Top 10 Largest Errors:")
        for idx, row in large_errors.head(10).iterrows():
            print(f"      {row['date'].strftime('%Y-%m-%d')}: {row['away_team']} @ {row['home_team']}")
            print(f"         Predicted: {row['model_prediction']:+.1f}, Actual: {row['actual_margin']:+.1f}, Error: {row['abs_error']:.1f}")

    print()

    # 8. Calibration analysis
    print("-"*60)
    print("ANALYSIS 8: Calibration (Are We Over/Under Predicting?)")
    print("-"*60)

    # Bin predictions and check if actual results match
    train_data['pred_bin'] = pd.cut(
        train_data['model_prediction'],
        bins=[-100, -10, -5, 0, 5, 10, 100],
        labels=['Away Blowout (<-10)', 'Away Moderate (-10 to -5)', 'Away Close (-5 to 0)',
                'Home Close (0 to 5)', 'Home Moderate (5 to 10)', 'Home Blowout (>10)']
    )

    calibration = train_data.groupby('pred_bin', observed=True).agg({
        'model_prediction': 'mean',
        'actual_margin': 'mean',
        'abs_error': 'mean'
    }).round(3)

    print(calibration)
    print()

    print("="*60)

    # Save detailed error data
    error_summary = pd.DataFrame({
        'Category': ['Overall', 'Close Games (<5)', 'Moderate (5-10)', 'Large (10-15)', 'Blowout (>15)'],
        'MAE': [
            train_data['abs_error'].mean(),
            train_data[np.abs(train_data['model_prediction']) < 5]['abs_error'].mean(),
            train_data[(np.abs(train_data['model_prediction']) >= 5) & (np.abs(train_data['model_prediction']) < 10)]['abs_error'].mean(),
            train_data[(np.abs(train_data['model_prediction']) >= 10) & (np.abs(train_data['model_prediction']) < 15)]['abs_error'].mean(),
            train_data[np.abs(train_data['model_prediction']) >= 15]['abs_error'].mean(),
        ]
    })

    # Ensure outputs directory exists
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    error_summary.to_csv(OUTPUTS_DIR / 'error_analysis_summary.csv', index=False)
    print(f"\nError summary saved to: outputs/error_analysis_summary.csv")

    return train_data


if __name__ == "__main__":
    train_data = load_and_predict()
    analyze_errors(train_data)
