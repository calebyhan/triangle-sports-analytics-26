"""
Train model on real historical game data
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from elo import EloRatingSystem
from models import ImprovedSpreadModel
from sklearn.model_selection import TimeSeriesSplit
import ssl
import urllib.request
from io import StringIO


def fetch_barttorvik_year(year):
    """Fetch team efficiency stats from Barttorvik"""
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    url = f"https://barttorvik.com/{year}_team_results.csv"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})

    with urllib.request.urlopen(req, context=ssl_context, timeout=30) as response:
        content = response.read().decode('utf-8')

    return pd.read_csv(StringIO(content))


def main():
    print("="*60)
    print("TRAINING ON REAL HISTORICAL GAME DATA")
    print("="*60)

    # 1. Load real historical games
    print("\n1. Loading real historical games...")
    games = pd.read_csv('data/raw/games/historical_games_2019_2025.csv', parse_dates=['date'])
    print(f"   ✓ Loaded {len(games)} games from {games['date'].min()} to {games['date'].max()}")

    # 2. Initialize Elo and process games chronologically
    print("\n2. Processing games chronologically through Elo system...")
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

    print(f"   ✓ Processed {len(elo_snapshots)} games, tracking {len(elo.ratings)} teams")

    # 3. Load efficiency stats
    print("\n3. Loading team efficiency stats from Barttorvik...")
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
    print("\n4. Creating training dataset...")
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

    print(f"   ✓ Created {len(train_data)} training samples")

    # 5. Train model
    print("\n5. Training model on real data...")
    feature_cols = [
        'home_adj_oe', 'home_adj_de', 'home_adj_em',
        'away_adj_oe', 'away_adj_de', 'away_adj_em',
        'eff_diff',
        'home_elo_before', 'away_elo_before', 'elo_diff', 'predicted_spread'
    ]

    X = train_data[feature_cols]
    y = train_data['actual_margin']

    model = ImprovedSpreadModel(
        ridge_alpha=1.0,
        lgbm_params={'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1},
        weights=(0.4, 0.6),
        use_lgbm=True
    )

    model.fit(X, y)
    print("   ✓ Model trained!")

    # 6. Cross-validation
    print("\n6. Running 5-fold time-series cross-validation...")
    tscv = TimeSeriesSplit(n_splits=5)
    cv_results = {'ridge': [], 'ensemble': []}

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        fold_model = ImprovedSpreadModel(weights=(0.4, 0.6))
        fold_model.fit(X_train, y_train)

        preds = fold_model.predict(X_val)
        components = fold_model.predict_components(X_val)

        ridge_mae = np.abs(components['ridge'] - y_val).mean()
        ensemble_mae = np.abs(preds - y_val).mean()

        cv_results['ridge'].append(ridge_mae)
        cv_results['ensemble'].append(ensemble_mae)

        print(f"   Fold {fold+1}: Ridge={ridge_mae:.3f}, Ensemble={ensemble_mae:.3f}")

    ridge_mean = np.mean(cv_results['ridge'])
    ridge_std = np.std(cv_results['ridge'])
    ensemble_mean = np.mean(cv_results['ensemble'])
    ensemble_std = np.std(cv_results['ensemble'])

    print(f"\n   Ridge CV MAE:    {ridge_mean:.3f} ± {ridge_std:.3f}")
    print(f"   Ensemble CV MAE: {ensemble_mean:.3f} ± {ensemble_std:.3f}")

    # 7. Generate 2026 predictions
    print("\n7. Generating 2026 predictions...")
    team_stats_2026 = pd.read_csv('data/processed/team_stats_2025_26.csv')
    template = pd.read_csv('tsa_pt_spread_template_2026 - Sheet1.csv')
    template = template.dropna(subset=['Home', 'Away'])

    team_dict = team_stats_2026.set_index('team').to_dict('index')
    pred_features = []
    valid_indices = []

    for idx, row in template.iterrows():
        home, away = row['Home'], row['Away']
        if home not in team_dict or away not in team_dict:
            continue

        home_stats = team_dict[home]
        away_stats = team_dict[away]

        home_oe = home_stats.get('off_efficiency', 100)
        home_de = home_stats.get('def_efficiency', 100)
        away_oe = away_stats.get('off_efficiency', 100)
        away_de = away_stats.get('def_efficiency', 100)

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

        pred_features.append(features)
        valid_indices.append(idx)

    X_pred = pd.DataFrame(pred_features)
    predictions = model.predict(X_pred)

    results = template.copy()
    for i, idx in enumerate(valid_indices):
        results.loc[idx, 'pt_spread'] = predictions[i]

    # 8. Save predictions
    print("\n8. Saving predictions...")
    submission = results[['Date', 'Away', 'Home', 'pt_spread']].copy()
    submission = submission.dropna(subset=['pt_spread'])

    submission['team_name'] = ''
    submission['team_member'] = ''
    submission['team_email'] = ''

    submission.loc[submission.index[0], 'team_name'] = 'CMMT'
    submission.loc[submission.index[0], 'team_member'] = 'Caleb Han'
    submission.loc[submission.index[0], 'team_email'] = 'calebhan@unc.edu'
    submission.loc[submission.index[1], 'team_member'] = 'Mason Mines'
    submission.loc[submission.index[1], 'team_email'] = 'mmines@unc.edu'
    submission.loc[submission.index[2], 'team_member'] = 'Mason Wang'
    submission.loc[submission.index[2], 'team_email'] = 'masonw@unc.edu'
    submission.loc[submission.index[3], 'team_member'] = 'Tony Wang'
    submission.loc[submission.index[3], 'team_email'] = 'tonyw@unc.edu'

    output_path = 'data/predictions/tsa_pt_spread_CMMT_2026_real_data.csv'
    submission.to_csv(output_path, index=False)
    print(f"   ✓ Saved: {output_path}")

    main_path = 'data/predictions/tsa_pt_spread_CMMT_2026.csv'
    submission.to_csv(main_path, index=False)
    print(f"   ✓ Updated: {main_path}")

    # Summary
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Training samples: {len(train_data)} real games")
    print(f"Predictions: {len(submission)} games")
    print(f"\nModel Performance (Cross-Validation):")
    print(f"  Ridge MAE:    {ridge_mean:.3f} ± {ridge_std:.3f}")
    print(f"  Ensemble MAE: {ensemble_mean:.3f} ± {ensemble_std:.3f}")

    baseline_mae = 11.41  # Naive baseline (predict 0)
    improvement = baseline_mae - ridge_mean
    print(f"\nImprovement from baseline ({baseline_mae} MAE):")
    print(f"  {improvement:+.3f} points ({abs(improvement)/baseline_mae*100:.1f}% better)")

    print("="*60)


if __name__ == "__main__":
    main()
