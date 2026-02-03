"""
Generate 2026 predictions using enhanced features.

This script should only be run if the A/B comparison shows that
enhanced features improve predictions over baseline.

Uses all 24 features (11 baseline + 13 enhanced) to generate
final predictions for the 2026 tournament.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from src.elo import EloRatingSystem
from src.models import ImprovedSpreadModel
from src.utils import fetch_barttorvik_year
from src.features.enhanced_pipeline import EnhancedFeatureEngine
from src import config


def main():
    """Generate 2026 predictions using enhanced model."""
    print("="*60)
    print("GENERATING 2026 PREDICTIONS (ENHANCED MODEL)")
    print("="*60)
    print("⚠ Only use if A/B test shows enhanced > baseline!")
    print("="*60)

    # ========================================================================
    # STEP 1: Train enhanced model on all historical data
    # ========================================================================
    print("\n1. Training enhanced model on historical data (2020-2025)...")

    # Load historical games
    games = pd.read_csv(config.HISTORICAL_GAMES_FILE, parse_dates=['date'])
    print(f"   ✓ Loaded {len(games)} historical games")

    # Process Elo
    elo = EloRatingSystem(
        k_factor=config.ELO_CONFIG['k_factor'],
        hca=config.ELO_CONFIG['home_court_advantage'],
        carryover=config.ELO_CONFIG['season_carryover']
    )
    elo.load_conference_mappings(config.CONFERENCE_MAPPINGS)
    elo_snapshots = elo.process_games(
        games, date_col='date', home_col='home_team', away_col='away_team',
        home_score_col='home_score', away_score_col='away_score',
        neutral_col='neutral_site', season_col='season', save_snapshots=True
    )
    print(f"   ✓ Processed Elo ratings through {games['date'].max()}")

    # Load efficiency stats
    all_stats = []
    for year in config.TRAINING_YEARS:
        df = fetch_barttorvik_year(year)
        df['season'] = year
        all_stats.append(df[['team', 'adjoe', 'adjde', 'season']])

    team_stats = pd.concat(all_stats, ignore_index=True)
    team_stats.columns = ['team', 'adj_oe', 'adj_de', 'season']
    team_stats['adj_em'] = team_stats['adj_oe'] - team_stats['adj_de']

    # Merge training data
    elo_snapshots['season'] = elo_snapshots['date'].dt.year
    train_data = elo_snapshots.merge(
        team_stats, left_on=['home_team', 'season'], right_on=['team', 'season'], how='left'
    ).rename(columns={'adj_oe': 'home_adj_oe', 'adj_de': 'home_adj_de', 'adj_em': 'home_adj_em'})
    train_data = train_data.drop(columns=['team'], errors='ignore').merge(
        team_stats, left_on=['away_team', 'season'], right_on=['team', 'season'], how='left'
    ).rename(columns={'adj_oe': 'away_adj_oe', 'adj_de': 'away_adj_de', 'adj_em': 'away_adj_em'})
    train_data = train_data.drop(columns=['team'], errors='ignore')
    train_data['eff_diff'] = train_data['home_adj_em'] - train_data['away_adj_em']
    train_data['elo_diff'] = train_data['home_elo_before'] - train_data['away_elo_before']
    train_data = train_data.dropna(subset=['home_adj_oe', 'away_adj_oe'])

    # Load or compute enhanced features
    enhanced_features_path = config.PROCESSED_DATA_DIR / 'enhanced_features_2020_2024.csv'
    if enhanced_features_path.exists():
        enhanced_features = pd.read_csv(enhanced_features_path)
        print(f"   ✓ Loaded pre-computed enhanced features")
    else:
        raise FileNotFoundError(
            f"Enhanced features not found: {enhanced_features_path}\n"
            f"Run train_model_enhanced.py first to compute features"
        )

    # Merge
    train_data = pd.concat([train_data.reset_index(drop=True), enhanced_features.reset_index(drop=True)], axis=1)

    # Train final model
    feature_cols = config.BASELINE_FEATURES + config.ENHANCED_FEATURES
    available_features = [f for f in feature_cols if f in train_data.columns]
    X = train_data[available_features]
    y = train_data['actual_margin']

    print(f"   Training with {len(available_features)} features on {len(X)} games...")

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
    print("   ✓ Model trained!")

    # ========================================================================
    # STEP 2: Prepare enhanced features for 2026 teams
    # ========================================================================
    print("\n2. Computing enhanced features for 2026 teams...")

    # Load 2026 team stats
    team_stats_2026 = pd.read_csv(config.PROCESSED_DATA_DIR / 'team_stats_2025_26.csv')
    print(f"   ✓ Loaded stats for {len(team_stats_2026)} teams")

    # Initialize feature engine with all historical data through 2025
    player_data_path = config.HISTORICAL_PLAYER_DATA
    if player_data_path.exists():
        player_data = pd.read_csv(player_data_path)
        player_data['game_date'] = pd.to_datetime(player_data['game_date'])
        print(f"   ✓ Loaded player data")
        use_player_features = True
    else:
        player_data = None
        print("   ⚠ Player data not found - skipping player features")
        use_player_features = False

    feature_engine = EnhancedFeatureEngine(
        games_df=games,
        player_data_df=player_data,
        use_player_features=use_player_features
    )

    # Cutoff date: end of 2025 season
    as_of_date = pd.Timestamp('2025-12-31')
    print(f"   Using data through {as_of_date}")

    # ========================================================================
    # STEP 3: Generate predictions for 2026 tournament
    # ========================================================================
    print("\n3. Generating predictions for 2026 tournament games...")

    # Load template
    template = pd.read_csv(config.DATA_DIR.parent / config.SUBMISSION_TEMPLATE)
    template = template.dropna(subset=['Home', 'Away'])
    print(f"   ✓ Loaded {len(template)} games to predict")

    # Create team dictionary
    team_dict = team_stats_2026.set_index('team').to_dict('index')

    predictions = []
    valid_indices = []

    for idx, row in template.iterrows():
        home, away = row['Home'], row['Away']

        if home not in team_dict or away not in team_dict:
            continue

        # Baseline features
        home_stats = team_dict[home]
        away_stats = team_dict[away]

        home_oe = home_stats.get('off_efficiency', 100)
        home_de = home_stats.get('def_efficiency', 100)
        away_oe = away_stats.get('off_efficiency', 100)
        away_de = away_stats.get('def_efficiency', 100)

        baseline_features = {
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

        # Enhanced features (using all historical data through 2025)
        game_row = pd.Series({
            'home_team': home,
            'away_team': away,
            'date': as_of_date
        })

        try:
            enhanced_features = feature_engine._compute_game_features(game_row, as_of_date)
        except Exception as e:
            print(f"   ⚠ Error computing features for {home} vs {away}: {e}")
            enhanced_features = feature_engine._get_default_features()

        # Combine all features
        all_features = {**baseline_features, **enhanced_features}

        # Predict
        feature_values = [all_features.get(f, 0.0) for f in available_features]
        pred = model.predict([feature_values])[0]

        predictions.append(pred)
        valid_indices.append(idx)

    print(f"   ✓ Generated {len(predictions)} predictions")

    # ========================================================================
    # STEP 4: Create submission file
    # ========================================================================
    print("\n4. Creating submission file...")

    results = template.copy()
    for i, idx in enumerate(valid_indices):
        results.loc[idx, 'pt_spread'] = predictions[i]

    submission = results[['Date', 'Away', 'Home', 'pt_spread']].copy()
    submission = submission.dropna(subset=['pt_spread'])

    submission['team_name'] = ''
    submission['team_member'] = ''
    submission['team_email'] = ''

    # Use team info from config
    team_members = config.TEAM_INFO['members']
    submission.loc[submission.index[0], 'team_name'] = config.TEAM_INFO['team_name']
    for i, member in enumerate(team_members):
        if i < len(submission):
            submission.loc[submission.index[i], 'team_member'] = member['name']
            submission.loc[submission.index[i], 'team_email'] = member['email']

    # Save
    output_path = config.PREDICTIONS_DIR / 'tsa_pt_spread_CMMT_2026_enhanced.csv'
    submission.to_csv(output_path, index=False)

    print(f"   ✓ Saved: {output_path}")

    # ========================================================================
    # STEP 5: Summary
    # ========================================================================
    print("\n" + "="*60)
    print("✓ ENHANCED PREDICTIONS COMPLETE")
    print("="*60)
    print(f"Predictions: {len(predictions)} games")
    print(f"Features used: {len(available_features)}")
    print(f"  - Baseline: {len(config.BASELINE_FEATURES)}")
    print(f"  - Enhanced: {len([f for f in available_features if f in config.ENHANCED_FEATURES])}")
    print(f"\nOutput: {output_path}")
    print("="*60)


if __name__ == '__main__':
    main()
