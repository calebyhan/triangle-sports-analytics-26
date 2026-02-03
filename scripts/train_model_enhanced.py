"""
Train model with enhanced features on real historical game data.

This script extends the baseline training pipeline by integrating:
- Momentum features (win streak, recent form)
- Blowout tendency (large margin patterns)
- Player features (star power, bench depth, balance, efficiency)
- Team-specific home court advantage

Maintains A/B testing capability against baseline.
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
from sklearn.model_selection import TimeSeriesSplit
from src import config


def main():
    print("="*60)
    print("TRAINING WITH ENHANCED FEATURES")
    print("="*60)

    # ========================================================================
    # STEP 1: Load real historical games
    # ========================================================================
    print("\n1. Loading real historical games...")
    games_path = config.HISTORICAL_GAMES_FILE

    if not games_path.exists():
        raise FileNotFoundError(
            f"Historical games file not found: {games_path}\n"
            f"Please ensure the data file exists or run the data collection script."
        )

    games = pd.read_csv(games_path, parse_dates=['date'])

    required_cols = ['date', 'home_team', 'away_team', 'home_score', 'away_score']
    missing_cols = [col for col in required_cols if col not in games.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in games data: {missing_cols}")

    if len(games) == 0:
        raise ValueError("Historical games file is empty")

    print(f"   ✓ Loaded {len(games)} games from {games['date'].min()} to {games['date'].max()}")

    # ========================================================================
    # STEP 2: Initialize Elo and process games chronologically
    # ========================================================================
    print("\n2. Processing games chronologically through Elo system...")
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

    # Add neutral_site column back to elo_snapshots (needed for HCA features)
    if 'neutral_site' not in elo_snapshots.columns and 'neutral_site' in games.columns:
        elo_snapshots['neutral_site'] = games['neutral_site'].values

    print(f"   ✓ Processed {len(elo_snapshots)} games, tracking {len(elo.ratings)} teams")

    # ========================================================================
    # STEP 3: Load efficiency stats
    # ========================================================================
    print("\n3. Loading team efficiency stats from Barttorvik...")
    all_stats = []
    for year in config.TRAINING_YEARS:
        df = fetch_barttorvik_year(year)
        df['season'] = year
        all_stats.append(df[['team', 'adjoe', 'adjde', 'season']])

    team_stats = pd.concat(all_stats, ignore_index=True)
    team_stats.columns = ['team', 'adj_oe', 'adj_de', 'season']
    team_stats['adj_em'] = team_stats['adj_oe'] - team_stats['adj_de']
    print(f"   ✓ Loaded {len(team_stats)} team-season efficiency records")

    # ========================================================================
    # STEP 4: Load or compute enhanced features (NEW!)
    # ========================================================================
    print("\n4. Loading or computing enhanced features...")
    enhanced_features_path = config.PROCESSED_DATA_DIR / 'enhanced_features_2020_2024.csv'

    if enhanced_features_path.exists():
        print("   Loading pre-computed enhanced features...")
        enhanced_features = pd.read_csv(enhanced_features_path)
        print(f"   ✓ Loaded {len(enhanced_features)} rows with {len(enhanced_features.columns)} features")
    else:
        print("   Computing enhanced features (this will take 3-4 minutes)...")

        # Check if player data exists
        player_data_path = config.HISTORICAL_PLAYER_DATA
        if player_data_path.exists():
            print(f"   Loading player data from {player_data_path}...")
            player_data = pd.read_csv(player_data_path)
            player_data['game_date'] = pd.to_datetime(player_data['game_date'])
            print(f"   ✓ Loaded {len(player_data)} player-game records")
            use_player_features = True
        else:
            print("   ⚠ Player data not found - skipping player features")
            player_data = None
            use_player_features = False

        # Initialize feature engine
        feature_engine = EnhancedFeatureEngine(
            games_df=elo_snapshots,
            player_data_df=player_data,
            use_player_features=use_player_features
        )

        # Compute all features
        enhanced_features = feature_engine.compute_all_features(elo_snapshots)

        # Save for future runs
        enhanced_features.to_csv(enhanced_features_path, index=False)
        print(f"\n   ✓ Saved enhanced features to {enhanced_features_path}")

    # ========================================================================
    # STEP 5: Merge to create training data
    # ========================================================================
    print("\n5. Creating training dataset...")
    elo_snapshots['season'] = elo_snapshots['date'].dt.year

    # Add a tracking index BEFORE any filtering
    elo_snapshots['_original_idx'] = range(len(elo_snapshots))

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

    # Filter to games with efficiency stats and keep track of which rows survived
    train_data = train_data.dropna(subset=['home_adj_oe', 'away_adj_oe'])
    valid_indices = train_data['_original_idx'].values

    # Merge enhanced features by filtering to matching rows
    print("   Merging enhanced features with training data...")
    print(f"   train_data: {len(train_data)} rows")
    print(f"   enhanced_features: {len(enhanced_features)} rows")

    # Filter enhanced_features to only rows that survived the efficiency filter
    enhanced_features_filtered = enhanced_features.iloc[valid_indices].reset_index(drop=True)
    train_data = train_data.reset_index(drop=True)

    # Remove tracking column
    train_data = train_data.drop(columns=['_original_idx'])

    # Now concat - lengths should match
    print(f"   After filtering: train_data={len(train_data)}, enhanced_features={len(enhanced_features_filtered)}")
    train_data = pd.concat([train_data, enhanced_features_filtered], axis=1)

    print(f"   ✓ Created {len(train_data)} training samples")

    # ========================================================================
    # STEP 6: Train model with enhanced features
    # ========================================================================
    print("\n6. Training model with enhanced features...")

    # Combine baseline + enhanced features
    feature_cols = config.BASELINE_FEATURES + config.ENHANCED_FEATURES

    # Check which features are available
    available_features = [f for f in feature_cols if f in train_data.columns]
    missing_features = [f for f in feature_cols if f not in train_data.columns]

    if missing_features:
        print(f"   ⚠ Warning: Missing features: {missing_features}")
        print(f"   Using {len(available_features)} available features")
        feature_cols = available_features

    X = train_data[feature_cols]
    y = train_data['actual_margin']

    print(f"\n   Feature breakdown:")
    print(f"   - Baseline features: {len(config.BASELINE_FEATURES)}")
    print(f"   - Enhanced features: {len([f for f in feature_cols if f in config.ENHANCED_FEATURES])}")
    print(f"   - Total features: {len(feature_cols)}")

    # Train model with same hyperparameters as baseline
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
    # STEP 7: Cross-validation
    # ========================================================================
    print(f"\n7. Running {config.CV_CONFIG['n_splits']}-fold time-series cross-validation...")
    tscv = TimeSeriesSplit(n_splits=config.CV_CONFIG['n_splits'])
    cv_results = {'ridge': [], 'ensemble': []}

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        fold_model = ImprovedSpreadModel(
            ridge_alpha=config.MODEL_CONFIG['ridge_alpha'],
            lgbm_params={
                'n_estimators': config.MODEL_CONFIG['n_estimators'],
                'max_depth': config.MODEL_CONFIG['max_depth'],
                'learning_rate': config.MODEL_CONFIG['learning_rate']
            },
            weights=(config.MODEL_CONFIG['ridge_weight'], config.MODEL_CONFIG['lgbm_weight'])
        )
        fold_model.fit(X_train, y_train)

        preds = fold_model.predict(X_val)
        components = fold_model.predict_components(X_val)

        ridge_mae = np.abs(components['ridge'] - y_val).mean()
        ensemble_mae = np.abs(preds - y_val).mean()

        cv_results['ridge'].append(ridge_mae)
        cv_results['ensemble'].append(ensemble_mae)

        print(f"   Fold {fold+1}: Ridge MAE={ridge_mae:.3f}, Ensemble MAE={ensemble_mae:.3f}")

    # ========================================================================
    # STEP 8: Summary
    # ========================================================================
    print(f"\n{'='*60}")
    print("ENHANCED MODEL TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Training data: {len(train_data)} games (2020-2025)")
    print(f"Total features: {len(feature_cols)}")
    print(f"  - Baseline: {len(config.BASELINE_FEATURES)}")
    print(f"  - Enhanced: {len([f for f in feature_cols if f in config.ENHANCED_FEATURES])}")
    print(f"\nCross-Validation Results:")
    print(f"  Ridge MAE:    {np.mean(cv_results['ridge']):.3f} ± {np.std(cv_results['ridge']):.3f}")
    print(f"  Ensemble MAE: {np.mean(cv_results['ensemble']):.3f} ± {np.std(cv_results['ensemble']):.3f}")
    print(f"\n{'='*60}")

    return {
        'model': model,
        'elo': elo,
        'cv_results': cv_results,
        'feature_cols': feature_cols,
        'X': X,
        'y': y
    }


if __name__ == '__main__':
    results = main()
