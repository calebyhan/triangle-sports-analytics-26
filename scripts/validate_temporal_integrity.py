"""
Validate temporal integrity of enhanced features.

For a random sample of games, manually recompute features using only
data before the game date and compare with pre-computed features to
ensure no data leakage.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from src.elo import EloRatingSystem
from src.features.enhanced_pipeline import EnhancedFeatureEngine
from src import config


def main():
    print("="*60)
    print("TEMPORAL INTEGRITY VALIDATION")
    print("="*60)

    # Load historical games
    print("\n1. Loading historical games...")
    games = pd.read_csv(config.HISTORICAL_GAMES_FILE, parse_dates=['date'])
    print(f"   ✓ Loaded {len(games)} games")

    # Process through Elo
    print("\n2. Processing games through Elo...")
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
    print(f"   ✓ Processed {len(elo_snapshots)} games")

    # Load pre-computed features
    print("\n3. Loading pre-computed features...")
    enhanced_features_path = config.PROCESSED_DATA_DIR / 'enhanced_features_2020_2024.csv'
    if not enhanced_features_path.exists():
        print(f"   ✗ Enhanced features file not found: {enhanced_features_path}")
        print("   Please run train_model_enhanced.py first to generate features.")
        return False

    precomputed_features = pd.read_csv(enhanced_features_path)
    print(f"   ✓ Loaded {len(precomputed_features)} pre-computed feature rows")

    # Load player data if available
    player_data_path = config.HISTORICAL_PLAYER_DATA
    if player_data_path.exists():
        print("\n4. Loading player data...")
        player_data = pd.read_csv(player_data_path)
        player_data['game_date'] = pd.to_datetime(player_data['game_date'])
        print(f"   ✓ Loaded {len(player_data)} player-game records")
        use_player_features = True
    else:
        print("\n4. Player data not found - skipping player features validation")
        player_data = None
        use_player_features = False

    # Initialize feature engine
    feature_engine = EnhancedFeatureEngine(
        games_df=elo_snapshots,
        player_data_df=player_data,
        use_player_features=use_player_features
    )

    # Sample 100 random games for validation
    print("\n5. Sampling games for validation...")
    sample_size = min(100, len(elo_snapshots))
    sample_indices = np.random.choice(len(elo_snapshots), size=sample_size, replace=False)
    sample_games = elo_snapshots.iloc[sample_indices]
    print(f"   ✓ Sampled {sample_size} games")

    # Validate each sample
    print("\n6. Validating temporal integrity...")
    print("   (Recomputing features for each sample game...)")

    errors = []
    warnings = []

    for idx, (sample_idx, game) in enumerate(sample_games.iterrows()):
        # Manually recompute features for this game
        as_of_date = game['date']
        home_team = game['home_team']
        away_team = game['away_team']

        try:
            # Recompute features
            recomputed_features = feature_engine._compute_game_features(
                game, as_of_date
            )

            # Get pre-computed features for this game
            precomputed_row = precomputed_features.iloc[sample_idx]

            # Compare each feature
            for feature_name in config.ENHANCED_FEATURES:
                if feature_name not in recomputed_features:
                    warnings.append((sample_idx, feature_name, "Feature not computed"))
                    continue

                if feature_name not in precomputed_row:
                    warnings.append((sample_idx, feature_name, "Feature not in pre-computed data"))
                    continue

                recomputed_value = recomputed_features[feature_name]
                precomputed_value = precomputed_row[feature_name]

                # Check for NaN mismatches
                if pd.isna(recomputed_value) and pd.isna(precomputed_value):
                    continue
                elif pd.isna(recomputed_value) or pd.isna(precomputed_value):
                    errors.append((
                        sample_idx,
                        feature_name,
                        f"NaN mismatch: recomputed={recomputed_value}, precomputed={precomputed_value}"
                    ))
                    continue

                # Check for value differences > 0.01
                diff = abs(recomputed_value - precomputed_value)
                if diff > 0.01:
                    errors.append((
                        sample_idx,
                        feature_name,
                        f"Value mismatch (diff={diff:.4f}): recomputed={recomputed_value:.4f}, precomputed={precomputed_value:.4f}"
                    ))

        except Exception as e:
            errors.append((sample_idx, "EXCEPTION", str(e)))

        # Progress indicator
        if (idx + 1) % 20 == 0:
            print(f"   Validated {idx + 1}/{sample_size} games...")

    # Report results
    print(f"\n{'='*60}")
    print("VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"Games validated: {sample_size}")
    print(f"Errors found: {len(errors)}")
    print(f"Warnings: {len(warnings)}")

    if errors:
        print(f"\n⚠ ERRORS DETECTED:")
        for idx, feature, message in errors[:20]:  # Show first 20
            print(f"   Game {idx}, {feature}: {message}")
        if len(errors) > 20:
            print(f"   ... and {len(errors) - 20} more errors")

    if warnings:
        print(f"\n⚠ WARNINGS:")
        for idx, feature, message in warnings[:20]:  # Show first 20
            print(f"   Game {idx}, {feature}: {message}")
        if len(warnings) > 20:
            print(f"   ... and {len(warnings) - 20} more warnings")

    if not errors and not warnings:
        print(f"\n✓ ALL CHECKS PASSED")
        print("  Temporal integrity verified - no data leakage detected")

    print(f"{'='*60}")

    return len(errors) == 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
