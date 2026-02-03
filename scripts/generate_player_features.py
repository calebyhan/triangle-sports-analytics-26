"""
Generate player-based features from collected box scores.

Creates star player, offensive balance, and bench depth metrics
for all 21 ACC teams.
"""

import sys
sys.path.append('..')

import pandas as pd
from pathlib import Path
from src.features.player_features import aggregate_all_player_features
from src.config import PROCESSED_DATA_DIR, ACC_TEAMS
from src.logger import setup_logger

logger = setup_logger(__name__)

# Input/Output paths
BOX_SCORES_FILE = PROCESSED_DATA_DIR / 'player_box_scores_2026.csv'
OUTPUT_FILE = PROCESSED_DATA_DIR / 'player_features_2026.csv'


def main():
    """Generate player features for all ACC teams."""
    print("="*60)
    print("GENERATING PLAYER FEATURES")
    print("="*60)
    print(f"Input: {BOX_SCORES_FILE}")
    print(f"Teams: {len(ACC_TEAMS)}")
    print("="*60)

    # Load box scores
    print("\nLoading box scores...")
    box_scores = pd.read_csv(BOX_SCORES_FILE)
    print(f"✓ Loaded {len(box_scores)} player-game records")
    print(f"  Unique players: {box_scores['player'].nunique()}")
    print(f"  Unique teams: {box_scores['team'].nunique()}")

    # Generate features
    print("\nGenerating player features...")
    features_df = aggregate_all_player_features(
        box_scores,
        ACC_TEAMS,
        output_path=OUTPUT_FILE
    )

    # Display summary
    print(f"\n{'='*60}")
    print("✓ FEATURE GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Teams: {len(features_df)}")
    print(f"Features per team: {len(features_df.columns) - 1}")  # Exclude 'team' column
    print(f"\nFeature categories:")
    print(f"  - Star Player Power (4 features)")
    print(f"  - Offensive Balance (3 features)")
    print(f"  - Bench Depth (3 features)")
    print(f"  - Key Player Efficiency (3 features)")
    print(f"\nTotal: 13 player-based features")
    print(f"\nSaved to: {OUTPUT_FILE}")
    print("="*60)

    # Show sample
    print("\nSample features (first 5 teams):")
    print(features_df.head())


if __name__ == '__main__':
    main()
