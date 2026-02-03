"""
Train Player-Based ELO Model

Simple script to train the complete player ELO system on historical data.

Usage:
    # Train on all years (2020-2025)
    python scripts/player_elo/train_model.py

    # Train on specific years
    python scripts/player_elo/train_model.py --years 2023 2024 2025

    # Force re-download of data
    python scripts/player_elo/train_model.py --no-cache
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.player_elo.training_pipeline import train_player_model
from src.player_elo.config import TRAINING_YEARS


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(
        description='Train Player-Based ELO Model'
    )

    parser.add_argument(
        '--years',
        type=int,
        nargs='+',
        default=TRAINING_YEARS,
        help=f'Training years (default: {TRAINING_YEARS})'
    )

    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Force re-download of player data'
    )

    parser.add_argument(
        '--cv-splits',
        type=int,
        default=5,
        help='Number of cross-validation splits (default: 5)'
    )

    args = parser.parse_args()

    print("\n" + "="*70)
    print("  PLAYER-BASED ELO MODEL TRAINING")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Training years: {args.years}")
    print(f"  Use cached data: {not args.no_cache}")
    print(f"  CV splits: {args.cv_splits}")
    print("\n" + "="*70 + "\n")

    # Run training
    try:
        results = train_player_model(
            years=args.years,
            use_cached_data=not args.no_cache,
            n_cv_splits=args.cv_splits
        )

        print("\n[SUCCESS] Training completed!")
        print(f"\nFinal Results:")
        print(f"  MAE: {results['mean_mae']:.4f} ± {results['std_mae']:.4f}")
        print(f"\nModel saved and ready for predictions.")

    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
