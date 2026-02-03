"""
Generate 2026 Predictions with Player-Based ELO Model

Simple script to generate predictions for 2026 ACC games using the trained
player-based ELO system.

Usage:
    # Generate predictions for 2026 season
    python scripts/player_elo/generate_predictions.py

    # Use specific model
    python scripts/player_elo/generate_predictions.py --model data/player_data/models/pytorch_model_fold1.pt

    # Specify output file
    python scripts/player_elo/generate_predictions.py --output my_predictions.csv
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.player_elo.prediction_pipeline import generate_predictions
from src.player_elo.config import MODELS_DIR, PROJECT_ROOT


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(
        description='Generate 2026 Predictions with Player-Based ELO Model'
    )

    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Path to trained model (default: data/player_data/models/pytorch_model.pt)'
    )

    parser.add_argument(
        '--elo-state',
        type=str,
        default=None,
        help='Path to ELO state JSON (default: data/player_data/models/player_elo_state.json)'
    )

    parser.add_argument(
        '--games',
        type=str,
        default=None,
        help='Path to games CSV (default: data/processed/acc_games_2026.csv)'
    )

    parser.add_argument(
        '--player-stats-year',
        type=int,
        default=2026,
        help='Year for player statistics (default: 2026)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output CSV path (default: data/predictions/tsa_pt_spread_PLAYER_ELO_2026.csv)'
    )

    parser.add_argument(
        '--team-name',
        type=str,
        default='CMMT',
        help='Team identifier for submission (default: CMMT)'
    )

    args = parser.parse_args()

    print("\n" + "="*70)
    print("  PLAYER-BASED ELO PREDICTION GENERATION")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Model: {args.model or 'default'}")
    print(f"  ELO state: {args.elo_state or 'default'}")
    print(f"  Games: {args.games or 'default'}")
    print(f"  Player stats year: {args.player_stats_year}")
    print(f"  Output: {args.output or 'default'}")
    print(f"  Team name: {args.team_name}")
    print("\n" + "="*70 + "\n")

    # Run prediction generation
    try:
        predictions = generate_predictions(
            model_path=args.model,
            elo_state_path=args.elo_state,
            games_file=args.games,
            player_stats_year=args.player_stats_year,
            output_file=args.output,
            team_name=args.team_name
        )

        print("\n[SUCCESS] Predictions generated!")
        print(f"\nSummary Statistics:")
        print(f"  Total games: {len(predictions)}")
        print(f"  Mean spread: {predictions['pt_spread'].mean():.2f} points")
        print(f"  Std spread: {predictions['pt_spread'].std():.2f} points")
        print(f"  Min spread: {predictions['pt_spread'].min():.2f} points")
        print(f"  Max spread: {predictions['pt_spread'].max():.2f} points")

        print(f"\nFirst 5 predictions:")
        print(predictions[['Date', 'Home', 'Away', 'pt_spread']].head().to_string(index=False))

        print(f"\nPredictions ready for submission!")

    except FileNotFoundError as e:
        print(f"\n[ERROR] File not found: {e}")
        print(f"\nPlease ensure:")
        print(f"  1. Model is trained: python scripts/player_elo/train_model.py")
        print(f"  2. Games file exists: {args.games or 'data/processed/acc_games_2026.csv'}")
        print(f"  3. Player data for {args.player_stats_year} exists in data/raw_pd/")
        sys.exit(1)

    except Exception as e:
        print(f"\n[ERROR] Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
