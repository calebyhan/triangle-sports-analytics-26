"""
Generate Predictions with Hybrid Player-Team Model

Uses the trained hybrid model that combines player ELO ratings with
team metrics for superior predictions.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np

from src.hybrid_model import HybridPredictionSystem
from src.player_elo.player_elo_system import PlayerEloSystem
from src.player_elo.player_data_collector import PlayerDataCollector
from src.elo import EloRatingSystem


def main():
    """Generate 2026 predictions with hybrid model"""
    print("\n" + "="*70)
    print("  HYBRID PLAYER-TEAM PREDICTIONS")
    print("="*70)
    print("\nCombining:")
    print("  [x] Player-level ELO ratings (individual skill)")
    print("  [x] Team-level metrics (synergy & coaching)")
    print("  [x] Neural network learned correlations")
    print("\n" + "="*70 + "\n")

    # Load model and scaler
    model_path = project_root / 'data' / 'player_data' / 'models' / 'hybrid_model.pt'
    scaler_path = project_root / 'data' / 'player_data' / 'models' / 'hybrid_scaler.pkl'

    if not model_path.exists():
        print(f"[ERROR] Hybrid model not found: {model_path}")
        print("Please train the model first: python scripts/train_hybrid_model.py")
        sys.exit(1)

    print(f"Loading hybrid model from: {model_path}")
    hybrid_system = HybridPredictionSystem(
        model_path=model_path,
        scaler_path=scaler_path if scaler_path.exists() else None
    )

    # Load player ELO system
    print("\nLoading player ELO system...")
    player_elo_system = PlayerEloSystem()
    elo_state_path = project_root / 'data' / 'player_data' / 'models' / 'player_elo_state.json'

    if elo_state_path.exists():
        player_elo_system.load_state(str(elo_state_path))
        print(f"  Loaded {len(player_elo_system.player_elos)} player ratings")
    else:
        print("  [WARNING] No player ELO state found")

    # Load team ELO system
    print("\nLoading team ELO system...")
    team_elo_system = EloRatingSystem()

    # Load player stats
    print("\nLoading player statistics for 2025...")
    data_collector = PlayerDataCollector()
    player_stats = data_collector.load_player_stats(2025)

    if player_stats is None or len(player_stats) == 0:
        print("[ERROR] Could not load player stats for 2025")
        sys.exit(1)

    print(f"  Loaded {len(player_stats)} player records")

    # Load team stats
    print("\nLoading team statistics...")
    team_stats = pd.read_csv(project_root / 'data' / 'processed' / 'team_stats_2025_26.csv')
    print(f"  Loaded {len(team_stats)} team records")

    # Initialize feature engine
    print("\nInitializing hybrid feature engine...")
    hybrid_system.initialize_feature_engine(
        player_elo_system,
        team_elo_system,
        player_stats,
        team_stats
    )

    # Load games to predict
    games_file = project_root / 'data' / 'processed' / 'games_to_predict.csv'
    print(f"\nLoading games from: {games_file}")
    games = pd.read_csv(games_file)
    print(f"  Loaded {len(games)} games to predict")

    # Generate predictions
    print(f"\nGenerating predictions...")
    predictions = []

    for idx, row in games.iterrows():
        try:
            pred = hybrid_system.predict(
                home_team=row['Home'],
                away_team=row['Away'],
                neutral=False
            )
            predictions.append(pred)

            if (idx + 1) % 10 == 0:
                print(f"  Processed {idx + 1}/{len(games)} games")

        except Exception as e:
            print(f"  [WARNING] Error predicting {row['Home']} vs {row['Away']}: {e}")
            predictions.append(0.0)  # Fallback

    # Create output DataFrame
    results = games.copy()
    results['pt_spread'] = predictions
    results['team_name'] = 'CMMT'

    # Statistics
    print("\n[SUCCESS] Hybrid predictions generated!")
    print(f"\nPrediction Statistics:")
    print(f"  Total games: {len(predictions)}")
    print(f"  Mean spread: {np.mean(predictions):.2f} points")
    print(f"  Std spread: {np.std(predictions):.2f} points")
    print(f"  Min spread: {np.min(predictions):.2f} points")
    print(f"  Max spread: {np.max(predictions):.2f} points")

    print(f"\nFirst 10 predictions:")
    display_cols = ['Date', 'Home', 'Away', 'pt_spread']
    print(results[display_cols].head(10).to_string(index=False))

    # Save predictions
    output_file = project_root / 'data' / 'predictions' / 'tsa_pt_spread_HYBRID_2026.csv'
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Competition format
    submission = results[['Date', 'Home', 'Away', 'pt_spread', 'team_name']].copy()
    submission.to_csv(output_file, index=False)
    print(f"\nSaved to: {output_file}")

    # Detailed output
    detailed_file = project_root / 'data' / 'predictions' / 'tsa_pt_spread_HYBRID_2026_detailed.csv'
    results.to_csv(detailed_file, index=False)
    print(f"Detailed: {detailed_file}")

    print(f"\nHybrid predictions ready for competition!")


if __name__ == "__main__":
    main()
