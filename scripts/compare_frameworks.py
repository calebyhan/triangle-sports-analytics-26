"""
Compare Team-Based vs Player-Based Prediction Frameworks

Backtests both systems on 2024-25 ACC conference games to evaluate
prediction accuracy.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from src.elo import EloRatingSystem
from src.player_elo.player_elo_system import PlayerEloSystem
from src.player_elo.features import PlayerFeatureEngine
from src.player_elo.pytorch_model import PlayerELONet
from src.player_elo.player_data_collector import PlayerDataCollector
import torch


def evaluate_predictions(predictions, actuals):
    """Calculate evaluation metrics"""
    errors = predictions - actuals
    abs_errors = np.abs(errors)

    mae = np.mean(abs_errors)
    rmse = np.sqrt(np.mean(errors ** 2))

    # Direction accuracy (correctly predict winner)
    correct_direction = ((predictions > 0) == (actuals > 0)).sum()
    direction_acc = (correct_direction / len(predictions)) * 100

    return {
        'mae': mae,
        'rmse': rmse,
        'direction_acc': direction_acc,
        'mean_error': np.mean(errors),
        'std_error': np.std(errors)
    }


def test_team_based_system(test_games):
    """Test team-based ELO system"""
    print("\n" + "="*70)
    print("  TESTING TEAM-BASED SYSTEM")
    print("="*70)

    # Load team stats
    team_stats = pd.read_csv(project_root / 'data' / 'processed' / 'team_stats_2025_26.csv')

    # Initialize ELO system
    elo_system = EloRatingSystem()

    predictions = []
    actuals = []

    for idx, row in test_games.iterrows():
        home_team = row['Home']
        away_team = row['Away']
        actual_spread = row['actual_spread']

        # Use ELO system's predict_spread method
        predicted_spread = elo_system.predict_spread(
            home_team=home_team,
            away_team=away_team,
            neutral=False
        )

        predictions.append(predicted_spread)
        actuals.append(actual_spread)

        if (idx + 1) % 20 == 0:
            print(f"  Processed {idx + 1}/{len(test_games)} games")

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    metrics = evaluate_predictions(predictions, actuals)

    print(f"\n  Results:")
    print(f"    MAE: {metrics['mae']:.2f} points")
    print(f"    RMSE: {metrics['rmse']:.2f} points")
    print(f"    Direction Accuracy: {metrics['direction_acc']:.2f}%")
    print(f"    Mean Error: {metrics['mean_error']:.2f} points")

    return metrics, predictions, actuals


def test_player_based_system(test_games):
    """Test player-based ELO system"""
    print("\n" + "="*70)
    print("  TESTING PLAYER-BASED SYSTEM")
    print("="*70)

    # Load model - use regular model (optimized model has BatchNorm layers not supported by PlayerELONet)
    model_path = project_root / 'data' / 'player_data' / 'models' / 'pytorch_model.pt'

    if not model_path.exists():
        print("  [ERROR] Player model not found")
        return None, None, None

    print(f"  Using model: {model_path.name}")

    # Load ELO state
    elo_state_path = project_root / 'data' / 'player_data' / 'models' / 'player_elo_state.json'
    elo_system = PlayerEloSystem()
    elo_system.load_state(str(elo_state_path))

    # Load model
    device = torch.device('cpu')
    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Infer architecture from state dict
    if 'network.0.weight' in state_dict:
        # Standard PlayerELONet architecture
        input_dim = state_dict['network.0.weight'].shape[1]
        hidden_dims = []

        # Find all Linear layers by looking for weights that aren't 1D (BatchNorm)
        # and aren't part of BatchNorm running stats
        for key in sorted(state_dict.keys()):
            if '.weight' in key and 'network.' in key:
                weight_shape = state_dict[key].shape
                # Linear layers have 2D weights, BatchNorm has 1D weights
                if len(weight_shape) == 2:
                    output_dim = weight_shape[0]
                    # Skip the final output layer (output_dim == 1)
                    if output_dim != 1:
                        hidden_dims.append(output_dim)

        if not hidden_dims:
            hidden_dims = [128, 64, 32]  # Default fallback
    else:
        # Try other possible architectures
        print(f"  Available keys: {list(state_dict.keys())[:5]}")
        input_dim = 65  # Default
        hidden_dims = [128, 64, 32]  # Default

    # Create model with correct architecture
    model = PlayerELONet(input_dim=input_dim, hidden_dims=hidden_dims)

    try:
        model.load_state_dict(state_dict, strict=False)
        print(f"  Loaded model with architecture: {input_dim} -> {hidden_dims} -> 1")
    except Exception as e:
        print(f"  [WARNING] Could not load state dict: {e}")

    model.eval()

    # Load player stats for 2025 (most recent available)
    data_collector = PlayerDataCollector()
    player_stats = data_collector.load_player_stats(2025)

    if player_stats is None or len(player_stats) == 0:
        print("  [ERROR] Player stats not found")
        return None, None, None

    # Create feature engine
    feature_engine = PlayerFeatureEngine(player_stats, elo_system)

    predictions = []
    actuals = []

    # Team name mapping
    TEAM_NAME_MAPPING = {
        'Florida State': 'Florida St.',
        'Florida St.': 'Florida St.',
        'Miami': 'Miami FL',
        'Miami FL': 'Miami FL',
        'NC State': 'N.C. State',
        'N.C. State': 'N.C. State',
        'Pitt': 'Pittsburgh',
        'Pittsburgh': 'Pittsburgh',
    }

    for idx, row in test_games.iterrows():
        home_team = TEAM_NAME_MAPPING.get(row['Home'], row['Home'])
        away_team = TEAM_NAME_MAPPING.get(row['Away'], row['Away'])
        actual_spread = row['actual_spread']

        # Get top 5 players by minutes for each team
        home_players = player_stats[player_stats['team'] == home_team].copy()
        away_players = player_stats[player_stats['team'] == away_team].copy()

        if len(home_players) == 0 or len(away_players) == 0:
            # Fallback to ELO difference
            home_elo = np.mean([elo_system.get_elo(p) for p in home_players['player_id'][:5]]) if len(home_players) > 0 else 1000
            away_elo = np.mean([elo_system.get_elo(p) for p in away_players['player_id'][:5]]) if len(away_players) > 0 else 1000
            predicted_spread = (home_elo - away_elo) / 25.0
        else:
            # Get lineups
            home_players = home_players.sort_values('minutes_per_game', ascending=False).head(5)
            away_players = away_players.sort_values('minutes_per_game', ascending=False).head(5)

            home_lineup = home_players['player_id'].tolist()
            away_lineup = away_players['player_id'].tolist()

            try:
                # Create features
                features = feature_engine.create_matchup_features(
                    home_lineup, away_lineup,
                    home_team, away_team,
                    is_neutral=False
                )

                # Predict
                with torch.no_grad():
                    features_tensor = torch.FloatTensor(features).unsqueeze(0)
                    predicted_spread = model(features_tensor).item()
            except Exception as e:
                # Fallback
                predicted_spread = 0.0

        predictions.append(predicted_spread)
        actuals.append(actual_spread)

        if (idx + 1) % 20 == 0:
            print(f"  Processed {idx + 1}/{len(test_games)} games")

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    metrics = evaluate_predictions(predictions, actuals)

    print(f"\n  Results:")
    print(f"    MAE: {metrics['mae']:.2f} points")
    print(f"    RMSE: {metrics['rmse']:.2f} points")
    print(f"    Direction Accuracy: {metrics['direction_acc']:.2f}%")
    print(f"    Mean Error: {metrics['mean_error']:.2f} points")

    return metrics, predictions, actuals


def main():
    """Main comparison"""
    print("\n" + "="*70)
    print("  FRAMEWORK COMPARISON: Team-Based vs Player-Based")
    print("="*70)

    # Load test set
    test_file = project_root / 'data' / 'test_data' / 'acc_2025_test_set.csv'

    if not test_file.exists():
        print(f"\n[ERROR] Test set not found: {test_file}")
        print("Run the data extraction step first.")
        sys.exit(1)

    test_games = pd.read_csv(test_file)
    print(f"\nTest set: {len(test_games)} ACC conference games from 2024-25 season")

    # Test team-based system
    team_metrics, team_preds, actuals = test_team_based_system(test_games)

    # Test player-based system
    player_metrics, player_preds, _ = test_player_based_system(test_games)

    # Comparison
    print("\n" + "="*70)
    print("  COMPARISON SUMMARY")
    print("="*70)

    print(f"\n{'Metric':<25} {'Team-Based':<15} {'Player-Based':<15} {'Winner':<10}")
    print("-"*70)

    if player_metrics is not None:
        mae_winner = "Team" if team_metrics['mae'] < player_metrics['mae'] else "Player"
        rmse_winner = "Team" if team_metrics['rmse'] < player_metrics['rmse'] else "Player"
        acc_winner = "Team" if team_metrics['direction_acc'] > player_metrics['direction_acc'] else "Player"

        print(f"{'MAE (points)':<25} {team_metrics['mae']:<15.2f} {player_metrics['mae']:<15.2f} {mae_winner:<10}")
        print(f"{'RMSE (points)':<25} {team_metrics['rmse']:<15.2f} {player_metrics['rmse']:<15.2f} {rmse_winner:<10}")
        print(f"{'Direction Accuracy (%)':<25} {team_metrics['direction_acc']:<15.2f} {player_metrics['direction_acc']:<15.2f} {acc_winner:<10}")

        # Overall winner
        team_wins = sum([
            team_metrics['mae'] < player_metrics['mae'],
            team_metrics['rmse'] < player_metrics['rmse'],
            team_metrics['direction_acc'] > player_metrics['direction_acc']
        ])

        print("\n" + "="*70)
        if team_wins >= 2:
            print("  WINNER: Team-Based System")
            improvement = ((player_metrics['mae'] - team_metrics['mae']) / player_metrics['mae']) * 100
            print(f"  Team-based MAE is {improvement:.1f}% better")
        else:
            print("  WINNER: Player-Based System")
            improvement = ((team_metrics['mae'] - player_metrics['mae']) / team_metrics['mae']) * 100
            print(f"  Player-based MAE is {improvement:.1f}% better")
        print("="*70)

    # Save results
    results_df = test_games.copy()
    results_df['team_prediction'] = team_preds
    if player_preds is not None:
        results_df['player_prediction'] = player_preds
    results_df['team_error'] = team_preds - actuals
    if player_preds is not None:
        results_df['player_error'] = player_preds - actuals

    results_df.to_csv(project_root / 'data' / 'test_data' / 'framework_comparison_results.csv', index=False)
    print(f"\nDetailed results saved to: data/test_data/framework_comparison_results.csv")


if __name__ == "__main__":
    main()
