"""
Train Hybrid Player-Team Model

Combines player ELO ratings with team metrics to create superior predictions.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import pickle

from src.hybrid_model import HybridNet, HybridFeatureEngine
from src.player_elo.player_elo_system import PlayerEloSystem
from src.player_elo.player_data_collector import PlayerDataCollector
from src.elo import EloRatingSystem


class GameDataset(Dataset):
    """Dataset for game predictions"""

    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


def prepare_training_data(years: list):
    """
    Prepare training data from historical games

    Returns:
        X: Feature matrix
        y: Target spreads
        feature_engine: For extracting features
    """
    print("\n" + "="*70)
    print("  PREPARING HYBRID TRAINING DATA")
    print("="*70)

    # Load historical games
    games_file = project_root / 'data' / 'raw' / 'games' / 'historical_games_2019_2025.csv'
    print(f"\nLoading games from: {games_file}")
    all_games = pd.read_csv(games_file)

    # Filter for specified seasons
    games = all_games[all_games['season'].isin(years)].copy()
    print(f"  Games in training seasons: {len(games)}")

    # Calculate spreads
    games['spread'] = games['home_score'] - games['away_score']

    # Load player data
    print("\nLoading player data...")
    data_collector = PlayerDataCollector()

    all_player_stats = []
    for year in years:
        stats = data_collector.load_player_stats(year)
        if stats is not None:
            stats['season'] = year
            all_player_stats.append(stats)

    if not all_player_stats:
        raise ValueError("No player stats found")

    player_stats = pd.concat(all_player_stats, ignore_index=True)
    print(f"  Loaded {len(player_stats)} player records")

    # Load team stats
    print("\nLoading team stats...")
    team_stats = pd.read_csv(project_root / 'data' / 'processed' / 'team_stats_2025_26.csv')
    print(f"  Loaded {len(team_stats)} team records")

    # Initialize ELO systems
    print("\nInitializing ELO systems...")
    player_elo_system = PlayerEloSystem()

    elo_state_path = project_root / 'data' / 'player_data' / 'models' / 'player_elo_state.json'
    if elo_state_path.exists():
        player_elo_system.load_state(str(elo_state_path))
        print("  Loaded player ELO state")
    else:
        print("  [WARNING] No player ELO state found, using defaults")

    team_elo_system = EloRatingSystem()

    # ACC teams filter
    acc_teams = [
        'Boston College', 'California', 'Clemson', 'Duke', 'Florida St.',
        'Georgia Tech', 'Louisville', 'Miami FL', 'N.C. State',
        'North Carolina', 'Notre Dame', 'Pittsburgh', 'SMU',
        'Stanford', 'Syracuse', 'Virginia', 'Virginia Tech', 'Wake Forest'
    ]

    # Filter for ACC games
    games = games[
        games['home_team'].isin(acc_teams) &
        games['away_team'].isin(acc_teams)
    ].copy()
    print(f"\nACC conference games: {len(games)}")

    # Extract features for each game
    print("\nExtracting features...")

    features = []
    targets = []

    total_games = len(games)
    for game_num, (idx, row) in enumerate(games.iterrows(), 1):
        # Get season-specific player stats
        season_player_stats = player_stats[player_stats['season'] == row['season']]

        if len(season_player_stats) == 0:
            continue

        # Create feature engine
        feature_engine = HybridFeatureEngine(
            player_elo_system,
            team_elo_system,
            season_player_stats,
            team_stats
        )

        try:
            # Extract features
            game_features = feature_engine.create_hybrid_features(
                row['home_team'],
                row['away_team'],
                neutral=row.get('neutral_site', False)
            )

            features.append(game_features)
            targets.append(row['spread'])

        except Exception as e:
            # Skip games with errors
            if game_num <= 5:  # Debug first few errors
                print(f"  [DEBUG] Error for {row['home_team']} vs {row['away_team']}: {str(e)[:100]}")
            continue

        # Progress update
        if game_num % 100 == 0:
            print(f"  Processed {game_num}/{total_games} games ({len(features)} valid)")

    if not features:
        raise ValueError("No features extracted! Check if player stats match game teams.")

    X = np.array(features)
    y = np.array(targets)

    print(f"\nPrepared {len(X)} samples with {X.shape[1]} features")
    print(f"  Target mean: {y.mean():.2f}")
    print(f"  Target std: {y.std():.2f}")

    # Cache training data for optimization
    cache_file = project_root / 'data' / 'player_data' / 'models' / 'training_cache.pkl'
    with open(cache_file, 'wb') as f:
        pickle.dump({'X': X, 'y': y}, f)
    print(f"\nCached training data to: {cache_file}")

    return X, y


def train_hybrid_model(
    X, y,
    n_splits: int = 5,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 0.001
):
    """
    Train hybrid model with cross-validation

    Args:
        X: Feature matrix
        y: Target spreads
        n_splits: Number of CV folds
        epochs: Training epochs per fold
        batch_size: Batch size
        lr: Learning rate

    Returns:
        results dict with metrics
    """
    print("\n" + "="*70)
    print("  TRAINING HYBRID MODEL")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Feature normalization
    print("\nNormalizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save scaler
    scaler_path = project_root / 'data' / 'player_data' / 'models' / 'hybrid_scaler.pkl'
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"  Saved scaler to: {scaler_path}")

    # Cross-validation
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_scaled), 1):
        print(f"\nFold {fold}/{n_splits}")
        print(f"  Train: {len(train_idx)} samples, Val: {len(val_idx)} samples")

        # Split data
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Create datasets
        train_dataset = GameDataset(X_train, y_train)
        val_dataset = GameDataset(X_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Create model
        model = HybridNet(input_dim=X.shape[1], hidden_dims=[128, 64, 32])
        model = model.to(device)

        # Loss and optimizer
        criterion = nn.HuberLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5
        )

        # Training loop
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0

        for epoch in range(epochs):
            # Train
            model.train()
            train_loss = 0.0

            for features, targets in train_loader:
                features = features.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()
                predictions = model(features)
                loss = criterion(predictions, targets)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validate
            model.eval()
            val_loss = 0.0
            val_predictions = []
            val_targets = []

            with torch.no_grad():
                for features, targets in val_loader:
                    features = features.to(device)
                    targets = targets.to(device)

                    predictions = model(features)
                    loss = criterion(predictions, targets)

                    val_loss += loss.item()
                    val_predictions.extend(predictions.cpu().numpy())
                    val_targets.extend(targets.cpu().numpy())

            val_loss /= len(val_loader)

            # Metrics
            val_predictions = np.array(val_predictions)
            val_targets = np.array(val_targets)
            mae = np.mean(np.abs(val_predictions - val_targets))

            # Learning rate scheduling
            scheduler.step(val_loss)

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1:3d}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val MAE: {mae:.4f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                # Save best model for this fold
                model_path = project_root / 'data' / 'player_data' / 'models' / f'hybrid_model_fold{fold}.pt'
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'input_dim': X.shape[1],
                    'hidden_dims': [128, 64, 32],
                    'epoch': epoch,
                    'val_loss': val_loss
                }, model_path)

            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch + 1}")
                    break

        # Final validation metrics
        model.eval()
        val_predictions = []
        val_targets = []

        with torch.no_grad():
            for features, targets in val_loader:
                features = features.to(device)
                targets = targets.to(device)
                predictions = model(features)
                val_predictions.extend(predictions.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())

        val_predictions = np.array(val_predictions)
        val_targets = np.array(val_targets)

        mae = np.mean(np.abs(val_predictions - val_targets))
        rmse = np.sqrt(np.mean((val_predictions - val_targets) ** 2))
        direction_acc = np.mean((val_predictions > 0) == (val_targets > 0)) * 100

        print(f"\n  Fold {fold} Results:")
        print(f"    MAE: {mae:.4f}")
        print(f"    RMSE: {rmse:.4f}")
        print(f"    Direction Accuracy: {direction_acc:.2f}%")

        fold_results.append({
            'mae': mae,
            'rmse': rmse,
            'direction_acc': direction_acc
        })

    # Summary
    mean_mae = np.mean([r['mae'] for r in fold_results])
    std_mae = np.std([r['mae'] for r in fold_results])
    mean_rmse = np.mean([r['rmse'] for r in fold_results])
    mean_acc = np.mean([r['direction_acc'] for r in fold_results])

    print("\n" + "="*70)
    print("  CROSS-VALIDATION SUMMARY")
    print("="*70)
    print(f"  Mean MAE: {mean_mae:.4f} +/- {std_mae:.4f}")
    print(f"  Mean RMSE: {mean_rmse:.4f}")
    print(f"  Mean Direction Acc: {mean_acc:.2f}%")
    print("="*70)

    # Train final model on all data
    print("\nTraining final model on all data...")
    train_dataset = GameDataset(X_scaled, y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    final_model = HybridNet(input_dim=X.shape[1], hidden_dims=[128, 64, 32])
    final_model = final_model.to(device)

    optimizer = torch.optim.AdamW(final_model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.HuberLoss()

    for epoch in range(50):  # Fewer epochs for final model
        final_model.train()
        train_loss = 0.0

        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            predictions = final_model(features)
            loss = criterion(predictions, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(final_model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/50 | Loss: {train_loss/len(train_loader):.4f}")

    # Save final model
    final_model_path = project_root / 'data' / 'player_data' / 'models' / 'hybrid_model.pt'
    torch.save({
        'model_state_dict': final_model.state_dict(),
        'input_dim': X.shape[1],
        'hidden_dims': [128, 64, 32]
    }, final_model_path)
    print(f"\n  Saved final model to: {final_model_path}")

    return {
        'mean_mae': mean_mae,
        'std_mae': std_mae,
        'mean_rmse': mean_rmse,
        'mean_direction_acc': mean_acc
    }


def main():
    """Main execution"""
    print("\n" + "="*70)
    print("  HYBRID PLAYER-TEAM MODEL TRAINING")
    print("="*70)

    # Training years
    years = [2020, 2021, 2022, 2023, 2024, 2025]
    print(f"\nTraining years: {years}")

    # Prepare data
    X, y = prepare_training_data(years)

    # Train model
    results = train_hybrid_model(X, y, n_splits=5, epochs=100)

    print("\n[SUCCESS] Hybrid model training completed!")
    print(f"\nFinal Results:")
    print(f"  MAE: {results['mean_mae']:.4f} +/- {results['std_mae']:.4f}")
    print(f"  RMSE: {results['mean_rmse']:.4f}")
    print(f"  Direction Accuracy: {results['mean_direction_acc']:.2f}%")

    # Compare to individual systems
    print("\n" + "="*70)
    print("  COMPARISON TO INDIVIDUAL SYSTEMS")
    print("="*70)
    print(f"  Team-based MAE: ~11.99 (test set)")
    print(f"  Player-based MAE: ~12.26 (test set)")
    print(f"  Hybrid MAE: {results['mean_mae']:.4f} (cross-validation)")

    if results['mean_mae'] < 11.0:
        improvement = ((11.99 - results['mean_mae']) / 11.99) * 100
        print(f"\n  HYBRID WINS! {improvement:.1f}% better than team-based")
    print("="*70)


if __name__ == "__main__":
    main()
