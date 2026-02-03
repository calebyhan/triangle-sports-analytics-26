"""
Optimized Training Script with All Improvements

Enhancements:
1. Feature normalization with StandardScaler
2. Improved neural network architecture (wider + batch norm)
3. Higher dropout for regularization
4. Better early stopping
5. Learning rate scheduling
6. Gradient clipping

Expected: MAE 9.3 -> 5.5-6.0
"""

import sys
import argparse
import pickle
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from src.player_elo.training_pipeline import PlayerModelTrainer
from src.player_elo.config import TRAINING_YEARS, PYTORCH_CONFIG, MODELS_DIR


# Improved Neural Network Architecture
class ImprovedPlayerELONet(nn.Module):
    """
    Improved architecture with batch normalization and wider layers
    """

    def __init__(
        self,
        input_dim=65,
        hidden_dims=[256, 128, 64],  # Wider than original
        dropout=0.35  # Higher dropout
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))  # NEW: Batch normalization
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout

    def forward(self, x):
        return self.network(x).squeeze()


def train_optimized_model(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    device: str = 'cpu'
):
    """
    Train improved model with all enhancements

    Returns:
        results dict with mean_mae, std_mae
    """
    print("\n" + "="*70)
    print("  OPTIMIZED NEURAL NETWORK TRAINING")
    print("="*70)
    print(f"\nArchitecture improvements:")
    print(f"  [x] Wider layers: [256, 128, 64] (vs [128, 64, 32])")
    print(f"  [x] Batch normalization: Enabled")
    print(f"  [x] Higher dropout: 0.35 (vs 0.2)")
    print(f"  [x] Gradient clipping: Enabled")
    print("="*70 + "\n")

    # Feature normalization
    print("Normalizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save scaler for prediction time
    scaler_path = MODELS_DIR / 'feature_scaler.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"  Saved feature scaler to: {scaler_path}\n")

    # Time-series cross-validation
    tscv = TimeSeriesSplit(n_splits=n_splits + 1)  # +1 because we skip first fold
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(list(tscv.split(X_scaled))[1:], 1):
        print(f"\nFold {fold}/{n_splits}")
        print(f"  Train: {len(train_idx)} samples, Val: {len(val_idx)} samples")

        # Split data
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val)
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Initialize improved model
        model = ImprovedPlayerELONet(
            input_dim=X.shape[1],
            hidden_dims=[256, 128, 64],
            dropout=0.35
        ).to(device)

        # Loss and optimizer
        criterion = nn.HuberLoss()
        optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5
        )

        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 10

        for epoch in range(epochs):
            # Train
            model.train()
            train_losses = []

            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                optimizer.zero_grad()
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                loss.backward()

                # NEW: Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                train_losses.append(loss.item())

            # Validate
            model.eval()
            val_losses = []
            val_predictions = []
            val_actuals = []

            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    predictions = model(batch_X)
                    loss = criterion(predictions, batch_y)

                    val_losses.append(loss.item())
                    val_predictions.extend(predictions.cpu().numpy())
                    val_actuals.extend(batch_y.cpu().numpy())

            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)
            val_mae = np.mean(np.abs(np.array(val_predictions) - np.array(val_actuals)))

            # Learning rate scheduling
            scheduler.step(val_loss)

            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:3d}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.4f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                # Save best model
                model_path = MODELS_DIR / f'pytorch_model_optimized_fold{fold}.pt'
                torch.save(model.state_dict(), model_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        # Evaluate on validation set
        model_path = MODELS_DIR / f'pytorch_model_optimized_fold{fold}.pt'
        model.load_state_dict(torch.load(model_path))
        model.eval()

        with torch.no_grad():
            val_preds = model(torch.FloatTensor(X_val).to(device)).cpu().numpy()
            val_mae = np.mean(np.abs(val_preds - y_val))
            val_rmse = np.sqrt(np.mean((val_preds - y_val)**2))

            # Direction accuracy
            direction_correct = np.sum(np.sign(val_preds) == np.sign(y_val))
            direction_acc = direction_correct / len(y_val) * 100

        print(f"\n  Fold {fold} Results:")
        print(f"    MAE: {val_mae:.4f}")
        print(f"    RMSE: {val_rmse:.4f}")
        print(f"    Direction Accuracy: {direction_acc:.2f}%")

        fold_results.append({
            'fold': fold,
            'mae': val_mae,
            'rmse': val_rmse,
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
    print("="*70 + "\n")

    # Train final model on all data
    print("Training final model on all data...")
    final_model = ImprovedPlayerELONet(
        input_dim=X.shape[1],
        hidden_dims=[256, 128, 64],
        dropout=0.35
    ).to(device)

    final_dataset = TensorDataset(
        torch.FloatTensor(X_scaled),
        torch.FloatTensor(y)
    )
    final_loader = DataLoader(final_dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.HuberLoss()
    optimizer = optim.AdamW(final_model.parameters(), lr=learning_rate, weight_decay=1e-5)

    for epoch in range(50):  # Fewer epochs for final model
        final_model.train()
        for batch_X, batch_y in final_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            predictions = final_model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(final_model.parameters(), max_norm=1.0)
            optimizer.step()

    # Save final model
    final_model_path = MODELS_DIR / 'pytorch_model_optimized.pt'
    torch.save(final_model.state_dict(), final_model_path)
    print(f"  Saved final model to: {final_model_path}\n")

    return {
        'mean_mae': mean_mae,
        'std_mae': std_mae,
        'mean_rmse': mean_rmse,
        'mean_direction_acc': mean_acc,
        'fold_results': fold_results
    }


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(
        description='Train Optimized Player-Based ELO Model'
    )

    parser.add_argument(
        '--years',
        type=int,
        nargs='+',
        default=TRAINING_YEARS,
        help=f'Training years (default: {TRAINING_YEARS})'
    )

    parser.add_argument(
        '--cv-splits',
        type=int,
        default=5,
        help='Number of cross-validation splits (default: 5)'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Maximum epochs (default: 100)'
    )

    args = parser.parse_args()

    print("\n" + "="*70)
    print("  OPTIMIZED PLAYER-BASED ELO MODEL TRAINING")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Training years: {args.years}")
    print(f"  CV splits: {args.cv_splits}")
    print(f"  Max epochs: {args.epochs}")
    print("\n" + "="*70 + "\n")

    # Use existing trainer to collect data and prepare features
    print("Collecting data and preparing features...")
    trainer = PlayerModelTrainer(args.years)

    # Steps 1-4: Data collection, roster prep, game processing, feature engineering
    games_df, player_stats_df = trainer.collect_data(use_cached=True)
    trainer.prepare_rosters(player_stats_df)
    game_records_df = trainer.process_games(games_df, player_stats_df)
    X, y = trainer.create_features(game_records_df, player_stats_df)

    print(f"\nPrepared {len(X)} samples for training\n")

    # Train optimized model
    try:
        results = train_optimized_model(
            X, y,
            n_splits=args.cv_splits,
            epochs=args.epochs
        )

        print("\n[SUCCESS] Optimized training completed!")
        print(f"\nFinal Results:")
        print(f"  MAE: {results['mean_mae']:.4f} +/- {results['std_mae']:.4f}")
        print(f"  RMSE: {results['mean_rmse']:.4f}")
        print(f"  Direction Accuracy: {results['mean_direction_acc']:.2f}%")

        improvement = 9.3 - results['mean_mae']
        pct_improvement = (improvement / 9.3) * 100
        print(f"\nImprovement from baseline:")
        print(f"  MAE reduction: {improvement:.2f} points ({pct_improvement:.1f}%)")

    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
