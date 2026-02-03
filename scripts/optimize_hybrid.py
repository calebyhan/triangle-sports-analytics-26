"""
Hyperparameter Optimization for Hybrid Model

Performs grid search to find optimal architecture and training parameters
while preventing overfitting through rigorous cross-validation.
"""

import sys
from pathlib import Path

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
from itertools import product

from src.hybrid_model import HybridNet, HybridFeatureEngine
from src.player_elo.player_elo_system import PlayerEloSystem
from src.player_elo.player_data_collector import PlayerDataCollector
from src.elo import EloRatingSystem


class GameDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


def train_with_config(X, y, config, n_splits=5):
    """
    Train model with specific hyperparameter configuration

    Returns:
        mean_mae, std_mae
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Feature normalization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Cross-validation
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_maes = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_scaled), 1):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        train_dataset = GameDataset(X_train, y_train)
        val_dataset = GameDataset(X_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])

        # Create model
        model = HybridNet(
            input_dim=X.shape[1],
            hidden_dims=config['hidden_dims']
        )
        model = model.to(device)

        # Optimizer with weight decay
        criterion = nn.HuberLoss(delta=config['huber_delta'])
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay']
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=config['scheduler_patience'],
            factor=config['scheduler_factor']
        )

        # Training
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(config['max_epochs']):
            # Train
            model.train()
            for features, targets in train_loader:
                features, targets = features.to(device), targets.to(device)
                optimizer.zero_grad()
                predictions = model(features)
                loss = criterion(predictions, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['grad_clip'])
                optimizer.step()

            # Validate
            model.eval()
            val_loss = 0.0
            val_predictions = []
            val_targets = []

            with torch.no_grad():
                for features, targets in val_loader:
                    features, targets = features.to(device), targets.to(device)
                    predictions = model(features)
                    loss = criterion(predictions, targets)
                    val_loss += loss.item()
                    val_predictions.extend(predictions.cpu().numpy())
                    val_targets.extend(targets.cpu().numpy())

            val_loss /= len(val_loader)
            scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config['early_stop_patience']:
                    break

        # Final MAE
        val_predictions = np.array(val_predictions)
        val_targets = np.array(val_targets)
        mae = np.mean(np.abs(val_predictions - val_targets))
        fold_maes.append(mae)

    return np.mean(fold_maes), np.std(fold_maes)


def hyperparameter_search(X, y):
    """
    Grid search over hyperparameter space
    """
    print("\n" + "="*70)
    print("  HYPERPARAMETER OPTIMIZATION")
    print("="*70)

    # Define search space
    search_space = {
        'hidden_dims': [
            [96, 48, 24],      # Smaller (less overfitting)
            [128, 64, 32],     # Baseline
            [160, 80, 40],     # Larger
            [128, 64, 32, 16]  # Deeper
        ],
        'lr': [0.0005, 0.001, 0.002],
        'weight_decay': [1e-4, 1e-5, 1e-6],
        'batch_size': [32, 64],
        'huber_delta': [1.0, 1.5],
        'grad_clip': [1.0],
        'scheduler_patience': [5],
        'scheduler_factor': [0.5],
        'early_stop_patience': [15],
        'max_epochs': [100]
    }

    # Generate all combinations
    keys = list(search_space.keys())
    values = list(search_space.values())

    all_configs = []
    for combo in product(*values):
        config = dict(zip(keys, combo))
        all_configs.append(config)

    print(f"\nTotal configurations to test: {len(all_configs)}")
    print(f"This will take approximately {len(all_configs) * 2} minutes")

    # Test each configuration
    results = []

    for i, config in enumerate(all_configs, 1):
        print(f"\n[{i}/{len(all_configs)}] Testing configuration:")
        print(f"  Architecture: {config['hidden_dims']}")
        print(f"  LR: {config['lr']}, Weight Decay: {config['weight_decay']}")
        print(f"  Batch Size: {config['batch_size']}, Huber Delta: {config['huber_delta']}")

        try:
            mean_mae, std_mae = train_with_config(X, y, config, n_splits=5)

            results.append({
                'config': config,
                'mean_mae': mean_mae,
                'std_mae': std_mae,
                'score': mean_mae + 0.5 * std_mae  # Penalize high variance
            })

            print(f"  Result: MAE = {mean_mae:.4f} +/- {std_mae:.4f}")

        except Exception as e:
            print(f"  [ERROR] Configuration failed: {e}")
            continue

    # Sort by score
    results.sort(key=lambda x: x['score'])

    print("\n" + "="*70)
    print("  TOP 5 CONFIGURATIONS")
    print("="*70)

    for i, result in enumerate(results[:5], 1):
        config = result['config']
        print(f"\n[{i}] MAE: {result['mean_mae']:.4f} +/- {result['std_mae']:.4f}")
        print(f"    Architecture: {config['hidden_dims']}")
        print(f"    LR: {config['lr']}, WD: {config['weight_decay']}")
        print(f"    Batch: {config['batch_size']}, Huber: {config['huber_delta']}")

    # Save results
    results_df = pd.DataFrame([
        {
            'rank': i,
            'mae': r['mean_mae'],
            'std': r['std_mae'],
            'hidden_dims': str(r['config']['hidden_dims']),
            'lr': r['config']['lr'],
            'weight_decay': r['config']['weight_decay'],
            'batch_size': r['config']['batch_size'],
            'huber_delta': r['config']['huber_delta']
        }
        for i, r in enumerate(results, 1)
    ])

    results_file = project_root / 'data' / 'player_data' / 'models' / 'hyperparameter_results.csv'
    results_df.to_csv(results_file, index=False)
    print(f"\n\nResults saved to: {results_file}")

    return results[0]['config'], results[0]['mean_mae']


def train_final_optimized_model(X, y, best_config):
    """Train final model with best configuration"""
    print("\n" + "="*70)
    print("  TRAINING FINAL OPTIMIZED MODEL")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save scaler
    scaler_path = project_root / 'data' / 'player_data' / 'models' / 'hybrid_scaler_optimized.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    # Create dataset
    dataset = GameDataset(X_scaled, y)
    loader = DataLoader(dataset, batch_size=best_config['batch_size'], shuffle=True)

    # Create model
    model = HybridNet(
        input_dim=X.shape[1],
        hidden_dims=best_config['hidden_dims']
    )
    model = model.to(device)

    # Optimizer
    criterion = nn.HuberLoss(delta=best_config['huber_delta'])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=best_config['lr'],
        weight_decay=best_config['weight_decay']
    )

    # Train
    for epoch in range(50):
        model.train()
        total_loss = 0.0

        for features, targets in loader:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad()
            predictions = model(features)
            loss = criterion(predictions, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=best_config['grad_clip'])
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/50 | Loss: {total_loss/len(loader):.4f}")

    # Save
    model_path = project_root / 'data' / 'player_data' / 'models' / 'hybrid_model_optimized.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': best_config,
        'input_dim': X.shape[1],
        'hidden_dims': best_config['hidden_dims']
    }, model_path)

    print(f"\n  Saved optimized model to: {model_path}")

    return model_path


def main():
    """Main execution"""
    print("\n" + "="*70)
    print("  HYBRID MODEL OPTIMIZATION")
    print("="*70)
    print("\nObjective: Minimize MAE without overfitting")
    print("Strategy:")
    print("  1. Hyperparameter grid search")
    print("  2. 5-fold time-series cross-validation")
    print("  3. Penalize high variance (overfitting)")
    print("  4. Train final model with best config")

    # Load training data (reuse from train_hybrid_model.py)
    print("\nLoading training data...")

    # For speed, load preprocessed data if available
    cache_file = project_root / 'data' / 'player_data' / 'models' / 'training_cache.pkl'

    if cache_file.exists():
        print("  Loading from cache...")
        with open(cache_file, 'rb') as f:
            cache = pickle.load(f)
        X = cache['X']
        y = cache['y']
        print(f"  Loaded {len(X)} samples with {X.shape[1]} features")
    else:
        print("  [ERROR] Training cache not found")
        print("  Run train_hybrid_model.py first to create cache")
        sys.exit(1)

    # Hyperparameter search
    best_config, best_mae = hyperparameter_search(X, y)

    print("\n" + "="*70)
    print("  BEST CONFIGURATION")
    print("="*70)
    print(f"  MAE: {best_mae:.4f}")
    print(f"  Architecture: {best_config['hidden_dims']}")
    print(f"  Learning rate: {best_config['lr']}")
    print(f"  Weight decay: {best_config['weight_decay']}")
    print(f"  Batch size: {best_config['batch_size']}")
    print(f"  Huber delta: {best_config['huber_delta']}")

    # Train final model
    model_path = train_final_optimized_model(X, y, best_config)

    print("\n[SUCCESS] Optimization complete!")
    print(f"Optimized model: {model_path}")
    print(f"Expected MAE: {best_mae:.4f}")


if __name__ == "__main__":
    main()
