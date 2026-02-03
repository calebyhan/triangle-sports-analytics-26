"""
PyTorch Neural Network for Player-Based Spread Prediction

Architecture:
- Input: 65D feature vector
- Hidden layers: [128, 64, 32] with ReLU + Dropout
- Output: 1D (point spread prediction)

Loss: Huber Loss (robust to outliers/blowouts)
Optimizer: AdamW with weight decay (L2 regularization)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, List, Dict
import logging

from .config import PYTORCH_CONFIG, VALIDATION_CONFIG

# Set up logger
logger = logging.getLogger(__name__)


# ============================================================================
# PYTORCH DATASET
# ============================================================================

class PlayerELODataset(Dataset):
    """
    PyTorch Dataset for player ELO features
    """

    def __init__(self, features: np.ndarray, targets: np.ndarray):
        """
        Initialize dataset

        Args:
            features: Feature matrix (n_samples, n_features)
            targets: Target vector (n_samples,)
        """
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(targets, dtype=torch.float32)

        logger.info(f"Dataset created: {len(self)} samples, {self.X.shape[1]} features")

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


# ============================================================================
# NEURAL NETWORK ARCHITECTURE
# ============================================================================

class PlayerELONet(nn.Module):
    """
    Neural network for predicting point spreads from player ELOs

    Architecture:
    - Input layer: 65 features
    - Hidden layer 1: 128 neurons (ReLU + Dropout)
    - Hidden layer 2: 64 neurons (ReLU + Dropout)
    - Hidden layer 3: 32 neurons (ReLU + Dropout)
    - Output layer: 1 neuron (point spread, no activation)
    """

    def __init__(
        self,
        input_dim: int = None,
        hidden_dims: List[int] = None,
        dropout: float = None
    ):
        """
        Initialize neural network

        Args:
            input_dim: Input feature dimension (defaults to config)
            hidden_dims: List of hidden layer sizes (defaults to config)
            dropout: Dropout probability (defaults to config)
        """
        super(PlayerELONet, self).__init__()

        # Use config defaults if not provided
        self.input_dim = input_dim or PYTORCH_CONFIG['input_dim']
        self.hidden_dims = hidden_dims or PYTORCH_CONFIG['hidden_dims']
        self.dropout = dropout or PYTORCH_CONFIG['dropout']

        # Build network layers
        layers = []
        prev_dim = self.input_dim

        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout))
            prev_dim = hidden_dim

        # Output layer (no activation for regression)
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._initialize_weights()

        logger.info(f"PlayerELONet initialized: {self.input_dim} → {self.hidden_dims} → 1")
        logger.info(f"  Total parameters: {self.count_parameters():,}")

    def _initialize_weights(self):
        """Xavier initialization for better convergence"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.network(x)

    def count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def create_data_loaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Create PyTorch data loaders

    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        batch_size: Batch size (defaults to config)

    Returns:
        (train_loader, val_loader)
    """
    if batch_size is None:
        batch_size = PYTORCH_CONFIG['batch_size']

    train_dataset = PlayerELODataset(X_train, y_train)
    val_dataset = PlayerELODataset(X_val, y_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle training data
        num_workers=0  # Windows compatibility
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle validation
        num_workers=0
    )

    return train_loader, val_loader


def train_player_elo_net(
    model: PlayerELONet,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = None,
    lr: float = None,
    device: str = None,
    save_path: Optional[Path] = None
) -> Tuple[PlayerELONet, Dict]:
    """
    Train PyTorch model

    Args:
        model: PlayerELONet model
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of epochs (defaults to config)
        lr: Learning rate (defaults to config)
        device: Device ('cuda' or 'cpu', defaults to auto)
        save_path: Path to save best model

    Returns:
        (trained_model, training_history)
    """
    # Use config defaults
    if epochs is None:
        epochs = PYTORCH_CONFIG['epochs']
    if lr is None:
        lr = PYTORCH_CONFIG['learning_rate']

    # Determine device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = model.to(device)
    logger.info(f"Training on device: {device}")

    # Loss function: Huber Loss (robust to outliers)
    criterion = nn.HuberLoss(delta=PYTORCH_CONFIG['huber_delta'])

    # Optimizer: AdamW with weight decay
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=PYTORCH_CONFIG['weight_decay']
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=PYTORCH_CONFIG['lr_scheduler_factor'],
        patience=PYTORCH_CONFIG['lr_scheduler_patience']
    )

    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    patience = PYTORCH_CONFIG['early_stopping_patience']

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mae': [],
        'learning_rate': []
    }

    logger.info(f"Starting training: {epochs} epochs, lr={lr}")

    for epoch in range(epochs):
        # ====================================================================
        # TRAINING PHASE
        # ====================================================================
        model.train()
        train_loss = 0.0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ====================================================================
        # VALIDATION PHASE
        # ====================================================================
        model.eval()
        val_loss = 0.0
        val_mae = 0.0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)

                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                mae = torch.abs(outputs.squeeze() - batch_y).mean()

                val_loss += loss.item()
                val_mae += mae.item()

        val_loss /= len(val_loader)
        val_mae /= len(val_loader)

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])

        # ====================================================================
        # LOGGING & EARLY STOPPING
        # ====================================================================
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val MAE: {val_mae:.4f}"
            )

        # Early stopping check
        if val_loss < best_val_loss - PYTORCH_CONFIG['min_delta']:
            best_val_loss = val_loss
            patience_counter = 0

            # Save best model
            if save_path:
                torch.save(model.state_dict(), save_path)
                logger.info(f"  ✓ Saved best model (val_loss={val_loss:.4f})")

        else:
            patience_counter += 1

            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    # Load best model if available
    if save_path and save_path.exists():
        model.load_state_dict(torch.load(save_path))
        logger.info(f"Loaded best model from: {save_path}")

    logger.info(f"Training complete. Best val loss: {best_val_loss:.4f}")

    return model, history


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def evaluate_model(
    model: PlayerELONet,
    data_loader: DataLoader,
    device: str = None
) -> Dict[str, float]:
    """
    Evaluate model performance

    Args:
        model: Trained model
        data_loader: Data loader for evaluation
        device: Device ('cuda' or 'cpu')

    Returns:
        Dictionary with metrics
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = model.to(device)
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_X)
            all_preds.extend(outputs.squeeze().cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())

    preds = np.array(all_preds)
    targets = np.array(all_targets)

    # Calculate metrics
    mae = np.abs(preds - targets).mean()
    rmse = np.sqrt(((preds - targets) ** 2).mean())
    direction_accuracy = ((preds > 0) == (targets > 0)).mean()

    metrics = {
        'mae': mae,
        'rmse': rmse,
        'direction_accuracy': direction_accuracy,
        'n_samples': len(preds)
    }

    return metrics


def predict(
    model: PlayerELONet,
    X: np.ndarray,
    device: str = None
) -> np.ndarray:
    """
    Make predictions with trained model

    Args:
        model: Trained model
        X: Feature matrix (n_samples, n_features)
        device: Device ('cuda' or 'cpu')

    Returns:
        Predictions array (n_samples,)
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = model.to(device)
    model.eval()

    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

    with torch.no_grad():
        predictions = model(X_tensor).squeeze().cpu().numpy()

    return predictions


# ============================================================================
# STANDALONE TESTING
# ============================================================================

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    print("\n" + "="*60)
    print("PYTORCH MODEL TEST")
    print("="*60)

    # Create synthetic data
    n_samples = 1000
    n_features = 65

    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.random.randn(n_samples).astype(np.float32) * 10  # Simulate spreads

    # Split data
    split_idx = int(0.8 * n_samples)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    print(f"\nData shapes:")
    print(f"  Train: X={X_train.shape}, y={y_train.shape}")
    print(f"  Val: X={X_val.shape}, y={y_val.shape}")

    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        X_train, y_train, X_val, y_val, batch_size=32
    )

    print(f"\nData loaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")

    # Create model
    model = PlayerELONet()

    print(f"\nModel architecture:")
    print(model)

    # Test forward pass
    sample_batch = next(iter(train_loader))
    sample_X, sample_y = sample_batch
    output = model(sample_X)

    print(f"\nForward pass test:")
    print(f"  Input shape: {sample_X.shape}")
    print(f"  Output shape: {output.shape}")

    # Quick training test (2 epochs)
    print(f"\nQuick training test (2 epochs)...")
    model, history = train_player_elo_net(
        model, train_loader, val_loader,
        epochs=2, lr=0.001
    )

    print(f"\nTraining history:")
    print(f"  Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"  Final val loss: {history['val_loss'][-1]:.4f}")
    print(f"  Final val MAE: {history['val_mae'][-1]:.4f}")

    # Evaluation
    metrics = evaluate_model(model, val_loader)

    print(f"\nFinal evaluation:")
    print(f"  MAE: {metrics['mae']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  Direction accuracy: {metrics['direction_accuracy']:.2%}")

    print("\n✓ PyTorch model test passed!")
    print("="*60)
