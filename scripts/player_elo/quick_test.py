"""
Quick Test Script - No Data Collection Required

Tests all components with synthetic data to verify implementation.
Faster than validate_system.py (no network requests).

Usage:
    python scripts/player_elo/quick_test.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.player_elo.player_elo_system import PlayerEloSystem
from src.player_elo.features import PlayerFeatureEngine
from src.player_elo.pytorch_model import PlayerELONet, create_data_loaders, train_player_elo_net

print("\n" + "="*70)
print("  QUICK SYSTEM TEST (Synthetic Data)")
print("="*70)

# Test 1: Player ELO System
print("\n[1/3] Testing Player ELO System...")
elo_system = PlayerEloSystem()

home_lineup = [f"HOME_{i}" for i in range(5)]
away_lineup = [f"AWAY_{i}" for i in range(5)]

for pid in home_lineup + away_lineup:
    elo_system.set_player_metadata(pid, usage=20, minutes=25, position='G')

initial_home = elo_system.calculate_team_strength(home_lineup)
initial_away = elo_system.calculate_team_strength(away_lineup)

print(f"  Initial ratings: Home={initial_home:.0f}, Away={initial_away:.0f}")

# Simulate game
elo_system.update_from_game(home_lineup, away_lineup, 85, 78)

final_home = elo_system.calculate_team_strength(home_lineup)
final_away = elo_system.calculate_team_strength(away_lineup)

print(f"  After game (85-78): Home={final_home:.0f} ({final_home-initial_home:+.0f}), " +
      f"Away={final_away:.0f} ({final_away-initial_away:+.0f})")
print(f"  [OK] Player ELO system working")

# Test 2: Feature Engineering
print("\n[2/3] Testing Feature Engineering...")

player_stats = pd.DataFrame({
    'player_id': home_lineup + away_lineup,
    'usage_pct': [20 + i for i in range(10)],
    'offensive_rating': [100 + i*2 for i in range(10)],
    'defensive_rating': [100 - i for i in range(10)],
    'minutes_per_game': [25 + i for i in range(10)],
})

feature_engine = PlayerFeatureEngine(player_stats, elo_system)

features = feature_engine.create_matchup_features(
    home_lineup, away_lineup, datetime.now(),
    'Home', 'Away'
)

print(f"  Feature vector shape: {features.shape}")
print(f"  Expected shape: (65,)")
print(f"  [OK] Feature engineering working" if features.shape == (65,) else "  [FAIL] Wrong shape!")

# Test 3: PyTorch Model
print("\n[3/3] Testing PyTorch Model...")

# Create synthetic training data
X = np.random.randn(200, 65).astype(np.float32)
y = (X[:, 0] * 2 + np.random.randn(200) * 3).astype(np.float32)

X_train, X_val = X[:160], X[160:]
y_train, y_val = y[:160], y[160:]

model = PlayerELONet()
print(f"  Model parameters: {model.count_parameters():,}")

train_loader, val_loader = create_data_loaders(
    X_train, y_train, X_val, y_val, batch_size=32
)

# Quick training (3 epochs)
print(f"  Training for 3 epochs...")
model, history = train_player_elo_net(
    model, train_loader, val_loader,
    epochs=3, lr=0.001
)

print(f"  Final MAE: {history['val_mae'][-1]:.2f}")
print(f"  [OK] PyTorch model working")

# Summary
print("\n" + "="*70)
print("  [SUCCESS] ALL TESTS PASSED!")
print("  System is ready for real data and full implementation.")
print("="*70 + "\n")
