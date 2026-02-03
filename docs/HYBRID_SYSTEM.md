# Hybrid Player-Team Prediction System

## Overview

The hybrid system combines the best of both worlds:
- **Player-level skill** (from player ELO ratings)
- **Team-level synergy** (from team statistics)
- **Learned correlations** (via neural network)

This creates a prediction system that captures individual talent while also accounting for coaching, team chemistry, and system effects.

## How It Works

### 1. Feature Engineering (38 Features)

The system creates a comprehensive feature vector for each game combining:

#### Player ELO Aggregates (10 features)
- **Mean ELO**: Average skill level of the lineup
- **Max ELO**: Star player (best player ELO)
- **Min ELO**: Bench depth (weakest rotation player)
- **Std ELO**: Team consistency
- **Weighted ELO**: Usage-weighted team strength

For both home and away teams (5 × 2 = 10 features)

#### Team Metrics (10 features)
- **Team ELO**: Traditional team-based rating
- **Offensive Efficiency**: Points per 100 possessions
- **Defensive Efficiency**: Points allowed per 100 possessions
- **Pace**: Possessions per game
- **Power Rating**: Overall team strength

For both home and away teams (5 × 2 = 10 features)

#### Player-Team Interactions (8 features)
- **ELO Residual**: Team ELO - Player ELO (coaching effect)
- **Chemistry**: How much better the team is than individual parts
- **Depth**: Min ELO / Max ELO ratio (rotation quality)
- **Star Power**: Max ELO - Mean ELO (star dependency)

For both home and away teams (4 × 2 = 8 features)

#### Matchup Features (10 features)
- **Team ELO Difference**: Traditional team matchup
- **Player ELO Difference**: Player-based matchup
- **Offense vs Defense**: Home OFF - Away DEF
- **Defense vs Offense**: Home DEF - Away OFF
- **Pace Difference**: Tempo matchup
- **Power Difference**: Overall strength gap
- **Consistency Difference**: Variance in performance
- **Chemistry Difference**: Synergy comparison
- **Depth Difference**: Rotation quality gap
- **Star Difference**: Star player comparison

### 2. Neural Network Architecture

```
Input (38 features)
  ├─ Linear(38 → 128)
  ├─ BatchNorm1d(128)
  ├─ ReLU
  ├─ Dropout(0.3)
  ├─ Linear(128 → 64)
  ├─ BatchNorm1d(64)
  ├─ ReLU
  ├─ Dropout(0.3)
  ├─ Linear(64 → 32)
  ├─ BatchNorm1d(32)
  ├─ ReLU
  ├─ Dropout(0.3)
  └─ Linear(32 → 1) → Point Spread
```

The network learns to:
- Weight player contributions vs team effects
- Identify favorable matchups
- Capture non-linear interactions
- Balance star power vs depth

### 3. Training Process

**Data**: 705 ACC conference games (2020-2025 seasons)

**Cross-Validation**: 5-fold time-series split
- Respects temporal ordering (no data leakage)
- Each fold trained on earlier games, validated on later games

**Optimization**:
- Loss: Huber Loss (robust to outliers/blowouts)
- Optimizer: AdamW (lr=0.001, weight_decay=1e-5)
- Scheduler: ReduceLROnPlateau (adaptive learning rate)
- Early Stopping: Patience=10 epochs
- Gradient Clipping: Max norm=1.0

**Regularization**:
- Batch normalization (reduces internal covariate shift)
- Dropout (0.3 rate prevents overfitting)
- L2 weight decay (encourages smaller weights)

## Key Innovations

### 1. Player-Team Correlation Learning

The hybrid system learns how aggregated player ELOs map to team performance:

```
Player ELOs → Team ELO + Residual
```

The **residual** captures:
- Coaching effects (good coach = positive residual)
- Team chemistry (whole > sum of parts)
- System fit (players thriving in certain systems)
- Home court advantage utilization

### 2. Matchup-Specific Adjustments

The network learns opponent-specific effects:
- **Pace matching**: Fast team vs slow team dynamics
- **Style matchups**: Offensive powerhouse vs defensive specialist
- **Star vs Depth**: Star-heavy team vs balanced team
- **Consistency**: Volatile team vs consistent team

### 3. Dual Information Sources

By combining player and team data:
- **Handles roster changes**: New players immediately affect predictions
- **Captures team effects**: Coaching, chemistry beyond individual skill
- **Robust to missing data**: Can fall back to team ELO if player data incomplete

## Usage

### Training
```bash
python scripts/train_hybrid_model.py
```

Outputs:
- `data/player_data/models/hybrid_model.pt` - Trained model
- `data/player_data/models/hybrid_model_fold{1-5}.pt` - CV models
- `data/player_data/models/hybrid_scaler.pkl` - Feature scaler

### Prediction
```bash
python scripts/generate_hybrid_predictions.py
```

Outputs:
- `data/predictions/tsa_pt_spread_HYBRID_2026.csv` - Competition submission
- `data/predictions/tsa_pt_spread_HYBRID_2026_detailed.csv` - With metadata

## Expected Performance

**Hypothesis**: The hybrid system should outperform both individual systems because:

1. **Better than team-only** (MAE ~12.0):
   - Captures roster changes and injuries
   - Player-level granularity for close matchups
   - Individual performance trends

2. **Better than player-only** (MAE ~12.3):
   - Leverages team statistics (coaching, system)
   - Less sensitive to player stat noise
   - Captures team chemistry effects

**Target MAE**: <10.0 points (improvement over both systems)

## Theoretical Advantages

### vs Team-Based System
| Aspect | Team-Based | Hybrid |
|--------|-----------|--------|
| Roster changes | Lags (ELO adjusts slowly) | Immediate (new player ELOs) |
| Injuries | Can't detect | Missing star = lower max ELO |
| Transfers | Invisible until games played | Portal player ELO transfers |
| Depth | Not captured | Min/Max ELO ratio |

### vs Player-Based System
| Aspect | Player-Based | Hybrid |
|--------|-------------|--------|
| Coaching | Not modeled | Team ELO residual |
| Chemistry | Implicit in outcomes | Explicit interaction features |
| System fit | Not considered | Learned from team metrics |
| Data noise | High (small samples) | Smoothed by team stats |

## Feature Importance

The neural network implicitly learns feature importance. Expected key features:

1. **Team ELO difference** - Strongest baseline signal
2. **Weighted player ELO** - Individual skill aggregate
3. **Chemistry metrics** - Team > parts synergy
4. **Star power** - Impact of best player
5. **Matchup styles** - Offense vs Defense, Pace

## Interpretability

While the neural network is a "black box," the features are interpretable:

- **High chemistry** = Good coaching / Team culture
- **Large ELO residual** = System effects dominate
- **High star power** = Star-dependent team
- **Low depth** = Weak bench / injury risk

## Future Enhancements

Potential improvements:
1. **Lineup diversity**: Model multiple possible lineups probabilistically
2. **Temporal features**: Recent form, fatigue, schedule density
3. **Venue effects**: Specific arena advantages
4. **Historical matchups**: Head-to-head history features
5. **Momentum**: Win/loss streak indicators
6. **Attention mechanism**: Learn which players matter most per matchup

## Comparison Framework

To evaluate the hybrid system:

```python
# Test on held-out 2024-25 games
results = {
    'Team-based': 11.99 MAE,
    'Player-based': 12.26 MAE,
    'Hybrid': ??? MAE  # Target: <10.0
}
```

Success criteria:
- **MAE improvement**: >10% better than team-based
- **Direction accuracy**: >60%
- **Consistency**: Low std across CV folds

---

**Status**: 🚧 Training in progress
**Architecture**: 38 → [128, 64, 32] → 1
**Training data**: 705 ACC games (2020-2025)
**Expected training time**: ~10-15 minutes
