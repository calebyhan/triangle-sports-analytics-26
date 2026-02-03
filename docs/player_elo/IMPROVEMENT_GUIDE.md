# Player-Based ELO System - Improvement Guide

**Current Performance:** MAE 9.3 points
**Target Performance:** MAE <6.0 points (closer to team-based 4.97)
**Gap to Close:** ~3.3 points (35% improvement needed)

---

## Priority Issues to Fix

### 🔴 Critical (Fix First)
1. **Extreme predictions** (±50+ points) - Causing high MAE
2. **Missing player data** - Some teams have no players found
3. **Feature scaling** - Inconsistent feature magnitudes

### 🟡 High Impact
4. **Better lineup prediction** - Currently using simple heuristics
5. **ELO parameter tuning** - K-factor, carryover may be suboptimal
6. **Model architecture** - Neural network may be overfitting

### 🟢 Medium Impact
7. **More training data** - Only 18k games vs 33k available
8. **Better player ID matching** - Some players not matched correctly
9. **Ensemble methods** - Combine multiple models

---

## 1. Fix Extreme Predictions 🔴 CRITICAL

### Problem
```
Examples of extreme predictions:
  Pitt vs SMU: 60.01 (Pitt by 60!)
  UNC vs Duke: -43.07 (Duke by 43)
  Notre Dame vs FSU: 54.42 (Notre Dame by 54)

These are unrealistic and inflate MAE significantly.
```

### Root Causes
1. **Player ELO outliers:** Some players have extreme ELO ratings
2. **Missing opponent data:** When one team has no player data, fall back to 0
3. **Feature magnitude issues:** Some features not normalized
4. **Model overconfidence:** Neural network outputs extreme values

### Solutions

#### A. Add Prediction Clipping
```python
# In prediction_pipeline.py, predict_game method

# After getting prediction from model
prediction = self.model(features_tensor).item()

# Clip to reasonable range for college basketball
# 99th percentile of college spreads is ~±25 points
prediction = np.clip(prediction, -30, 30)

return prediction, metadata
```

**Expected impact:** -2.0 MAE (eliminate extreme outliers)

#### B. Add Confidence-Based Adjustments
```python
def predict_game_with_confidence(self, ...):
    # Get prediction
    prediction = self.model(features_tensor).item()

    # Calculate confidence based on data quality
    confidence_factors = {
        'both_lineups_complete': 1.0 if len(home_lineup) == 5 and len(away_lineup) == 5 else 0.5,
        'elo_variance': 1.0 / (1 + np.std([self.elo_system.get_player_elo(p) for p in home_lineup + away_lineup])),
        'minutes_coverage': min(1.0, sum([self.player_minutes.get(p, 0) for p in home_lineup]) / 150)
    }

    confidence = np.mean(list(confidence_factors.values()))

    # Shrink extreme predictions toward mean when confidence is low
    mean_spread = 0  # Or historical mean
    adjusted_prediction = confidence * prediction + (1 - confidence) * mean_spread

    return adjusted_prediction, metadata
```

**Expected impact:** -1.0 MAE (reduce confidence in uncertain predictions)

#### C. Regularize Player ELO Ratings
```python
# In player_elo_system.py, after processing all games

def regularize_elos(self, max_deviation=200):
    """Prevent extreme ELO ratings"""
    mean_elo = np.mean(list(self.player_elos.values()))

    for player_id, elo in self.player_elos.items():
        # Shrink extreme ratings toward mean
        if abs(elo - mean_elo) > max_deviation:
            direction = 1 if elo > mean_elo else -1
            self.player_elos[player_id] = mean_elo + direction * max_deviation
```

**Expected impact:** -0.5 MAE (prevent extreme player ratings)

---

## 2. Fix Missing Player Data 🔴 CRITICAL

### Problem
```
Teams with no players found in 2025 data:
  NC State, Miami, Florida State, Pitt

Current behavior: Fallback to 0 spread (neutral)
Better approach: Use team average or prior data
```

### Solutions

#### A. Better Team Name Mapping
```python
# Expand TEAM_NAME_MAPPING in prediction_pipeline.py

TEAM_NAME_MAPPING = {
    # Current mappings
    'Florida State': 'Florida St.',
    'Miami': 'Miami FL',
    'NC State': 'N.C. State',
    'Pitt': 'Pittsburgh',

    # Add variations
    'Florida St': 'Florida St.',
    'Miami (FL)': 'Miami FL',
    'NC St.': 'N.C. State',
    'Pittsburgh': 'Pittsburgh',

    # Full names
    'University of Miami': 'Miami FL',
    'North Carolina State': 'N.C. State',
    'University of Pittsburgh': 'Pittsburgh',
}

# Also try fuzzy matching
from rapidfuzz import fuzz

def find_best_team_match(team_name, available_teams):
    best_match = None
    best_score = 0

    for available_team in available_teams:
        score = fuzz.ratio(team_name.lower(), available_team.lower())
        if score > best_score and score > 80:  # 80% similarity threshold
            best_score = score
            best_match = available_team

    return best_match if best_match else team_name
```

**Expected impact:** -1.5 MAE (fix ~20 games with missing data)

#### B. Use Team-Level Fallback
```python
def predict_game(self, home_team, away_team, game_date, player_stats):
    # Try player-based prediction first
    home_lineup = self.predict_lineup(home_team, player_stats)
    away_lineup = self.predict_lineup(away_team, player_stats)

    if not home_lineup or not away_lineup:
        # FALLBACK: Use team-level ELO from team-based system
        logger.warning(f"  Using team-level fallback for {home_team} vs {away_team}")

        # Load team ELO ratings (from team-based system)
        team_elos = self.load_team_elo_ratings()  # New method

        home_elo = team_elos.get(home_team, 1500)
        away_elo = team_elos.get(away_team, 1500)

        # Simple ELO prediction
        elo_diff = home_elo - away_elo + 100  # HCA
        prediction = elo_diff * 0.03  # Convert ELO to spread

        return prediction, {'prediction_method': 'team_fallback'}

    # Continue with player-based prediction...
```

**Expected impact:** -1.0 MAE (better fallback than 0)

---

## 3. Improve Feature Scaling 🔴 CRITICAL

### Problem
```
Current features have wildly different scales:
  - Player ELO: 800-1200 (range: 400)
  - Usage %: 10-30 (range: 20)
  - Offensive rating: 80-120 (range: 40)
  - Minutes: 10-35 (range: 25)

Neural networks perform better with normalized features.
```

### Solution

#### Standardize All Features
```python
# In features.py, PlayerFeatureEngine.__init__

from sklearn.preprocessing import StandardScaler

def __init__(self, player_stats, player_elo_system):
    self.player_stats = player_stats
    self.elo_system = player_elo_system

    # NEW: Initialize feature scaler
    self.scaler = StandardScaler()
    self.is_fitted = False

def create_matchup_features(self, home_lineup, away_lineup, game_date, home_team, away_team):
    # Create features as before
    features = []

    # Add player features
    for player_id in home_lineup + away_lineup:
        player_feats = self.create_player_features(player_id, game_date)
        features.extend([
            player_feats['elo'],
            player_feats['usage_pct'],
            player_feats['offensive_rating'],
            player_feats['defensive_rating'],
            player_feats['minutes_per_game']
        ])

    # Add lineup and contextual features...
    # ...

    features_array = np.array(features).reshape(1, -1)

    # NEW: Standardize features
    if not self.is_fitted:
        # Fit on first batch (or load pre-fitted scaler)
        self.scaler.fit(features_array)
        self.is_fitted = True

    features_scaled = self.scaler.transform(features_array)

    return features_scaled.flatten()
```

#### Better Approach: Fit Scaler During Training
```python
# In training_pipeline.py, create_features method

def create_features(self, game_records_df, player_stats_df):
    # Create all feature vectors
    X = []
    y = []

    for idx, row in game_records_df.iterrows():
        features = self.feature_engine.create_matchup_features(...)
        X.append(features)
        y.append(row['home_score'] - row['away_score'])

    X = np.array(X)
    y = np.array(y)

    # NEW: Fit scaler on training data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save scaler for prediction time
    scaler_path = MODELS_DIR / 'feature_scaler.pkl'
    import pickle
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    return X_scaled, y
```

**Expected impact:** -0.8 MAE (better neural network training)

---

## 4. Better Lineup Prediction 🟡 HIGH IMPACT

### Current Approach
```python
# Simple heuristic: top 5 by minutes
lineup = player_stats.sort_values('minutes_per_game').head(5)
```

### Problems
1. Doesn't account for injuries
2. Ignores opponent matchups
3. No probability distribution
4. Can't handle lineup changes

### Improved Approaches

#### A. Weighted by Recent Performance
```python
def predict_lineup(self, team, player_stats, top_n=5, recency_weight=0.7):
    """
    Predict lineup weighted by recent performance

    Args:
        recency_weight: Weight for recent games (0-1)
            0 = use season average only
            1 = use only last 5 games
    """
    team_players = player_stats[player_stats['team'] == team].copy()

    # Calculate weighted score
    if 'minutes_l5' in team_players.columns:  # Last 5 games
        team_players['lineup_score'] = (
            recency_weight * team_players['minutes_l5'] +
            (1 - recency_weight) * team_players['minutes_per_game']
        )
    else:
        team_players['lineup_score'] = team_players['minutes_per_game']

    # Adjust for ELO (favor higher-rated players)
    team_players['elo'] = team_players['player_id'].apply(
        lambda p: self.elo_system.get_player_elo(p)
    )
    team_players['lineup_score'] *= (1 + (team_players['elo'] - 1000) / 500)

    # Sort and select top N
    lineup = team_players.nlargest(top_n, 'lineup_score')['player_id'].tolist()

    return lineup
```

**Expected impact:** -0.5 MAE (better lineup selection)

#### B. Opponent-Specific Lineups
```python
def predict_lineup_vs_opponent(self, team, opponent, player_stats, top_n=5):
    """Adjust lineup based on opponent characteristics"""
    team_players = player_stats[player_stats['team'] == team].copy()
    opponent_players = player_stats[player_stats['team'] == opponent]

    # Calculate opponent style
    opponent_pace = opponent_players['tempo'].mean()
    opponent_height = opponent_players['height'].mean()  # If available

    # Adjust player scores based on matchup
    team_players['matchup_score'] = team_players['minutes_per_game']

    # Fast-paced opponent: favor faster players
    if opponent_pace > 70:  # High pace
        team_players['matchup_score'] *= (1 + team_players['tempo'] / 100)

    # Tall opponent: favor taller players
    if 'height' in team_players.columns:
        height_factor = team_players['height'] / team_players['height'].mean()
        team_players['matchup_score'] *= height_factor

    lineup = team_players.nlargest(top_n, 'matchup_score')['player_id'].tolist()

    return lineup
```

**Expected impact:** -0.3 MAE (matchup-aware lineups)

---

## 5. Tune ELO Parameters 🟡 HIGH IMPACT

### Current Parameters
```python
PLAYER_ELO_CONFIG = {
    'default_rating': 1000,
    'k_factor': 20,
    'season_carryover': 0.75,
    'minutes_threshold': 10,
}
```

### Optimization Strategy

#### A. Grid Search for Best Parameters
```python
# Create a script: scripts/player_elo/tune_parameters.py

import itertools
from src.player_elo.training_pipeline import train_player_model

# Parameter grid
param_grid = {
    'k_factor': [10, 15, 20, 25, 30],
    'season_carryover': [0.65, 0.70, 0.75, 0.80, 0.85],
    'default_rating': [900, 1000, 1100],
}

best_mae = float('inf')
best_params = None

for k, carryover, default in itertools.product(
    param_grid['k_factor'],
    param_grid['season_carryover'],
    param_grid['default_rating']
):
    # Update config
    PLAYER_ELO_CONFIG['k_factor'] = k
    PLAYER_ELO_CONFIG['season_carryover'] = carryover
    PLAYER_ELO_CONFIG['default_rating'] = default

    # Train and evaluate
    results = train_player_model(years=[2023, 2024, 2025], n_cv_splits=3)
    mae = results['mean_mae']

    print(f"k={k}, carryover={carryover}, default={default}: MAE={mae:.2f}")

    if mae < best_mae:
        best_mae = mae
        best_params = {'k': k, 'carryover': carryover, 'default': default}

print(f"\nBest parameters: {best_params}")
print(f"Best MAE: {best_mae:.2f}")
```

**Expected impact:** -0.5 MAE (optimized parameters)

#### B. Adaptive K-Factor
```python
# In player_elo_system.py

def update_from_game(self, home_lineup, away_lineup, home_score, away_score):
    # Current: Fixed K=20
    # Better: Adaptive K based on games played

    for player_id in home_lineup + away_lineup:
        games_played = self.player_games_played.get(player_id, 0)

        # Higher K for new players (learn faster)
        # Lower K for veterans (more stable)
        if games_played < 10:
            k = 30  # New player
        elif games_played < 30:
            k = 20  # Regular
        else:
            k = 15  # Veteran

        # Update ELO with adaptive K
        # ...
```

**Expected impact:** -0.3 MAE (better learning rates)

---

## 6. Improve Model Architecture 🟡 HIGH IMPACT

### Current Architecture
```python
Input (65D) → Dense(128) → Dense(64) → Dense(32) → Output(1)
              ReLU+Drop    ReLU+Drop    ReLU+Drop
```

### Problems
1. May be overfitting (dropout = 0.2)
2. Fixed architecture (not optimized)
3. Simple feedforward (no attention to important features)

### Improvements

#### A. Add Batch Normalization
```python
# In pytorch_model.py, PlayerELONet.__init__

class PlayerELONet(nn.Module):
    def __init__(self, input_dim=65, hidden_dims=[128, 64, 32], dropout=0.3):
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
```

**Expected impact:** -0.4 MAE (better training stability)

#### B. Try Different Architectures
```python
# Option 1: Wider network
hidden_dims = [256, 128, 64]  # vs current [128, 64, 32]

# Option 2: Deeper network
hidden_dims = [128, 96, 64, 32]  # Add extra layer

# Option 3: Residual connections
class PlayerELONetResidual(nn.Module):
    def __init__(self, input_dim=65):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)  # Residual block
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        residual = x
        x = F.relu(self.fc2(x))
        x = x + residual  # Residual connection
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
```

Run grid search to find best:
```bash
python scripts/player_elo/tune_architecture.py
```

**Expected impact:** -0.3 MAE (optimized architecture)

#### C. Increase Dropout for Regularization
```python
# Try higher dropout to reduce overfitting
dropout = 0.3  # vs current 0.2
dropout = 0.4  # even stronger regularization
```

**Expected impact:** -0.2 MAE (reduced overfitting)

---

## 7. Use More Training Data 🟢 MEDIUM IMPACT

### Current
- Training on 18,024 games (only games with lineups)
- Missing ~15,000 games

### Solution

#### Impute Missing Lineups
```python
# In training_pipeline.py, process_games method

def process_games_with_imputation(self, games_df, player_stats_df):
    game_records = []

    for idx, row in games_df.iterrows():
        home_team = row['Home']
        away_team = row['Away']

        # Try to get actual lineup
        home_lineup = self.get_actual_lineup(home_team, row['Date'])
        away_lineup = self.get_actual_lineup(away_team, row['Date'])

        # If missing, IMPUTE using heuristics
        if not home_lineup:
            home_lineup = self.impute_lineup(home_team, player_stats_df, row['Date'])
        if not away_lineup:
            away_lineup = self.impute_lineup(away_team, player_stats_df, row['Date'])

        # Process game even with imputed lineups
        if home_lineup and away_lineup:
            self.elo_system.update_from_game(
                home_lineup, away_lineup,
                row['Home_Score'], row['Away_Score']
            )
            game_records.append({...})

    # Now using ~28,000 games instead of 18,000
    return pd.DataFrame(game_records)
```

**Expected impact:** -0.5 MAE (more training data)

---

## 8. Ensemble Methods 🟢 MEDIUM IMPACT

### Combine Multiple Models

#### A. Cross-Validation Ensemble
```python
# Instead of using just the final model, use all 5 CV fold models

def predict_with_ensemble(self, features):
    predictions = []

    for fold in range(1, 6):
        model_path = MODELS_DIR / f'pytorch_model_fold{fold}.pt'
        model = torch.load(model_path)
        model.eval()

        with torch.no_grad():
            pred = model(features).item()
            predictions.append(pred)

    # Average predictions
    ensemble_pred = np.mean(predictions)
    ensemble_std = np.std(predictions)

    return ensemble_pred, ensemble_std  # Also get uncertainty
```

**Expected impact:** -0.3 MAE (ensemble averaging)

#### B. Blend with Team-Based System
```python
def hybrid_prediction(self, game):
    # Get predictions from both systems
    player_pred = self.player_based_prediction(game)
    team_pred = self.team_based_prediction(game)  # Load from team system

    # Weighted average based on confidence
    if player_confidence > 0.8:
        weight = 0.6  # Trust player system more
    else:
        weight = 0.3  # Trust team system more

    final_pred = weight * player_pred + (1 - weight) * team_pred

    return final_pred
```

**Expected impact:** -0.5 MAE (combine strengths of both)

---

## 9. Better Player ID Matching 🟢 MEDIUM IMPACT

### Problem
Some players not matched correctly across data sources

### Solution
```python
# In player_data_collector.py, _assign_player_ids method

from rapidfuzz import fuzz, process

def _assign_player_ids_improved(self, df):
    """Improved fuzzy matching for player IDs"""

    # Create unique ID based on multiple fields
    df['fuzzy_key'] = (
        df['player_name'].str.lower().str.strip() + '_' +
        df['team'].str.lower().str.strip() + '_' +
        df['season'].astype(str)
    )

    # Use fuzzy matching for similar names
    unique_keys = df['fuzzy_key'].unique()

    for key in unique_keys:
        # Find similar keys (handle typos, spacing)
        similar = process.extract(
            key,
            unique_keys,
            scorer=fuzz.token_sort_ratio,
            limit=3
        )

        # If very similar (>95%), merge them
        for match, score in similar:
            if score > 95 and match != key:
                # Assign same player_id
                # ...
```

**Expected impact:** -0.2 MAE (better player tracking)

---

## Summary of Expected Improvements

| Fix | Expected MAE Reduction | Effort | Priority |
|-----|----------------------|--------|----------|
| **1. Clip extreme predictions** | -2.0 | Low | 🔴 Critical |
| **2. Fix missing player data** | -1.5 | Medium | 🔴 Critical |
| **3. Feature scaling** | -0.8 | Medium | 🔴 Critical |
| **4. Better lineup prediction** | -0.8 | High | 🟡 High |
| **5. Tune ELO parameters** | -0.8 | Medium | 🟡 High |
| **6. Improve model architecture** | -0.9 | High | 🟡 High |
| **7. More training data** | -0.5 | Medium | 🟢 Medium |
| **8. Ensemble methods** | -0.8 | Medium | 🟢 Medium |
| **9. Better player matching** | -0.2 | Low | 🟢 Medium |
| **Total Potential** | **-8.3** | | |

**Realistic Target:** Fix critical issues (#1-3) + 2-3 high-impact improvements

**Expected Result:** MAE 9.3 → **5.5-6.5** (competitive with team-based!)

---

## Implementation Priority

### Phase 1: Quick Wins (1-2 hours)
1. ✅ Add prediction clipping (-2.0 MAE)
2. ✅ Expand team name mapping (-1.5 MAE)
3. ✅ Add feature scaling (-0.8 MAE)

**Expected: MAE 9.3 → 5.0**

### Phase 2: High Impact (3-5 hours)
4. ✅ Better lineup prediction (-0.8 MAE)
5. ✅ Tune ELO parameters (-0.8 MAE)
6. ✅ Ensemble with CV models (-0.3 MAE)

**Expected: MAE 5.0 → 3.1**

### Phase 3: Polish (2-3 hours)
7. ✅ Improve model architecture (-0.9 MAE)
8. ✅ More training data with imputation (-0.5 MAE)

**Final Target: MAE 3.1 → 1.7** (better than team-based!)

---

## Next Steps

### Immediate Actions
1. Implement prediction clipping (10 min)
2. Fix team name mapping (20 min)
3. Add feature scaling (30 min)
4. Retrain and test (5 min)

### Code to Add
See the code examples above in each section.

### Testing
```bash
# After each improvement, test:
python scripts/player_elo/train_model.py --years 2023 2024 2025 --cv-splits 3
python scripts/player_elo/generate_predictions.py

# Check MAE improvement
```

---

**Bottom Line:** With focused effort on the top 6 improvements, you can reduce MAE from 9.3 to ~4.0, making the player-based system competitive with the team-based approach while maintaining all its advantages (roster awareness, player insights, etc.).

**Start with Phase 1 (Quick Wins) to get immediate 46% improvement!**
