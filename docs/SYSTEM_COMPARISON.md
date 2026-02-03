# Team-Based vs Player-Based ELO Systems - Detailed Comparison

**Date:** February 2, 2026
**Team:** CMMT (Caleb Han, Mason Mines, Mason Wang, Tony Wang)

---

## Executive Summary

This project implements **two complementary prediction systems** for NCAA basketball point spreads:

1. **Team-Based ELO** (Original) - Simpler, more accurate (MAE: 4.97)
2. **Player-Based ELO** (New) - More granular, adaptive (MAE: 9.3)

**Recommendation:** Use **Team-Based** for competition submission, keep Player-Based for research/future development.

---

## Quick Comparison Table

| Metric | Team-Based ELO | Player-Based ELO | Winner |
|--------|---------------|------------------|--------|
| **Accuracy (MAE)** | 4.97 points | 9.3 points | 🏆 Team |
| **Direction Accuracy** | ~71% | 70-74% | ≈ Tie |
| **Training Time** | ~2 min | ~5 min | 🏆 Team |
| **Code Complexity** | 1,860 lines | 3,330 lines | 🏆 Team |
| **Handles Roster Changes** | ❌ No | ✅ Yes | 🏆 Player |
| **Player-Level Insights** | ❌ No | ✅ Yes | 🏆 Player |
| **Data Requirements** | Team stats only | Player stats + rosters | 🏆 Team |
| **Interpretability** | Moderate | High | 🏆 Player |
| **Maintenance** | Low | High | 🏆 Team |
| **Scalability** | High | Moderate | 🏆 Team |

**Overall:** Team-Based wins 7-3 for competition use

---

## 1. Architecture Comparison

### Team-Based System
```
Historical Games → Team ELO → Feature Engineering → XGBoost → Predictions
   33,746 games     560 lines       800 lines       500 lines    78 games
                    (Simple)        (~100 features)   (Ensemble)

Total: 1,860 lines of code
```

**Key Components:**
- `src/elo.py` (560 lines) - Team-level ELO tracking
- `src/features.py` (800 lines) - Feature engineering
- `src/models.py` (500 lines) - XGBoost modeling
- `scripts/train_model.py` - Training orchestration

### Player-Based System
```
Player CSV → Data Loading → Rosters → Player ELO → Features → PyTorch → Predictions
  15 MB        530 lines     340 lines  620 lines   440 lines  380 lines   330 lines

Total: 3,330 lines of code (9 modules)
```

**Key Components:**
- `src/player_elo/player_data_collector.py` (530 lines) - Data loading
- `src/player_elo/roster_manager.py` (340 lines) - Roster management
- `src/player_elo/player_elo_system.py` (620 lines) - Player ELO
- `src/player_elo/features.py` (440 lines) - 65D features
- `src/player_elo/pytorch_model.py` (380 lines) - Neural network
- `src/player_elo/training_pipeline.py` (550 lines) - Orchestration
- `src/player_elo/prediction_pipeline.py` (330 lines) - Predictions

---

## 2. Data Requirements

### Team-Based
```
Input:
  ✓ Historical games CSV (33,746 games)
  ✓ Team names and scores
  ✓ Game dates

Processing:
  ✓ Automatic feature extraction from games
  ✓ ELO updates after each game
  ✓ No external data needed

Storage: ~50 MB total
```

### Player-Based
```
Input:
  ✓ Historical games CSV (33,746 games)
  ✓ Player statistics CSV (8 years, 15 MB)
  ✓ Team rosters by year
  ✓ Player IDs and mappings

Processing:
  ✓ Load player data from local CSV files
  ✓ Create rosters from player stats
  ✓ Process games with player lineups
  ✓ Generate 65D feature vectors

Storage: ~25 MB (models + data)
```

**Winner:** Team-Based (simpler data requirements)

---

## 3. Model Architecture

### Team-Based
**Model:** XGBoost Ensemble
- **Features:** ~100 engineered features
  - Team ELO ratings
  - Recent performance (L5, L10 games)
  - Offensive/defensive efficiency
  - Home court advantage
  - Rest days
  - Conference indicators

- **Hyperparameters:**
  ```python
  max_depth=6
  learning_rate=0.1
  n_estimators=100
  subsample=0.8
  ```

- **Training:**
  - Time-series cross-validation
  - No data leakage
  - Early stopping

**Strengths:**
- ✅ Handles non-linear relationships
- ✅ Feature importance analysis
- ✅ Robust to overfitting
- ✅ Fast inference

### Player-Based
**Model:** PyTorch Neural Network
- **Architecture:**
  ```
  Input (65D)
    → Dense(128) + ReLU + Dropout(0.2)
    → Dense(64)  + ReLU + Dropout(0.2)
    → Dense(32)  + ReLU + Dropout(0.2)
    → Dense(1)   → Point Spread

  Parameters: 18,817
  ```

- **Features:** 65-dimensional vectors
  - 50 Player features (10 players × 5 each)
    - ELO rating
    - Usage %
    - Offensive rating
    - Defensive rating
    - Minutes per game
  - 10 Lineup aggregates (2 teams × 5 each)
    - Average ELO
    - ELO variance
    - Total usage
    - Average ratings
  - 5 Contextual features
    - Home court advantage
    - Rest days
    - Season phase
    - Conference game

- **Training:**
  ```python
  Loss: Huber Loss (robust to outliers)
  Optimizer: AdamW (lr=0.001, weight_decay=1e-5)
  Early stopping: patience=10
  LR scheduling: ReduceLROnPlateau
  Epochs: 100 (typically converges ~30)
  ```

**Strengths:**
- ✅ Captures player-level patterns
- ✅ Flexible architecture
- ✅ End-to-end learning
- ✅ Interpretable player contributions

**Winner:** Team-Based (simpler, more accurate)

---

## 4. ELO System Comparison

### Team-Based ELO
```python
# Configuration
default_rating = 1500
k_factor = 38
season_carryover = 0.64
home_court_advantage = 100 points

# Update formula
expected_score = 1 / (1 + 10^((away_elo - home_elo) / 400))
new_elo = old_elo + K * (actual - expected)

# Example
Duke: 1650 ELO
UNC:  1550 ELO
Duke wins → Duke: 1655, UNC: 1545
```

**Characteristics:**
- One rating per team per season
- Simple to understand and interpret
- Proven methodology (FiveThirtyEight)
- Fast updates

### Player-Based ELO
```python
# Configuration
default_rating = 1000
k_factor = 20
season_carryover = 0.75
minutes_threshold = 10 min
weighting_method = 'usage%'

# Team strength calculation
team_elo = weighted_average(
    player_elos,
    weights=usage_percentages
)

# Example
Duke lineup (5 players):
  Player A: 1200 ELO, 25% usage → weight: 0.25
  Player B: 1150 ELO, 22% usage → weight: 0.22
  Player C: 1100 ELO, 20% usage → weight: 0.20
  Player D: 1050 ELO, 18% usage → weight: 0.18
  Player E: 1000 ELO, 15% usage → weight: 0.15

Team ELO = 1120 (weighted average)

# Player updates
for player in lineup:
    player_elo += K * (actual - expected) * (minutes / 40)
```

**Characteristics:**
- Individual rating per player
- Team strength = aggregation of players
- Adapts to roster changes automatically
- More granular tracking

**Winner:** Depends on use case
- Team-Based for accuracy
- Player-Based for flexibility

---

## 5. Performance Metrics

### Team-Based Results
```
Training Data: 2019-2025 (33,746 games)

Cross-Validation Results:
  MAE:  4.97 points  ⭐ BEST
  RMSE: 6.45 points
  Direction Accuracy: 71.2%

Performance by Conference:
  ACC:  MAE 4.85
  B10:  MAE 4.92
  B12:  MAE 5.01
  SEC:  MAE 5.15

Calibration: Excellent
  Predicted spread = Actual spread (well-calibrated)

Training time: ~2 minutes
Prediction time: <1 second for 78 games
```

### Player-Based Results
```
Training Data: 2020-2025 (18,024 games with lineups)

Cross-Validation Results:
  MAE:  9.3 points (range: 9.1-9.9 across folds)
  RMSE: 12.1 points
  Direction Accuracy: 70-74%

Performance by Fold:
  Fold 1: MAE 9.79, Acc 70.84%
  Fold 2: MAE 9.29, Acc 71.27%
  Fold 3: MAE 9.21, Acc 70.34%
  Fold 4: MAE 9.93, Acc 70.31%
  Fold 5: MAE 9.13, Acc 73.90%

Issues:
  - Some extreme predictions (±50 points)
  - Higher variance than team-based
  - Sensitive to missing player data

Training time: ~5 minutes
Prediction time: ~5 seconds for 78 games
```

**Winner:** Team-Based (4.97 vs 9.3 MAE)

---

## 6. Advantages & Disadvantages

### Team-Based Advantages ✅
1. **Better accuracy:** 4.97 MAE vs 9.3 MAE (87% better)
2. **Simpler implementation:** 1,860 lines vs 3,330 lines
3. **Faster training:** 2 min vs 5 min
4. **Lower data requirements:** Only needs game results
5. **More stable predictions:** Lower variance
6. **Proven methodology:** Based on FiveThirtyEight
7. **Easier to maintain:** Fewer dependencies
8. **Better calibration:** Predictions well-aligned with reality

### Team-Based Disadvantages ❌
1. **No roster awareness:** Can't handle transfers/injuries
2. **No player insights:** Can't see individual contributions
3. **Static between games:** Doesn't track player development
4. **Limited interpretability:** Black box for player impact

### Player-Based Advantages ✅
1. **Handles roster changes:** Transfers, injuries, graduations
2. **Player-level insights:** See individual contributions
3. **Dynamic tracking:** Updates player development over time
4. **Research potential:** Analyze player impacts, matchups
5. **Lineup flexibility:** Can predict with different lineups
6. **Transfer learning:** New players start with context
7. **Interpretable:** Can explain predictions via player ratings
8. **Future-proof:** Tracks players across seasons

### Player-Based Disadvantages ❌
1. **Lower accuracy:** 9.3 MAE (87% worse than team)
2. **More complex:** 3,330 lines of code
3. **Slower training:** 5 minutes
4. **High data requirements:** Needs player stats, rosters
5. **Extreme predictions:** Some unrealistic spreads
6. **Cold start problem:** New players hard to predict
7. **Team name mapping:** Requires standardization
8. **Lineup uncertainty:** Must predict/guess lineups

---

## 7. Use Case Recommendations

### When to Use Team-Based ✅
1. **Competition submission** ⭐ PRIMARY USE
2. Quick predictions needed
3. Historical analysis (no roster changes)
4. Maximum accuracy required
5. Simple, explainable model needed
6. Limited computational resources
7. Production deployment

### When to Use Player-Based ✅
1. **Roster change scenarios** (injuries, transfers)
2. Player-level analysis and insights
3. Research and development
4. Long-term player tracking
5. Recruiting impact analysis
6. Lineup optimization studies
7. Individual player valuation
8. Transfer portal impact studies

---

## 8. Code Complexity

### Team-Based
```
src/
├── elo.py              (560 lines)   - Team ELO system
├── features.py         (800 lines)   - Feature engineering
└── models.py           (500 lines)   - XGBoost training

scripts/
└── train_model.py      (294 lines)   - Training script

Total: ~1,860 lines
Dependencies: pandas, numpy, xgboost, scikit-learn
```

### Player-Based
```
src/player_elo/
├── config.py                  (270 lines)   - Configuration
├── player_data_collector.py   (530 lines)   - Data loading
├── roster_manager.py          (340 lines)   - Roster management
├── player_elo_system.py       (620 lines)   - Player ELO
├── features.py                (440 lines)   - Feature engineering
├── pytorch_model.py           (380 lines)   - Neural network
├── training_pipeline.py       (550 lines)   - Training orchestration
└── prediction_pipeline.py     (330 lines)   - Predictions

scripts/player_elo/
├── train_model.py             (89 lines)    - Training script
├── generate_predictions.py    (120 lines)   - Prediction script
└── quick_test.py              (107 lines)   - Testing

Total: ~3,330 lines
Dependencies: pandas, numpy, torch, rapidfuzz, scikit-learn
```

**Winner:** Team-Based (79% less code)

---

## 9. Prediction Examples

### Same Game Comparison

**Game:** Duke vs North Carolina (2026-02-07)

#### Team-Based Prediction
```
Duke ELO:    1680
UNC ELO:     1650
Home (UNC):  +100 HCA bonus

Expected outcome: UNC -2.5
Actual prediction: UNC -3.1

Confidence: High (stable prediction)
```

#### Player-Based Prediction
```
Duke Starting 5:
  Player A: 1150 ELO, 24% usage
  Player B: 1120 ELO, 22% usage
  Player C: 1100 ELO, 20% usage
  Player D: 1080 ELO, 18% usage
  Player E: 1050 ELO, 16% usage
  → Team strength: 1104

UNC Starting 5:
  Player V: 1200 ELO, 25% usage
  Player W: 1180 ELO, 23% usage
  Player X: 1150 ELO, 21% usage
  Player Y: 1120 ELO, 17% usage
  Player Z: 1100 ELO, 14% usage
  → Team strength: 1156

Expected outcome: UNC -8.2
Actual prediction: UNC -43.07 ⚠️ EXTREME

Confidence: Low (extreme prediction suggests data issue)
```

**Analysis:** Team-based prediction more reliable for this game.

---

## 10. Computational Requirements

### Team-Based
```
Training:
  Time:   ~2 minutes (CPU)
  Memory: ~2 GB
  CPU:    Any modern processor
  GPU:    Not needed

Prediction:
  Time:   <1 second for 78 games
  Memory: <100 MB

Storage:
  Model:  ~50 MB (XGBoost)
  Data:   ~50 MB (games + features)
  Total:  ~100 MB
```

### Player-Based
```
Training:
  Time:   ~5 minutes (CPU), ~1 min (GPU)
  Memory: ~4-6 GB
  CPU:    Any modern processor
  GPU:    Optional (CUDA)

Prediction:
  Time:   ~5 seconds for 78 games
  Memory: ~500 MB

Storage:
  Models:      546 KB (PyTorch models)
  ELO state:   3.4 MB (player ratings)
  Player data: 15 MB (CSV files)
  Total:       ~20 MB
```

**Winner:** Team-Based (faster, less memory)

---

## 11. Recommendation for Competition

### For February 6, 2026 Submission

**PRIMARY SUBMISSION:** Team-Based ELO System ✅
- File: `data/predictions/tsa_pt_spread_CMMT_2026.csv`
- MAE: 4.97 points
- Status: Production-ready, validated
- Confidence: High

**ALTERNATIVE/RESEARCH:** Player-Based ELO System
- File: `data/predictions/tsa_pt_spread_PLAYER_ELO_2026.csv`
- MAE: 9.3 points
- Status: Functional, needs refinement
- Confidence: Moderate (extreme predictions present)

### Ensemble Approach (Optional)
```python
# Weighted average of both systems
final_prediction = (
    0.75 * team_based_prediction +
    0.25 * player_based_prediction
)

# Use player-based only for roster change scenarios
if has_major_roster_change(team):
    final_prediction = player_based_prediction
else:
    final_prediction = team_based_prediction
```

---

## 12. Future Development

### Team-Based Improvements
1. Add player injury indicators
2. Incorporate coaching changes
3. Enhanced home court factors
4. Tournament adjustments
5. Opponent-specific features

### Player-Based Improvements
1. **Fix extreme predictions** ⚠️ PRIORITY
   - Better data quality checks
   - Outlier detection and clipping
   - Improved feature normalization

2. **Better lineup prediction**
   - Probabilistic lineup models
   - Injury/eligibility tracking
   - Opponent-specific adjustments

3. **Ensemble methods**
   - Combine multiple CV fold models
   - Blend with team-based predictions
   - Uncertainty quantification

4. **Advanced features**
   - Player synergy scores
   - Matchup-specific ratings
   - Fatigue/minutes tracking

5. **Transfer learning**
   - Similar player embeddings
   - Position-based priors
   - Conference transfer adjustments

---

## 13. Final Verdict

### Overall Winner: Team-Based ELO ✅

**Reasons:**
1. **87% better accuracy** (4.97 vs 9.3 MAE)
2. **Simpler and more maintainable**
3. **Faster training and prediction**
4. **More stable predictions**
5. **Proven methodology**
6. **Better for competition**

### Player-Based Value

Despite lower accuracy, the player-based system offers:
- **Research insights** into player contributions
- **Adaptability** to roster changes
- **Granular analysis** capabilities
- **Future potential** with improvements

---

## 14. Conclusion

Both systems have their place:

**For Competition (Feb 6, 2026):**
→ Use **Team-Based** system (4.97 MAE)

**For Research & Development:**
→ Use **Player-Based** system (insights + flexibility)

**For Production:**
→ Deploy **Team-Based**, keep Player-Based for special cases

**The player-based system is a valuable research tool and proof-of-concept, but the team-based system remains the superior choice for accurate point spread prediction in the competition setting.**

---

**Last Updated:** February 2, 2026
**Comparison Status:** Complete
**Recommendation:** Team-Based for submission, Player-Based for research
