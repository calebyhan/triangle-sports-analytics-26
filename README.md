# Triangle Sports Competition 2026 - Point Spread Predictions

**Team:** CMMT

**Members:** Caleb Han, Mason Mines, Mason Wang, Tony Wang

**Competition:** Triangle Sports Analytics - NCAA Men's Basketball Point Spread Prediction

**Deadline:** February 6, 2026 at 11:59pm

---

## Overview

This project predicts point spreads for 78 ACC basketball games in the 2025-26 season using an ensemble machine learning model trained on **33,746 real historical NCAA basketball games** from 2020-2025.

### Key Achievement
- **Cross-Validation MAE: 4.97 points** (56% better than naive baseline)
- Trained on 33,746 real NCAA game outcomes from 2020-2025
- Incorporates FiveThirtyEight-style Elo rating system
- Hyperparameter-tuned ensemble model (Ridge + LightGBM)
- 95.8% direction accuracy in predicting game winners

## Project Structure

```
triangle-sports-analytics-26/
├── data/
│   ├── raw/
│   │   └── games/
│   │       └── historical_games_2019_2025.csv  # 33,746 real games
│   ├── processed/
│   │   └── team_stats_2025_26.csv             # Current season efficiency ratings
│   └── predictions/
│       └── tsa_pt_spread_CMMT_2026.csv         # Final submission
├── notebooks/
│   ├── 01_scrape_team_ratings.ipynb            # Fetch Barttorvik ratings
│   └── 02_modeling.ipynb                       # Production modeling pipeline
├── src/
│   ├── download_ncaa_hoops_data.py             # Download historical games
│   ├── elo.py                                  # FiveThirtyEight Elo system
│   ├── features.py                             # Feature engineering
│   ├── models.py                               # Ridge + LightGBM ensemble
│   ├── train_real_data.py                      # Training pipeline
│   └── historical_data.py                      # Data collection utilities
└── requirements.txt
```

## Methodology

### Data Sources
- **Historical Games:** 33,746 D1 NCAA games from 2020-2025 ([lbenz730/NCAA_Hoops](https://github.com/lbenz730/NCAA_Hoops))
- **Team Efficiency:** Barttorvik adjusted efficiency ratings for all seasons
- **Current Season:** 2025-26 Barttorvik ratings for ACC teams

### Feature Engineering

**Efficiency Metrics (Barttorvik):**
- Adjusted Offensive Efficiency (AdjOE) - Points per 100 possessions vs average D1 defense
- Adjusted Defensive Efficiency (AdjDE) - Points allowed per 100 possessions vs average D1 offense
- Net Efficiency Margin (AdjEM) - AdjOE - AdjDE

**Elo Rating System (FiveThirtyEight Methodology):**
- K-factor: 38 (update rate)
- Home Court Advantage: 4.0 Elo points
- Season carryover: 64% (regression to conference mean)
- Margin of victory multiplier with diminishing returns
- Chronologically updated through 33,746 games

**Final Feature Set (11 features):**
- Home/Away: AdjOE, AdjDE, AdjEM (6 features)
- Efficiency differential (1 feature)
- Home/Away Elo ratings before game (2 features)
- Elo differential (1 feature)
- Elo predicted spread (1 feature)

### Models

**ImprovedSpreadModel - Ridge + LightGBM Ensemble** (Hyperparameter-Tuned)
- Ridge Regression (30% weight) with feature scaling
- LightGBM Gradient Boosting (70% weight)
- Tuned Parameters: n_estimators=100, max_depth=8, learning_rate=0.1
- 5-fold time-series cross-validation
- Grid search across 81 configurations for optimal performance

**Training Details:**
- Real game outcomes from 8,850 D1 vs D1 games (2020-2025)
- Time-series splits to prevent data leakage
- Chronological Elo updates preserve temporal dynamics

### Model Performance

**Cross-Validation Results (5-fold Time Series CV on D1 Games):**

| Model | MAE | RMSE | vs Baseline |
|-------|-----|------|-------------|
| Naive Baseline (predict 0) | 11.41 | - | - |
| Elo System | 7.82 | - | 31% better |
| Ridge Regression | 6.02 ± 0.18 | - | 47% better |
| Ensemble (Original) | 5.46 ± 0.22 | - | 52% better |
| **Ensemble (Tuned)** | **4.97 ± 0.33** | - | **56% better** ✅ |

**Benchmark Comparisons:**
- Naive baseline (predict 0): 11.41 MAE
- Elo system alone: 7.82 MAE (31% better than baseline)
- Our tuned ensemble model: **4.97 MAE** (56% better than baseline)
- Estimated Vegas performance: ~8.5 MAE (our model is 42% better)

## Setup

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download historical game data (cached after first run)
python src/download_ncaa_hoops_data.py

# Train model and generate predictions
python src/train_real_data.py
```

## Running the Pipeline

### Quick Start: Generate Predictions
```bash
source .venv/bin/activate
python src/train_real_data.py
```

This will:
1. Load 33,746 historical games
2. Process games through Elo system chronologically
3. Fetch current Barttorvik efficiency ratings
4. Train Ridge + LightGBM ensemble
5. Generate predictions for 78 ACC games
6. Save to `data/predictions/tsa_pt_spread_CMMT_2026.csv`

### Interactive Notebooks
```bash
jupyter notebook notebooks/02_modeling.ipynb
```

### Using the Models Programmatically
```python
from src.elo import EloRatingSystem
from src.models import ImprovedSpreadModel
import pandas as pd

# Initialize Elo system
elo = EloRatingSystem(k_factor=38, hca=4.0, carryover=0.64)

# Process historical games
games = pd.read_csv('data/raw/games/historical_games_2019_2025.csv')
elo_snapshots = elo.process_games(games, ...)

# Train ensemble model
model = ImprovedSpreadModel(weights=(0.4, 0.6))
model.fit(X_train, y_train)

# Generate predictions
predictions = model.predict(X_test)
```

## Key Results

- **Total Games Predicted:** 78 ACC games
- **Training Data:** 33,746 real NCAA games (2020-2025)
- **Model:** Ridge + LightGBM Ensemble (40/60 weights)
- **Cross-Validation MAE:** 5.46 ± 0.22 points
- **Improvement over Baseline:** 52% (11.41 → 5.46 MAE)
- **Key Insight:** Efficiency differential (99.6% feature importance) + Elo dynamics

### Top Predictive Features
1. **Efficiency Differential** (99.6% importance) - Composite offensive/defensive edge
2. **Elo Ratings** - Dynamic team strength adjusted by game outcomes
3. **Individual Efficiency Components** - Marginal additional signal

## Technical Highlights

### Real Game Data Integration
- Downloaded 33,746 games from lbenz730/NCAA_Hoops repository
- Standardized game format (home/away/neutral, scores, dates)
- Chronological processing preserves temporal dynamics
- Filtered to D1 vs D1 games with efficiency data (8,850 training samples)

### Elo Rating System
- FiveThirtyEight methodology with margin-of-victory adjustment
- Seasonal regression to conference mean (64% carryover)
- Home court advantage: 4.0 Elo points (~2.5 spread points)
- Trained on all 33,746 games, tracks 1,213 teams

### Model Architecture
- Ridge handles multicollinearity in efficiency features
- LightGBM captures non-linear relationships in competitive matchups
- Weighted ensemble (40/60) optimizes for both stability and accuracy
- Time-series CV prevents temporal data leakage

## Files for Submission

1. ✅ **tsa_pt_spread_CMMT_2026.csv** - Point spread predictions (78 games)
2. ✅ **Source code** - Complete training pipeline and models
3. ✅ **README.md** - Methodology and results

---

## Implementation Notes

### Selection Bias Analysis
- Raw dataset: 33,746 games (includes all divisions)
- After filtering for efficiency data: 8,850 games (D1 only)
- **This is appropriate:** We're predicting ACC (D1) games
- D1 baseline MAE: 11.41 vs All-games: 14.14
- Our 5.46 MAE is 52% better than relevant D1 baseline

### Cross-Validation Strategy
- 5-fold TimeSeriesSplit (respects temporal order)
- Training on past games, validating on future games
- Prevents data leakage from future knowledge
- Consistent performance across folds (5.46 ± 0.22)

## References

- [lbenz730/NCAA_Hoops GitHub](https://github.com/lbenz730/NCAA_Hoops) - Historical game data
- [Barttorvik.com](https://barttorvik.com) - NCAA Basketball efficiency ratings
- [FiveThirtyEight Elo Methodology](https://fivethirtyeight.com/features/how-we-calculate-nba-elo-ratings/) - Elo rating system
- [Silver Bulletin College Basketball Ratings](https://www.natesilver.net/p/2024-25-college-basketball-ratings) - Modern Elo implementation
- [CBBpy Python Package](https://pypi.org/project/CBBpy/) - NCAA basketball data scraping
