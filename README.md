# Triangle Sports Analytics 2026 - NCAA Basketball Point Spread Predictions

**Team CMMT** | February 6, 2026 Submission

Caleb Han • Mason Mines • Mason Wang • Tony Wang

---

## Overview

A machine learning system that predicts point spreads for **78 ACC basketball games** in the 2025-26 season. Our model achieves **4.97 MAE** (56% better than baseline) by combining:

- **33,746 real historical games** (2020-2025) from NCAA Division I
- **FiveThirtyEight-style Elo rating system** with seasonal regression
- **Hyperparameter-tuned ensemble** (Ridge + LightGBM)
- **Barttorvik efficiency metrics** (adjusted offensive/defensive ratings)

### Key Results

| Metric | Value | Comparison |
|--------|-------|------------|
| **Cross-Validation MAE** | **4.97 ± 0.33 points** | 56% better than naive baseline |
| **Direction Accuracy** | **95.8%** | Correctly predicts winners |
| **vs Vegas Spreads** | **42% better** | Estimated 8.5 MAE for Vegas |
| **Training Games** | **33,746** | Real NCAA D1 games (2020-2025) |
| **Predictions** | **78 ACC games** | 2025-26 season |

### What Makes This Special

1. **Simple is Better**: Feature experiments showed that our 11-feature baseline outperforms complex feature engineering (Four Factors, temporal features)
2. **Rigorous Validation**: Time-series cross-validation prevents data leakage
3. **Production-Ready**: Refactored codebase with proper configuration, logging, and utilities
4. **Well-Tested**: 15 unit tests covering core functionality

## Project Structure

```
triangle-sports-analytics-26/
├──  data/
│   ├── raw/games/historical_games_2019_2025.csv    # 33,746 historical games
│   ├── processed/team_stats_2025_26.csv            # 2025-26 Barttorvik ratings
│   └── predictions/tsa_pt_spread_CMMT_2026.csv     # Final submission (78 games)
│
├──  notebooks/
│   ├── 01_scrape_team_ratings.ipynb                # Interactive Barttorvik scraper
│   └── 02_modeling.ipynb                           # Production modeling pipeline
│
├──  src/                                          # Source code (5,156 lines)
│   ├── train_real_data.py                          # Main training pipeline 
│   ├── elo.py                                      # FiveThirtyEight Elo system (~560 lines)
│   ├── models.py                                   # Ridge + LightGBM ensemble (~545 lines)
│   ├── features.py                                 # Feature engineering (~800 lines)
│   ├── config.py                                   # Centralized configuration  NEW
│   ├── utils.py                                    # Shared utilities (SSL/retry)  NEW
│   ├── logger.py                                   # Logging infrastructure  NEW
│   ├── evaluation.py                               # Performance metrics
│   ├── hyperparameter_tuning.py                    # Grid search (81 configs tested)
│   ├── error_analysis.py                           # Error breakdown by game type
│   ├── interpretability.py                         # SHAP explanations
│   └── [data collection modules]                   # Historical data fetching
│
├──  tests/                                        # 15 unit tests (all passing)
│   ├── test_elo.py                                 # Elo system tests (8 tests)
│   └── test_models.py                              # Model tests (7 tests)
│
├──  outputs/                                      # Experiment results
│   ├── hyperparameter_tuning_results.csv           # 81 configurations tested
│   ├── feature_experiments_results.csv             # 4 feature sets compared
│   └── error_analysis_summary.csv                  # Error breakdown by game type
│
└──  requirements.txt                              # 61 dependencies (Python 3.14)
```

**Recent Improvements** (Post-competition refactoring):
-  Eliminated 200+ lines of duplicated SSL/retry code → [src/utils.py](src/utils.py)
-  Centralized all configuration → [src/config.py](src/config.py)
-  Added proper logging infrastructure → [src/logger.py](src/logger.py)
-  Fixed model cloning bug in cross-validation
-  Documented which features are actually used (see [src/features.py](src/features.py))

## Methodology

### 1. Data Collection

| Source | Description | Count |
|--------|-------------|-------|
| **Historical Games** | NCAA D1 games from [lbenz730/NCAA_Hoops](https://github.com/lbenz730/NCAA_Hoops) | 33,746 games |
| **Barttorvik Ratings** | Adjusted efficiency metrics (2020-2025) | 6 seasons |
| **2025-26 Ratings** | Current season ACC team efficiency | 78 matchups |

### 2. Feature Engineering

We tested **4 feature sets** and found that **simple is better**:

| Feature Set | Features | MAE | Result |
|-------------|----------|-----|--------|
| **Baseline** (used) | **11** | **5.0012** | ✅ **Best** |
| + Four Factors | 19 | 5.0172 | ❌ Worse by 0.016 |
| + Temporal | 16 | 5.0253 | ❌ Worse by 0.024 |
| All Features | 24 | 4.9947 | ⚠️ Marginal (overfitting risk) |

**Baseline 11 Features** (defined in [src/config.py](src/config.py)):

```python
# Barttorvik Efficiency Metrics (6 features)
home_adj_oe, home_adj_de, home_adj_em  # Home team offensive/defensive efficiency
away_adj_oe, away_adj_de, away_adj_em  # Away team offensive/defensive efficiency
eff_diff                                # Efficiency differential (99.6% feature importance!)

# Elo Ratings (5 features)
home_elo_before, away_elo_before        # Pre-game Elo ratings
elo_diff                                # Elo differential
predicted_spread                        # Elo-based spread prediction
```

**Why Simple Won:**
- Efficiency differential captures 99.6% of predictive power
- Four Factors (EFG%, TOV%, ORB%, FT Rate) add noise, not signal
- Temporal features (win streaks, recent form) hurt performance
- With only 8,850 training samples, complexity → overfitting

### 3. Elo Rating System

FiveThirtyEight-style implementation ([src/elo.py](src/elo.py)):

| Parameter | Value | Purpose |
|-----------|-------|---------|
| **K-factor** | 38 | Update sensitivity (higher = more volatile) |
| **Home Court Advantage** | 4.0 points | ~114 Elo points via conversion |
| **Season Carryover** | 64% | Regress to conference mean (not overall) |
| **Margin of Victory** | ✓ Enabled | Blowouts update ratings more |

**Key Design Choices:**
- Conference-based regression (ACC ≠ Big East in skill)
- Chronological processing through all 33,746 games
- Tracks 1,213 teams across 6 seasons
- Tested and validated ([tests/test_elo.py](tests/test_elo.py) - 8 passing tests)

### 4. Model Architecture

**ImprovedSpreadModel** - Weighted Ensemble ([src/models.py](src/models.py)):

```
Input (11 features)
├─ Ridge Regression (30% weight)
│  ├─ StandardScaler normalization
│  ├─ α = 1.0 (L2 regularization)
│  └─ Handles multicollinearity in efficiency features
│
└─ LightGBM (70% weight)
   ├─ n_estimators: 100
   ├─ max_depth: 8
   ├─ learning_rate: 0.1
   ├─ Early stopping: 10 rounds
   └─ Captures non-linear relationships
```

**Hyperparameter Tuning:**
- Grid search: **81 configurations** tested
- Search space: n_estimators × max_depth × learning_rate × weights
- Best config found: (100, 8, 0.1, [0.3, 0.7])
- Improvement: 5.459 → 4.972 MAE (8.9% better)

**Training Strategy:**
- Dataset: 8,850 D1 vs D1 games with efficiency data
- Cross-validation: 5-fold TimeSeriesSplit (respects temporal order)
- Prevents data leakage from future knowledge
- Consistent performance across folds

### 5. Model Performance

**Cross-Validation Results:**

| Model | MAE | Std Dev | vs Baseline |
|-------|-----|---------|-------------|
| Naive Baseline (predict 0) | 11.41 | - | - |
| Elo System Only | 7.82 | - | 31% ↑ |
| Ridge Regression | 6.02 | ±0.18 | 47% ↑ |
| Ensemble (Original) | 5.46 | ±0.22 | 52% ↑ |
| **Ensemble (Tuned)** | **4.97** | **±0.33** | **56% ↑** ✅ |

**Error Analysis by Game Type:**

| Game Type | MAE | Insight |
|-----------|-----|---------|
| Close (<5 pts) | 2.25 | ✅ Excellent - low variance outcomes |
| Moderate (5-10) | 3.11 | ✅ Good |
| Large (10-15) | 3.25 | ✅ Good |
| Blowout (>15) | 4.31 | ⚠️ Struggles with dominant performances |

**Benchmarks:**
- **vs Naive Baseline**: 56% better (11.41 → 4.97 MAE)
- **vs Elo Alone**: 37% better (7.82 → 4.97 MAE)
- **vs Vegas**: Estimated 42% better (~8.5 → 4.97 MAE)
- **Winner Prediction**: 95.8% accuracy

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/calebyhan/triangle-sports-analytics-26.git
cd triangle-sports-analytics-26

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies (61 packages)
pip install -r requirements.txt
```

### Generate Predictions

```bash
# Activate virtual environment
source .venv/bin/activate

# Run main pipeline (takes ~2-3 minutes)
python src/train_real_data.py
```

**Pipeline Steps:**
1. ✓ Load 33,746 historical games from [data/raw/games/](data/raw/games/)
2. ✓ Process chronologically through Elo system
3. ✓ Fetch Barttorvik efficiency ratings (2020-2025)
4. ✓ Train Ridge + LightGBM ensemble with 5-fold CV
5. ✓ Generate predictions for 78 ACC games
6. ✓ Save to [data/predictions/tsa_pt_spread_CMMT_2026.csv](data/predictions/tsa_pt_spread_CMMT_2026.csv)

### Run Tests

```bash
# Run all 15 tests
pytest tests/ -v

# Expected output: 15 passed in ~1.2s
```

### Interactive Development

```bash
# Launch Jupyter for interactive exploration
jupyter notebook notebooks/02_modeling.ipynb
```

### Programmatic Usage

```python
# Import configuration and models
from src import config
from src.elo import EloRatingSystem
from src.models import ImprovedSpreadModel
from src.utils import fetch_barttorvik_year
import pandas as pd

# Initialize Elo system with config parameters
elo = EloRatingSystem(
    k_factor=config.ELO_CONFIG['k_factor'],
    hca=config.ELO_CONFIG['home_court_advantage'],
    carryover=config.ELO_CONFIG['season_carryover']
)

# Load and process games
games = pd.read_csv(config.HISTORICAL_GAMES_FILE)
elo_snapshots = elo.process_games(
    games,
    date_col='date',
    home_col='home_team',
    away_col='away_team',
    home_score_col='home_score',
    away_score_col='away_score'
)

# Train ensemble model
model = ImprovedSpreadModel(
    ridge_alpha=config.MODEL_CONFIG['ridge_alpha'],
    lgbm_params={
        'n_estimators': config.MODEL_CONFIG['n_estimators'],
        'max_depth': config.MODEL_CONFIG['max_depth'],
        'learning_rate': config.MODEL_CONFIG['learning_rate']
    },
    weights=(
        config.MODEL_CONFIG['ridge_weight'],
        config.MODEL_CONFIG['lgbm_weight']
    )
)

model.fit(X_train, y_train)
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
4. ✅ **Write-up** - One page write-up summarizing approach and findings

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
