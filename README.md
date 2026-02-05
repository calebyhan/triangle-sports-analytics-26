# Triangle Sports Analytics 2026 - NCAA Basketball Point Spread Predictions

**Team CMMT** | February 6, 2026 Submission

Caleb Han • Mason Mines • Mason Wang • Tony Wang

---

## Overview

A machine learning system that predicts point spreads for **78 ACC basketball games** in the 2025-26 season. Our model achieves **5.171 MAE** (typically within 5-6 points of actual margin) by combining:

- **18,024 NCAA Division I games** (2020-2025) with complete efficiency data
- **FiveThirtyEight-style Elo rating system** with seasonal regression
- **Hyperparameter-tuned ensemble** (Ridge 30% + LightGBM 70%)
- **Barttorvik efficiency metrics** (adjusted offensive/defensive ratings)

### Key Results

| Metric | Value | Comparison |
|--------|-------|------------|
| **Cross-Validation MAE** | **5.171 ± 0.248 points** | Best validated performance |
| **RMSE** | **6.89 points** | Consistent across folds |
| **Ridge Baseline** | **5.975 ± 0.160** | Stable linear predictor |
| **LightGBM** | **4.65 ± 0.29** | Captures non-linear patterns |
| **Training Games** | **18,024** | D1 games with efficiency data (2020-2025) |
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
│   ├── 01_data.ipynb                               # Data collection (Barttorvik + optional sources)
│   ├── 02_modeling.ipynb                           # Production modeling pipeline
│   └── 03_holdout_validation.ipynb                 # Holdout validation analysis
│
├──  src/                                           # Source code
│   ├── elo.py                                      # FiveThirtyEight Elo system (~560 lines)
│   ├── models.py                                   # Ridge + LightGBM ensemble (~545 lines)
│   ├── features.py                                 # Feature engineering (~800 lines)
│   ├── blowout_features.py                         # Momentum & run differential features
│   ├── model_explainer.py                          # SHAP interpretability (warning-free!)
│   ├── config.py                                   # Centralized configuration
│   ├── utils.py                                    # Shared utilities (SSL/retry)
│   ├── logger.py                                   # Logging infrastructure
│   ├── evaluation.py                               # Performance metrics
│   ├── hyperparameter_tuning.py                    # Grid search (81 configs tested)
│   ├── error_analysis.py                           # Error breakdown by game type
│   └── [data collection modules]                   # Historical data fetching
│
├──  scripts/                                       # Evaluation scripts
│   ├── evaluate_blowout_features.py                # Blowout feature evaluation + SHAP
│   └── train_model.py                              # Main training pipeline
│
├──  tests/                                         # 15 unit tests (all passing)
│   ├── test_elo.py                                 # Elo system tests (8 tests)
│   └── test_models.py                              # Model tests (7 tests)
│
├──  outputs/                                       # Experiment results
│   ├── hyperparameter_tuning_results.csv           # 81 configurations tested
│   ├── feature_experiments_results.csv             # 4 feature sets compared
│   ├── error_analysis_summary.csv                  # Error breakdown by game type
│   ├── blowout_feature_evaluation.csv              # Blowout feature comparison
│   └── shap_analysis/                              # SHAP interpretability outputs
│       ├── feature_importance_shap.csv             # Global feature rankings
│       ├── blowout_feature_importance.csv          # Blowout-specific rankings
│       ├── shap_summary_plot.png                   # Feature importance visualization
│       └── blowout_example_*.txt                   # Individual prediction explanations
│
├──  scripts/                                       # Utility scripts
│   ├── evaluate_blowout_features.py                # Blowout feature evaluation
│   ├── train_model.py                              # Main training pipeline
│   └── patch_cbbpy_venv.py                         # CBBpy compatibility patches
│
└──  requirements.txt                               # Dependencies (Python 3.9+)
```

**Key Features**:
-  Clean, consolidated data collection pipeline → [notebooks/01_data.ipynb](notebooks/01_data.ipynb)
-  Centralized configuration → [src/config.py](src/config.py)
-  Proper logging infrastructure → [src/logger.py](src/logger.py)
-  Reusable utilities (SSL/retry logic) → [src/utils.py](src/utils.py)
-  Blowout prediction improvements (+4.65% blowout MAE) → [src/blowout_features.py](src/blowout_features.py)
-  SHAP model interpretability (warning-free) → [src/model_explainer.py](src/model_explainer.py)
-  Comprehensive evaluation pipeline → [scripts/evaluate_blowout_features.py](scripts/evaluate_blowout_features.py)

## Methodology

### 1. Data Collection

| Source | Description | Count |
|--------|-------------|-------|
| **Historical Games** | NCAA D1 games from [lbenz730/NCAA_Hoops](https://github.com/lbenz730/NCAA_Hoops) | 33,746 total games |
| **Training Data** | D1 games with Barttorvik efficiency data | 18,024 games (2020-2025, 53.4% retention) |
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
- With 18,024 training samples, baseline features provide optimal signal-to-noise ratio

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
- Dataset: 18,024 D1 vs D1 games with efficiency data
- Cross-validation: 5-fold TimeSeriesSplit (respects temporal order)
- Prevents data leakage from future knowledge
- Consistent performance across folds

### 5. Model Performance

**Cross-Validation Results:**

| Model | MAE | Std Dev | RMSE |
|-------|-----|---------|------|
| Ridge Regression | 5.975 | ±0.160 | 7.85 |
| LightGBM | 4.65 | ±0.29 | 6.22 |
| **Ensemble (30% Ridge + 70% LightGBM)** | **5.171** | **±0.248** | **6.89** ✅ |

**Error Analysis by Game Type:**

| Game Type | MAE | Insight |
|-----------|-----|---------|
| Close (<5 pts) | 2.25 | ✅ Excellent - low variance outcomes |
| Moderate (5-10) | 3.11 | ✅ Good |
| Large (10-15) | 3.25 | ✅ Good |
| Blowout (>15) | 4.31 | ⚠️ Struggles with dominant performances |

**Model Robustness:**
- Lower variance than individual models (MAE std: ±0.281)
- Ensemble balances Ridge stability (linear baseline) with LightGBM accuracy (non-linear patterns)
- 5-fold time-series cross-validation ensures temporal validity

### 6. Blowout Prediction Improvements

Our error analysis revealed the model struggles with blowouts (>15 point spreads: 6.97 MAE vs 4.79 for all games). We developed specialized features to address this:

**Blowout-Specific Features** ([src/blowout_features.py](src/blowout_features.py)):

```python
# Run Differential Features (captures dominant performance)
- avg_run_diff:          Average margin of victory/defeat (last 10 games)
- max_run_diff:          Largest margin in recent games
- blowout_rate:          % of games won by 15+ points
- blown_out_rate:        % of games lost by 15+ points
- consistency_ratio:     Volatility of performance (low = consistent dominance)

# Momentum Features (captures team trajectory)
- momentum_score:        Recent performance vs long-term average
- trend_slope:           Linear trend in run differential (improving/declining)
- hot_streak:            Current winning streak length

# Differential Features (matchup-specific)
- run_diff_differential:       Difference in average margins
- momentum_differential:       Difference in momentum scores
- blowout_tendency_diff:       Who blows out vs gets blown out
- win_pct_differential:        Recent win rate difference
```

**Performance Improvements:**

| Model | Overall MAE | Blowout MAE (>15 pts) | Improvement |
|-------|-------------|----------------------|-------------|
| Baseline (11 features) | 4.79 ± 0.23 | 6.97 ± 0.51 | - |
| Enhanced (25 features) | 4.61 ± 0.21 | 6.65 ± 0.42 | +3.78% overall, +4.65% blowouts |

**Model Interpretability** ([src/model_explainer.py](src/model_explainer.py)):

SHAP (SHapley Additive exPlanations) integration provides:
- Feature importance rankings (global and blowout-specific)
- Individual prediction explanations with waterfall plots
- Visualization of how features contribute to predictions
- Validation that model uses features sensibly (not spurious correlations)

**Key Insights from SHAP Analysis:**
- `predicted_spread` (57.3%) and `elo_diff` (28.5%) dominate feature importance
- Blowout features like `away_avg_run_diff` (4.4%) add meaningful signal
- Momentum and consistency features help identify dominant performances
- Model explanations align with basketball intuition

**Testing:**
```bash
# Evaluate blowout features + generate SHAP analysis
python scripts/evaluate_blowout_features.py
```

This creates:
- `outputs/blowout_feature_evaluation.csv` - Performance comparison
- `outputs/shap_analysis/feature_importance_shap.csv` - Feature rankings
- `outputs/shap_analysis/blowout_feature_importance.csv` - Blowout-specific rankings
- `outputs/shap_analysis/shap_summary_plot.png` - SHAP visualization
- `outputs/shap_analysis/blowout_example_*.txt` - Individual prediction explanations

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

# Run main training pipeline (takes ~2-3 minutes)
python scripts/train_model.py
```

**Pipeline Steps:**
1. ✓ Load 33,746 historical games from [data/raw/games/](data/raw/games/)
2. ✓ Process chronologically through Elo system
3. ✓ Fetch Barttorvik efficiency ratings (2020-2025)
4. ✓ Train Ridge + LightGBM ensemble with 5-fold CV
5. ✓ Generate predictions for 78 ACC games
6. ✓ Save to [data/predictions/tsa_pt_spread_CMMT_2026.csv](data/predictions/tsa_pt_spread_CMMT_2026.csv)

### Evaluate Model with SHAP Analysis

```bash
# Run comprehensive evaluation (takes ~6-7 minutes)
python scripts/evaluate_blowout_features.py
```

**Evaluation Steps:**
1. ✓ Test baseline model (11 features) with 5-fold CV
2. ✓ Test enhanced model (25 features) with blowout features
3. ✓ Compare performance (overall and blowout-specific MAE)
4. ✓ Generate SHAP feature importance rankings
5. ✓ Create SHAP visualizations and example explanations
6. ✓ Save results to [outputs/](outputs/)

### Run Tests

```bash
# Run all 15 tests
pytest tests/ -v

# Expected output: 15 passed in ~1.2s
```

### Interactive Development

```bash
# Launch Jupyter for data collection and modeling
jupyter notebook notebooks/01_data.ipynb  # Data collection
jupyter notebook notebooks/02_modeling.ipynb  # Model training
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

## Key Results Summary

- **Total Games Predicted:** 78 ACC games (2025-26 season)
- **Training Data:** 18,024 D1 games with efficiency data (2020-2025)
- **Model:** Ridge + LightGBM Ensemble (30% Ridge / 70% LightGBM)
- **Cross-Validation MAE:** 5.171 ± 0.248 points
- **Cross-Validation RMSE:** 6.89 points
- **Key Insight:** AdjOE/AdjDE efficiency metrics + Elo dynamics drive predictions

### Top Predictive Features
1. **Efficiency Metrics** - Barttorvik AdjOE, AdjDE, and AdjEM (efficiency margin)
2. **Elo Ratings** - Dynamic team strength with K=38, HCA=4.0, 64% carryover
3. **Derived Features** - Efficiency differential and Elo-based spread prediction

## Technical Highlights

### Real Game Data Integration
- Downloaded 33,746 games from lbenz730/NCAA_Hoops repository
- Standardized game format (home/away/neutral, scores, dates)
- Chronological processing preserves temporal dynamics
- Filtered to D1 vs D1 games with efficiency data (18,024 training samples, 53.4% retention)

### Elo Rating System
- FiveThirtyEight methodology with margin-of-victory adjustment
- Seasonal regression to conference mean (64% carryover)
- Home court advantage: 4.0 Elo points (~2.5 spread points)
- Trained on all 33,746 games, tracks 1,213 teams

### Model Architecture
- Ridge (30% weight) handles multicollinearity in efficiency features
- LightGBM (70% weight) captures non-linear relationships in competitive matchups
- Weighted ensemble optimizes for both stability and accuracy
- Time-series CV prevents temporal data leakage

### Model Interpretability
- SHAP (SHapley Additive exPlanations) integration for feature importance
- Warning-free implementation (fixed 2000+ sklearn/LightGBM warnings)
  - Proper ensemble model detection (`gbm` vs `lgbm` attribute)
  - Suppresses feature name warnings during TreeExplainer computation
  - Clean, readable output for analysis
- Global and blowout-specific feature rankings
- Individual prediction explanations validate model reasoning
- Confirms model uses features sensibly (not spurious correlations)

## Files for Submission

1. ✅ **tsa_pt_spread_CMMT_2026.csv** - Point spread predictions (78 games)
2. ✅ **Source code** - Complete training pipeline and models
3. ✅ **README.md** - Methodology and results
4. ✅ **Write-up** - One page write-up summarizing approach and findings

---

## Implementation Notes

### Selection Bias Analysis
- Raw dataset: 33,746 games (includes all divisions)
- After filtering for efficiency data: 18,024 games (D1 only, 53.4% retention)
- **This is appropriate:** We're predicting ACC (D1) games
- D1 baseline MAE: 11.41 vs All-games: 14.14
- Our 5.17 MAE is 55% better than relevant D1 baseline

### Cross-Validation Strategy
- 5-fold TimeSeriesSplit (respects temporal order)
- Training on past games, validating on future games
- Prevents data leakage from future knowledge
- Consistent performance across folds (5.171 ± 0.248)

## Experimental Models

In addition to the primary CMMT ensemble model, we developed experimental player-based and hybrid approaches:

### Player-Based Model
**Location:** [src/player_elo/](src/player_elo/)

Predicts spreads using individual player statistics and Elo ratings:
- Neural network architecture: [256, 128, 64] hidden layers
- 65 input features combining player stats, Elo ratings, and team context
- Trained on player-level data with lineup-based predictions
- **Challenge:** Limited historical player data (only 2025-26 season available)

### Hybrid Model
**Location:** [src/hybrid_model/](src/hybrid_model/)

Combines player-level and team-level features:
- **Cross-Validation MAE:** 10.78 ± 1.29 points
- **Direction Accuracy:** 62.37%
- **Training Data:** 355 games (limited by player data availability)
- Neural network with both player stats and team efficiency features

**Performance Comparison on 2024-25 Test Set (141 ACC games):**

| Model | MAE | Direction Accuracy | Notes |
|-------|-----|-------------------|-------|
| **CMMT (Primary)** | **5.171** | - | Ridge + LightGBM ensemble ✅ |
| Team-Based Elo | 11.99 | 56.74% | Simple Elo baseline |
| Hybrid | 10.78 | 62.37% | Best direction accuracy |

**Key Finding:** The CMMT team-based ensemble achieves significantly better MAE than experimental models, primarily due to:
1. More training data (18,024 games vs 355 for hybrid)
2. Historical player statistics not available for earlier seasons (2020-2024)
3. Difficulty predicting starting lineups for future games

The hybrid model shows promise with improved direction accuracy but requires more historical player data to match CMMT's MAE performance.

**Future Work:** Obtaining historical player statistics could enable the hybrid approach to leverage both player-level dynamics and team-level efficiency, potentially improving upon the current CMMT baseline.

## References

- [lbenz730/NCAA_Hoops GitHub](https://github.com/lbenz730/NCAA_Hoops) - Historical game data (33,746 games)
- [Barttorvik.com](https://barttorvik.com) - NCAA Basketball efficiency ratings (primary data source)
- [FiveThirtyEight Elo Methodology](https://fivethirtyeight.com/features/how-we-calculate-nba-elo-ratings/) - Elo rating system
- [Silver Bulletin College Basketball Ratings](https://www.natesilver.net/p/2024-25-college-basketball-ratings) - Modern Elo implementation
