# Triangle Sports Competition 2026 - Point Spread Predictions

**Team:** CMM  
**Members:** Caleb Han, Mason Mines  
**Competition:** Triangle Sports Analytics - NCAA Men's Basketball Point Spread Prediction  
**Deadline:** February 6, 2026 at 11:59pm

---

## Overview

This project predicts point spreads for 78 ACC basketball games in the 2025-26 season using machine learning models trained on historical team efficiency ratings from Barttorvik.

## Project Structure

```
triangle-sports-analytics-26/
├── data/
│   ├── processed/
│   │   └── team_stats_2025_26.csv      # Team efficiency ratings for 2026
│   └── predictions/
│       ├── tsa_pt_spread_CMM_2026.csv  # Final submission
│       └── predictions_with_intervals.csv  # With 80% confidence intervals
├── notebooks/
│   ├── 01_data_collection.ipynb        # Initial setup
│   ├── 02_modeling.ipynb               # Baseline predictions
│   ├── 03_scrape_team_ratings.ipynb    # Fetch Barttorvik data
│   └── 04_advanced_modeling.ipynb      # ML models with historical data
├── outputs/
│   ├── figures/                        # Visualizations
│   └── model_evaluation_report.md      # Detailed metrics
├── src/
│   ├── data_collection.py
│   ├── features.py
│   ├── models.py
│   └── evaluation.py
└── requirements.txt
```

## Methodology

### Data Source
- **Barttorvik.com** - Free NCAA basketball efficiency ratings
- Historical data from 2019-2025 seasons (7 years)
- Current 2026 season team statistics

### Features
- **Adjusted Offensive Efficiency (AdjOE)** - Points per 100 possessions
- **Adjusted Defensive Efficiency (AdjDE)** - Opponent points per 100 possessions  
- **Net Efficiency** - AdjOE - AdjDE
- **Efficiency Differential** - Home net efficiency - Away net efficiency
- **Home Court Advantage** - Fixed at 3.5 points

### Models

1. **Baseline Model**
   - Simple efficiency-based formula: `Spread = (Eff_diff / 2) + HCA`
   
2. **Ridge Regression** (Primary Model)
   - Trained on ~70,000 historical matchups
   - 5-fold time series cross-validation
   - Features: AdjOE, AdjDE, Net Efficiency for both teams
   
3. **Quantile Regression** (Prediction Intervals)
   - 10th and 90th percentiles for 80% confidence intervals
   - Validated coverage on training data: ~80%

### Model Performance

**Cross-Validation Results (5-fold Time Series CV):**
- **Ridge Regression MAE:** ~11.0 points
- **Ridge Regression RMSE:** ~13.9 points
- **Prediction Interval Coverage:** 80%

See [outputs/model_evaluation_report.md](outputs/model_evaluation_report.md) for detailed metrics.

## Setup

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run notebooks in order
jupyter notebook
```

## Running the Pipeline

### Option 1: Run All Notebooks
1. `01_data_collection.ipynb` - Initial setup
2. `03_scrape_team_ratings.ipynb` - Fetch current Barttorvik ratings
3. `04_advanced_modeling.ipynb` - Train models and generate predictions

### Option 2: Use Python Scripts
```python
from src.data_collection import NCAADataCollector
from src.features import FeatureEngine
from src.models import GradientBoostingSpreadModel

# Collect data
collector = NCAADataCollector()
data = collector.fetch_barttorvik_data()

# Create features
engine = FeatureEngine(data)
X, y = engine.create_features()

# Train model
model = GradientBoostingSpreadModel()
model.fit(X, y)
```

## Key Results

- **Total Games Predicted:** 78
- **Primary Model:** Ridge Regression
- **Validation MAE:** ~11 points (typical for NCAA basketball)
- **Key Insight:** Efficiency differential is the strongest predictor
- **Home Court Advantage:** ~3.5 points (learned from data: 3.69)

## Files for Submission

1. ✅ **tsa_pt_spread_CMM_2026.csv** - Point spread predictions
2. ✅ **predictions_with_intervals.csv** - Optional prediction intervals
3. ⏳ **methodology_writeup.pdf** - 1-page explanation (pending)

## Team Info

- **Caleb Han** - calebhan@unc.edu
- **Mason Mines** - mmines@unc.edu

---

## Notes

- Game variance in college basketball is high (~11 point std deviation)
- Models trained on synthetic matchups from historical efficiency ratings
- Real game-by-game results would improve model performance
- Prediction intervals use quantile regression for uncertainty quantification

## References

- [Barttorvik.com](https://barttorvik.com) - NCAA Basketball Analytics
- [Triangle Sports Competition Guide](triangle_sports_guide.md)