# Player-Based ELO System - Usage Guide

**Quick Start:** Train a player-level ELO model and generate predictions for 2026 ACC games

---

## 🚀 Quick Start (3 Commands)

### 1. Test the System (10 seconds)
```bash
cd "e:\Triangle Sports Analytics\triangle-sports-analytics-26"
python scripts/player_elo/quick_test.py
```
Verifies all components work with synthetic data.

### 2. Train the Model (30-60 minutes) ⭐ **START HERE**
```bash
python scripts/player_elo/train_model.py
```
**What it does:**
- Downloads 2020-2025 player stats from Barttorvik (~15-20 min)
- Processes 33k+ games through player ELO system (~10 min)
- Creates 65D feature vectors (~5 min)
- Trains PyTorch neural network (~10-30 min)
- Validates with 5-fold cross-validation

**Expected output:** MAE ~5.0-5.5 points

### 3. Generate 2026 Predictions (COMING SOON)
```bash
python scripts/player_elo/generate_predictions.py
```
⚠️ **Status:** Prediction pipeline not yet implemented (~250 lines remaining)

---

## 📋 Training Options

### Basic Training (Default)
```bash
# Train on all years (2020-2025)
python scripts/player_elo/train_model.py
```

### Quick Training (Faster)
```bash
# Train on recent years only
python scripts/player_elo/train_model.py --years 2023 2024 2025
```
Trains in ~20 minutes instead of ~60 minutes

### Force Fresh Data
```bash
# Re-download all player stats (if data seems stale)
python scripts/player_elo/train_model.py --no-cache
```

### Adjust Cross-Validation
```bash
# Use 3 folds instead of 5 (faster)
python scripts/player_elo/train_model.py --cv-splits 3
```

---

## 📊 Expected Training Output

```
======================================================================
  PLAYER-BASED ELO MODEL TRAINING PIPELINE
======================================================================

======================================================================
STEP 1: DATA COLLECTION
======================================================================
Loading historical games from: data/raw/games/historical_games_2019_2025.csv
  ✓ Loaded 33746 games

Collecting player statistics from Barttorvik...
  Using cached player stats: data/player_data/raw/player_stats/barttorvik_stats_2020_2025.csv
  ✓ Loaded 24000 player records

======================================================================
STEP 2: ROSTER PREPARATION
======================================================================
  ✓ Created roster for 2020: 350 teams
  ✓ Created roster for 2021: 350 teams
  ... (continues for each year)

======================================================================
STEP 3: PROCESS GAMES THROUGH PLAYER ELO SYSTEM
======================================================================
Processing 33746 games chronologically...
  Processed 1000 games...
  Processed 2000 games...
  ... (continues)
  ✓ Processed 28500 games with lineups

======================================================================
STEP 4: FEATURE ENGINEERING
======================================================================
Creating features for 28500 games...
  ✓ Created features: X shape (28500, 65), y shape (28500,)

======================================================================
STEP 5: TRAIN PYTORCH MODEL
======================================================================

Fold 1/5
  Train: 22800 samples, Val: 5700 samples
  Epoch 10/100 | Train Loss: 45.2341 | Val Loss: 46.1234 | Val MAE: 5.1234
  Epoch 20/100 | Train Loss: 42.3456 | Val Loss: 43.2345 | Val MAE: 4.9876
  ... (continues)
  ✓ Saved best model (val_loss=41.2345)
  Fold 1 Results:
    MAE: 4.9876
    RMSE: 6.5432
    Direction Accuracy: 72.34%

... (Folds 2-5)

======================================================================
CROSS-VALIDATION SUMMARY
======================================================================
  Mean MAE: 5.1234 ± 0.1234
  Mean RMSE: 6.7890 ± 0.2345
  Mean Direction Acc: 71.23%

Training final model on all data...
  ✓ Final model saved to: data/player_data/models/pytorch_model.pt

======================================================================
STEP 6: SAVE ARTIFACTS
======================================================================
  ✓ Saved ELO state to: data/player_data/models/player_elo_state.json
  ✓ Saved transfer data
  All artifacts saved successfully!

======================================================================
  TRAINING COMPLETE!
======================================================================

  Final Performance:
    MAE: 5.1234 ± 0.1234
    Saved model: data/player_data/models/pytorch_model.pt
    Saved ELO state: data/player_data/models/player_elo_state.json
======================================================================

[SUCCESS] Training completed!
```

---

## 🔍 What Gets Created

After training, you'll have:

### Model Files
```
data/player_data/models/
├── pytorch_model.pt              # Final trained neural network
├── pytorch_model_fold1.pt        # CV fold 1 model (for ensemble)
├── pytorch_model_fold2.pt        # CV fold 2 model
├── ... (folds 3-5)
└── player_elo_state.json         # Player ELO ratings
```

### Data Files
```
data/player_data/
├── raw/player_stats/
│   └── barttorvik_stats_2020_2025.csv  # Downloaded player stats
└── processed/
    └── transfer_tracker.csv             # Player transfers
```

---

## ⚠️ Troubleshooting

### Error: "Historical games file not found"
**Solution:** Make sure you're in the project root directory
```bash
cd "e:\Triangle Sports Analytics\triangle-sports-analytics-26"
```

### Error: "No module named 'torch'"
**Solution:** PyTorch not installed (should have been installed earlier)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Error: "Connection timeout" during data collection
**Solution:** Barttorvik is slow/blocking - use cached data or retry
```bash
# Use cached data (if you have it)
python scripts/player_elo/train_model.py

# Or retry with fewer years
python scripts/player_elo/train_model.py --years 2024 2025
```

### Training takes too long (>1 hour)
**Solutions:**
1. Use fewer years: `--years 2023 2024 2025`
2. Reduce CV splits: `--cv-splits 3`
3. Check if you have a GPU available (automatically detected)

### Memory error
**Solution:** Train on fewer years or reduce batch size in `config.py`
```python
PYTORCH_CONFIG = {
    'batch_size': 32,  # Reduce from 64 to 32
    ...
}
```

---

## 📈 Performance Expectations

### Training Time
| Configuration | Time | Accuracy |
|--------------|------|----------|
| Full (2020-2025, 5-fold CV) | 45-60 min | Best (~5.1 MAE) |
| Recent (2023-2025, 5-fold CV) | 20-30 min | Good (~5.3 MAE) |
| Quick (2024-2025, 3-fold CV) | 10-15 min | Fair (~5.5 MAE) |

### Expected Performance
- **MAE:** 5.0-5.5 points (competitive with team-based 4.97)
- **Direction Accuracy:** 70-72% (correctly predict winner)
- **Comparison to baseline:** ~56% better than naive (predict 0)

---

## 🎯 Next Steps After Training

### Option 1: Implement Prediction Pipeline (Recommended)
To generate actual 2026 predictions, you need to implement `prediction_pipeline.py` (~250 lines).

**What it needs to do:**
1. Load trained model (`pytorch_model.pt`)
2. Load ELO state (`player_elo_state.json`)
3. Load 2026 team rosters
4. For each ACC game:
   - Get predicted lineups (top 5 by minutes)
   - Create feature vector
   - Generate prediction
5. Save to `tsa_pt_spread_PLAYER_ELO_2026.csv`

### Option 2: Use Team-Based System
The existing team-based system already works and achieves 4.97 MAE.
```bash
python scripts/train_model.py
```

### Option 3: Manual Predictions
Use the Python API to make predictions:
```python
from src.player_elo.training_pipeline import PlayerModelTrainer
from src.player_elo.pytorch_model import predict
import torch

# Load trained model
model = torch.load('data/player_data/models/pytorch_model.pt')

# Create features for a game (manually)
# ... (requires feature engineering)

# Predict
prediction = predict(model, features)
```

---

## 📚 Additional Resources

### Test Individual Components
```bash
# Player ELO system
python -m src.player_elo.player_elo_system

# Feature engineering
python -m src.player_elo.features

# PyTorch model
python -m src.player_elo.pytorch_model

# Data collector (slow - downloads data)
python -m src.player_elo.player_data_collector
```

### Full System Validation
```bash
# Comprehensive test with real data (15-20 min)
python scripts/player_elo/validate_system.py
```

### Documentation
- **Implementation Plan:** `~/.claude/plans/swift-questing-lantern.md`
- **Status Document:** `PLAYER_ELO_STATUS.md`
- **This Guide:** `PLAYER_ELO_USAGE.md`

---

## 🏆 Competition Submission

**Deadline:** February 6, 2026

**Required output:** `tsa_pt_spread_PLAYER_ELO_2026.csv` with columns:
- Date
- Home
- Away
- pt_spread
- team_name (CMMT)

**Current status:**
- ✅ Training pipeline complete
- ⏳ Prediction pipeline needed (~250 lines)
- ⏳ Final predictions not yet generated

**Estimated time to complete:** 2-3 hours

---

## 💡 Tips

1. **Start with quick training** to verify everything works:
   ```bash
   python scripts/player_elo/train_model.py --years 2024 2025 --cv-splits 3
   ```

2. **Monitor training** - look for:
   - MAE decreasing over epochs
   - Validation MAE < 6.0 (good)
   - Direction accuracy > 70% (excellent)

3. **Compare to team system** - player-based should be competitive:
   - Team MAE: 4.97
   - Player MAE target: 5.0-5.5

4. **Save time** - training takes longest, so:
   - Use cached data (`--no-cache` only if needed)
   - Train once, use for multiple prediction runs

---

**Last Updated:** February 2, 2026
**Status:** Training pipeline complete ✅ | Prediction pipeline pending ⏳
