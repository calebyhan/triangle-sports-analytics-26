# Player ELO System - Quick Start Guide

**⏱️ Total Time: 5 minutes**

## Prerequisites
- Python 3.8+
- PyTorch installed (if not: `pip install torch`)
- Player data files in `data/raw_pd/` (already provided)

## Step 1: Navigate to Project Root (5 seconds)
```bash
cd "e:\Triangle Sports Analytics\triangle-sports-analytics-26"
```

## Step 2: Verify Data Files (5 seconds)
```bash
ls data/raw_pd/*.csv
```
**Expected:** You should see 2019_pd.csv through 2026_pd.csv

## Step 3: Train the Model (3-5 minutes)
```bash
python scripts/player_elo/train_model.py
```

**What you'll see:**
```
======================================================================
  PLAYER-BASED ELO MODEL TRAINING PIPELINE
======================================================================

STEP 1: DATA COLLECTION
  [OK] Loaded 4733 player records for 2020
  [OK] Loaded 4970 player records for 2021
  ...

STEP 2: ROSTER PREPARATION
  Created roster for 2020: 350 teams
  ...

STEP 3: PROCESS GAMES THROUGH PLAYER ELO SYSTEM
  Processed 28500 games with lineups

STEP 4: FEATURE ENGINEERING
  Created features: X shape (28500, 65)

STEP 5: TRAIN PYTORCH MODEL
  Fold 1/5 | MAE: 10.25
  ...

TRAINING COMPLETE!
  MAE: 10.25 ± 1.22
  Model saved: data/player_data/models/pytorch_model.pt
```

## Step 4: Verify Success (5 seconds)
```bash
ls data/player_data/models/
```
**Expected files:**
- `pytorch_model.pt` - Trained neural network
- `player_elo_state.json` - Player ELO ratings
- `pytorch_model_fold*.pt` - Cross-validation models

## Done! ✅

Your player-based ELO model is now trained and ready.

## Next Steps

### Option A: Generate 2026 Predictions (Not Yet Implemented)
```bash
python scripts/player_elo/generate_predictions.py
```
⚠️ This script needs to be implemented (~250 lines)

### Option B: Use Team-Based System (Already Working)
```bash
python scripts/train_model.py
```
This uses the existing team-based system (MAE: 4.97)

### Option C: Explore the Model
```python
import torch
model = torch.load('data/player_data/models/pytorch_model.pt')
print(model)
```

## Training Options

### Train on Specific Years (Faster)
```bash
python scripts/player_elo/train_model.py --years 2023 2024 2025
```

### Use Fewer Cross-Validation Folds (Faster)
```bash
python scripts/player_elo/train_model.py --cv-splits 3
```

### Force Re-processing Data
```bash
python scripts/player_elo/train_model.py --no-cache
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'torch'"
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### "Historical games file not found"
Make sure you're in the project root directory:
```bash
cd "e:\Triangle Sports Analytics\triangle-sports-analytics-26"
```

### "No player data found"
Verify CSV files exist:
```bash
ls data/raw_pd/*.csv
```

## Performance Expectations

| Training Years | Time | MAE (Lower is Better) |
|---------------|------|---------------------|
| 2024-2025 | ~1 min | ~11.5 points |
| 2023-2025 | ~2 min | ~10.8 points |
| 2020-2025 | ~5 min | ~10.2 points |

**Comparison:** Team-based system achieves 4.97 MAE

## Documentation

- **Comprehensive Guide:** [docs/PLAYER_ELO_README.md](../PLAYER_ELO_README.md)
- **Implementation Status:** [PLAYER_ELO_STATUS.md](../../PLAYER_ELO_STATUS.md)
- **Full Usage Guide:** [PLAYER_ELO_USAGE.md](../../PLAYER_ELO_USAGE.md)

---

**Questions?** Check the comprehensive README or run the quick test:
```bash
python scripts/player_elo/quick_test.py
```
