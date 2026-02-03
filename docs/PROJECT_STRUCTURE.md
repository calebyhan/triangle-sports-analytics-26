# Triangle Sports Analytics - Project Structure

**Competition:** Triangle Sports Analytics Point Spread Prediction
**Team:** CMMT (Caleb Han, Mason Mines, Mason Wang, Tony Wang)
**Deadline:** February 6, 2026

---

## Overview

This project implements **two complementary prediction systems** for NCAA basketball point spreads:

1. **Team-Based ELO System** (Primary) - MAE: 4.97 points ✅
2. **Player-Based ELO System** (Advanced) - MAE: ~9.3 points ✅

---

## 📁 Directory Structure

```
triangle-sports-analytics-26/
│
├── 📚 docs/                          # All documentation
│   ├── PLAYER_ELO_README.md         # Comprehensive player system guide
│   ├── PROJECT_STRUCTURE.md         # This file
│   └── player_elo/
│       ├── INDEX.md                 # Documentation index
│       ├── QUICKSTART.md            # 5-minute quick start
│       ├── PLAYER_ELO_STATUS.md     # Technical implementation status
│       ├── PLAYER_ELO_USAGE.md      # Usage guide
│       └── CREATE_SAMPLE_DATA.md    # Data collection help
│
├── 🔧 src/                           # Source code
│   ├── elo.py                       # Team-based ELO system (560 lines)
│   ├── features.py                  # Team feature engineering (800 lines)
│   ├── models.py                    # Team ML models (500 lines)
│   ├── data_collection.py           # Team data collection
│   ├── utils.py                     # Shared utilities
│   │
│   └── player_elo/                  # Player-based ELO system ⭐ NEW
│       ├── __init__.py              # Package initialization
│       ├── config.py                # Configuration (270 lines)
│       ├── player_data_collector.py # Data loading (530 lines)
│       ├── roster_manager.py        # Roster management (340 lines)
│       ├── player_elo_system.py     # Player ELO logic (620 lines)
│       ├── features.py              # Feature engineering (440 lines)
│       ├── pytorch_model.py         # Neural network (380 lines)
│       └── training_pipeline.py     # Training orchestration (550 lines)
│
│       Total: 2,760 lines of production code
│
├── 🎯 scripts/                       # Executable scripts
│   ├── train_model.py               # Train team-based system
│   ├── generate_predictions.py      # Generate team predictions
│   │
│   └── player_elo/                  # Player system scripts
│       ├── train_model.py           # Train player system ⭐ MAIN
│       ├── quick_test.py            # Fast validation (10 sec)
│       └── validate_system.py       # Full validation (15 min)
│
├── 📊 data/                          # Data directories
│   ├── raw/
│   │   └── games/
│   │       └── historical_games_2019_2025.csv  # 33,746 games
│   │
│   ├── raw_pd/                      # Player data (manually provided)
│   │   ├── 2019_pd.csv              # 1.8 MB
│   │   ├── 2020_pd.csv              # 1.8 MB (4,733 players)
│   │   ├── 2021_pd.csv              # 1.9 MB (4,970 players)
│   │   ├── 2022_pd.csv              # 2.0 MB
│   │   ├── 2023_pd.csv              # 2.0 MB
│   │   ├── 2024_pd.csv              # 2.0 MB
│   │   ├── 2025_pd.csv              # 2.0 MB
│   │   └── 2026_pd.csv              # 1.9 MB (for predictions)
│   │
│   ├── player_data/                 # Player system data
│   │   ├── raw/
│   │   │   ├── player_stats/        # Processed stats
│   │   │   │   └── barttorvik_stats_2020_2025.csv
│   │   │   └── rosters/             # Team rosters by year
│   │   ├── processed/               # Transfer tracking
│   │   └── models/                  # Trained models ⭐
│   │       ├── pytorch_model.pt              # 78 KB - Final model
│   │       ├── pytorch_model_fold1.pt        # 78 KB - CV fold 1
│   │       ├── pytorch_model_fold2.pt        # 78 KB - CV fold 2
│   │       ├── pytorch_model_fold3.pt        # 78 KB - CV fold 3
│   │       ├── pytorch_model_fold4.pt        # 78 KB - CV fold 4
│   │       ├── pytorch_model_fold5.pt        # 78 KB - CV fold 5
│   │       └── player_elo_state.json         # 3.4 MB - Player ratings
│   │
│   ├── processed/                   # Team-based processed data
│   └── predictions/                 # Final predictions
│       └── tsa_pt_spread_CMMT_2026.csv  # Team-based predictions
│
├── 🧪 tests/                         # Test files
│   └── test_player_elo/
│       └── (tests to be added)
│
├── 📋 Configuration Files
│   ├── requirements.txt             # Python dependencies
│   ├── .gitignore
│   └── README.md                    # Main project README
│
└── 📝 Root Documentation
    └── (moved to docs/ folder)
```

---

## 🎯 Quick Access

### Train Team-Based System (Primary)
```bash
python scripts/train_model.py
```
**Output:** MAE ~4.97 points

### Train Player-Based System (Advanced)
```bash
python scripts/player_elo/train_model.py
```
**Output:** MAE ~9.3 points, 18,024 games processed

### Generate Predictions
```bash
# Team-based (working)
python scripts/generate_predictions.py

# Player-based (not yet implemented)
python scripts/player_elo/generate_predictions.py
```

---

## 📊 System Comparison

| Feature | Team-Based | Player-Based |
|---------|-----------|--------------|
| **Accuracy (MAE)** | 4.97 points ⭐ | 9.3 points |
| **Direction Acc** | ~71% | 70-74% |
| **Training Time** | ~2 min | ~5 min |
| **Model Type** | XGBoost | PyTorch NN |
| **Parameters** | ~100 features | 18,817 params |
| **Handles Roster Changes** | ❌ No | ✅ Yes |
| **Player-Level Insights** | ❌ No | ✅ Yes |
| **Complexity** | Low | High |
| **Status** | ✅ Complete | ⏳ 90% Complete |

---

## 📈 Data Flow

### Team-Based System
```
Historical Games → Team ELO → Team Features → XGBoost → Predictions
   33,746 games     560 lines    800 lines     500 lines    78 games
```

### Player-Based System
```
Player CSV → Load Data → Rosters → Player ELO → Features → PyTorch → Predictions
  15 MB       530 lines   340 lines  620 lines    440 lines  380 lines   78 games

  9,703 players → 18,024 games → 65D vectors → 18,817 params → Spreads
```

---

## 🚀 Implementation Status

### ✅ Completed (90%)
- [x] Team-based ELO system (100%)
- [x] Player data collection module (100%)
- [x] Player ELO tracking system (100%)
- [x] Roster management (100%)
- [x] Feature engineering (65D) (100%)
- [x] PyTorch neural network (100%)
- [x] Training pipeline (100%)
- [x] Full documentation (100%)
- [x] Training on real data (100%)

### ⏳ Remaining (10%)
- [ ] Lineup prediction module (optional, 300 lines)
- [ ] Prediction pipeline (required, 250 lines)
- [ ] Generate 2026 predictions (required)

**Estimated time to completion:** 2-3 hours

---

## 📚 Documentation Guide

### For Quick Start
→ [docs/player_elo/QUICKSTART.md](player_elo/QUICKSTART.md)

### For Comprehensive Guide
→ [docs/PLAYER_ELO_README.md](PLAYER_ELO_README.md)

### For Technical Details
→ [docs/player_elo/PLAYER_ELO_STATUS.md](player_elo/PLAYER_ELO_STATUS.md)

### For All Documentation
→ [docs/player_elo/INDEX.md](player_elo/INDEX.md)

---

## 💾 Storage Requirements

| Component | Size |
|-----------|------|
| Player CSV files (8 years) | 15 MB |
| Processed player stats | 5 MB |
| Trained models (7 files) | 546 KB |
| Player ELO state | 3.4 MB |
| **Total** | **~25 MB** |

---

## 🔧 Key Configuration Files

### Team System
- `src/elo.py` - Team ELO parameters
- `src/features.py` - Feature engineering
- `src/models.py` - Model configurations

### Player System
- `src/player_elo/config.py` - All player system parameters
  - Player ELO: K=20, default=1000, carryover=75%
  - PyTorch: 128-64-32 layers, dropout=0.2
  - Training: batch_size=64, lr=0.001

---

## 📦 Dependencies

### Core Dependencies
```txt
pandas>=1.5.0
numpy>=1.24.0
scikit-learn>=1.2.0
xgboost>=1.7.0
```

### Player System Additional
```txt
torch>=2.0.0
torchvision>=0.15.0
rapidfuzz>=3.0.0
```

**Install all:**
```bash
pip install -r requirements.txt
```

---

## 🎓 Learning Resources

### Understand the Team System
1. Read: `src/elo.py` (560 lines, well-commented)
2. Run: `python scripts/train_model.py`
3. Study: FiveThirtyEight ELO methodology

### Understand the Player System
1. Read: [docs/player_elo/QUICKSTART.md](player_elo/QUICKSTART.md)
2. Run: `python scripts/player_elo/quick_test.py`
3. Study: [docs/PLAYER_ELO_README.md](PLAYER_ELO_README.md)
4. Explore: Source code in `src/player_elo/`

---

## 🏆 Competition Deliverables

### Required Output
```csv
Date,Home,Away,pt_spread,team_name
2026-01-15,Duke,UNC,-5.2,CMMT
2026-01-18,Virginia,Louisville,3.1,CMMT
...
```

### Current Status
- ✅ Team-based predictions: Complete
- ⏳ Player-based predictions: Pipeline needed (~250 lines)

---

## 🔍 Key Insights

### Why Player System Has Higher MAE
1. **More complex:** Player-level tracking vs team-level
2. **Cold start:** New/transfer players start at team average
3. **Lineup uncertainty:** Using heuristics for starting lineups
4. **More parameters:** 18,817 params vs simpler team model

### Advantages of Player System
1. **Handles roster changes:** Transfers, injuries, graduations
2. **Player-level insights:** See individual contributions
3. **More flexible:** Can predict with different lineups
4. **Future-proof:** Tracks player development over time

---

## 📞 Support

### For Team System
- Check main `README.md`
- Review `src/elo.py` comments

### For Player System
- Quick help: [docs/player_elo/QUICKSTART.md](player_elo/QUICKSTART.md)
- Full guide: [docs/PLAYER_ELO_README.md](PLAYER_ELO_README.md)
- All docs: [docs/player_elo/INDEX.md](player_elo/INDEX.md)

---

**Last Updated:** February 2, 2026
**Project Status:** 90% Complete
**Team:** CMMT
