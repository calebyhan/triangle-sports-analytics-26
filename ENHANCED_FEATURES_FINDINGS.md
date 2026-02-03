# Enhanced Features Investigation - Findings Report

## Executive Summary

**Recommendation: STICK WITH BASELINE MODEL**

Enhanced features (momentum, blowout tendency, player stats, team HCA) **significantly hurt** model performance by **-2.3%** (p=0.0131). The baseline 11-feature model should be used for production predictions.

---

## Performance Comparison

| Model | Features | MAE | Std Dev | vs Baseline |
|-------|----------|-----|---------|-------------|
| **Baseline** | 11 | **5.535** | ±0.281 | - |
| Enhanced | 24 | 5.662 | ±0.300 | **-2.3% worse** |

**Statistical Significance**: YES (t=-4.256, p=0.0131)

The regression is **consistent across all spread buckets**:
- Close games (<5): -2.3% worse
- Moderate (5-15): -2.3% worse
- Blowouts (>15): -2.3% worse

---

## Root Cause Analysis

### 1. Data Quality Issues

#### 29.7% of games have all-zero momentum features
- Early season games lack historical data
- Default values (all zeros) add noise to the model
- No meaningful signal for ~10,000 games

#### 35.2% of games have all-zero player features
- Missing player data for early season games
- Empty features provide no predictive value

#### 19.6% of games have extreme `bench_depth_diff` values (>1000)
- **Bug in calculation**: Values range from -3098 to +3098
- Should be ~-20 to +20 (points per game differential)
- Caused by summing averaged stats across all historical players
- Source: `player_features.py:165` - `bench_ppg = bench_players['pts'].sum()`

### 2. Feature Engineering Problems

- **No normalization**: Features on wildly different scales
  - `star_power_diff`: ~-100 to +100
  - `bench_depth_diff`: ~-3000 to +3000
  - `momentum_diff`: ~-70 to +100

- **Default values everywhere**: HCA defaults to 3.5 for most teams
- **Insufficient history**: Early season features have no data to work with

### 3. Model Impact

Adding 13 noisy/buggy features:
- Increases model complexity without adding signal
- Introduces noise that hurts generalization
- Regularization cannot fully compensate
- Result: Statistically significant **-2.3% regression**

---

## Technical Implementation Details

### What Was Attempted

1. **Data Collection** (2 hours):
   - Collected 1.15M player-game records (2020-2024)
   - Fixed missing game_date column
   - Processed 1,919 team-season combinations

2. **Feature Computation** (~3 minutes):
   - Computed 13 enhanced features for 33,746 games
   - Momentum: win streaks, recent form, margin trends
   - Blowout: consistency, hot streaks, blowout tendency
   - Player: star power, bench depth, balance, efficiency
   - HCA: Team-specific home court advantage

3. **Training & Validation**:
   - Fixed data alignment issues (train_data vs enhanced_features)
   - Fixed neutral_site KeyError
   - Ran 5-fold time-series cross-validation
   - Performed statistical significance testing

### What Worked

- ✓ Data pipeline successfully collected historical player data
- ✓ Feature computation ran without crashes
- ✓ Temporal integrity maintained (no future data leakage)
- ✓ Proper train/test splitting with time-series CV

### What Didn't Work

- ✗ Enhanced features hurt performance instead of helping
- ✗ Bench depth calculation has critical bugs
- ✗ Too many games with default/zero features (30-35%)
- ✗ Feature scaling inconsistent across features
- ✗ Player data availability insufficient for many games

---

## Lessons Learned

1. **More features ≠ better model**
   - Adding noisy features actively hurts performance
   - Quality > Quantity

2. **Feature engineering is hard**
   - Small bugs (like bench_depth sum) can break everything
   - Validation beyond "does it run" is critical
   - Unit tests for feature ranges would have caught bugs

3. **Data availability matters**
   - 30-35% of games lack sufficient historical data
   - Default values add noise, not signal
   - Early season predictions are challenging

4. **Baseline models are underrated**
   - Elo + Efficiency stats (11 features) is robust
   - Simple features with good data > complex features with poor data
   - The baseline was already well-tuned

---

## Recommendations

### Immediate Action
✓ **Use baseline model (11 features) for 2026 predictions**
- Proven performance: 5.535 MAE
- No data quality issues
- Simple, interpretable, robust

### Future Work (If pursuing enhanced features)

1. **Fix bench_depth calculation**:
   ```python
   # Current (WRONG):
   bench_ppg = bench_players['pts'].sum()  # Sums all players' averages

   # Should be:
   bench_ppg = bench_players['pts'].mean()  # Average per-player production
   ```

2. **Add feature normalization**:
   - StandardScaler or MinMaxScaler
   - Consistent scaling across all enhanced features

3. **Handle missing data better**:
   - Skip enhanced features for games with <5 historical games
   - Use only baseline for early season
   - Implement confidence intervals

4. **Add validation**:
   - Unit tests for feature value ranges
   - Sanity checks (e.g., bench_depth should be -50 to +50)
   - Per-feature validation before training

5. **Feature selection**:
   - Try features individually, not all at once
   - Use feature importance to identify useful features
   - Drop features that don't improve validation performance

6. **Consider simpler alternatives**:
   - Just add recent form (last 5 games average margin)
   - Just add star player PPG differential
   - Start with 1-2 features, not 13

---

## Files Modified

### New Files Created
1. `scripts/collect_historical_players.py` - Player data collection
2. `scripts/fetch_game_dates_from_schedules.py` - Game date extraction
3. `src/features/enhanced_pipeline.py` - Enhanced feature computation
4. `scripts/train_model_enhanced.py` - Enhanced model training
5. `scripts/compare_baseline_vs_enhanced.py` - A/B comparison
6. `scripts/validate_temporal_integrity.py` - Validation framework
7. `data/processed/historical_player_box_scores_2020_2024.csv` - 1.15M records
8. `data/processed/enhanced_features_2020_2024.csv` - 33,746 rows × 13 features

### Files Modified
1. `src/config.py` - Added ENHANCED_FEATURES configuration

### Existing Files with Issues
1. `src/features/player_features.py:165` - Bench depth calculation bug

---

## Performance Metrics Summary

```
BASELINE (11 features):
  Overall MAE: 5.535 ± 0.281
  Close games (<5): 5.397 MAE
  Moderate (5-15): 3.913 MAE
  Blowouts (>15): 8.057 MAE

ENHANCED (24 features):
  Overall MAE: 5.662 ± 0.300  ⚠ WORSE
  Close games (<5): 5.521 MAE  ⚠ WORSE
  Moderate (5-15): 4.005 MAE  ⚠ WORSE
  Blowouts (>15): 8.239 MAE  ⚠ WORSE

Regression: -0.127 MAE (-2.3%)
Statistical significance: p=0.0131 (highly significant)
```

---

## Conclusion

The enhanced features experiment demonstrates that adding features without careful validation and quality control can significantly hurt model performance. The baseline model's simplicity and robustness make it the clear choice for production use.

**Stick with the baseline model.**

---

*Generated: 2026-02-03*
*Total investigation time: ~6 hours*
*Training data: 8,850 games (2020-2025)*
