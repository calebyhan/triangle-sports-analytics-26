
# Model Evaluation Report
Generated: 2026-01-15 15:13

## Training Data
- Synthetic matchups: 16882
- Based on Barttorvik efficiency ratings
- Game variance modeled as N(0, 11)

## Cross-Validation Results (5-fold Time Series CV)

| Model | MAE | RMSE |
|-------|-----|------|
| Linear Regression | 8.825 ± 0.182 | 11.082 ± 0.261 |
| Ridge Regression | 8.825 ± 0.182 | 11.082 ± 0.261 |
| Gradient Boosting | 8.913 ± 0.206 | 11.207 ± 0.288 |

## Prediction Intervals
- Method: Quantile Regression (10th, 50th, 90th percentiles)
- Coverage: 80.0% (target: 80%)
- Average Width: 28.31 points

## Model Parameters
- Selected Model: Ridge Regression (alpha=1.0)
- Intercept (HCA): 3.624

## Feature Importance
- eff_diff: 0.2858
- away_net_eff: -0.1440
- home_net_eff: 0.1418
- away_adj_de: 0.0723
- away_adj_oe: -0.0717
- home_adj_de: -0.0715
- home_adj_oe: 0.0703