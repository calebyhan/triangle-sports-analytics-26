"""
Compare baseline vs enhanced model performance.

Analyzes:
- Overall MAE comparison
- Performance by spread bucket
- Feature importance
- Statistical significance
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from src.elo import EloRatingSystem
from src.models import ImprovedSpreadModel
from src.utils import fetch_barttorvik_year
from src.features.enhanced_pipeline import EnhancedFeatureEngine
from sklearn.model_selection import TimeSeriesSplit
from scipy import stats
from src import config


def train_and_evaluate(feature_cols, X, y, label):
    """Train model and evaluate with cross-validation."""
    tscv = TimeSeriesSplit(n_splits=config.CV_CONFIG['n_splits'])
    cv_results = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx][feature_cols], X.iloc[val_idx][feature_cols]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = ImprovedSpreadModel(
            ridge_alpha=config.MODEL_CONFIG['ridge_alpha'],
            lgbm_params={
                'n_estimators': config.MODEL_CONFIG['n_estimators'],
                'max_depth': config.MODEL_CONFIG['max_depth'],
                'learning_rate': config.MODEL_CONFIG['learning_rate']
            },
            weights=(config.MODEL_CONFIG['ridge_weight'], config.MODEL_CONFIG['lgbm_weight'])
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_val)

        mae = np.abs(preds - y_val).mean()
        cv_results.append({
            'fold': fold + 1,
            'mae': mae,
            'predictions': preds,
            'actuals': y_val
        })

    return cv_results


def analyze_by_spread_bucket(cv_results, label):
    """Analyze performance by absolute spread magnitude."""
    all_preds = np.concatenate([r['predictions'] for r in cv_results])
    all_actuals = np.concatenate([r['actuals'] for r in cv_results])

    abs_actuals = np.abs(all_actuals)

    buckets = [
        (0, 5, "Close (<5)"),
        (5, 15, "Moderate (5-15)"),
        (15, np.inf, "Blowout (>15)")
    ]

    results = []
    for low, high, name in buckets:
        mask = (abs_actuals >= low) & (abs_actuals < high)
        if mask.sum() > 0:
            bucket_mae = np.abs(all_preds[mask] - all_actuals[mask]).mean()
            results.append({
                'bucket': name,
                'mae': bucket_mae,
                'count': mask.sum()
            })

    return pd.DataFrame(results)


def main():
    print("="*60)
    print("BASELINE VS ENHANCED MODEL COMPARISON")
    print("="*60)

    # ========================================================================
    # 1. Load and prepare data
    # ========================================================================
    print("\n1. Loading and preparing data...")
    games = pd.read_csv(config.HISTORICAL_GAMES_FILE, parse_dates=['date'])

    elo = EloRatingSystem(
        k_factor=config.ELO_CONFIG['k_factor'],
        hca=config.ELO_CONFIG['home_court_advantage'],
        carryover=config.ELO_CONFIG['season_carryover']
    )
    elo.load_conference_mappings(config.CONFERENCE_MAPPINGS)
    elo_snapshots = elo.process_games(
        games,
        date_col='date',
        home_col='home_team',
        away_col='away_team',
        home_score_col='home_score',
        away_score_col='away_score',
        neutral_col='neutral_site',
        season_col='season',
        save_snapshots=True
    )

    # Add neutral_site back
    if 'neutral_site' not in elo_snapshots.columns and 'neutral_site' in games.columns:
        elo_snapshots['neutral_site'] = games['neutral_site'].values

    print(f"   ✓ Processed {len(elo_snapshots)} games")

    # Load efficiency stats
    print("\n2. Loading efficiency stats...")
    all_stats = []
    for year in config.TRAINING_YEARS:
        df = fetch_barttorvik_year(year)
        df['season'] = year
        all_stats.append(df[['team', 'adjoe', 'adjde', 'season']])

    team_stats = pd.concat(all_stats, ignore_index=True)
    team_stats.columns = ['team', 'adj_oe', 'adj_de', 'season']
    team_stats['adj_em'] = team_stats['adj_oe'] - team_stats['adj_de']
    print(f"   ✓ Loaded {len(team_stats)} team-season efficiency records")

    # Load enhanced features
    print("\n3. Loading enhanced features...")
    enhanced_features_path = config.PROCESSED_DATA_DIR / 'enhanced_features_2020_2024.csv'
    enhanced_features = pd.read_csv(enhanced_features_path)
    print(f"   ✓ Loaded {len(enhanced_features)} enhanced feature rows")

    # Create training data
    print("\n4. Creating training dataset...")
    elo_snapshots['season'] = elo_snapshots['date'].dt.year
    elo_snapshots['_original_idx'] = range(len(elo_snapshots))

    train_data = elo_snapshots.merge(
        team_stats,
        left_on=['home_team', 'season'],
        right_on=['team', 'season'],
        how='left'
    ).rename(columns={'adj_oe': 'home_adj_oe', 'adj_de': 'home_adj_de', 'adj_em': 'home_adj_em'})

    train_data = train_data.drop(columns=['team'], errors='ignore').merge(
        team_stats,
        left_on=['away_team', 'season'],
        right_on=['team', 'season'],
        how='left'
    ).rename(columns={'adj_oe': 'away_adj_oe', 'adj_de': 'away_adj_de', 'adj_em': 'away_adj_em'})

    train_data = train_data.drop(columns=['team'], errors='ignore')
    train_data['eff_diff'] = train_data['home_adj_em'] - train_data['away_adj_em']
    train_data['elo_diff'] = train_data['home_elo_before'] - train_data['away_elo_before']
    train_data = train_data.dropna(subset=['home_adj_oe', 'away_adj_oe'])

    # Filter enhanced features to match
    valid_indices = train_data['_original_idx'].values
    enhanced_features_filtered = enhanced_features.iloc[valid_indices].reset_index(drop=True)
    train_data = train_data.reset_index(drop=True).drop(columns=['_original_idx'])
    train_data = pd.concat([train_data, enhanced_features_filtered], axis=1)

    print(f"   ✓ Created {len(train_data)} training samples")

    # ========================================================================
    # 5. Train and evaluate both models
    # ========================================================================
    y = train_data['actual_margin']

    print("\n5. Training baseline model (11 features)...")
    baseline_results = train_and_evaluate(config.BASELINE_FEATURES, train_data, y, "Baseline")
    baseline_maes = [r['mae'] for r in baseline_results]
    baseline_mean = np.mean(baseline_maes)
    baseline_std = np.std(baseline_maes)
    print(f"   ✓ Baseline MAE: {baseline_mean:.3f} ± {baseline_std:.3f}")

    print("\n6. Training enhanced model (24 features)...")
    all_features = config.BASELINE_FEATURES + config.ENHANCED_FEATURES
    available_features = [f for f in all_features if f in train_data.columns]
    enhanced_results = train_and_evaluate(available_features, train_data, y, "Enhanced")
    enhanced_maes = [r['mae'] for r in enhanced_results]
    enhanced_mean = np.mean(enhanced_maes)
    enhanced_std = np.std(enhanced_maes)
    print(f"   ✓ Enhanced MAE: {enhanced_mean:.3f} ± {enhanced_std:.3f}")

    # ========================================================================
    # 7. Statistical significance test
    # ========================================================================
    print("\n7. Statistical significance test...")
    t_stat, p_value = stats.ttest_rel(baseline_maes, enhanced_maes)

    improvement = baseline_mean - enhanced_mean
    improvement_pct = (improvement / baseline_mean) * 100

    print(f"   Mean improvement: {improvement:.3f} ({improvement_pct:+.1f}%)")
    print(f"   t-statistic: {t_stat:.3f}")
    print(f"   p-value: {p_value:.4f}")

    if p_value < 0.05:
        if improvement > 0:
            print("   ✓ Enhanced model is SIGNIFICANTLY BETTER (p < 0.05)")
        else:
            print("   ⚠ Enhanced model is SIGNIFICANTLY WORSE (p < 0.05)")
    else:
        print("   ≈ No significant difference (p >= 0.05)")

    # ========================================================================
    # 8. Performance by spread bucket
    # ========================================================================
    print("\n8. Performance by spread bucket...")
    baseline_buckets = analyze_by_spread_bucket(baseline_results, "Baseline")
    enhanced_buckets = analyze_by_spread_bucket(enhanced_results, "Enhanced")

    comparison = baseline_buckets.merge(
        enhanced_buckets,
        on='bucket',
        suffixes=('_baseline', '_enhanced')
    )
    comparison['improvement'] = comparison['mae_baseline'] - comparison['mae_enhanced']
    comparison['improvement_pct'] = (comparison['improvement'] / comparison['mae_baseline']) * 100

    print("\n   Bucket Analysis:")
    for _, row in comparison.iterrows():
        print(f"   {row['bucket']:20s}: Baseline={row['mae_baseline']:.3f}, Enhanced={row['mae_enhanced']:.3f}, "
              f"Improvement={row['improvement']:+.3f} ({row['improvement_pct']:+.1f}%)")

    # ========================================================================
    # 9. Feature importance (enhanced model only)
    # ========================================================================
    print("\n9. Training full enhanced model for feature importance...")
    X_full = train_data[available_features]
    full_model = ImprovedSpreadModel(
        ridge_alpha=config.MODEL_CONFIG['ridge_alpha'],
        lgbm_params={
            'n_estimators': config.MODEL_CONFIG['n_estimators'],
            'max_depth': config.MODEL_CONFIG['max_depth'],
            'learning_rate': config.MODEL_CONFIG['learning_rate']
        },
        weights=(config.MODEL_CONFIG['ridge_weight'], config.MODEL_CONFIG['lgbm_weight'])
    )
    full_model.fit(X_full, y)

    # Get LightGBM feature importance
    if hasattr(full_model, 'lgbm_model') and full_model.lgbm_model is not None:
        importance = full_model.lgbm_model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': available_features,
            'importance': importance
        }).sort_values('importance', ascending=False)

        print("\n   Top 15 Features (LightGBM importance):")
        for idx, row in feature_importance.head(15).iterrows():
            feature_type = "ENHANCED" if row['feature'] in config.ENHANCED_FEATURES else "baseline"
            print(f"      {row['feature']:30s}: {row['importance']:8.1f}  ({feature_type})")

        # Count baseline vs enhanced in top 10
        top10 = feature_importance.head(10)
        baseline_in_top10 = sum(1 for f in top10['feature'] if f in config.BASELINE_FEATURES)
        enhanced_in_top10 = sum(1 for f in top10['feature'] if f in config.ENHANCED_FEATURES)

        print(f"\n   Top 10 breakdown: {baseline_in_top10} baseline, {enhanced_in_top10} enhanced")

    # ========================================================================
    # 10. Summary
    # ========================================================================
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Baseline (11 features):  {baseline_mean:.3f} ± {baseline_std:.3f} MAE")
    print(f"Enhanced (24 features):  {enhanced_mean:.3f} ± {enhanced_std:.3f} MAE")
    print(f"Improvement:             {improvement:+.3f} ({improvement_pct:+.1f}%)")
    print(f"Statistical significance: {'YES' if p_value < 0.05 else 'NO'} (p={p_value:.4f})")

    print(f"\nDecision:")
    if p_value < 0.05 and improvement > 0.02:  # At least 2% improvement and significant
        print("✓ USE ENHANCED MODEL")
        print("  Enhanced features provide meaningful improvement")
    elif improvement < -0.01:  # Regression
        print("✗ STICK WITH BASELINE")
        print("  Enhanced features hurt performance")
    else:
        print("≈ STICK WITH BASELINE")
        print("  Enhanced features don't provide sufficient improvement")

    print(f"{'='*60}")


if __name__ == '__main__':
    main()
