"""
Train model on real historical game data
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
from sklearn.model_selection import TimeSeriesSplit
from src import config


def main():
    print("="*60)
    print("TRAINING ON REAL HISTORICAL GAME DATA")
    print("="*60)

    # 1. Load real historical games
    print("\n1. Loading real historical games...")
    games_path = config.HISTORICAL_GAMES_FILE

    # Validate file exists
    if not games_path.exists():
        raise FileNotFoundError(
            f"Historical games file not found: {games_path}\n"
            f"Please ensure the data file exists or run the data collection script."
        )

    games = pd.read_csv(games_path, parse_dates=['date'])

    # Validate data loaded correctly
    required_cols = ['date', 'home_team', 'away_team', 'home_score', 'away_score']
    missing_cols = [col for col in required_cols if col not in games.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in games data: {missing_cols}")

    if len(games) == 0:
        raise ValueError("Historical games file is empty")

    print(f"   ✓ Loaded {len(games)} games from {games['date'].min()} to {games['date'].max()}")

    # 2. Initialize Elo and process games chronologically
    print("\n2. Processing games chronologically through Elo system...")
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

    print(f"   ✓ Processed {len(elo_snapshots)} games, tracking {len(elo.ratings)} teams")

    # 3. Load efficiency stats
    print("\n3. Loading team efficiency stats from Barttorvik...")
    all_stats = []
    for year in config.TRAINING_YEARS:
        df = fetch_barttorvik_year(year)
        df['season'] = year
        all_stats.append(df[['team', 'adjoe', 'adjde', 'season']])

    team_stats = pd.concat(all_stats, ignore_index=True)
    team_stats.columns = ['team', 'adj_oe', 'adj_de', 'season']
    team_stats['adj_em'] = team_stats['adj_oe'] - team_stats['adj_de']
    print(f"   ✓ Loaded {len(team_stats)} team-season efficiency records")

    # 4. Merge to create training data
    print("\n4. Creating training dataset...")
    elo_snapshots['season'] = elo_snapshots['date'].dt.year

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

    print(f"   ✓ Created {len(train_data)} training samples")

    # 5. Train model
    print("\n5. Training model on real data...")
    feature_cols = [
        'home_adj_oe', 'home_adj_de', 'home_adj_em',
        'away_adj_oe', 'away_adj_de', 'away_adj_em',
        'eff_diff',
        'home_elo_before', 'away_elo_before', 'elo_diff', 'predicted_spread'
    ]

    X = train_data[feature_cols]
    y = train_data['actual_margin']

    # Updated with tuned hyperparameters (8.9% improvement: 5.459 -> 4.972 MAE)
    model = ImprovedSpreadModel(
        ridge_alpha=config.MODEL_CONFIG['ridge_alpha'],
        lgbm_params={
            'n_estimators': config.MODEL_CONFIG['n_estimators'],
            'max_depth': config.MODEL_CONFIG['max_depth'],
            'learning_rate': config.MODEL_CONFIG['learning_rate']
        },
        weights=(config.MODEL_CONFIG['ridge_weight'], config.MODEL_CONFIG['lgbm_weight']),
        use_lgbm=True
    )

    model.fit(X, y)
    print("   ✓ Model trained!")

    # 6. Cross-validation
    print(f"\n6. Running {config.CV_CONFIG['n_splits']}-fold time-series cross-validation...")
    tscv = TimeSeriesSplit(n_splits=config.CV_CONFIG['n_splits'])
    cv_results = {'ridge': [], 'ensemble': []}

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        fold_model = ImprovedSpreadModel(
            lgbm_params={
                'n_estimators': config.MODEL_CONFIG['n_estimators'],
                'max_depth': config.MODEL_CONFIG['max_depth'],
                'learning_rate': config.MODEL_CONFIG['learning_rate']
            },
            weights=(config.MODEL_CONFIG['ridge_weight'], config.MODEL_CONFIG['lgbm_weight'])
        )
        fold_model.fit(X_train, y_train)

        preds = fold_model.predict(X_val)
        components = fold_model.predict_components(X_val)

        ridge_mae = np.abs(components['ridge'] - y_val).mean()
        ensemble_mae = np.abs(preds - y_val).mean()

        cv_results['ridge'].append(ridge_mae)
        cv_results['ensemble'].append(ensemble_mae)

        print(f"   Fold {fold+1}: Ridge={ridge_mae:.3f}, Ensemble={ensemble_mae:.3f}")

    ridge_mean = np.mean(cv_results['ridge'])
    ridge_std = np.std(cv_results['ridge'])
    ensemble_mean = np.mean(cv_results['ensemble'])
    ensemble_std = np.std(cv_results['ensemble'])

    print(f"\n   Ridge CV MAE:    {ridge_mean:.3f} ± {ridge_std:.3f}")
    print(f"   Ensemble CV MAE: {ensemble_mean:.3f} ± {ensemble_std:.3f}")

    # 7. Generate 2026 predictions
    print(f"\n7. Generating {config.PREDICTION_YEAR} predictions...")
    team_stats_2026 = pd.read_csv(config.PROCESSED_DATA_DIR / 'team_stats_2025_26.csv')
    template = pd.read_csv(config.PROJECT_ROOT / config.SUBMISSION_TEMPLATE)
    template = template.dropna(subset=['Home', 'Away'])

    team_dict = team_stats_2026.set_index('team').to_dict('index')
    pred_features = []
    valid_indices = []

    for idx, row in template.iterrows():
        home, away = row['Home'], row['Away']
        if home not in team_dict or away not in team_dict:
            continue

        home_stats = team_dict[home]
        away_stats = team_dict[away]

        home_oe = home_stats.get('off_efficiency', 100)
        home_de = home_stats.get('def_efficiency', 100)
        away_oe = away_stats.get('off_efficiency', 100)
        away_de = away_stats.get('def_efficiency', 100)

        features = {
            'home_adj_oe': home_oe,
            'home_adj_de': home_de,
            'home_adj_em': home_oe - home_de,
            'away_adj_oe': away_oe,
            'away_adj_de': away_de,
            'away_adj_em': away_oe - away_de,
            'eff_diff': (home_oe - home_de) - (away_oe - away_de),
            'home_elo_before': elo.get_rating(home),
            'away_elo_before': elo.get_rating(away),
            'elo_diff': elo.get_rating(home) - elo.get_rating(away),
            'predicted_spread': elo.predict_spread(home, away),
        }

        pred_features.append(features)
        valid_indices.append(idx)

    X_pred = pd.DataFrame(pred_features)
    predictions = model.predict(X_pred)

    results = template.copy()
    for i, idx in enumerate(valid_indices):
        results.loc[idx, 'pt_spread'] = predictions[i]

    # 8. Save predictions
    print("\n8. Saving predictions...")
    submission = results[['Date', 'Away', 'Home', 'pt_spread']].copy()
    submission = submission.dropna(subset=['pt_spread'])

    submission['team_name'] = ''
    submission['team_member'] = ''
    submission['team_email'] = ''

    # Add team info from config
    for i, member in enumerate(config.TEAM_INFO['members'][:len(submission)]):
        if i == 0:
            submission.loc[submission.index[i], 'team_name'] = config.TEAM_INFO['team_name']
        submission.loc[submission.index[i], 'team_member'] = member['name']
        submission.loc[submission.index[i], 'team_email'] = member['email']

    main_path = config.PREDICTION_OUTPUT_FILE
    # Ensure directory exists
    main_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(main_path, index=False)
    print(f"   ✓ Created: {main_path}")

    # 9. Generate SHAP explanations (optional)
    try:
        import shap
        from src.model_explainer import ModelExplainer

        print("\n9. Generating model interpretability report...")
        # Sample background data for SHAP
        X_sample = X.sample(min(500, len(X)), random_state=42)

        explainer = ModelExplainer(model, X_sample)

        # Ensure outputs directory exists
        config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

        # Generate summary plot
        explainer.summary_plot(X_sample, save_path=str(config.OUTPUTS_DIR / 'shap_summary.png'))

        print("   ✓ Model interpretability analysis complete")

    except ImportError:
        print("\n9. SHAP not installed, skipping interpretability")
        print("   Install with: pip install shap")
    except Exception as e:
        print(f"\n9. ⚠ SHAP analysis failed: {e}")
        print("   Continuing without interpretability report...")

    # Summary
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Training samples: {len(train_data)} real games")
    print(f"Predictions: {len(submission)} games")
    print(f"\nModel Performance (Cross-Validation):")
    print(f"  Ridge MAE:    {ridge_mean:.3f} ± {ridge_std:.3f}")
    print(f"  Ensemble MAE: {ensemble_mean:.3f} ± {ensemble_std:.3f}")

    baseline_mae = 11.41  # Naive baseline (predict 0)
    improvement = baseline_mae - ridge_mean
    print(f"\nImprovement from baseline ({baseline_mae} MAE):")
    print(f"  {improvement:+.3f} points ({abs(improvement)/baseline_mae*100:.1f}% better)")

    print("="*60)


if __name__ == "__main__":
    main()
