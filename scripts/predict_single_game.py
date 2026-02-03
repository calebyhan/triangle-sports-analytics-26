"""
Predict a single game using the trained model.

Usage:
    python scripts/predict_single_game.py --home "North Carolina" --away "Notre Dame" --date 2025-01-21
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import pandas as pd
from datetime import datetime

from src import config
from src.elo import EloRatingSystem
from src.models import ImprovedSpreadModel
from src.utils import fetch_barttorvik_year
from src.logger import setup_logger

logger = setup_logger(__name__)


def train_model():
    """Train the model on historical data."""
    logger.info("Loading historical games for model training...")
    games_path = config.HISTORICAL_GAMES_FILE

    if not games_path.exists():
        raise FileNotFoundError(f"Historical games file not found: {games_path}")

    games = pd.read_csv(games_path, parse_dates=['date'])

    # Initialize Elo
    logger.info("Processing games through Elo system...")
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

    # Load efficiency stats for training years
    logger.info("Loading team efficiency stats...")
    all_stats = []
    for year in config.TRAINING_YEARS:
        df = fetch_barttorvik_year(year)
        df['season'] = year
        all_stats.append(df[['team', 'adjoe', 'adjde', 'season']])

    team_stats = pd.concat(all_stats, ignore_index=True)
    team_stats.columns = ['team', 'adj_oe', 'adj_de', 'season']
    team_stats['adj_em'] = team_stats['adj_oe'] - team_stats['adj_de']

    # Merge to create training data
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

    # Train model
    logger.info(f"Training model on {len(train_data)} games...")
    feature_cols = config.BASELINE_FEATURES

    X = train_data[feature_cols]
    y = train_data['actual_margin']

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
    logger.info("Model trained successfully")

    return model, elo


def predict_game(home_team: str, away_team: str, game_date: str):
    """
    Predict a single game.

    Args:
        home_team: Home team name
        away_team: Away team name
        game_date: Game date (YYYY-MM-DD)
    """
    # Train model
    model, elo = train_model()

    # Parse date to get season
    date_obj = datetime.strptime(game_date, '%Y-%m-%d')
    season = date_obj.year

    # Load current season efficiency stats
    logger.info(f"Loading {season} season efficiency stats...")
    team_stats = fetch_barttorvik_year(season)

    # Find team stats
    home_stats = team_stats[team_stats['team'] == home_team]
    away_stats = team_stats[team_stats['team'] == away_team]

    if home_stats.empty:
        logger.error(f"Could not find stats for {home_team}")
        print(f"\nAvailable teams containing '{home_team.split()[0]}':")
        matching = team_stats[team_stats['team'].str.contains(home_team.split()[0], case=False)]
        for team in matching['team'].values[:10]:
            print(f"  - {team}")
        return

    if away_stats.empty:
        logger.error(f"Could not find stats for {away_team}")
        print(f"\nAvailable teams containing '{away_team.split()[0]}':")
        matching = team_stats[team_stats['team'].str.contains(away_team.split()[0], case=False)]
        for team in matching['team'].values[:10]:
            print(f"  - {team}")
        return

    home_stats = home_stats.iloc[0]
    away_stats = away_stats.iloc[0]

    # Build features
    home_oe = home_stats['adjoe']
    home_de = home_stats['adjde']
    away_oe = away_stats['adjoe']
    away_de = away_stats['adjde']

    features = {
        'home_adj_oe': home_oe,
        'home_adj_de': home_de,
        'home_adj_em': home_oe - home_de,
        'away_adj_oe': away_oe,
        'away_adj_de': away_de,
        'away_adj_em': away_oe - away_de,
        'eff_diff': (home_oe - home_de) - (away_oe - away_de),
        'home_elo_before': elo.get_rating(home_team),
        'away_elo_before': elo.get_rating(away_team),
        'elo_diff': elo.get_rating(home_team) - elo.get_rating(away_team),
        'predicted_spread': elo.predict_spread(home_team, away_team),
    }

    X = pd.DataFrame([features])[config.BASELINE_FEATURES]
    prediction = model.predict(X)[0]

    # Print results
    print("\n" + "="*70)
    print(f"GAME PREDICTION: {home_team} vs {away_team}")
    print(f"Date: {game_date}")
    print("="*70)
    print(f"\nHome Team: {home_team}")
    print(f"  AdjOE: {home_oe:.1f}")
    print(f"  AdjDE: {home_de:.1f}")
    print(f"  AdjEM: {home_oe - home_de:.1f}")
    print(f"  Elo Rating: {elo.get_rating(home_team):.1f}")

    print(f"\nAway Team: {away_team}")
    print(f"  AdjOE: {away_oe:.1f}")
    print(f"  AdjDE: {away_de:.1f}")
    print(f"  AdjEM: {away_oe - away_de:.1f}")
    print(f"  Elo Rating: {elo.get_rating(away_team):.1f}")

    print(f"\n{'PREDICTED SPREAD:':<25} {home_team} {prediction:+.1f}")
    if prediction > 0:
        print(f"{'PREDICTION:':<25} {home_team} by {prediction:.1f}")
    elif prediction < 0:
        print(f"{'PREDICTION:':<25} {away_team} by {-prediction:.1f}")
    else:
        print(f"{'PREDICTION:':<25} Even game")

    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict a single game')
    parser.add_argument('--home', required=True, help='Home team name')
    parser.add_argument('--away', required=True, help='Away team name')
    parser.add_argument('--date', required=True, help='Game date (YYYY-MM-DD)')

    args = parser.parse_args()

    predict_game(args.home, args.away, args.date)
