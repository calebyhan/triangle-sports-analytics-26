"""
Update historical games file with current season (2025-26) games.

This script:
1. Fetches 2025-26 season games using CBBpy
2. Standardizes format to match historical_games_2019_2025.csv
3. Saves to a combined file for use in predictions
4. Updates Elo ratings with current season results

Usage:
    python scripts/update_current_season_games.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from datetime import datetime

from src import config
from src.fetch_real_games import fetch_season_schedule
from src.logger import setup_logger

logger = setup_logger(__name__)


def standardize_cbbpy_format(df: pd.DataFrame, season: int) -> pd.DataFrame:
    """
    Standardize CBBpy game data to match historical games format.

    Args:
        df: Raw CBBpy DataFrame
        season: Season year (e.g., 2026 for 2025-26)

    Returns:
        Standardized DataFrame with columns: date, home_team, away_team,
        home_score, away_score, neutral_site, season, margin
    """
    logger.info(f"Standardizing {len(df)} games from CBBpy format...")

    # CBBpy columns may vary, let's check what we have
    logger.info(f"Available columns: {df.columns.tolist()}")

    # Common CBBpy column mappings (check actual column names)
    column_mapping = {
        'DATE': 'date',
        'AWAY': 'away_team',
        'HOME': 'home_team',
        'AWAY_SCORE': 'away_score',
        'HOME_SCORE': 'home_score',
    }

    # Try different possible column names
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns:
            df = df.rename(columns={old_col: new_col})
        # Try lowercase versions
        elif old_col.lower() in df.columns:
            df = df.rename(columns={old_col.lower(): new_col})

    # Ensure required columns exist
    required_cols = ['date', 'home_team', 'away_team', 'home_score', 'away_score']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        # Print first few rows to help debug
        logger.error(f"Missing columns: {missing_cols}")
        logger.info(f"First row:\n{df.head(1).to_dict('records')}")
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Convert date to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])

    # Add neutral site flag (CBBpy may have this as 'LOCATION' or similar)
    if 'neutral_site' not in df.columns:
        if 'LOCATION' in df.columns:
            df['neutral_site'] = df['LOCATION'].str.upper() == 'N'
        elif 'location' in df.columns:
            df['neutral_site'] = df['location'].str.upper() == 'N'
        else:
            # Default to False if no location info
            df['neutral_site'] = False

    # Calculate margin (home_score - away_score)
    df['margin'] = df['home_score'] - df['away_score']

    # Add season
    df['season'] = season

    # Select and order columns to match historical format
    output_cols = ['date', 'home_team', 'away_team', 'home_score', 'away_score',
                   'neutral_site', 'season', 'margin']

    df = df[output_cols].copy()

    # Remove any games without scores (future games)
    df = df.dropna(subset=['home_score', 'away_score'])

    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)

    logger.info(f"Standardized {len(df)} completed games")

    return df


def update_games_file():
    """
    Fetch current season games and combine with historical data.
    """
    logger.info("="*70)
    logger.info("UPDATING GAMES DATA WITH CURRENT SEASON")
    logger.info("="*70)

    # Load historical games
    historical_path = config.HISTORICAL_GAMES_FILE
    logger.info(f"Loading historical games from {historical_path}...")

    if not historical_path.exists():
        raise FileNotFoundError(f"Historical games file not found: {historical_path}")

    historical_games = pd.read_csv(historical_path, parse_dates=['date'])
    logger.info(f"Loaded {len(historical_games)} historical games (2019-2025)")
    logger.info(f"Date range: {historical_games['date'].min()} to {historical_games['date'].max()}")

    # Fetch current season games
    current_season = config.PREDICTION_YEAR
    logger.info(f"\nFetching {current_season} season games using CBBpy...")

    try:
        current_games = fetch_season_schedule(season=current_season, delay=1.0)

        if current_games.empty:
            logger.warning("No games fetched for current season!")
            return

        logger.info(f"Fetched {len(current_games)} games from CBBpy")

        # Standardize format
        current_games = standardize_cbbpy_format(current_games, season=current_season)
        logger.info(f"After filtering completed games: {len(current_games)} games")

        if current_games.empty:
            logger.warning("No completed games found for current season yet")
            return

        # Combine with historical
        logger.info("\nCombining with historical data...")
        combined = pd.concat([historical_games, current_games], ignore_index=True)
        combined = combined.sort_values('date').reset_index(drop=True)

        # Remove duplicates (in case of overlap)
        before_dedup = len(combined)
        combined = combined.drop_duplicates(
            subset=['date', 'home_team', 'away_team'],
            keep='first'
        )
        if len(combined) < before_dedup:
            logger.info(f"Removed {before_dedup - len(combined)} duplicate games")

        # Save combined file
        output_path = config.DATA_DIR / "raw" / "games" / "historical_games_with_current.csv"
        combined.to_csv(output_path, index=False)

        logger.info("\n" + "="*70)
        logger.info("SUMMARY")
        logger.info("="*70)
        logger.info(f"Historical games (2019-2025):  {len(historical_games):,}")
        logger.info(f"Current season games ({current_season}):  {len(current_games):,}")
        logger.info(f"Total combined games:           {len(combined):,}")
        logger.info(f"Date range:                     {combined['date'].min()} to {combined['date'].max()}")
        logger.info(f"\nSaved to: {output_path}")
        logger.info("="*70)

        # Show most recent games
        logger.info("\nMost recent 10 games:")
        recent = combined.tail(10)[['date', 'home_team', 'away_team', 'home_score', 'away_score', 'margin']]
        for _, row in recent.iterrows():
            logger.info(f"  {row['date'].strftime('%Y-%m-%d')}: {row['away_team']} @ {row['home_team']} "
                       f"({row['away_score']:.0f}-{row['home_score']:.0f})")

    except Exception as e:
        logger.error(f"Error fetching/processing current season games: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    update_games_file()
