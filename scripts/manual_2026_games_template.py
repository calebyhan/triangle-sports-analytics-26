"""
Template for manually adding 2025-26 season games.

INSTRUCTIONS:
1. Visit https://barttorvik.com/schedule.php?year=2025
2. Download the schedule data (or copy key games)
3. Add the games to the games_data list below
4. Run this script

Or alternatively, create a CSV file directly and load it.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src import config
from src.logger import setup_logger

logger = setup_logger(__name__)


def create_2026_games_from_manual_data():
    """
    Manually add 2025-26 season games here.

    Format: (date, home_team, away_team, home_score, away_score, neutral_site)
    """

    # ADD REAL GAMES HERE - these are just examples
    games_data = [
        # Format: ('YYYY-MM-DD', 'Home Team', 'Away Team', home_score, away_score, neutral_bool)
        ('2025-11-04', 'Duke', 'Maine', 96, 62, False),
        ('2025-11-04', 'North Carolina', 'Elon', 90, 76, False),
        ('2025-11-05', 'Virginia', 'Coppin St.', 85, 58, False),
        # ADD MORE GAMES HERE...
    ]

    # Convert to DataFrame
    df = pd.DataFrame(games_data, columns=[
        'date', 'home_team', 'away_team', 'home_score', 'away_score', 'neutral_site'
    ])

    # Process
    df['date'] = pd.to_datetime(df['date'])
    df['margin'] = df['home_score'] - df['away_score']
    df['season'] = 2026

    logger.info(f"Created {len(df)} manual games for 2025-26 season")

    return df[['date', 'home_team', 'away_team', 'home_score', 'away_score',
               'neutral_site', 'season', 'margin']]


def load_2026_games_from_csv(filepath: str):
    """
    Load 2025-26 games from a CSV file.

    Expected columns: date, home_team, away_team, home_score, away_score
    """
    logger.info(f"Loading 2025-26 games from {filepath}...")

    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])

    # Add season and neutral_site if missing
    if 'season' not in df.columns:
        df['season'] = 2026

    if 'neutral_site' not in df.columns:
        df['neutral_site'] = False

    if 'margin' not in df.columns:
        df['margin'] = df['home_score'] - df['away_score']

    logger.info(f"Loaded {len(df)} games")

    return df[['date', 'home_team', 'away_team', 'home_score', 'away_score',
               'neutral_site', 'season', 'margin']]


def main():
    logger.info("="*70)
    logger.info("ADDING 2025-26 SEASON GAMES")
    logger.info("="*70)

    # Option 1: Load from CSV (if you have it)
    csv_path = config.DATA_DIR / "raw" / "games" / "2025-26_results.csv"

    if csv_path.exists():
        logger.info(f"Found CSV file: {csv_path}")
        games_2026 = load_2026_games_from_csv(csv_path)
    else:
        logger.info("No CSV file found. Using manual data from script...")
        games_2026 = create_2026_games_from_manual_data()

    # Load historical games
    historical_path = config.HISTORICAL_GAMES_FILE
    historical_games = pd.read_csv(historical_path, parse_dates=['date'])

    logger.info(f"Historical games (2019-2025): {len(historical_games):,}")

    # Combine
    combined = pd.concat([historical_games, games_2026], ignore_index=True)

    # Remove duplicates
    before_dedup = len(combined)
    combined = combined.drop_duplicates(
        subset=['date', 'home_team', 'away_team'],
        keep='first'
    )
    if len(combined) < before_dedup:
        logger.info(f"Removed {before_dedup - len(combined)} duplicates")

    # Sort
    combined = combined.sort_values('date').reset_index(drop=True)

    # Save
    output_path = config.DATA_DIR / "raw" / "games" / "historical_games_with_current.csv"
    combined.to_csv(output_path, index=False)

    logger.info("\n" + "="*70)
    logger.info("SUMMARY")
    logger.info("="*70)
    logger.info(f"Historical (2019-2025): {len(historical_games):,}")
    logger.info(f"Current season (2026):  {len(games_2026):,}")
    logger.info(f"Total combined:         {len(combined):,}")
    logger.info(f"\nDate range: {combined['date'].min()} to {combined['date'].max()}")
    logger.info(f"\nSaved to: {output_path}")
    logger.info("="*70)

    # Show recent games
    if len(games_2026) > 0:
        logger.info("\nMost recent 2025-26 games added:")
        recent = games_2026.tail(5)
        for _, row in recent.iterrows():
            logger.info(
                f"  {row['date'].strftime('%Y-%m-%d')}: "
                f"{row['away_team']} @ {row['home_team']} "
                f"({int(row['away_score'])}-{int(row['home_score'])})"
            )


if __name__ == "__main__":
    main()
