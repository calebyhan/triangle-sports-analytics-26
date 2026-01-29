"""
Convert Barttorvik 2025-26 game results to our standard format.

This script converts team-centric game data (one row per team per game)
to game-centric format (one row per game with home/away).

Usage:
    1. Download from: https://barttorvik.com/gamestat.php?year=2026&csv=1
    2. Save to: data/raw/games/2025-26_results_raw.csv
    3. Run: python scripts/convert_barttorvik_2026_games.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src import config
from src.logger import setup_logger

logger = setup_logger(__name__)


def convert_team_centric_to_game_centric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert team-centric format to game-centric format.

    Team-centric: Each row is one team's perspective of a game
    Game-centric: Each row is one game with home_team, away_team

    Input columns (barttorvik format):
        year, month, day, team, opponent, location, teamscore, oppscore, etc.

    Output columns:
        date, home_team, away_team, home_score, away_score, neutral_site, season, margin
    """
    logger.info(f"Converting {len(df)} team-game records to game-centric format...")

    # Create date column
    df['date'] = pd.to_datetime(df[['year', 'month', 'day']])

    # Determine home/away based on location
    # H = Home, V = Away/Visitor, N = Neutral
    home_games = df[df['location'] == 'H'].copy()
    away_games = df[df['location'] == 'V'].copy()
    neutral_games = df[df['location'] == 'N'].copy()

    logger.info(f"  Home games: {len(home_games)}")
    logger.info(f"  Away games: {len(away_games)}")
    logger.info(f"  Neutral games: {len(neutral_games)}")

    # Process home games
    home_games_formatted = pd.DataFrame({
        'date': home_games['date'],
        'home_team': home_games['team'],
        'away_team': home_games['opponent'],
        'home_score': home_games['teamscore'],
        'away_score': home_games['oppscore'],
        'neutral_site': False,
        'season': 2026,
    })

    # Process away games (flip perspective)
    away_games_formatted = pd.DataFrame({
        'date': away_games['date'],
        'home_team': away_games['opponent'],
        'away_team': away_games['team'],
        'home_score': away_games['oppscore'],
        'away_score': away_games['teamscore'],
        'neutral_site': False,
        'season': 2026,
    })

    # Process neutral games
    # For neutral games, we need to deduplicate by picking one team as "home"
    # We'll arbitrarily use alphabetical order
    neutral_games_formatted = []

    if len(neutral_games) > 0:
        # Group neutral games by date and team pair
        for _, row in neutral_games.iterrows():
            team1 = row['team']
            team2 = row['opponent']

            # Use alphabetical order to determine "home" vs "away"
            if team1 < team2:
                neutral_games_formatted.append({
                    'date': row['date'],
                    'home_team': team1,
                    'away_team': team2,
                    'home_score': row['teamscore'],
                    'away_score': row['oppscore'],
                    'neutral_site': True,
                    'season': 2026,
                })

        neutral_games_formatted = pd.DataFrame(neutral_games_formatted)
        logger.info(f"  Processed {len(neutral_games_formatted)} unique neutral site games")
    else:
        neutral_games_formatted = pd.DataFrame(columns=[
            'date', 'home_team', 'away_team', 'home_score', 'away_score', 'neutral_site', 'season'
        ])

    # Combine all games
    all_games = pd.concat([
        home_games_formatted,
        away_games_formatted,
        neutral_games_formatted
    ], ignore_index=True)

    # Remove duplicates (from neutral games appearing twice)
    before_dedup = len(all_games)
    all_games = all_games.drop_duplicates(
        subset=['date', 'home_team', 'away_team'],
        keep='first'
    )

    if len(all_games) < before_dedup:
        logger.info(f"  Removed {before_dedup - len(all_games)} duplicate games")

    # Remove games without scores (future games)
    all_games = all_games.dropna(subset=['home_score', 'away_score'])

    # Calculate margin
    all_games['margin'] = all_games['home_score'] - all_games['away_score']

    # Sort by date
    all_games = all_games.sort_values('date').reset_index(drop=True)

    logger.info(f"✓ Converted to {len(all_games)} unique games")

    return all_games[['date', 'home_team', 'away_team', 'home_score', 'away_score',
                     'neutral_site', 'season', 'margin']]


def main():
    logger.info("="*70)
    logger.info("CONVERTING BARTTORVIK 2025-26 GAMES TO STANDARD FORMAT")
    logger.info("="*70)

    # Input file (manually downloaded)
    input_file = config.DATA_DIR / "raw" / "games" / "2025-26_results_raw.csv"

    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        logger.info("\nPlease download the file:")
        logger.info("1. Visit: https://barttorvik.com/gamestat.php?year=2026&csv=1")
        logger.info(f"2. Save to: {input_file}")
        logger.info("3. Run this script again")
        return

    # Load raw data
    logger.info(f"Loading data from {input_file}...")
    raw_data = pd.read_csv(input_file)

    logger.info(f"Loaded {len(raw_data)} team-game records")
    logger.info(f"Columns: {raw_data.columns.tolist()}")

    # Check for required columns
    required_cols = ['year', 'month', 'day', 'team', 'opponent', 'location', 'teamscore', 'oppscore']
    missing_cols = [col for col in required_cols if col not in raw_data.columns]

    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        logger.info(f"Available columns: {raw_data.columns.tolist()}")
        return

    # Convert to game-centric format
    games_2026 = convert_team_centric_to_game_centric(raw_data)

    # Save converted file
    output_file = config.DATA_DIR / "raw" / "games" / "2025-26_results.csv"
    games_2026.to_csv(output_file, index=False)
    logger.info(f"\n✓ Saved to: {output_file}")

    # Combine with historical data
    logger.info("\nCombining with historical data...")
    historical_path = config.HISTORICAL_GAMES_FILE
    historical_games = pd.read_csv(historical_path, parse_dates=['date'])

    logger.info(f"Historical games (2019-2025): {len(historical_games):,}")

    combined = pd.concat([historical_games, games_2026], ignore_index=True)
    combined = combined.sort_values('date').reset_index(drop=True)

    # Save combined file
    combined_path = config.DATA_DIR / "raw" / "games" / "historical_games_with_current.csv"
    combined.to_csv(combined_path, index=False)

    logger.info("\n" + "="*70)
    logger.info("SUMMARY")
    logger.info("="*70)
    logger.info(f"2025-26 games:          {len(games_2026):,}")
    logger.info(f"Historical games:       {len(historical_games):,}")
    logger.info(f"Total combined:         {len(combined):,}")
    logger.info(f"\nDate range: {combined['date'].min()} to {combined['date'].max()}")
    logger.info(f"\nSaved to:")
    logger.info(f"  1. {output_file}")
    logger.info(f"  2. {combined_path}")
    logger.info("="*70)

    # Show recent games
    logger.info("\nMost recent 10 games from 2025-26:")
    recent = games_2026.tail(10)
    for _, row in recent.iterrows():
        logger.info(
            f"  {row['date'].strftime('%Y-%m-%d')}: "
            f"{row['away_team']} @ {row['home_team']} "
            f"({int(row['away_score'])}-{int(row['home_score'])})"
        )

    logger.info("\n✓ Ready for holdout validation!")
    logger.info("Next: Run notebooks/03_holdout_validation.ipynb")


if __name__ == "__main__":
    main()
