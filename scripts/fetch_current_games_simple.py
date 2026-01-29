"""
Simple script to fetch current season games from Sports Reference.

Since CBBpy is hanging and other sources don't have 2025-26 data yet,
this script uses web scraping to get current season games.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import time

from src import config
from src.logger import setup_logger

logger = setup_logger(__name__)


def fetch_sports_reference_games(year: int = 2026) -> pd.DataFrame:
    """
    Scrape game results from Sports Reference for current season.

    Args:
        year: Season year (2026 for 2025-26 season)

    Returns:
        DataFrame with game results
    """
    url = f"https://www.sports-reference.com/cbb/seasons/men/{year}-schedule.html"
    logger.info(f"Fetching games from {url}")

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        # Find the schedule table
        table = soup.find('table', {'id': 'schedule'})

        if not table:
            logger.error("Could not find schedule table")
            return pd.DataFrame()

        # Parse table rows
        games = []
        rows = table.find('tbody').find_all('tr')

        for row in rows:
            # Skip header rows
            if row.get('class') and 'thead' in row.get('class'):
                continue

            cells = row.find_all('td')
            if len(cells) < 7:
                continue

            try:
                date_cell = row.find('th', {'data-stat': 'date_game'})
                date_str = date_cell.get('csk', '') if date_cell else ''

                if not date_str:
                    continue

                # Parse date
                game_date = datetime.strptime(date_str, '%Y-%m-%d')

                # Get teams and scores
                visitor_cell = row.find('td', {'data-stat': 'visitor_school_name'})
                home_cell = row.find('td', {'data-stat': 'home_school_name'})
                visitor_pts_cell = row.find('td', {'data-stat': 'visitor_pts'})
                home_pts_cell = row.find('td', {'data-stat': 'home_pts'})

                if not all([visitor_cell, home_cell, visitor_pts_cell, home_pts_cell]):
                    continue

                visitor_team = visitor_cell.get_text(strip=True)
                home_team = home_cell.get_text(strip=True)

                # Check if game has been played (has scores)
                visitor_pts = visitor_pts_cell.get_text(strip=True)
                home_pts = home_pts_cell.get_text(strip=True)

                if not visitor_pts or not home_pts:
                    # Game hasn't been played yet
                    continue

                visitor_score = float(visitor_pts)
                home_score = float(home_pts)

                # Check for neutral site
                neutral_icon = row.find('td', {'data-stat': 'game_location'})
                neutral_site = bool(neutral_icon and neutral_icon.get_text(strip=True) == 'N')

                games.append({
                    'date': game_date,
                    'home_team': home_team,
                    'away_team': visitor_team,
                    'home_score': home_score,
                    'away_score': visitor_score,
                    'neutral_site': neutral_site,
                    'season': year,
                    'margin': home_score - visitor_score
                })

            except Exception as e:
                logger.warning(f"Error parsing row: {e}")
                continue

        df = pd.DataFrame(games)
        logger.info(f"Fetched {len(df)} completed games")

        return df

    except Exception as e:
        logger.error(f"Error fetching games: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


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
    logger.info(f"\nFetching {current_season} season games from Sports Reference...")

    current_games = fetch_sports_reference_games(year=current_season)

    if current_games.empty:
        logger.warning("No games fetched for current season!")
        return

    logger.info(f"Fetched {len(current_games)} completed games")

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


if __name__ == "__main__":
    update_games_file()
