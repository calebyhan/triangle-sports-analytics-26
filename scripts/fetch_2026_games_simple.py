"""
Simple script to fetch 2025-26 season games using web scraping.
Falls back to manual CSV creation if scraping fails.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import requests
from datetime import datetime
from src import config
from src.logger import setup_logger

logger = setup_logger(__name__)


def fetch_games_from_sports_reference(season: int = 2026) -> pd.DataFrame:
    """
    Fetch games from Sports Reference (simple HTML scraping).

    For 2025-26 season, we need to fetch from the schedule page.
    """
    logger.info(f"Attempting to fetch {season} season games from Sports Reference...")

    # Sports Reference URL pattern
    # For 2025-26 season, games started in November 2025
    url = f"https://www.sports-reference.com/cbb/seasons/men/{season}-schedule.html"

    logger.info(f"URL: {url}")

    try:
        # Add headers to avoid being blocked
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        # Parse HTML tables using pandas
        tables = pd.read_html(response.text)

        if not tables:
            logger.warning("No tables found on page")
            return pd.DataFrame()

        # The schedule table should be one of the first tables
        for i, df in enumerate(tables):
            logger.info(f"Table {i} columns: {df.columns.tolist()}")
            logger.info(f"Table {i} shape: {df.shape}")

            # Look for tables with date, teams, and scores
            if 'Date' in df.columns or 'Visitor' in df.columns or 'Home' in df.columns:
                logger.info(f"Found potential schedule table at index {i}")
                return df

        logger.warning("Could not identify schedule table")
        return pd.DataFrame()

    except Exception as e:
        logger.error(f"Error fetching from Sports Reference: {e}")
        return pd.DataFrame()


def standardize_sports_ref_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize Sports Reference data to our format.
    """
    if df.empty:
        return df

    logger.info(f"Standardizing Sports Reference data...")
    logger.info(f"Columns: {df.columns.tolist()}")

    # Common Sports Reference columns
    # Typical format: Date, Visitor, PTS, Home, PTS

    standardized = pd.DataFrame()

    # Try to map columns
    try:
        # Date
        if 'Date' in df.columns:
            standardized['date'] = pd.to_datetime(df['Date'], errors='coerce')

        # Teams and scores
        if 'Visitor' in df.columns and 'Home' in df.columns:
            standardized['away_team'] = df['Visitor'].astype(str).str.strip()
            standardized['home_team'] = df['Home'].astype(str).str.strip()

            # Scores - may be in PTS columns or next to team names
            if 'PTS' in df.columns:
                # Multiple PTS columns - first is visitor, second is home
                pts_cols = [col for col in df.columns if 'PTS' in str(col)]
                if len(pts_cols) >= 2:
                    standardized['away_score'] = pd.to_numeric(df[pts_cols[0]], errors='coerce')
                    standardized['home_score'] = pd.to_numeric(df[pts_cols[1]], errors='coerce')

        # Neutral site - usually marked with @ or N
        if 'Location' in df.columns:
            standardized['neutral_site'] = df['Location'].str.contains('N', na=False)
        else:
            standardized['neutral_site'] = False

        # Calculate margin
        if 'home_score' in standardized.columns and 'away_score' in standardized.columns:
            standardized['margin'] = standardized['home_score'] - standardized['away_score']

        # Season
        standardized['season'] = 2026

        # Remove games without scores (future games)
        standardized = standardized.dropna(subset=['home_score', 'away_score'])

        logger.info(f"Standardized {len(standardized)} games")

        return standardized[['date', 'home_team', 'away_team', 'home_score',
                           'away_score', 'neutral_site', 'season', 'margin']]

    except Exception as e:
        logger.error(f"Error standardizing data: {e}")
        logger.info(f"Sample row: {df.head(1).to_dict('records')}")
        return pd.DataFrame()


def create_sample_2026_games() -> pd.DataFrame:
    """
    Create a sample dataset for demonstration if web scraping fails.

    Note: This is a PLACEHOLDER. For production, you should manually
    add real game results or use a working data source.
    """
    logger.warning("Creating PLACEHOLDER dataset - you need to add real 2025-26 games!")

    # This is just for structure demonstration
    sample_games = {
        'date': ['2025-11-04', '2025-11-05', '2025-11-06'],
        'home_team': ['Duke', 'North Carolina', 'Virginia'],
        'away_team': ['Maine', 'Elon', 'Coppin St.'],
        'home_score': [96, 90, 85],
        'away_score': [62, 76, 58],
        'neutral_site': [False, False, False],
        'season': [2026, 2026, 2026],
        'margin': [34, 14, 27]
    }

    df = pd.DataFrame(sample_games)
    df['date'] = pd.to_datetime(df['date'])

    return df


def main():
    """Main execution."""
    logger.info("="*70)
    logger.info("FETCHING 2025-26 SEASON GAMES")
    logger.info("="*70)

    # Try Sports Reference first
    games_2026 = fetch_games_from_sports_reference(2026)

    if not games_2026.empty:
        games_2026 = standardize_sports_ref_format(games_2026)

    # If that failed, create placeholder
    if games_2026.empty:
        logger.warning("\nCould not fetch games from Sports Reference")
        logger.info("\nOPTIONS:")
        logger.info("1. Manually create CSV with 2025-26 games")
        logger.info("2. Use cbbpy (if working): pip install cbbpy")
        logger.info("3. Download from barttorvik.com manually")
        logger.info("\nCreating PLACEHOLDER file for now...")
        games_2026 = create_sample_2026_games()

    # Load historical games
    historical_path = config.HISTORICAL_GAMES_FILE
    historical_games = pd.read_csv(historical_path, parse_dates=['date'])

    # Combine
    combined = pd.concat([historical_games, games_2026], ignore_index=True)
    combined = combined.sort_values('date').reset_index(drop=True)

    # Save
    output_path = config.DATA_DIR / "raw" / "games" / "historical_games_with_current.csv"
    combined.to_csv(output_path, index=False)

    logger.info("\n" + "="*70)
    logger.info("SUMMARY")
    logger.info("="*70)
    logger.info(f"Historical games: {len(historical_games):,}")
    logger.info(f"2025-26 games:    {len(games_2026):,}")
    logger.info(f"Total:            {len(combined):,}")
    logger.info(f"\nSaved to: {output_path}")
    logger.info("="*70)

    if len(games_2026) <= 10:
        logger.warning("\n⚠️  WARNING: Very few 2025-26 games found!")
        logger.warning("You need to manually add real game results for accurate validation.")


if __name__ == "__main__":
    main()
