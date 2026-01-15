"""
Fetch real historical NCAA basketball game results using CBBpy
"""

import pandas as pd
import numpy as np
from cbbpy import mens_scraper
from datetime import datetime
import os
import time
from typing import List, Optional


def fetch_season_schedule(season: int, delay: float = 1.0) -> pd.DataFrame:
    """
    Fetch all D1 games for a season using CBBpy

    Args:
        season: Season year (e.g., 2024 for 2023-24 season)
        delay: Delay between requests to respect rate limits

    Returns:
        DataFrame with game results
    """
    print(f"Fetching {season} season games...")

    try:
        # CBBpy's get_games_season gets all D1 games for a season
        games_df = mens_scraper.get_games_season(season=season)

        if games_df is not None and not games_df.empty:
            print(f"  ✓ Fetched {len(games_df)} games for {season}")
            time.sleep(delay)
            return games_df
        else:
            print(f"  ✗ No games found for {season}")
            return pd.DataFrame()

    except Exception as e:
        print(f"  ✗ Error fetching {season}: {e}")
        return pd.DataFrame()


def fetch_conference_schedule(
    conference: str,
    season: int,
    delay: float = 1.0
) -> pd.DataFrame:
    """
    Fetch schedule for a specific conference

    Args:
        conference: Conference name (e.g., 'ACC', 'SEC')
        season: Season year
        delay: Delay between requests

    Returns:
        DataFrame with conference game results
    """
    print(f"Fetching {conference} schedule for {season}...")

    try:
        # Get conference schedule
        games_df = mens_scraper.get_games_conference(conference=conference, season=season)

        if games_df is not None and not games_df.empty:
            print(f"  ✓ Fetched {len(games_df)} games")
            time.sleep(delay)
            return games_df
        else:
            print(f"  ✗ No games found")
            return pd.DataFrame()

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return pd.DataFrame()


def standardize_game_data(games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize CBBpy game data to our format

    Expected output columns:
    - date: Game date
    - season: Season year
    - home_team: Home team name
    - away_team: Away team name
    - home_score: Home team score
    - away_score: Away team score
    - neutral_site: Boolean for neutral site games
    - margin: Point differential (home - away)
    """
    if games_df is None or games_df.empty:
        return pd.DataFrame()

    print(f"\nStandardizing {len(games_df)} games...")
    print(f"Available columns: {games_df.columns.tolist()}")

    # Map CBBpy columns to our standard format
    # CBBpy typically has: DATE, HOME, AWAY, HOME_SCORE, AWAY_SCORE, etc.
    standardized = pd.DataFrame()

    # Try common column name variations
    date_cols = ['DATE', 'Date', 'date', 'GAME_DATE']
    home_cols = ['HOME', 'Home', 'home', 'HOME_TEAM', 'home_team']
    away_cols = ['AWAY', 'Away', 'away', 'AWAY_TEAM', 'away_team']
    home_score_cols = ['HOME_SCORE', 'Home_Score', 'home_score', 'PTS_HOME']
    away_score_cols = ['AWAY_SCORE', 'Away_Score', 'away_score', 'PTS_AWAY']
    neutral_cols = ['NEUTRAL', 'Neutral', 'neutral', 'NEUTRAL_SITE', 'neutral_site']

    def find_column(df, possible_names):
        """Find first matching column name"""
        for name in possible_names:
            if name in df.columns:
                return name
        return None

    # Map columns
    date_col = find_column(games_df, date_cols)
    home_col = find_column(games_df, home_cols)
    away_col = find_column(games_df, away_cols)
    home_score_col = find_column(games_df, home_score_cols)
    away_score_col = find_column(games_df, away_score_cols)
    neutral_col = find_column(games_df, neutral_cols)

    if not all([date_col, home_col, away_col, home_score_col, away_score_col]):
        print(f"⚠️ Missing required columns!")
        print(f"  Date: {date_col}")
        print(f"  Home: {home_col}")
        print(f"  Away: {away_col}")
        print(f"  Home Score: {home_score_col}")
        print(f"  Away Score: {away_score_col}")
        return pd.DataFrame()

    standardized['date'] = pd.to_datetime(games_df[date_col])
    standardized['home_team'] = games_df[home_col].astype(str).str.strip()
    standardized['away_team'] = games_df[away_col].astype(str).str.strip()
    standardized['home_score'] = pd.to_numeric(games_df[home_score_col], errors='coerce')
    standardized['away_score'] = pd.to_numeric(games_df[away_score_col], errors='coerce')

    # Neutral site (default to False if not available)
    if neutral_col:
        standardized['neutral_site'] = games_df[neutral_col].fillna(False).astype(bool)
    else:
        standardized['neutral_site'] = False

    # Calculate margin
    standardized['margin'] = standardized['home_score'] - standardized['away_score']

    # Extract season from date if not provided
    if 'season' in games_df.columns:
        standardized['season'] = games_df['season']
    else:
        # Assume season year is the year if month >= 11, else previous year
        standardized['season'] = standardized['date'].apply(
            lambda x: x.year if x.month >= 11 else x.year
        )

    # Remove games with missing scores
    standardized = standardized.dropna(subset=['home_score', 'away_score'])

    print(f"✓ Standardized to {len(standardized)} valid games")

    return standardized


def fetch_multi_year_games(
    start_year: int = 2020,
    end_year: int = 2025,
    conferences: Optional[List[str]] = None,
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Fetch games across multiple years

    Args:
        start_year: First season to fetch
        end_year: Last season to fetch
        conferences: Optional list of conferences to filter (if None, gets all D1)
        save_path: Optional path to save results

    Returns:
        Combined DataFrame with all games
    """
    all_games = []

    for year in range(start_year, end_year + 1):
        if conferences:
            # Fetch specific conferences
            for conf in conferences:
                conf_games = fetch_conference_schedule(conf, year)
                if not conf_games.empty:
                    all_games.append(conf_games)
        else:
            # Fetch all D1 games
            season_games = fetch_season_schedule(year)
            if not season_games.empty:
                all_games.append(season_games)

    if not all_games:
        print("⚠️ No games fetched!")
        return pd.DataFrame()

    # Combine all games
    combined = pd.concat(all_games, ignore_index=True)

    # Standardize format
    standardized = standardize_game_data(combined)

    # Sort by date
    standardized = standardized.sort_values('date').reset_index(drop=True)

    print(f"\n{'='*60}")
    print(f"TOTAL GAMES FETCHED: {len(standardized)}")
    print(f"Date range: {standardized['date'].min()} to {standardized['date'].max()}")
    print(f"Seasons: {sorted(standardized['season'].unique())}")
    print(f"{'='*60}")

    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        standardized.to_csv(save_path, index=False)
        print(f"\n✓ Saved to: {save_path}")

    return standardized


def load_cached_games(filepath: str) -> Optional[pd.DataFrame]:
    """Load previously fetched games from cache"""
    if os.path.exists(filepath):
        print(f"Loading cached games from: {filepath}")
        df = pd.read_csv(filepath, parse_dates=['date'])
        print(f"✓ Loaded {len(df)} cached games")
        return df
    return None


if __name__ == "__main__":
    # Configure
    DATA_DIR = "data/raw/games"
    CACHE_FILE = os.path.join(DATA_DIR, "historical_games_2020_2025.csv")

    # Try loading cache first
    games = load_cached_games(CACHE_FILE)

    if games is None:
        # Fetch fresh data
        print("No cache found. Fetching fresh data from CBBpy...")
        print("This may take several minutes due to rate limiting...\n")

        # Fetch all D1 games for past 5 years
        games = fetch_multi_year_games(
            start_year=2020,
            end_year=2025,
            conferences=None,  # All D1 games
            save_path=CACHE_FILE
        )

    # Display summary
    if games is not None and not games.empty:
        print("\n" + "="*60)
        print("GAMES SUMMARY")
        print("="*60)
        print(f"Total games: {len(games)}")
        print(f"\nGames per season:")
        print(games['season'].value_counts().sort_index())
        print(f"\nNeutral site games: {games['neutral_site'].sum()}")
        print(f"Average margin: {games['margin'].mean():.2f} ± {games['margin'].std():.2f}")
        print("\nSample games:")
        print(games.head(10))
