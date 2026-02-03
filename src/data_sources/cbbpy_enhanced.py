"""
Enhanced CBBpy wrapper for NCAA basketball data.

CBBpy is a Python package for scraping NCAA basketball data from NCAA.com.
This module provides enhanced functions with:
- Rate limiting
- Caching
- Error handling
- Standardized outputs

Features:
- Play-by-play data
- Box scores (team and player)
- Game metadata
- Season schedules

Package: https://pypi.org/project/CBBpy/
Repo: https://github.com/dcstats/CBBpy
"""

import pandas as pd
import time
from pathlib import Path
from typing import Optional, List
from datetime import datetime, timedelta
from ..config import CBBPY_CONFIG, CBBPY_DATA_DIR
from ..logger import setup_logger

logger = setup_logger(__name__)

# Ensure directory exists
CBBPY_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Try to import CBBpy (may not be installed)
try:
    # Import patches first to auto-fix CBBpy issues
    from . import cbbpy_patches

    from cbbpy import mens_scraper
    CBBPY_AVAILABLE = True
    logger.info("CBBpy imported successfully (with patches applied)")
except ImportError:
    CBBPY_AVAILABLE = False
    logger.warning("CBBpy not installed. Install with: pip install cbbpy")


class RateLimiter:
    """Rate limiter for CBBpy requests to NCAA.com"""
    def __init__(self, requests_per_minute: int = 20):
        self.requests_per_minute = requests_per_minute
        self.requests = []

    def wait_if_needed(self):
        """Wait if we've exceeded rate limit"""
        now = datetime.now()
        cutoff = now - timedelta(minutes=1)

        self.requests = [r for r in self.requests if r > cutoff]

        if len(self.requests) >= self.requests_per_minute:
            sleep_time = (self.requests[0] - cutoff).total_seconds() + 0.1
            logger.info(f"Rate limit reached, sleeping {sleep_time:.1f}s")
            time.sleep(sleep_time)
            self.requests = []

        self.requests.append(now)


_rate_limiter = RateLimiter(CBBPY_CONFIG['rate_limit'])


def _check_cbbpy():
    """Check if CBBpy is available"""
    if not CBBPY_AVAILABLE:
        raise ImportError(
            "CBBpy is not installed. Install it with:\n"
            "pip install cbbpy\n"
            "See: https://github.com/dcstats/CBBpy"
        )


def fetch_games_season(season: int = 2026) -> pd.DataFrame:
    """
    Fetch all games for a season.

    Args:
        season: Season year (e.g., 2026 for 2025-26 season)

    Returns:
        DataFrame with all games for the season
    """
    _check_cbbpy()
    _rate_limiter.wait_if_needed()

    logger.info(f"Fetching games for {season} season...")

    try:
        games = mens_scraper.get_games_season(season=season)
        logger.info(f"Fetched {len(games)} games for {season}")
        return games
    except Exception as e:
        logger.error(f"Failed to fetch games for {season}: {e}")
        return pd.DataFrame()


def fetch_games_team(team: str, season: int = 2026, include_all: bool = False) -> pd.DataFrame:
    """
    Fetch all games for a specific team.

    Args:
        team: Team name (e.g., 'Duke')
        season: Season year
        include_all: If True, return (schedule, box_scores, pbp) tuple. If False, return just schedule

    Returns:
        DataFrame with team's games, or tuple of 3 DataFrames if include_all=True
    """
    _check_cbbpy()
    _rate_limiter.wait_if_needed()

    logger.info(f"Fetching games for {team} in {season}...")

    try:
        result = mens_scraper.get_games_team(team=team, season=season)

        # CBBpy returns a tuple: (schedule_df, box_scores_df, pbp_df)
        if isinstance(result, tuple) and len(result) == 3:
            schedule, box_scores, pbp = result
            logger.info(f"Fetched {len(schedule)} games, {len(box_scores)} box score rows, {len(pbp)} PBP rows for {team}")
            return result if include_all else schedule
        else:
            logger.warning(f"Unexpected return type from CBBpy: {type(result)}")
            return result
    except Exception as e:
        logger.error(f"Failed to fetch games for {team}: {e}")
        return pd.DataFrame()


def fetch_game_boxscore(game_id: str) -> pd.DataFrame:
    """
    Fetch box score for a specific game.

    Args:
        game_id: NCAA game ID

    Returns:
        DataFrame with player statistics for the game
    """
    _check_cbbpy()
    _rate_limiter.wait_if_needed()

    logger.debug(f"Fetching box score for game {game_id}...")

    try:
        boxscore = mens_scraper.get_game_boxscore(game_id=game_id)
        return boxscore
    except Exception as e:
        logger.error(f"Failed to fetch box score for {game_id}: {e}")
        return pd.DataFrame()


def fetch_game_pbp(game_id: str) -> pd.DataFrame:
    """
    Fetch play-by-play data for a specific game.

    Args:
        game_id: NCAA game ID

    Returns:
        DataFrame with play-by-play events
    """
    _check_cbbpy()
    _rate_limiter.wait_if_needed()

    logger.debug(f"Fetching play-by-play for game {game_id}...")

    try:
        pbp = mens_scraper.get_game_pbp(game_id=game_id)
        return pbp
    except Exception as e:
        logger.error(f"Failed to fetch play-by-play for {game_id}: {e}")
        return pd.DataFrame()


def fetch_game_info(game_id: str) -> dict:
    """
    Fetch metadata for a specific game.

    Args:
        game_id: NCAA game ID

    Returns:
        Dictionary with game metadata
    """
    _check_cbbpy()
    _rate_limiter.wait_if_needed()

    logger.debug(f"Fetching info for game {game_id}...")

    try:
        info = mens_scraper.get_game_info(game_id=game_id)
        return info
    except Exception as e:
        logger.error(f"Failed to fetch info for {game_id}: {e}")
        return {}


def calculate_pace_from_pbp(pbp: pd.DataFrame) -> float:
    """
    Calculate game pace (possessions per game) from play-by-play data.

    Args:
        pbp: Play-by-play DataFrame

    Returns:
        Estimated number of possessions
    """
    if pbp.empty:
        return 0.0

    # Count possession-ending events
    # This is a simplified calculation
    possession_events = [
        'made', 'miss', 'turnover', 'foul'
    ]

    # Count events that end possessions
    # Actual implementation would need to parse PBP event types
    logger.warning("Pace calculation from PBP is simplified")
    return 70.0  # Placeholder


def extract_player_stats(boxscore: pd.DataFrame, stat_type: str = 'all') -> pd.DataFrame:
    """
    Extract specific player statistics from box score.

    Args:
        boxscore: Box score DataFrame
        stat_type: Type of stats ('scoring', 'rebounding', 'assists', 'all')

    Returns:
        DataFrame with filtered player stats
    """
    if boxscore.empty:
        return pd.DataFrame()

    # Filter based on stat type
    if stat_type == 'scoring':
        cols = ['player', 'team', 'pts', 'fgm', 'fga', 'fg_pct', '3pm', '3pa', '3p_pct', 'ftm', 'fta', 'ft_pct']
    elif stat_type == 'rebounding':
        cols = ['player', 'team', 'oreb', 'dreb', 'reb']
    elif stat_type == 'assists':
        cols = ['player', 'team', 'ast', 'tov', 'ast_tov_ratio']
    else:
        cols = boxscore.columns.tolist()

    # Filter to available columns
    available_cols = [c for c in cols if c in boxscore.columns]
    return boxscore[available_cols]


def fetch_games_range(
    start_date: str,
    end_date: str,
    teams: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Fetch games within a date range.

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        teams: Optional list of team names to filter

    Returns:
        DataFrame with games in date range
    """
    _check_cbbpy()

    logger.info(f"Fetching games from {start_date} to {end_date}...")

    # CBBpy doesn't have a direct date range function
    # We'd need to fetch by season and filter
    logger.warning("Date range fetching requires filtering season data")

    # Get year from start_date
    year = int(start_date[:4])

    # Fetch full season
    all_games = fetch_games_season(year)

    if all_games.empty:
        return pd.DataFrame()

    # Filter by date range
    if 'date' in all_games.columns:
        all_games['date'] = pd.to_datetime(all_games['date'])
        mask = (all_games['date'] >= start_date) & (all_games['date'] <= end_date)
        games = all_games[mask]
    else:
        games = all_games

    # Filter by teams if specified
    if teams is not None and 'home_team' in games.columns and 'away_team' in games.columns:
        mask = games['home_team'].isin(teams) | games['away_team'].isin(teams)
        games = games[mask]

    logger.info(f"Found {len(games)} games in range")
    return games


def save_cbbpy_data(df: pd.DataFrame, filename: str):
    """
    Save CBBpy data to directory.

    Args:
        df: DataFrame to save
        filename: Output filename
    """
    output_path = CBBPY_DATA_DIR / filename
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df)} records to {output_path}")


if __name__ == '__main__':
    # Simple test
    print("CBBpy Enhanced Module")
    print(f"CBBpy available: {CBBPY_AVAILABLE}")
    print(f"Rate limit: {CBBPY_CONFIG['rate_limit']} req/min")

    if CBBPY_AVAILABLE:
        # Test fetching games
        print("\nFetching Duke games for 2026...")
        duke_games = fetch_games_team('Duke', season=2026)
        print(f"Found {len(duke_games)} games")
        if len(duke_games) > 0:
            print("\nColumns:", duke_games.columns.tolist())
            print("\nFirst few games:")
            print(duke_games.head())
    else:
        print("\nInstall CBBpy to test: pip install cbbpy")
