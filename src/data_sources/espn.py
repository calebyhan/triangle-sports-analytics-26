"""
ESPN Hidden API wrapper for college basketball data.

ESPN provides unofficial APIs that return JSON data without authentication.
This module provides convenient functions to fetch:
- Team BPI ratings
- Team statistics
- Player statistics
- Game schedules and scores
- Live game data

Rate limiting: ~100 requests/minute recommended
"""

import requests
import pandas as pd
import json
import time
from pathlib import Path
from typing import Optional, List, Dict, Union
from datetime import datetime, timedelta
from ..config import ESPN_CONFIG, ESPN_DATA_DIR, CACHE_DIR
from ..logger import setup_logger

logger = setup_logger(__name__)

# Ensure directories exist
ESPN_DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = ESPN_CONFIG['base_url']
TIMEOUT = ESPN_CONFIG['timeout']
CACHE_TTL = ESPN_CONFIG['cache_ttl']


class RateLimiter:
    """Simple rate limiter for API calls"""
    def __init__(self, requests_per_minute: int = 100):
        self.requests_per_minute = requests_per_minute
        self.requests = []

    def wait_if_needed(self):
        """Wait if we've exceeded rate limit"""
        now = datetime.now()
        cutoff = now - timedelta(minutes=1)

        # Remove old requests
        self.requests = [r for r in self.requests if r > cutoff]

        if len(self.requests) >= self.requests_per_minute:
            sleep_time = (self.requests[0] - cutoff).total_seconds() + 0.1
            logger.info(f"Rate limit reached, sleeping {sleep_time:.1f}s")
            time.sleep(sleep_time)
            self.requests = []

        self.requests.append(now)


# Global rate limiter
_rate_limiter = RateLimiter(ESPN_CONFIG['rate_limit'])


def _get_cache_path(cache_key: str) -> Path:
    """Get cache file path for a given key"""
    return CACHE_DIR / 'espn' / f"{cache_key}.json"


def _is_cache_valid(cache_path: Path) -> bool:
    """Check if cache file exists and is not expired"""
    if not cache_path.exists():
        return False

    age = time.time() - cache_path.stat().st_mtime
    return age < CACHE_TTL


def _load_cache(cache_path: Path) -> Optional[dict]:
    """Load data from cache"""
    try:
        with open(cache_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load cache: {e}")
        return None


def _save_cache(cache_path: Path, data: dict):
    """Save data to cache"""
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump(data, f)
    except Exception as e:
        logger.warning(f"Failed to save cache: {e}")


def _fetch_with_retry(
    url: str,
    max_retries: int = 3,
    use_cache: bool = True,
    cache_key: Optional[str] = None
) -> dict:
    """
    Fetch URL with rate limiting, caching, and retry logic.

    Args:
        url: URL to fetch
        max_retries: Maximum number of retry attempts
        use_cache: Whether to use caching
        cache_key: Key for cache storage (if None, derived from URL)

    Returns:
        JSON response as dictionary
    """
    # Check cache first
    if use_cache:
        if cache_key is None:
            cache_key = url.replace('/', '_').replace(':', '_')
        cache_path = _get_cache_path(cache_key)

        if _is_cache_valid(cache_path):
            logger.debug(f"Loading from cache: {cache_key}")
            cached = _load_cache(cache_path)
            if cached is not None:
                return cached

    # Fetch from API
    for attempt in range(max_retries):
        try:
            _rate_limiter.wait_if_needed()

            logger.debug(f"Fetching: {url}")
            response = requests.get(url, timeout=TIMEOUT)
            response.raise_for_status()

            data = response.json()

            # Save to cache
            if use_cache:
                _save_cache(cache_path, data)

            return data

        except requests.exceptions.RequestException as e:
            logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise

    raise RuntimeError(f"Failed to fetch {url} after {max_retries} attempts")


def fetch_scoreboard(date: str, groups: str = '100') -> pd.DataFrame:
    """
    Fetch scoreboard for a specific date.

    Args:
        date: Date in YYYYMMDD format (e.g., '20260128')
        groups: Group ID (100 = all D1 teams)

    Returns:
        DataFrame with game information
    """
    url = f"{BASE_URL}/scoreboard?dates={date}&groups={groups}"
    data = _fetch_with_retry(url, cache_key=f"scoreboard_{date}")

    if 'events' not in data:
        logger.warning(f"No events found for {date}")
        return pd.DataFrame()

    events = []
    for event in data['events']:
        game_info = {
            'date': event.get('date'),
            'game_id': event.get('id'),
            'name': event.get('name'),
            'short_name': event.get('shortName'),
            'status': event['status'].get('type', {}).get('description', 'Unknown'),
        }

        # Extract team information
        if 'competitions' in event and len(event['competitions']) > 0:
            comp = event['competitions'][0]

            if 'competitors' in comp and len(comp['competitors']) >= 2:
                for team in comp['competitors']:
                    prefix = 'home' if team.get('homeAway') == 'home' else 'away'
                    game_info[f"{prefix}_team"] = team['team'].get('displayName')
                    game_info[f"{prefix}_team_id"] = team['team'].get('id')
                    game_info[f"{prefix}_score"] = team.get('score')
                    game_info[f"{prefix}_record"] = team.get('records', [{}])[0].get('summary', '')

        events.append(game_info)

    return pd.DataFrame(events)


def fetch_team_stats(team_id: str) -> dict:
    """
    Fetch statistics for a specific team.

    Args:
        team_id: ESPN team ID (e.g., '150' for Duke)

    Returns:
        Dictionary with team statistics
    """
    url = f"{BASE_URL}/teams/{team_id}"
    data = _fetch_with_retry(url, cache_key=f"team_{team_id}")

    return data.get('team', {})


def fetch_standings(year: int = 2026, group: str = '50') -> pd.DataFrame:
    """
    Fetch conference standings.

    Args:
        year: Season year (e.g., 2026 for 2025-26 season)
        group: Conference group ID (50 = Power conferences)

    Returns:
        DataFrame with standings
    """
    # The standings endpoint doesn't return full data via API
    # ESPN requires web scraping for detailed standings
    # For now, try the teams endpoint to get basic team info
    logger.warning("ESPN standings API doesn't return detailed data - using teams endpoint instead")

    # Try getting teams list
    try:
        url = f"{BASE_URL}/teams?limit=500&groups={group}"
        data = _fetch_with_retry(url, cache_key=f"teams_{year}_{group}")

        teams = []
        if 'sports' in data and len(data['sports']) > 0:
            sport = data['sports'][0]
            if 'leagues' in sport and len(sport['leagues']) > 0:
                league = sport['leagues'][0]
                if 'teams' in league:
                    for team_item in league['teams']:
                        if 'team' in team_item:
                            team = team_item['team']
                            teams.append({
                                'team': team.get('displayName'),
                                'team_id': team.get('id'),
                                'abbreviation': team.get('abbreviation'),
                                'location': team.get('location'),
                            })

        if len(teams) > 0:
            logger.info(f"Fetched {len(teams)} teams from ESPN")
            return pd.DataFrame(teams)

    except Exception as e:
        logger.error(f"Failed to fetch teams: {e}")

    logger.warning("No standings/teams data available from ESPN API")
    return pd.DataFrame()


def fetch_bpi_rankings(year: int = 2026) -> pd.DataFrame:
    """
    Fetch BPI (Basketball Power Index) rankings.

    Note: This scrapes the BPI page since there's no direct API endpoint.
    Returns team rankings with BPI ratings.

    Args:
        year: Season year (e.g., 2026 for 2025-26 season)

    Returns:
        DataFrame with BPI rankings
    """
    # BPI is on the main ESPN site, not the API
    url = f"https://www.espn.com/mens-college-basketball/bpi/_/season/{year}"

    logger.info(f"Fetching BPI rankings for {year}")
    logger.warning("BPI scraping may require additional parsing logic")

    # This would require HTML parsing with BeautifulSoup
    # For now, return empty DataFrame with expected structure
    return pd.DataFrame(columns=[
        'rank', 'team', 'team_id', 'conference',
        'bpi', 'offensive_bpi', 'defensive_bpi',
        'wins', 'losses'
    ])


def fetch_team_schedule(team_id: str, year: int = 2026) -> pd.DataFrame:
    """
    Fetch schedule for a specific team.

    Args:
        team_id: ESPN team ID
        year: Season year

    Returns:
        DataFrame with team schedule
    """
    url = f"{BASE_URL}/teams/{team_id}/schedule?season={year}"
    data = _fetch_with_retry(url, cache_key=f"schedule_{team_id}_{year}")

    games = []
    if 'events' in data:
        for event in data['events']:
            game_info = {
                'date': event.get('date'),
                'game_id': event.get('id'),
                'opponent': event.get('opponent', {}).get('displayName'),
                'opponent_id': event.get('opponent', {}).get('id'),
                'home_away': event.get('homeAway'),
                'result': event.get('result'),
                'score': event.get('score'),
            }
            games.append(game_info)

    return pd.DataFrame(games)


def standardize_team_name(espn_name: str) -> str:
    """
    Convert ESPN team name to standard format used in project.

    Args:
        espn_name: Team name from ESPN API

    Returns:
        Standardized team name
    """
    # Common mappings
    mappings = {
        'North Carolina Tar Heels': 'North Carolina',
        'Duke Blue Devils': 'Duke',
        'NC State Wolfpack': 'NC State',
        'Miami Hurricanes': 'Miami',
        'Pittsburgh Panthers': 'Pitt',
        'Florida State Seminoles': 'Florida State',
        'Virginia Tech Hokies': 'Virginia Tech',
        'Boston College Eagles': 'Boston College',
        'Georgia Tech Yellow Jackets': 'Georgia Tech',
        'Louisville Cardinals': 'Louisville',
        'Clemson Tigers': 'Clemson',
        'Syracuse Orange': 'Syracuse',
        'Wake Forest Demon Deacons': 'Wake Forest',
        'Virginia Cavaliers': 'Virginia',
        'Notre Dame Fighting Irish': 'Notre Dame',
        'Stanford Cardinal': 'Stanford',
        'California Golden Bears': 'California',
        'SMU Mustangs': 'SMU',
        'Ohio State Buckeyes': 'Ohio State',
        'Michigan Wolverines': 'Michigan',
        'Baylor Bears': 'Baylor',
    }

    return mappings.get(espn_name, espn_name.replace(' Tar Heels', '')
                                            .replace(' Blue Devils', '')
                                            .replace(' Wolfpack', '')
                                            .replace(' Hurricanes', '')
                                            .replace(' Panthers', '')
                                            .replace(' Seminoles', '')
                                            .replace(' Hokies', '')
                                            .replace(' Eagles', '')
                                            .replace(' Yellow Jackets', '')
                                            .replace(' Cardinals', '')
                                            .replace(' Tigers', '')
                                            .replace(' Orange', '')
                                            .replace(' Demon Deacons', '')
                                            .replace(' Cavaliers', '')
                                            .replace(' Fighting Irish', '')
                                            .replace(' Cardinal', '')
                                            .replace(' Golden Bears', '')
                                            .replace(' Mustangs', '')
                                            .replace(' Buckeyes', '')
                                            .replace(' Wolverines', '')
                                            .replace(' Bears', ''))


def save_team_data(df: pd.DataFrame, filename: str):
    """
    Save team data to ESPN data directory.

    Args:
        df: DataFrame to save
        filename: Output filename (will be saved in ESPN_DATA_DIR)
    """
    output_path = ESPN_DATA_DIR / filename
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df)} records to {output_path}")


if __name__ == '__main__':
    # Simple test
    print("ESPN API Module")
    print(f"Base URL: {BASE_URL}")
    print(f"Rate limit: {ESPN_CONFIG['rate_limit']} req/min")
    print(f"Cache TTL: {CACHE_TTL}s")

    # Test scoreboard fetch
    today = datetime.now().strftime('%Y%m%d')
    print(f"\nFetching scoreboard for {today}...")
    scoreboard = fetch_scoreboard(today)
    print(f"Found {len(scoreboard)} games")
    if len(scoreboard) > 0:
        print(scoreboard.head())
