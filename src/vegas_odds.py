"""
Vegas odds fetching and processing utilities.

This module provides functionality to fetch real Vegas odds from The Odds API
and convert them to point spreads for comparison against model predictions.
"""

import os
import json
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import requests

from src.logger import setup_logger

logger = setup_logger(__name__)


class VegasOddsFetcher:
    """
    Fetches and processes Vegas odds from The Odds API.

    Requires an API key from https://the-odds-api.com (free tier: 500 requests/month)
    Set the API key via environment variable: ODDS_API_KEY
    """

    BASE_URL = "https://api.the-odds-api.com/v4"
    SPORT = "basketball_ncaab"
    CACHE_DIR = Path(__file__).parent.parent / "data" / "cache" / "odds"

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Vegas odds fetcher.

        Args:
            api_key: The Odds API key. If not provided, reads from ODDS_API_KEY env var.
        """
        self.api_key = api_key or os.environ.get("ODDS_API_KEY")
        if not self.api_key:
            logger.warning(
                "No API key provided. Set ODDS_API_KEY environment variable or pass api_key parameter. "
                "Get a free key at https://the-odds-api.com"
            )

        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self._team_name_mappings = self._build_team_mappings()

    def _build_team_mappings(self) -> Dict[str, str]:
        """Build mappings between API team names and our team names."""
        return {
            # Common variations
            "North Carolina Tar Heels": "North Carolina",
            "Duke Blue Devils": "Duke",
            "Virginia Cavaliers": "Virginia",
            "Miami Hurricanes": "Miami",
            "Florida State Seminoles": "Florida State",
            "NC State Wolfpack": "NC State",
            "Boston College Eagles": "Boston College",
            "Clemson Tigers": "Clemson",
            "Georgia Tech Yellow Jackets": "Georgia Tech",
            "Louisville Cardinals": "Louisville",
            "Notre Dame Fighting Irish": "Notre Dame",
            "Pittsburgh Panthers": "Pitt",
            "Syracuse Orange": "Syracuse",
            "Virginia Tech Hokies": "Virginia Tech",
            "Wake Forest Demon Deacons": "Wake Forest",
            "SMU Mustangs": "SMU",
            "California Golden Bears": "California",
            "Stanford Cardinal": "Stanford",
        }

    def _normalize_team_name(self, api_name: str) -> str:
        """Convert API team name to our standard team name format."""
        if api_name in self._team_name_mappings:
            return self._team_name_mappings[api_name]

        # Try removing common suffixes
        for suffix in [" Tar Heels", " Blue Devils", " Cavaliers", " Hurricanes",
                       " Seminoles", " Wolfpack", " Eagles", " Tigers", " Yellow Jackets",
                       " Cardinals", " Fighting Irish", " Panthers", " Orange", " Hokies",
                       " Demon Deacons", " Mustangs", " Golden Bears", " Cardinal"]:
            if api_name.endswith(suffix):
                return api_name[:-len(suffix)]

        return api_name

    def _get_cache_key(self, params: Dict) -> str:
        """Generate a cache key from request parameters."""
        param_str = json.dumps(params, sort_keys=True)
        return hashlib.md5(param_str.encode()).hexdigest()

    def _get_cached_response(self, cache_key: str, max_age_hours: int = 24) -> Optional[Dict]:
        """Get cached API response if available and not expired."""
        cache_file = self.CACHE_DIR / f"{cache_key}.json"

        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cached = json.load(f)

                cached_time = datetime.fromisoformat(cached['timestamp'])
                if datetime.now() - cached_time < timedelta(hours=max_age_hours):
                    logger.debug(f"Using cached response for {cache_key}")
                    return cached['data']
            except (json.JSONDecodeError, KeyError):
                pass

        return None

    def _save_to_cache(self, cache_key: str, data: Dict):
        """Save API response to cache."""
        cache_file = self.CACHE_DIR / f"{cache_key}.json"

        cached = {
            'timestamp': datetime.now().isoformat(),
            'data': data
        }

        with open(cache_file, 'w') as f:
            json.dump(cached, f)

    def fetch_odds(
        self,
        markets: str = "spreads",
        regions: str = "us",
        odds_format: str = "american",
        date_from: Optional[str] = None,
        date_to: Optional[str] = None
    ) -> List[Dict]:
        """
        Fetch current odds from The Odds API.

        Args:
            markets: Type of odds - 'spreads', 'h2h' (moneyline), 'totals'
            regions: Bookmaker regions - 'us', 'uk', 'eu', 'au'
            odds_format: Format - 'american', 'decimal'
            date_from: ISO format date string for historical odds
            date_to: ISO format date string for historical odds

        Returns:
            List of game odds dictionaries
        """
        if not self.api_key:
            logger.error("Cannot fetch odds without API key")
            return []

        params = {
            'apiKey': self.api_key,
            'regions': regions,
            'markets': markets,
            'oddsFormat': odds_format,
        }

        if date_from:
            params['commenceTimeFrom'] = date_from
        if date_to:
            params['commenceTimeTo'] = date_to

        cache_key = self._get_cache_key({k: v for k, v in params.items() if k != 'apiKey'})
        cached = self._get_cached_response(cache_key)
        if cached is not None:
            return cached

        url = f"{self.BASE_URL}/sports/{self.SPORT}/odds"

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()
            self._save_to_cache(cache_key, data)

            remaining = response.headers.get('x-requests-remaining', 'unknown')
            logger.info(f"Fetched odds for {len(data)} games. API requests remaining: {remaining}")

            return data

        except requests.RequestException as e:
            logger.error(f"Failed to fetch odds: {e}")
            return []

    def fetch_historical_odds(
        self,
        event_id: str,
        date: str
    ) -> Optional[Dict]:
        """
        Fetch historical odds for a specific event.

        Note: Historical odds require a paid API plan.

        Args:
            event_id: The Odds API event ID
            date: ISO format date string

        Returns:
            Historical odds dictionary or None
        """
        if not self.api_key:
            return None

        url = f"{self.BASE_URL}/historical/sports/{self.SPORT}/events/{event_id}/odds"
        params = {
            'apiKey': self.api_key,
            'regions': 'us',
            'markets': 'spreads',
            'date': date,
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Failed to fetch historical odds: {e}")
            return None

    def extract_spread(self, game_odds: Dict, bookmaker: str = None) -> Optional[Dict]:
        """
        Extract point spread from game odds data.

        Args:
            game_odds: Game odds dictionary from API
            bookmaker: Specific bookmaker to use (e.g., 'fanduel', 'draftkings')
                      If None, uses consensus (average) spread.

        Returns:
            Dictionary with home_team, away_team, home_spread, away_spread
            or None if spreads not available
        """
        home_team = self._normalize_team_name(game_odds.get('home_team', ''))
        away_team = self._normalize_team_name(game_odds.get('away_team', ''))

        bookmakers = game_odds.get('bookmakers', [])
        if not bookmakers:
            return None

        spreads = []

        for bm in bookmakers:
            if bookmaker and bm['key'] != bookmaker:
                continue

            for market in bm.get('markets', []):
                if market['key'] != 'spreads':
                    continue

                for outcome in market.get('outcomes', []):
                    team = self._normalize_team_name(outcome.get('name', ''))
                    point = outcome.get('point', 0)

                    if team == home_team:
                        spreads.append(point)

        if not spreads:
            return None

        home_spread = sum(spreads) / len(spreads)

        return {
            'home_team': home_team,
            'away_team': away_team,
            'home_spread': home_spread,
            'away_spread': -home_spread,
            'commence_time': game_odds.get('commence_time'),
            'num_bookmakers': len(spreads),
        }

    def get_spreads_for_games(
        self,
        games: List[Tuple[str, str, str]],
        bookmaker: str = None
    ) -> Dict[Tuple[str, str], float]:
        """
        Get Vegas spreads for a list of games.

        Args:
            games: List of (date, home_team, away_team) tuples
            bookmaker: Specific bookmaker or None for consensus

        Returns:
            Dictionary mapping (home_team, away_team) to spread
        """
        current_odds = self.fetch_odds(markets='spreads')

        spreads = {}
        for game_odds in current_odds:
            spread_info = self.extract_spread(game_odds, bookmaker)
            if spread_info:
                key = (spread_info['home_team'], spread_info['away_team'])
                spreads[key] = spread_info['home_spread']

        return spreads


def american_to_implied_prob(odds: int) -> float:
    """
    Convert American odds to implied probability.

    Args:
        odds: American odds (e.g., -110, +150)

    Returns:
        Implied probability (0-1)
    """
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    else:
        return 100 / (odds + 100)


def implied_prob_to_spread(home_prob: float, away_prob: float) -> float:
    """
    Estimate point spread from moneyline implied probabilities.

    Uses an empirical relationship: spread ~ 13.5 * log(home_prob / away_prob)

    Args:
        home_prob: Home team implied probability
        away_prob: Away team implied probability

    Returns:
        Estimated point spread (positive = home favored)
    """
    import math

    if home_prob <= 0 or away_prob <= 0:
        return 0.0

    ratio = home_prob / away_prob
    spread = 13.5 * math.log(ratio)

    return spread


def load_cached_odds(cache_dir: Path = None) -> Dict:
    """
    Load all cached odds data.

    Args:
        cache_dir: Directory containing cached odds files

    Returns:
        Dictionary of cached odds by cache key
    """
    if cache_dir is None:
        cache_dir = Path(__file__).parent.parent / "data" / "cache" / "odds"

    if not cache_dir.exists():
        return {}

    cached = {}
    for cache_file in cache_dir.glob("*.json"):
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
            cached[cache_file.stem] = data
        except json.JSONDecodeError:
            continue

    return cached


def load_historical_odds_from_excel(
    odds_dir: Path = None,
    seasons: List[str] = None
) -> 'pd.DataFrame':
    """
    Load historical Vegas odds from Sportsbookreviewsonline Excel files.

    Files should be named like: ncaa-basketball-2015-16.xlsx
    Download from: https://www.sportsbookreviewsonline.com/scoresoddsarchives/

    Args:
        odds_dir: Directory containing Excel files. Defaults to data/raw/odds/
        seasons: List of seasons to load (e.g., ['2015-16', '2021-22']).
                If None, loads all available files.

    Returns:
        DataFrame with columns: date, away_team, home_team, away_score, home_score,
                               vegas_spread, actual_margin, season
    """
    import pandas as pd
    import numpy as np

    if odds_dir is None:
        odds_dir = Path(__file__).parent.parent / "data" / "raw" / "odds"

    if not odds_dir.exists():
        logger.warning(f"Odds directory not found: {odds_dir}")
        return pd.DataFrame()

    # Find all Excel files
    excel_files = list(odds_dir.glob("ncaa-basketball-*.xlsx"))
    if not excel_files:
        logger.warning(f"No odds files found in {odds_dir}")
        return pd.DataFrame()

    all_games = []

    for file_path in excel_files:
        # Extract season from filename (e.g., "2015-16" from "ncaa-basketball-2015-16.xlsx")
        season = file_path.stem.replace("ncaa-basketball-", "")

        if seasons is not None and season not in seasons:
            continue

        logger.info(f"Loading odds from {file_path.name}")

        try:
            df = pd.read_excel(file_path)
        except Exception as e:
            logger.error(f"Failed to read {file_path}: {e}")
            continue

        # Process pairs of rows (Visitor, Home)
        for i in range(0, len(df) - 1, 2):
            visitor = df.iloc[i]
            home = df.iloc[i + 1]

            # Validate row pairing
            if visitor.get('VH') != 'V' or home.get('VH') != 'H':
                continue

            # Extract spread - should be a small number (< 40)
            # Home row typically has the spread, but data can be inconsistent
            home_close = pd.to_numeric(home.get('Close'), errors='coerce')
            visitor_close = pd.to_numeric(visitor.get('Close'), errors='coerce')

            if pd.notna(home_close) and home_close < 40:
                spread = home_close
            elif pd.notna(visitor_close) and visitor_close < 40:
                spread = -visitor_close  # Visitor spread, flip sign
            else:
                spread = np.nan

            # Parse date (format: MMDD, e.g., 1113 = Nov 13)
            date_val = visitor.get('Date')
            if pd.notna(date_val):
                date_str = str(int(date_val)).zfill(4)
                month = int(date_str[:2]) if len(date_str) >= 2 else 0
                day = int(date_str[2:]) if len(date_str) >= 4 else 0

                # Determine year from season (e.g., "2015-16")
                # Nov-Dec = first year, Jan-Apr = second year
                start_year = int(season.split('-')[0])
                year = start_year if month >= 10 else start_year + 1

                try:
                    game_date = pd.Timestamp(year=year, month=month, day=day)
                except:
                    game_date = pd.NaT
            else:
                game_date = pd.NaT

            game = {
                'date': game_date,
                'away_team': visitor.get('Team', ''),
                'home_team': home.get('Team', ''),
                'away_score': visitor.get('Final'),
                'home_score': home.get('Final'),
                'vegas_spread': spread,
                'season': season,
            }

            # Calculate actual margin (positive = home win)
            if pd.notna(game['home_score']) and pd.notna(game['away_score']):
                game['actual_margin'] = game['home_score'] - game['away_score']
            else:
                game['actual_margin'] = np.nan

            all_games.append(game)

    if not all_games:
        return pd.DataFrame()

    games_df = pd.DataFrame(all_games)
    games_df = games_df.sort_values('date').reset_index(drop=True)

    # Log summary
    valid_spreads = games_df['vegas_spread'].notna().sum()
    logger.info(f"Loaded {len(games_df)} games with {valid_spreads} valid Vegas spreads")

    return games_df


# Team name mapping for matching between datasets
TEAM_NAME_MAPPINGS = {
    # Sportsbookreviewsonline -> Standard names
    'IowaState': 'Iowa St.',
    'MichiganState': 'Michigan St.',
    'FlaAtlantic': 'Florida Atlantic',
    'CharlotteU': 'Charlotte',
    'St.Josephs': "St. Joseph's",
    'JamesMadison': 'James Madison',
    'SetonHall': 'Seton Hall',
    'NorthCarolina': 'North Carolina',
    'GeorgiaTech': 'Georgia Tech',
    'LoyolaChicago': 'Loyola Chicago',
    'BallState': 'Ball St.',
    'SoIllinois': 'Southern Illinois',
    'William&Mary': 'William & Mary',
    'NCState': 'NC State',
    'SanFrancisco': 'San Francisco',
    'IllinoisChicago': 'Illinois-Chicago',
    'SanDiego': 'San Diego',
    'TexSanAntonio': 'UTSA',
    'WiscGreenBay': 'Green Bay',
    'AirForce': 'Air Force',
    'MiamiOhio': 'Miami (OH)',
    'MiamiFlorida': 'Miami (FL)',
    'FloridaState': 'Florida St.',
    'OhioState': 'Ohio St.',
    'PennState': 'Penn St.',
    'MissState': 'Mississippi St.',
    'OklahomaSt': 'Oklahoma St.',
    'KansasState': 'Kansas St.',
    'TexasA&M': 'Texas A&M',
    'BoiseState': 'Boise St.',
    'SanDiegoSt': 'San Diego St.',
    'FresnoState': 'Fresno St.',
    'ColoradoSt': 'Colorado St.',
    'UtahState': 'Utah St.',
    'ArizonaSt': 'Arizona St.',
    'WashState': 'Washington St.',
    'OregonState': 'Oregon St.',
    'BostonColl': 'Boston College',
    'WakeForest': 'Wake Forest',
    'VirginiaTech': 'Virginia Tech',
    'NotreDame': 'Notre Dame',
    'SouthCarolina': 'South Carolina',
    'MissouriSt': 'Missouri St.',
    'WichitaSt': 'Wichita St.',
    'CentralFla': 'UCF',
    'SouthFlorida': 'South Florida',
}


def normalize_team_name(name: str) -> str:
    """
    Normalize team name to standard format.

    Args:
        name: Raw team name from data source

    Returns:
        Normalized team name
    """
    if name in TEAM_NAME_MAPPINGS:
        return TEAM_NAME_MAPPINGS[name]
    return name
