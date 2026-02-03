"""
Player Data Collection Module

Collects player-level statistics from multiple sources:
- CBBpy: Game-by-game boxscores and player stats
- Barttorvik: Season-aggregated advanced metrics (usage%, offensive/defensive ratings)
- ESPN: Starting lineups for games

Handles player ID assignment, name normalization, and fuzzy matching across data sources.
"""

import time
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import logging
from rapidfuzz import fuzz, process
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils import fetch_url_with_retry
from .config import (
    PLAYER_STATS_DIR, BOXSCORES_DIR, LINEUPS_DIR,
    BARTTORVIK_PLAYER_URL_TEMPLATE, SCRAPING_DELAY,
    BARTTORVIK_MAX_RETRIES, BARTTORVIK_RETRY_DELAY,
    TRAINING_YEARS, LOGGING_CONFIG
)

# Try to import CBBpy (may not be installed in all environments)
try:
    from cbbpy import mens_scraper
    CBBPY_AVAILABLE = True
except ImportError:
    CBBPY_AVAILABLE = False
    print("Warning: CBBpy not available. Player data collection will be limited.")

# Set up logging
logger = logging.getLogger(__name__)


class PlayerDataCollector:
    """
    Collects and processes player-level data from multiple sources
    """

    def __init__(self, data_dir: Path = None):
        """
        Initialize data collector

        Args:
            data_dir: Root directory for player data (defaults to config path)
        """
        self.data_dir = data_dir or Path(PLAYER_STATS_DIR).parent
        self.player_stats_dir = PLAYER_STATS_DIR
        self.boxscores_dir = BOXSCORES_DIR
        self.lineups_dir = LINEUPS_DIR

        # Player ID mapping: (normalized_name, first_team, first_season) -> player_id
        self.player_id_map: Dict[Tuple[str, str, int], str] = {}
        self.player_counter = 0

        logger.info(f"PlayerDataCollector initialized with data_dir: {self.data_dir}")

    # ========================================================================
    # BARTTORVIK DATA COLLECTION
    # ========================================================================

    def collect_player_stats_barttorvik(
        self,
        years: List[int] = None
    ) -> pd.DataFrame:
        """
        Collect season-aggregated player stats from Barttorvik

        Args:
            years: List of years to collect (defaults to config.TRAINING_YEARS)

        Returns:
            DataFrame with columns: player_name, team, season, games_played,
            usage_pct, offensive_rating, defensive_rating, minutes_per_game, etc.
        """
        if years is None:
            years = TRAINING_YEARS

        all_stats = []

        for year in years:
            logger.info(f"Collecting Barttorvik player stats for {year}...")

            try:
                # Fetch from Barttorvik
                url = BARTTORVIK_PLAYER_URL_TEMPLATE.format(year=year)
                df = fetch_url_with_retry(
                    url,
                    max_retries=BARTTORVIK_MAX_RETRIES,
                    retry_delay=BARTTORVIK_RETRY_DELAY,
                    parse_csv=True
                )

                # Add season column
                df['season'] = year

                # Standardize column names
                df = self._standardize_barttorvik_columns(df)

                all_stats.append(df)
                logger.info(f"  ✓ Collected {len(df)} player records for {year}")

                # Rate limiting
                time.sleep(SCRAPING_DELAY)

            except Exception as e:
                logger.error(f"  ✗ Failed to collect Barttorvik data for {year}: {e}")
                continue

        if not all_stats:
            logger.warning("No Barttorvik data collected")
            return pd.DataFrame()

        # Combine all years
        combined = pd.concat(all_stats, ignore_index=True)
        logger.info(f"Total Barttorvik records collected: {len(combined)}")

        # Assign player IDs
        combined = self._assign_player_ids(combined)

        # Save to file
        output_file = self.player_stats_dir / f"barttorvik_stats_{min(years)}_{max(years)}.csv"
        combined.to_csv(output_file, index=False)
        logger.info(f"Saved to: {output_file}")

        return combined

    def _standardize_barttorvik_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize Barttorvik column names to consistent format

        Args:
            df: Raw Barttorvik dataframe

        Returns:
            DataFrame with standardized columns
        """
        # Common Barttorvik columns (may vary by year)
        column_mapping = {
            'Player': 'player_name',
            'Team': 'team',
            'Conf': 'conference',
            'G': 'games_played',
            'Min%': 'minutes_pct',
            'Usage': 'usage_pct',
            'ORtg': 'offensive_rating',
            'DRtg': 'defensive_rating',
            'eFG%': 'efg_pct',
            'TO%': 'tov_pct',
            'ORB%': 'orb_pct',
            'FTR': 'ft_rate',
            'PRPG': 'prpg',  # Player Rating Per Game
            'BPM': 'bpm',    # Box Plus/Minus
            'Ht': 'height',
            'Yr': 'year_in_school',
        }

        # Rename columns that exist
        df = df.rename(columns={
            old: new for old, new in column_mapping.items()
            if old in df.columns
        })

        # Calculate minutes per game if available
        if 'minutes_pct' in df.columns and 'games_played' in df.columns:
            # Estimate: minutes_pct is percentage of team minutes
            # Assume 200 minutes per game (40 min game × 5 players)
            df['minutes_per_game'] = df['minutes_pct'] * 2.0
        elif 'Min' in df.columns:  # Sometimes direct minutes are provided
            df['minutes_per_game'] = df['Min'] / df['games_played'].replace(0, 1)

        return df

    def collect_player_stats_from_local(
        self,
        years: List[int] = None,
        local_data_dir: Path = None
    ) -> pd.DataFrame:
        """
        Load player stats from local CSV files (for manually collected data)

        Args:
            years: List of years to load (defaults to config.TRAINING_YEARS)
            local_data_dir: Directory containing CSV files (defaults to data/raw_pd/)

        Returns:
            DataFrame with standardized player stats columns
        """
        if years is None:
            years = TRAINING_YEARS

        if local_data_dir is None:
            # Default to data/raw_pd/ directory
            local_data_dir = Path(self.data_dir).parent.parent / "raw_pd"

        logger.info(f"Loading player stats from local directory: {local_data_dir}")

        all_stats = []

        # Define column names based on typical Barttorvik structure (67 columns)
        column_names = [
            'player_name', 'team', 'conference', 'games_played', 'minutes_pct',
            'offensive_rating', 'usage_pct', 'tempo', 'ts_pct', 'orb_pct',
            'drb_pct', 'ast_pct', 'tov_pct', 'ftm', 'fta', 'ft_pct',
            'fg2m', 'fg2a', 'fg2_pct', 'fg3m', 'fg3a', 'fg3_pct',
            'ftr', 'stl_pct', 'blk_pct', 'year_in_school', 'height', 'rank',
            'prpg', 'adj_oe', 'stops', 'season', 'player_id_raw', 'hometown',
            'high_school', 'bpm', 'fg2m_rim', 'fg2a_rim', 'fg2m_mid', 'fg2a_mid',
            'fg2_rim_pct', 'fg2_mid_pct', 'fg3m_c', 'fg3a_c', 'fg3_c_pct',
            'dunks_attempted', 'defensive_rating', 'adj_de', 'dbpm', 'porpag',
            'adj_tempo', 'wab', 'wab_rank', 'obpm', 'pick_prob', 'ppg', 'rpg',
            'apg', 'mpg', 'spg', 'bpg', 'tpg', 'ftpg', 'position', 'combo',
            'birthdate', 'col_66_unknown'
        ]

        for year in years:
            csv_file = local_data_dir / f"{year}_pd.csv"

            if not csv_file.exists():
                logger.warning(f"  File not found: {csv_file}")
                continue

            try:
                # Load CSV without header and without index column
                df = pd.read_csv(csv_file, header=None, names=column_names, index_col=False)

                logger.info(f"  [OK] Loaded {len(df)} player records for {year} from {csv_file.name}")

                # Standardize column names (use existing method)
                df = self._standardize_local_columns(df)

                all_stats.append(df)

            except Exception as e:
                logger.error(f"  [FAIL] Failed to load {csv_file.name}: {e}")
                continue

        if not all_stats:
            logger.warning("No local player data loaded")
            return pd.DataFrame()

        # Combine all years
        combined = pd.concat(all_stats, ignore_index=True)
        logger.info(f"Total local records loaded: {len(combined)}")

        # Assign player IDs
        combined = self._assign_player_ids(combined)

        # Save to standardized location
        output_file = self.player_stats_dir / f"barttorvik_stats_{min(years)}_{max(years)}.csv"
        combined.to_csv(output_file, index=False)
        logger.info(f"Saved to: {output_file}")

        return combined

    def _standardize_local_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize columns from local CSV files

        Args:
            df: Raw dataframe from local CSV

        Returns:
            DataFrame with standardized columns matching Barttorvik format
        """
        # Ensure season is integer
        if 'season' in df.columns:
            df['season'] = pd.to_numeric(df['season'], errors='coerce')

        # Calculate minutes per game if not already present
        if 'minutes_per_game' not in df.columns and 'minutes_pct' in df.columns:
            df['minutes_per_game'] = df['minutes_pct'] * 2.0
        elif 'mpg' in df.columns:
            df['minutes_per_game'] = df['mpg']

        # Ensure required columns exist with default values
        required_cols = {
            'player_name': '',
            'team': '',
            'season': 0,
            'games_played': 0,
            'usage_pct': 20.0,
            'offensive_rating': 100.0,
            'defensive_rating': 100.0,
            'minutes_per_game': 15.0
        }

        for col, default in required_cols.items():
            if col not in df.columns:
                df[col] = default

        return df

    # ========================================================================
    # CBBPY DATA COLLECTION
    # ========================================================================

    def collect_game_boxscores_cbbpy(
        self,
        season: int,
        max_games: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Collect game-by-game player boxscores using CBBpy

        Args:
            season: Season year (e.g., 2024 for 2023-24 season)
            max_games: Maximum number of games to collect (None = all)

        Returns:
            DataFrame with columns: game_id, date, player_name, team, minutes,
            points, rebounds, assists, etc.
        """
        if not CBBPY_AVAILABLE:
            logger.error("CBBpy not available. Cannot collect boxscores.")
            return pd.DataFrame()

        logger.info(f"Collecting CBBpy boxscores for {season} season...")

        try:
            # Get all games for the season
            games = mens_scraper.get_games_season(season)

            if games is None or len(games) == 0:
                logger.warning(f"No games found for season {season}")
                return pd.DataFrame()

            if max_games is not None:
                games = games.head(max_games)

            logger.info(f"  Processing {len(games)} games...")

            all_boxscores = []
            successful_games = 0
            failed_games = 0

            for idx, game in games.iterrows():
                try:
                    game_id = game.get('game_id', idx)
                    game_date = game.get('date', 'Unknown')

                    # Fetch boxscore
                    boxscore = mens_scraper.get_game_boxscore(game_id)

                    if boxscore is None or len(boxscore) == 0:
                        failed_games += 1
                        continue

                    # Add game metadata
                    boxscore['game_id'] = game_id
                    boxscore['date'] = game_date
                    boxscore['season'] = season

                    all_boxscores.append(boxscore)
                    successful_games += 1

                    # Progress logging
                    if successful_games % 100 == 0:
                        logger.info(f"    Processed {successful_games} games...")

                    # Rate limiting
                    time.sleep(SCRAPING_DELAY)

                except Exception as e:
                    logger.warning(f"    Failed to get boxscore for game {game_id}: {e}")
                    failed_games += 1
                    continue

            if not all_boxscores:
                logger.warning(f"No boxscores collected for season {season}")
                return pd.DataFrame()

            # Combine all boxscores
            combined = pd.concat(all_boxscores, ignore_index=True)
            logger.info(f"  ✓ Collected {len(combined)} player-game records")
            logger.info(f"  Successful: {successful_games}, Failed: {failed_games}")

            # Standardize columns
            combined = self._standardize_cbbpy_columns(combined)

            # Assign player IDs
            combined = self._assign_player_ids(combined)

            # Save to file
            output_file = self.boxscores_dir / f"boxscores_{season}.csv"
            combined.to_csv(output_file, index=False)
            logger.info(f"Saved to: {output_file}")

            return combined

        except Exception as e:
            logger.error(f"Error collecting CBBpy boxscores for {season}: {e}")
            return pd.DataFrame()

    def _standardize_cbbpy_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize CBBpy boxscore column names

        Args:
            df: Raw CBBpy boxscore dataframe

        Returns:
            DataFrame with standardized columns
        """
        column_mapping = {
            'Player': 'player_name',
            'player': 'player_name',
            'Team': 'team',
            'team': 'team',
            'Opp': 'opponent',
            'opponent': 'opponent',
            'MIN': 'minutes',
            'Min': 'minutes',
            'PTS': 'points',
            'Pts': 'points',
            'REB': 'rebounds',
            'Reb': 'rebounds',
            'AST': 'assists',
            'Ast': 'assists',
            'STL': 'steals',
            'Stl': 'steals',
            'BLK': 'blocks',
            'Blk': 'blocks',
            'TO': 'turnovers',
            'TOV': 'turnovers',
            'FG': 'fg_made',
            'FGA': 'fg_att',
            '3P': 'three_made',
            '3PA': 'three_att',
            'FT': 'ft_made',
            'FTA': 'ft_att',
            '+/-': 'plus_minus',
        }

        # Rename columns that exist
        df = df.rename(columns={
            old: new for old, new in column_mapping.items()
            if old in df.columns
        })

        # Convert minutes to numeric (handle MM:SS format)
        if 'minutes' in df.columns:
            df['minutes'] = df['minutes'].apply(self._parse_minutes)

        return df

    @staticmethod
    def _parse_minutes(minutes_str) -> float:
        """
        Parse minutes from MM:SS format to float

        Args:
            minutes_str: Minutes string (e.g., "32:15" or "32")

        Returns:
            Minutes as float
        """
        if pd.isna(minutes_str):
            return 0.0

        minutes_str = str(minutes_str).strip()

        if ':' in minutes_str:
            try:
                parts = minutes_str.split(':')
                mins = float(parts[0])
                secs = float(parts[1]) if len(parts) > 1 else 0
                return mins + secs / 60.0
            except:
                return 0.0
        else:
            try:
                return float(minutes_str)
            except:
                return 0.0

    # ========================================================================
    # PLAYER ID ASSIGNMENT
    # ========================================================================

    def _assign_player_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Assign unique player IDs based on fuzzy name matching

        Strategy:
        1. Normalize player names (lowercase, remove punctuation)
        2. Match by (normalized_name, team, season)
        3. For exact matches, reuse existing ID
        4. For fuzzy matches (e.g., "J. Smith" vs "John Smith"), merge if confidence > 90%
        5. Otherwise, create new ID

        Args:
            df: DataFrame with 'player_name', 'team', and 'season' columns

        Returns:
            DataFrame with added 'player_id' column
        """
        if 'player_name' not in df.columns or 'team' not in df.columns:
            logger.warning("Cannot assign player IDs: missing required columns")
            return df

        logger.info("Assigning player IDs using fuzzy matching...")

        player_ids = []

        for idx, row in df.iterrows():
            player_name = row.get('player_name', '')
            team = row.get('team', '')
            season = row.get('season', 0)

            if pd.isna(player_name) or player_name == '':
                player_ids.append(None)
                continue

            # Normalize name
            normalized_name = self._normalize_name(player_name)

            # Create key
            key = (normalized_name, team, season)

            # Check for exact match
            if key in self.player_id_map:
                player_ids.append(self.player_id_map[key])
                continue

            # Fuzzy matching within same team/season
            best_match, best_score = self._fuzzy_match_player(
                normalized_name, team, season
            )

            if best_match is not None and best_score > 90:
                # High confidence match - reuse ID
                player_ids.append(self.player_id_map[best_match])
                # Add this variant to the map
                self.player_id_map[key] = self.player_id_map[best_match]
            else:
                # Create new ID
                new_id = f"PLY{self.player_counter:06d}"
                self.player_counter += 1
                self.player_id_map[key] = new_id
                player_ids.append(new_id)

        df['player_id'] = player_ids
        logger.info(f"  Assigned IDs to {len(df)} records ({self.player_counter} unique players)")

        return df

    @staticmethod
    def _normalize_name(name: str) -> str:
        """
        Normalize player name for matching

        Args:
            name: Raw player name

        Returns:
            Normalized name (lowercase, no punctuation)
        """
        import re
        name = str(name).lower().strip()
        name = re.sub(r'[^\w\s]', '', name)  # Remove punctuation
        name = re.sub(r'\s+', ' ', name)     # Normalize whitespace
        return name

    def _fuzzy_match_player(
        self,
        normalized_name: str,
        team: str,
        season: int,
        threshold: float = 85.0
    ) -> Tuple[Optional[Tuple], float]:
        """
        Find best fuzzy match for a player name within same team/season

        Args:
            normalized_name: Normalized player name
            team: Team name
            season: Season year
            threshold: Minimum similarity score (0-100)

        Returns:
            (best_match_key, score) or (None, 0) if no good match
        """
        # Get all players from same team/season
        candidates = [
            (key, name) for (name, t, s), key in self.player_id_map.items()
            if t == team and s == season
        ]

        if not candidates:
            return None, 0.0

        # Fuzzy match against candidate names
        candidate_names = [name for _, name in candidates]
        best_match = process.extractOne(
            normalized_name,
            candidate_names,
            scorer=fuzz.ratio
        )

        if best_match is None:
            return None, 0.0

        matched_name, score, _ = best_match

        if score >= threshold:
            # Find the key for the matched name
            for (name, t, s), key in self.player_id_map.items():
                if name == matched_name and t == team and s == season:
                    return (name, t, s), score

        return None, 0.0

    # ========================================================================
    # DATA LOADING
    # ========================================================================

    def load_player_stats(self, year: int) -> pd.DataFrame:
        """
        Load previously collected player stats for a year

        Args:
            year: Season year

        Returns:
            DataFrame with player stats
        """
        file_pattern = f"barttorvik_stats_*_{year}.csv"
        files = list(self.player_stats_dir.glob(file_pattern))

        if not files:
            logger.warning(f"No player stats file found for {year}")
            return pd.DataFrame()

        # Load the most recent file
        file = max(files, key=lambda p: p.stat().st_mtime)
        logger.info(f"Loading player stats from: {file}")

        return pd.read_csv(file)

    def load_boxscores(self, year: int) -> pd.DataFrame:
        """
        Load previously collected boxscores for a year

        Args:
            year: Season year

        Returns:
            DataFrame with boxscores
        """
        file = self.boxscores_dir / f"boxscores_{year}.csv"

        if not file.exists():
            logger.warning(f"No boxscores file found for {year}: {file}")
            return pd.DataFrame()

        logger.info(f"Loading boxscores from: {file}")
        return pd.read_csv(file)

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def get_player_stats_summary(self, df: pd.DataFrame) -> Dict:
        """
        Get summary statistics for collected player data

        Args:
            df: Player stats dataframe

        Returns:
            Dictionary with summary statistics
        """
        if df.empty:
            return {}

        summary = {
            'total_records': len(df),
            'unique_players': df['player_id'].nunique() if 'player_id' in df.columns else 0,
            'unique_teams': df['team'].nunique() if 'team' in df.columns else 0,
            'seasons': sorted(df['season'].unique().tolist()) if 'season' in df.columns else [],
            'avg_games_played': df['games_played'].mean() if 'games_played' in df.columns else 0,
            'missing_values': df.isnull().sum().to_dict(),
        }

        return summary


# ============================================================================
# STANDALONE EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Set up logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format=LOGGING_CONFIG['format'],
        datefmt=LOGGING_CONFIG['date_format']
    )

    # Create collector
    collector = PlayerDataCollector()

    # Collect Barttorvik data for all training years
    print("\n" + "="*60)
    print("COLLECTING BARTTORVIK PLAYER STATS")
    print("="*60)

    barttorvik_stats = collector.collect_player_stats_barttorvik(TRAINING_YEARS)

    if not barttorvik_stats.empty:
        print(f"\n✓ Collected {len(barttorvik_stats)} Barttorvik records")
        print(f"✓ Unique players: {barttorvik_stats['player_id'].nunique()}")
        print(f"✓ Seasons: {sorted(barttorvik_stats['season'].unique())}")
    else:
        print("\n✗ No Barttorvik data collected")

    print("\n" + "="*60)
    print("DATA COLLECTION COMPLETE")
    print("="*60)
