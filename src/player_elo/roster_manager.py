"""
Roster Management Module

Tracks player rosters, transfers, injuries, and eligibility status.
Maintains historical record of roster changes across seasons.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Set, Tuple
import logging

from .config import ROSTERS_DIR, PROCESSED_DATA_DIR

# Set up logger
logger = logging.getLogger(__name__)


class RosterManager:
    """
    Manages team rosters, player transfers, and eligibility tracking
    """

    def __init__(self, rosters_dir: Path = None):
        """
        Initialize Roster Manager

        Args:
            rosters_dir: Directory for roster data (defaults to config)
        """
        self.rosters_dir = rosters_dir or ROSTERS_DIR

        # Current rosters: season -> team -> [player_ids]
        self.rosters: Dict[int, Dict[str, List[str]]] = {}

        # Transfer tracking
        self.transfers: List[Dict] = []

        # Injury/eligibility status
        self.injuries: pd.DataFrame = pd.DataFrame()

        # Player metadata: player_id -> {team, position, year, etc.}
        self.player_info: Dict[str, Dict] = {}

        logger.info(f"RosterManager initialized with rosters_dir: {self.rosters_dir}")

    # ========================================================================
    # ROSTER MANAGEMENT
    # ========================================================================

    def load_rosters(self, seasons: List[int]) -> None:
        """
        Load roster data for multiple seasons

        Args:
            seasons: List of season years to load
        """
        logger.info(f"Loading rosters for seasons: {seasons}")

        for season in seasons:
            roster_file = self.rosters_dir / f"rosters_{season}.csv"

            if not roster_file.exists():
                logger.warning(f"  No roster file found for {season}: {roster_file}")
                continue

            try:
                df = pd.read_csv(roster_file)
                self._load_roster_from_df(df, season)
                logger.info(f"  ✓ Loaded {len(df)} players for {season}")
            except Exception as e:
                logger.error(f"  ✗ Failed to load rosters for {season}: {e}")

    def _load_roster_from_df(self, df: pd.DataFrame, season: int) -> None:
        """
        Load roster data from DataFrame

        Args:
            df: Roster DataFrame with columns: team, player_id, player_name, position, etc.
            season: Season year
        """
        if season not in self.rosters:
            self.rosters[season] = {}

        # Group by team
        for team, team_players in df.groupby('team'):
            player_ids = team_players['player_id'].tolist()
            self.rosters[season][team] = player_ids

            # Update player info
            for _, player in team_players.iterrows():
                player_id = player['player_id']
                self.player_info[player_id] = {
                    'player_name': player.get('player_name', ''),
                    'team': team,
                    'season': season,
                    'position': player.get('position', 'Unknown'),
                    'year_in_school': player.get('year_in_school', ''),
                    'height': player.get('height', ''),
                    'weight': player.get('weight', ''),
                }

    def get_team_roster(self, team: str, season: int) -> List[str]:
        """
        Get roster for a team in a specific season

        Args:
            team: Team name
            season: Season year

        Returns:
            List of player IDs on roster
        """
        if season not in self.rosters:
            logger.warning(f"No rosters loaded for season {season}")
            return []

        return self.rosters.get(season, {}).get(team, [])

    def get_player_info(self, player_id: str) -> Dict:
        """
        Get metadata for a player

        Args:
            player_id: Player ID

        Returns:
            Dictionary with player info (empty if not found)
        """
        return self.player_info.get(player_id, {})

    # ========================================================================
    # TRANSFER TRACKING
    # ========================================================================

    def track_transfer(
        self,
        player_id: str,
        from_team: str,
        to_team: str,
        season: int,
        player_name: Optional[str] = None
    ) -> None:
        """
        Record a player transfer

        Args:
            player_id: Player ID
            from_team: Origin team
            to_team: Destination team
            season: Season of transfer
            player_name: Optional player name
        """
        transfer = {
            'player_id': player_id,
            'player_name': player_name or self.player_info.get(player_id, {}).get('player_name', ''),
            'from_team': from_team,
            'to_team': to_team,
            'season': season,
            'date_recorded': datetime.now()
        }

        self.transfers.append(transfer)
        logger.info(f"  Transfer recorded: {player_name} ({from_team} → {to_team})")

    def detect_transfers(self, season1: int, season2: int) -> List[Dict]:
        """
        Detect transfers between two consecutive seasons

        Args:
            season1: Earlier season
            season2: Later season

        Returns:
            List of detected transfers
        """
        if season1 not in self.rosters or season2 not in self.rosters:
            logger.warning(f"Cannot detect transfers: missing roster data")
            return []

        transfers = []

        # Build player -> team mappings
        season1_teams = {}
        for team, players in self.rosters[season1].items():
            for player_id in players:
                season1_teams[player_id] = team

        season2_teams = {}
        for team, players in self.rosters[season2].items():
            for player_id in players:
                season2_teams[player_id] = team

        # Find players who changed teams
        common_players = set(season1_teams.keys()) & set(season2_teams.keys())

        for player_id in common_players:
            team1 = season1_teams[player_id]
            team2 = season2_teams[player_id]

            if team1 != team2:
                self.track_transfer(player_id, team1, team2, season2)
                transfers.append({
                    'player_id': player_id,
                    'from_team': team1,
                    'to_team': team2
                })

        logger.info(f"  Detected {len(transfers)} transfers between {season1} and {season2}")
        return transfers

    def get_transfers_dataframe(self) -> pd.DataFrame:
        """
        Get all transfers as DataFrame

        Returns:
            DataFrame with transfer records
        """
        if not self.transfers:
            return pd.DataFrame()

        return pd.DataFrame(self.transfers)

    def save_transfers(self, filepath: Path = None) -> None:
        """
        Save transfer data to CSV

        Args:
            filepath: Output path (defaults to config path)
        """
        if filepath is None:
            filepath = PROCESSED_DATA_DIR / "transfer_tracker.csv"

        df = self.get_transfers_dataframe()
        if not df.empty:
            df.to_csv(filepath, index=False)
            logger.info(f"Saved {len(df)} transfers to: {filepath}")

    # ========================================================================
    # INJURY/ELIGIBILITY TRACKING
    # ========================================================================

    def update_injury_status(
        self,
        injury_df: pd.DataFrame,
        replace: bool = True
    ) -> None:
        """
        Update injury/eligibility status

        Args:
            injury_df: DataFrame with columns: player_id, team, date, status, description
            replace: If True, replace existing data; if False, append
        """
        if replace:
            self.injuries = injury_df
        else:
            self.injuries = pd.concat([self.injuries, injury_df], ignore_index=True)

        logger.info(f"Updated injury status: {len(self.injuries)} records")

    def get_injury_status(
        self,
        player_id: Optional[str] = None,
        team: Optional[str] = None,
        date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Get injury status for player(s)

        Args:
            player_id: Optional player ID filter
            team: Optional team filter
            date: Optional date filter

        Returns:
            Filtered injury DataFrame
        """
        df = self.injuries

        if df.empty:
            return df

        if player_id is not None:
            df = df[df['player_id'] == player_id]

        if team is not None:
            df = df[df['team'] == team]

        if date is not None:
            # Filter to injuries current as of date
            if 'date' in df.columns:
                df = df[pd.to_datetime(df['date']) <= date]

        return df

    def is_player_available(
        self,
        player_id: str,
        date: Optional[datetime] = None
    ) -> bool:
        """
        Check if player is available (not injured/ineligible)

        Args:
            player_id: Player ID
            date: Date to check (defaults to now)

        Returns:
            True if available, False if out/doubtful
        """
        if self.injuries.empty:
            return True  # Assume available if no injury data

        status = self.get_injury_status(player_id, date=date)

        if status.empty:
            return True  # No injury record = available

        # Get most recent status
        if 'date' in status.columns:
            status = status.sort_values('date', ascending=False)

        latest_status = status.iloc[0]['status'] if 'status' in status.columns else 'Unknown'

        # Consider 'Out' and 'Doubtful' as unavailable
        return latest_status not in ['Out', 'Doubtful']

    def get_available_players(
        self,
        team: str,
        season: int,
        date: Optional[datetime] = None
    ) -> List[str]:
        """
        Get list of available players for a team

        Args:
            team: Team name
            season: Season year
            date: Date to check availability

        Returns:
            List of available player IDs
        """
        roster = self.get_team_roster(team, season)

        if not roster:
            return []

        available = [
            player_id for player_id in roster
            if self.is_player_available(player_id, date)
        ]

        return available

    # ========================================================================
    # ROSTER STATISTICS
    # ========================================================================

    def get_roster_continuity(self, team: str, season1: int, season2: int) -> float:
        """
        Calculate roster continuity between seasons

        Args:
            team: Team name
            season1: Earlier season
            season2: Later season

        Returns:
            Continuity percentage (0-100)
        """
        roster1 = set(self.get_team_roster(team, season1))
        roster2 = set(self.get_team_roster(team, season2))

        if not roster1 or not roster2:
            return 0.0

        # Players who remained
        retained = len(roster1 & roster2)

        # Continuity as percentage of previous roster
        continuity = (retained / len(roster1)) * 100

        return continuity

    def get_roster_summary(self, team: str, season: int) -> Dict:
        """
        Get summary statistics for a team's roster

        Args:
            team: Team name
            season: Season year

        Returns:
            Dictionary with roster statistics
        """
        roster = self.get_team_roster(team, season)

        if not roster:
            return {}

        # Position breakdown
        positions = [
            self.player_info.get(pid, {}).get('position', 'Unknown')
            for pid in roster
        ]
        position_counts = pd.Series(positions).value_counts().to_dict()

        # Year breakdown
        years = [
            self.player_info.get(pid, {}).get('year_in_school', 'Unknown')
            for pid in roster
        ]
        year_counts = pd.Series(years).value_counts().to_dict()

        summary = {
            'team': team,
            'season': season,
            'roster_size': len(roster),
            'position_breakdown': position_counts,
            'year_breakdown': year_counts,
        }

        return summary

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def create_roster_from_stats(
        self,
        player_stats: pd.DataFrame,
        season: int,
        min_games: int = 5
    ) -> None:
        """
        Create roster from player stats (when roster data unavailable)

        Args:
            player_stats: DataFrame with player_id, team, games_played
            season: Season year
            min_games: Minimum games to be considered on roster
        """
        logger.info(f"Creating roster from stats for {season} (min_games={min_games})")

        # Filter to players with enough games
        roster_df = player_stats[player_stats.get('games_played', 0) >= min_games]

        if season not in self.rosters:
            self.rosters[season] = {}

        # Group by team
        for team, team_players in roster_df.groupby('team'):
            player_ids = team_players['player_id'].tolist()
            self.rosters[season][team] = player_ids

            logger.info(f"  {team}: {len(player_ids)} players")

    def export_rosters(self, season: int, filepath: Path = None) -> None:
        """
        Export rosters for a season to CSV

        Args:
            season: Season year
            filepath: Output path (defaults to config path)
        """
        if season not in self.rosters:
            logger.warning(f"No rosters loaded for season {season}")
            return

        if filepath is None:
            filepath = self.rosters_dir / f"rosters_{season}.csv"

        rows = []
        for team, player_ids in self.rosters[season].items():
            for player_id in player_ids:
                info = self.player_info.get(player_id, {})
                rows.append({
                    'team': team,
                    'player_id': player_id,
                    'player_name': info.get('player_name', ''),
                    'position': info.get('position', ''),
                    'year_in_school': info.get('year_in_school', ''),
                    'height': info.get('height', ''),
                    'weight': info.get('weight', ''),
                })

        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)
        logger.info(f"Exported {len(df)} players to: {filepath}")


# ============================================================================
# STANDALONE EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Create manager
    manager = RosterManager()

    print("\n" + "="*60)
    print("ROSTER MANAGER TEST")
    print("="*60)

    # Test with hypothetical data
    test_roster = pd.DataFrame({
        'team': ['Duke'] * 10,
        'player_id': [f'DUKE{i}' for i in range(10)],
        'player_name': [f'Player {i}' for i in range(10)],
        'position': ['G', 'G', 'F', 'F', 'C'] * 2,
        'year_in_school': ['Fr', 'So', 'Jr', 'Sr', 'Gr'] * 2,
    })

    manager._load_roster_from_df(test_roster, 2025)

    print(f"\nLoaded roster for Duke 2025: {len(manager.get_team_roster('Duke', 2025))} players")

    summary = manager.get_roster_summary('Duke', 2025)
    print(f"\nRoster summary:")
    print(f"  Size: {summary['roster_size']}")
    print(f"  Positions: {summary['position_breakdown']}")
    print(f"  Years: {summary['year_breakdown']}")

    print("\n" + "="*60)
