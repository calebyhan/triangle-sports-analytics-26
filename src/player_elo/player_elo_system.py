"""
Player-Based Elo Rating System for NCAA Basketball

Adapted from team-based src/elo.py to track individual player ratings.
Team strength is calculated as weighted average of player ELOs by usage% or minutes.

Key parameters:
- K-factor: 20 (lower than team's 38 - less volatile)
- Default rating: 1000 (lower than team's 1500)
- Season carryover: 0.75 (higher than team's 0.64 - players more stable)
- Home Court Advantage: 2.0 ELO points
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
import json
import logging
from pathlib import Path

from .config import PLAYER_ELO_CONFIG, CONFERENCE_MAPPINGS

# Set up logger
logger = logging.getLogger(__name__)


class PlayerEloSystem:
    """
    Player-level ELO rating system for NCAA basketball

    Tracks individual player ratings and aggregates to team strength.
    Updates are based on game performance, minutes played, and impact.
    """

    def __init__(self, config: Dict = None):
        """
        Initialize Player ELO System

        Args:
            config: Configuration dictionary (defaults to PLAYER_ELO_CONFIG)
        """
        self.config = config or PLAYER_ELO_CONFIG

        # Extract parameters
        self.default_rating = self.config['default_rating']
        self.k_factor = self.config['k_factor']
        self.hca = self.config['home_court_advantage']
        self.season_carryover = self.config['season_carryover']
        self.minutes_threshold = self.config['minutes_threshold']
        self.weighting_method = self.config['weighting_method']
        self.points_per_elo = self.config['points_per_elo']
        self.mov_enabled = self.config.get('mov_enabled', True)

        # Current ratings: player_id -> rating
        self.player_elos: Dict[str, float] = {}

        # Player metadata
        self.player_usage: Dict[str, float] = {}      # player_id -> usage%
        self.player_minutes: Dict[str, float] = {}    # player_id -> avg minutes
        self.player_position: Dict[str, str] = {}     # player_id -> position
        self.player_team: Dict[str, str] = {}         # player_id -> current team

        # Team rosters: team_name -> [player_ids]
        self.team_rosters: Dict[str, List[str]] = {}

        # Conference mappings (for position regression)
        self.team_conference: Dict[str, str] = {}
        self._load_conference_mappings()

        # Rating history for backtesting
        self.history: List[Dict] = []

        logger.info(f"PlayerEloSystem initialized (K={self.k_factor}, default={self.default_rating})")

    def _load_conference_mappings(self):
        """Load conference mappings from config"""
        for conf, teams in CONFERENCE_MAPPINGS.items():
            for team in teams:
                self.team_conference[team] = conf

    # ========================================================================
    # PLAYER ELO MANAGEMENT
    # ========================================================================

    def get_player_elo(self, player_id: str) -> float:
        """
        Get player's current ELO rating

        Args:
            player_id: Unique player identifier

        Returns:
            Player's ELO rating (default if new player)
        """
        if player_id not in self.player_elos:
            self.player_elos[player_id] = self.default_rating
        return self.player_elos[player_id]

    def set_player_metadata(
        self,
        player_id: str,
        usage: Optional[float] = None,
        minutes: Optional[float] = None,
        position: Optional[str] = None,
        team: Optional[str] = None
    ):
        """
        Set metadata for a player

        Args:
            player_id: Unique player identifier
            usage: Usage percentage (0-100)
            minutes: Average minutes per game
            position: Position (G, F, C)
            team: Current team
        """
        if usage is not None:
            self.player_usage[player_id] = usage
        if minutes is not None:
            self.player_minutes[player_id] = minutes
        if position is not None:
            self.player_position[player_id] = position
        if team is not None:
            self.player_team[player_id] = team

    def register_roster(self, team: str, player_ids: List[str]):
        """
        Register team roster

        Args:
            team: Team name
            player_ids: List of player IDs on roster
        """
        self.team_rosters[team] = player_ids

        # Set team for all players
        for player_id in player_ids:
            self.player_team[player_id] = team

    # ========================================================================
    # TEAM STRENGTH CALCULATION
    # ========================================================================

    def calculate_team_strength(
        self,
        lineup: List[str],
        weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Calculate team strength from lineup ELOs

        Args:
            lineup: List of player IDs (typically 5 starters)
            weights: Optional custom weights (player_id -> weight)

        Returns:
            Aggregate team ELO rating
        """
        if not lineup:
            logger.warning("Empty lineup provided")
            return self.default_rating

        # Get player ELOs
        player_elos = [self.get_player_elo(pid) for pid in lineup]

        # Determine weighting
        if weights is not None:
            # Custom weights provided
            weight_values = [weights.get(pid, 1.0) for pid in lineup]
        elif self.weighting_method == 'usage':
            # Weight by usage percentage
            weight_values = [self.player_usage.get(pid, 20.0) for pid in lineup]
        elif self.weighting_method == 'minutes':
            # Weight by minutes per game
            weight_values = [self.player_minutes.get(pid, 20.0) for pid in lineup]
        else:  # 'equal'
            # Equal weighting
            weight_values = [1.0] * len(lineup)

        # Normalize weights
        total_weight = sum(weight_values)
        if total_weight == 0:
            total_weight = len(lineup)
            weight_values = [1.0] * len(lineup)

        normalized_weights = [w / total_weight for w in weight_values]

        # Weighted average
        team_elo = sum(elo * weight for elo, weight in zip(player_elos, normalized_weights))

        return team_elo

    # ========================================================================
    # SPREAD PREDICTION
    # ========================================================================

    def predict_spread(
        self,
        home_lineup: List[str],
        away_lineup: List[str],
        neutral: bool = False
    ) -> float:
        """
        Predict point spread given two lineups

        Args:
            home_lineup: Home team's lineup (player IDs)
            away_lineup: Away team's lineup (player IDs)
            neutral: If True, no home court advantage

        Returns:
            Predicted spread (positive = home favored)
        """
        home_elo = self.calculate_team_strength(home_lineup)
        away_elo = self.calculate_team_strength(away_lineup)

        elo_diff = home_elo - away_elo
        hca = 0 if neutral else self.hca

        # Convert ELO difference to point spread
        spread = (elo_diff / self.points_per_elo) + hca

        return spread

    def predict_win_probability(
        self,
        home_lineup: List[str],
        away_lineup: List[str],
        neutral: bool = False
    ) -> float:
        """
        Predict win probability for home team

        Args:
            home_lineup: Home team's lineup (player IDs)
            away_lineup: Away team's lineup (player IDs)
            neutral: If True, no home court advantage

        Returns:
            Win probability for home team (0-1)
        """
        home_elo = self.calculate_team_strength(home_lineup)
        away_elo = self.calculate_team_strength(away_lineup)

        # Adjust for HCA
        hca_adjustment = 0 if neutral else self.hca
        home_elo_adj = home_elo + hca_adjustment

        # Logistic probability
        return self._expected_score(home_elo_adj, away_elo)

    @staticmethod
    def _expected_score(rating_a: float, rating_b: float) -> float:
        """
        Calculate expected score (win probability) for rating A

        Args:
            rating_a: Rating of team/player A
            rating_b: Rating of team/player B

        Returns:
            Expected probability of A winning
        """
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))

    # ========================================================================
    # RATING UPDATES
    # ========================================================================

    def update_from_game(
        self,
        home_lineup: List[str],
        away_lineup: List[str],
        home_score: float,
        away_score: float,
        player_stats: Optional[pd.DataFrame] = None,
        neutral: bool = False
    ) -> Dict[str, Tuple[float, float]]:
        """
        Update all player ELOs from a game result

        Args:
            home_lineup: Home team's lineup (player IDs)
            away_lineup: Away team's lineup (player IDs)
            home_score: Home team's final score
            away_score: Away team's final score
            player_stats: Optional game boxscore with 'player_id', 'minutes', 'plus_minus'
            neutral: If True, no home court advantage

        Returns:
            Dictionary of player_id -> (old_elo, new_elo)
        """
        # Calculate team strengths
        home_team_elo = self.calculate_team_strength(home_lineup)
        away_team_elo = self.calculate_team_strength(away_lineup)

        # Adjust for HCA
        hca_adjustment = 0 if neutral else self.hca
        home_elo_adj = home_team_elo + hca_adjustment

        # Expected scores
        home_expected = self._expected_score(home_elo_adj, away_team_elo)
        away_expected = 1 - home_expected

        # Actual result
        margin = home_score - away_score
        home_won = 1 if margin > 0 else (0 if margin < 0 else 0.5)
        away_won = 1 - home_won

        # Margin of victory multiplier
        if self.mov_enabled and margin != 0:
            winner_elo = home_team_elo if margin > 0 else away_team_elo
            loser_elo = away_team_elo if margin > 0 else home_team_elo
            mov_mult = self._margin_of_victory_multiplier(abs(margin), winner_elo, loser_elo)
        else:
            mov_mult = 1.0

        # Track updates
        updates = {}

        # Update home players
        for player_id in home_lineup:
            old_elo = self.get_player_elo(player_id)

            # Get player's minutes and impact
            minutes = self._get_player_minutes(player_id, player_stats)
            impact = self._get_player_impact(player_id, player_stats, margin)

            # Skip if below minutes threshold
            if minutes < self.minutes_threshold:
                continue

            # Calculate update
            new_elo = self._update_player_elo(
                old_elo, home_expected, home_won,
                minutes, impact, mov_mult
            )

            self.player_elos[player_id] = new_elo
            updates[player_id] = (old_elo, new_elo)

        # Update away players
        for player_id in away_lineup:
            old_elo = self.get_player_elo(player_id)

            # Get player's minutes and impact
            minutes = self._get_player_minutes(player_id, player_stats)
            impact = self._get_player_impact(player_id, player_stats, -margin)

            # Skip if below minutes threshold
            if minutes < self.minutes_threshold:
                continue

            # Calculate update
            new_elo = self._update_player_elo(
                old_elo, away_expected, away_won,
                minutes, impact, mov_mult
            )

            self.player_elos[player_id] = new_elo
            updates[player_id] = (old_elo, new_elo)

        return updates

    def _update_player_elo(
        self,
        current_elo: float,
        expected_score: float,
        actual_score: float,
        minutes: float,
        impact: float,
        mov_mult: float
    ) -> float:
        """
        Calculate updated player ELO

        Args:
            current_elo: Player's current ELO
            expected_score: Expected win probability
            actual_score: Actual result (1=win, 0=loss, 0.5=tie)
            minutes: Minutes played
            impact: Player's contribution to margin
            mov_mult: Margin of victory multiplier

        Returns:
            Updated ELO rating
        """
        # Weight by minutes played (0-1 scale, cap at 35 minutes)
        minutes_weight = min(minutes / 35.0, 1.0)

        # Impact multiplier (based on plus/minus or contribution)
        impact_mult = np.log(abs(impact) + 1) * 0.5 if self.mov_enabled else 1.0

        # Effective K-factor
        k_effective = self.k_factor * minutes_weight * impact_mult * mov_mult

        # Standard ELO update
        elo_change = k_effective * (actual_score - expected_score)

        return current_elo + elo_change

    @staticmethod
    def _margin_of_victory_multiplier(
        margin: float,
        winner_elo: float,
        loser_elo: float
    ) -> float:
        """
        Calculate margin of victory multiplier

        Args:
            margin: Absolute margin of victory
            winner_elo: Winner's ELO
            loser_elo: Loser's ELO

        Returns:
            Multiplier for K-factor
        """
        elo_diff = winner_elo - loser_elo

        # Log scaling
        mov_factor = np.log(abs(margin) + 1)

        # Autocorrelation adjustment
        autocorr = 2.2 / ((elo_diff * 0.001) + 2.2)

        return mov_factor * autocorr

    @staticmethod
    def _get_player_minutes(
        player_id: str,
        player_stats: Optional[pd.DataFrame]
    ) -> float:
        """
        Get minutes played for a player from boxscore

        Args:
            player_id: Player ID
            player_stats: Game boxscore

        Returns:
            Minutes played (default 20 if not found)
        """
        if player_stats is None or len(player_stats) == 0:
            return 20.0  # Default assumption

        player_row = player_stats[player_stats['player_id'] == player_id]
        if len(player_row) == 0:
            return 20.0

        return player_row.iloc[0].get('minutes', 20.0)

    @staticmethod
    def _get_player_impact(
        player_id: str,
        player_stats: Optional[pd.DataFrame],
        team_margin: float
    ) -> float:
        """
        Get player's impact/contribution

        Args:
            player_id: Player ID
            player_stats: Game boxscore
            team_margin: Team's margin (positive if won)

        Returns:
            Player's impact (plus/minus or estimated contribution)
        """
        if player_stats is None or len(player_stats) == 0:
            # Estimate: distribute team margin equally among starters
            return team_margin / 5.0

        player_row = player_stats[player_stats['player_id'] == player_id]
        if len(player_row) == 0:
            return team_margin / 5.0

        # Use plus/minus if available
        if 'plus_minus' in player_row.columns:
            return player_row.iloc[0]['plus_minus']

        # Otherwise estimate from team margin
        return team_margin / 5.0

    # ========================================================================
    # SEASON RESET
    # ========================================================================

    def season_reset(self, season: int):
        """
        Reset player ratings at season start (regression to position means)

        Args:
            season: New season year
        """
        logger.info(f"Performing season reset for {season}...")

        position_regression = self.config['position_regression']
        carryover = self.season_carryover

        reset_count = 0

        for player_id, old_elo in list(self.player_elos.items()):
            # Get player position
            position = self.player_position.get(player_id, 'Unknown')
            position_mean = position_regression.get(position, self.default_rating)

            # Regress to position mean
            new_elo = carryover * old_elo + (1 - carryover) * position_mean

            self.player_elos[player_id] = new_elo
            reset_count += 1

        logger.info(f"  Reset {reset_count} player ratings (carryover={carryover})")

    # ========================================================================
    # STATE MANAGEMENT
    # ========================================================================

    def save_state(self, filepath: Path):
        """
        Save player ELO state to JSON

        Args:
            filepath: Path to save JSON file
        """
        state = {
            'player_elos': self.player_elos,
            'player_usage': self.player_usage,
            'player_minutes': self.player_minutes,
            'player_position': self.player_position,
            'player_team': self.player_team,
            'team_rosters': self.team_rosters,
            'config': self.config,
        }

        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)

        logger.info(f"Saved player ELO state to: {filepath}")

    def load_state(self, filepath: Path):
        """
        Load player ELO state from JSON

        Args:
            filepath: Path to JSON file
        """
        with open(filepath, 'r') as f:
            state = json.load(f)

        self.player_elos = state.get('player_elos', {})
        self.player_usage = state.get('player_usage', {})
        self.player_minutes = state.get('player_minutes', {})
        self.player_position = state.get('player_position', {})
        self.player_team = state.get('player_team', {})
        self.team_rosters = state.get('team_rosters', {})

        logger.info(f"Loaded player ELO state from: {filepath}")
        logger.info(f"  {len(self.player_elos)} player ratings loaded")

    # ========================================================================
    # INITIALIZATION HELPERS
    # ========================================================================

    def initialize_player_elo(
        self,
        player_id: str,
        team: str,
        position: str = 'Unknown',
        previous_elo: Optional[float] = None,
        transfer_from: Optional[str] = None
    ) -> float:
        """
        Initialize ELO for new/transfer player

        Args:
            player_id: Player identifier
            team: Current team
            position: Player position
            previous_elo: Previous ELO if transferring
            transfer_from: Team transferring from

        Returns:
            Initialized ELO rating
        """
        if previous_elo is not None:
            # Transfer with known ELO: 90% previous + 10% new team avg
            team_avg = self._get_team_avg_elo(team)
            elo = 0.9 * previous_elo + 0.1 * team_avg

        elif transfer_from is not None:
            # Transfer from known team: use that team's average
            elo = self._get_team_avg_elo(transfer_from)

        else:
            # Freshman or unknown: position mean + team adjustment
            position_mean = self.config['position_regression'].get(position, self.default_rating)
            team_avg = self._get_team_avg_elo(team)
            team_adjustment = (team_avg - self.default_rating) * 0.3
            elo = position_mean + team_adjustment

        self.player_elos[player_id] = elo
        self.set_player_metadata(player_id, position=position, team=team)

        return elo

    def _get_team_avg_elo(self, team: str) -> float:
        """
        Calculate average ELO for a team's roster

        Args:
            team: Team name

        Returns:
            Average ELO (default rating if no roster)
        """
        roster = self.team_rosters.get(team, [])
        if not roster:
            return self.default_rating

        elos = [self.get_player_elo(pid) for pid in roster]
        return np.mean(elos) if elos else self.default_rating


# ============================================================================
# STANDALONE TESTING
# ============================================================================

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Create system
    elo_system = PlayerEloSystem()

    # Test with hypothetical game
    print("\n" + "="*60)
    print("TESTING PLAYER ELO SYSTEM")
    print("="*60)

    # Register two teams
    duke_lineup = [f"DUKE{i}" for i in range(5)]
    unc_lineup = [f"UNC{i}" for i in range(5)]

    # Set metadata (hypothetical)
    for i, pid in enumerate(duke_lineup):
        elo_system.set_player_metadata(pid, usage=20+i*2, minutes=25+i, position='G')

    for i, pid in enumerate(unc_lineup):
        elo_system.set_player_metadata(pid, usage=18+i*2, minutes=23+i, position='F')

    print("\nInitial Team Strengths:")
    print(f"  Duke: {elo_system.calculate_team_strength(duke_lineup):.1f}")
    print(f"  UNC: {elo_system.calculate_team_strength(unc_lineup):.1f}")

    # Simulate game: Duke 85, UNC 78
    print("\nSimulating game: Duke 85, UNC 78")
    updates = elo_system.update_from_game(
        duke_lineup, unc_lineup, 85, 78
    )

    print(f"\nUpdated {len(updates)} player ratings")
    print("\nFinal Team Strengths:")
    print(f"  Duke: {elo_system.calculate_team_strength(duke_lineup):.1f}")
    print(f"  UNC: {elo_system.calculate_team_strength(unc_lineup):.1f}")

    print("\n" + "="*60)
