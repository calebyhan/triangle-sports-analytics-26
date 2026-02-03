"""
Feature Engineering for Player-Based ELO System

Creates 65-dimensional feature vectors for PyTorch neural network:
- Player features (10 players × 5 = 50)
- Lineup aggregate features (2 teams × 5 = 10)
- Contextual features (5)
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import logging

from .player_elo_system import PlayerEloSystem
from .config import (
    PLAYER_FEATURES, LINEUP_AGGREGATE_FEATURES, CONTEXTUAL_FEATURES,
    TOTAL_FEATURES
)

# Set up logger
logger = logging.getLogger(__name__)


class PlayerFeatureEngine:
    """
    Creates feature vectors for player-based spread prediction
    """

    def __init__(
        self,
        player_stats: pd.DataFrame,
        player_elo_system: PlayerEloSystem
    ):
        """
        Initialize Feature Engine

        Args:
            player_stats: DataFrame with player statistics
            player_elo_system: Trained player ELO system
        """
        self.player_stats = player_stats
        self.elo_system = player_elo_system

        logger.info("PlayerFeatureEngine initialized")

    # ========================================================================
    # INDIVIDUAL PLAYER FEATURES
    # ========================================================================

    def create_player_features(
        self,
        player_id: str,
        as_of_date: Optional[datetime] = None
    ) -> Dict[str, float]:
        """
        Create features for a single player

        Args:
            player_id: Player identifier
            as_of_date: Date to extract features (for time-series consistency)

        Returns:
            Dictionary with player features
        """
        # Get player's current stats
        if as_of_date is not None:
            # Filter to data available before this date
            stats = self.player_stats[
                (self.player_stats['player_id'] == player_id) &
                (pd.to_datetime(self.player_stats.get('date', datetime.now())) <= as_of_date)
            ]
        else:
            stats = self.player_stats[self.player_stats['player_id'] == player_id]

        if len(stats) == 0:
            # No stats available - use defaults
            return self._default_player_features(player_id)

        # Get most recent stats
        if 'date' in stats.columns:
            stats = stats.sort_values('date', ascending=False)

        latest = stats.iloc[0]

        features = {
            'player_elo': self.elo_system.get_player_elo(player_id),
            'usage_pct': latest.get('usage_pct', 20.0),
            'offensive_rating': latest.get('offensive_rating', 100.0),
            'defensive_rating': latest.get('defensive_rating', 100.0),
            'minutes_per_game': latest.get('minutes_per_game', 20.0),
        }

        return features

    def _default_player_features(self, player_id: str) -> Dict[str, float]:
        """
        Default features for unknown player

        Args:
            player_id: Player identifier

        Returns:
            Default feature dictionary
        """
        return {
            'player_elo': self.elo_system.get_player_elo(player_id),
            'usage_pct': 20.0,
            'offensive_rating': 100.0,
            'defensive_rating': 100.0,
            'minutes_per_game': 20.0,
        }

    # ========================================================================
    # LINEUP AGGREGATE FEATURES
    # ========================================================================

    def create_lineup_features(
        self,
        lineup: List[str],
        as_of_date: Optional[datetime] = None
    ) -> Dict[str, float]:
        """
        Create aggregate features for a lineup

        Args:
            lineup: List of player IDs (typically 5 starters)
            as_of_date: Date to extract features

        Returns:
            Dictionary with lineup aggregate features
        """
        # Get features for all players in lineup
        player_features = [
            self.create_player_features(player_id, as_of_date)
            for player_id in lineup
        ]

        if not player_features:
            return self._default_lineup_features()

        # Extract individual feature arrays
        elos = [pf['player_elo'] for pf in player_features]
        usages = [pf['usage_pct'] for pf in player_features]
        off_ratings = [pf['offensive_rating'] for pf in player_features]
        def_ratings = [pf['defensive_rating'] for pf in player_features]

        # Aggregate statistics
        lineup_features = {
            'avg_elo': np.mean(elos),
            'elo_variance': np.var(elos),
            'total_usage': np.sum(usages),
            'avg_offensive_rating': np.mean(off_ratings),
            'avg_defensive_rating': np.mean(def_ratings),
        }

        return lineup_features

    @staticmethod
    def _default_lineup_features() -> Dict[str, float]:
        """Default lineup features"""
        return {
            'avg_elo': 1000.0,
            'elo_variance': 0.0,
            'total_usage': 100.0,
            'avg_offensive_rating': 100.0,
            'avg_defensive_rating': 100.0,
        }

    # ========================================================================
    # CONTEXTUAL FEATURES
    # ========================================================================

    def create_contextual_features(
        self,
        home_team: str,
        away_team: str,
        game_date: datetime,
        neutral: bool = False,
        conference_game: bool = False,
        games_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """
        Create contextual features for a game

        Args:
            home_team: Home team name
            away_team: Away team name
            game_date: Game date
            neutral: If True, neutral site
            conference_game: If True, conference matchup
            games_df: Optional DataFrame of all games (for rest days calculation)

        Returns:
            Dictionary with contextual features
        """
        # Home court advantage
        hca = 0.0 if neutral else 1.0

        # Rest days (days since last game)
        home_rest = self._calculate_rest_days(home_team, game_date, games_df)
        away_rest = self._calculate_rest_days(away_team, game_date, games_df)

        # Season phase (0 = early Nov, 1 = late March)
        season_phase = self._calculate_season_phase(game_date)

        # Conference game indicator
        conf_game = 1.0 if conference_game else 0.0

        features = {
            'home_court_advantage': hca,
            'home_rest_days': home_rest,
            'away_rest_days': away_rest,
            'season_phase': season_phase,
            'conference_game': conf_game,
        }

        return features

    @staticmethod
    def _calculate_rest_days(
        team: str,
        game_date: datetime,
        games_df: Optional[pd.DataFrame]
    ) -> float:
        """
        Calculate days of rest before a game

        Args:
            team: Team name
            game_date: Current game date
            games_df: DataFrame of all games

        Returns:
            Days since last game (default 2.0)
        """
        if games_df is None or len(games_df) == 0:
            return 2.0  # Default assumption

        # Filter to this team's games before this date
        team_games = games_df[
            ((games_df['home_team'] == team) | (games_df['away_team'] == team)) &
            (pd.to_datetime(games_df['date']) < game_date)
        ]

        if len(team_games) == 0:
            return 7.0  # First game of season

        # Get most recent game
        team_games = team_games.sort_values('date', ascending=False)
        last_game_date = pd.to_datetime(team_games.iloc[0]['date'])

        # Calculate rest days
        rest_days = (game_date - last_game_date).days

        return float(rest_days)

    @staticmethod
    def _calculate_season_phase(game_date: datetime) -> float:
        """
        Calculate season phase (0 = early, 1 = late)

        Args:
            game_date: Game date

        Returns:
            Season phase indicator (0-1)
        """
        # NCAA season typically: Nov 1 - Mar 31
        season_start = datetime(game_date.year, 11, 1)

        # If game is before November, it's preseason (previous year's season)
        if game_date.month < 11:
            season_start = datetime(game_date.year - 1, 11, 1)

        season_end = datetime(game_date.year, 3, 31)
        if game_date.month < 4:
            season_end = datetime(game_date.year, 3, 31)
        else:
            season_end = datetime(game_date.year + 1, 3, 31)

        # Calculate progress through season (0-1)
        total_days = (season_end - season_start).days
        days_elapsed = (game_date - season_start).days

        if total_days <= 0:
            return 0.5

        phase = days_elapsed / total_days

        # Clamp to [0, 1]
        return max(0.0, min(1.0, phase))

    # ========================================================================
    # COMPLETE FEATURE VECTOR
    # ========================================================================

    def create_matchup_features(
        self,
        home_lineup: List[str],
        away_lineup: List[str],
        game_date: datetime,
        home_team: str = None,
        away_team: str = None,
        neutral: bool = False,
        conference_game: bool = False,
        games_df: Optional[pd.DataFrame] = None
    ) -> np.ndarray:
        """
        Create complete 65D feature vector for a game

        Args:
            home_lineup: Home team lineup (5 player IDs)
            away_lineup: Away team lineup (5 player IDs)
            game_date: Game date
            home_team: Home team name (for rest days)
            away_team: Away team name (for rest days)
            neutral: Neutral site indicator
            conference_game: Conference game indicator
            games_df: DataFrame of all games (for rest days)

        Returns:
            NumPy array of shape (65,)
        """
        feature_vector = []

        # 1. Individual player features (50 features: 5 per player × 10 players)
        for player_id in home_lineup + away_lineup:
            player_feats = self.create_player_features(player_id, game_date)

            # Add in order defined by PLAYER_FEATURES
            for feat_name in PLAYER_FEATURES:
                feature_vector.append(player_feats.get(feat_name, 0.0))

        # 2. Lineup aggregate features (10 features: 5 per team × 2 teams)
        home_lineup_feats = self.create_lineup_features(home_lineup, game_date)
        away_lineup_feats = self.create_lineup_features(away_lineup, game_date)

        for feat_name in LINEUP_AGGREGATE_FEATURES:
            feature_vector.append(home_lineup_feats.get(feat_name, 0.0))

        for feat_name in LINEUP_AGGREGATE_FEATURES:
            feature_vector.append(away_lineup_feats.get(feat_name, 0.0))

        # 3. Contextual features (5 features)
        contextual_feats = self.create_contextual_features(
            home_team or 'Unknown',
            away_team or 'Unknown',
            game_date,
            neutral,
            conference_game,
            games_df
        )

        for feat_name in CONTEXTUAL_FEATURES:
            feature_vector.append(contextual_feats.get(feat_name, 0.0))

        # Convert to numpy array
        feature_array = np.array(feature_vector, dtype=np.float32)

        # Validate dimensions
        assert len(feature_array) == TOTAL_FEATURES, \
            f"Feature dimension mismatch: expected {TOTAL_FEATURES}, got {len(feature_array)}"

        return feature_array

    # ========================================================================
    # BATCH FEATURE CREATION
    # ========================================================================

    def create_features_batch(
        self,
        games_df: pd.DataFrame,
        lineup_df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create features for a batch of games

        Args:
            games_df: DataFrame with columns: game_id, date, home_team, away_team,
                     home_score, away_score, neutral, conference_game
            lineup_df: DataFrame with columns: game_id, team, player1-player5

        Returns:
            (X, y) where X is (n_games, 65) features and y is (n_games,) targets
        """
        logger.info(f"Creating features for {len(games_df)} games...")

        X_list = []
        y_list = []

        for idx, game in games_df.iterrows():
            game_id = game.get('game_id', idx)
            game_date = pd.to_datetime(game['date'])
            home_team = game['home_team']
            away_team = game['away_team']
            neutral = game.get('neutral', False)
            conference_game = game.get('conference_game', False)

            # Get lineups
            home_lineup = self._get_lineup_from_df(lineup_df, game_id, home_team)
            away_lineup = self._get_lineup_from_df(lineup_df, game_id, away_team)

            if not home_lineup or not away_lineup:
                logger.warning(f"  Skipping game {game_id}: missing lineup data")
                continue

            # Create features
            try:
                features = self.create_matchup_features(
                    home_lineup, away_lineup, game_date,
                    home_team, away_team,
                    neutral, conference_game,
                    games_df
                )

                # Target: point spread (home - away)
                target = game.get('home_score', 0) - game.get('away_score', 0)

                X_list.append(features)
                y_list.append(target)

            except Exception as e:
                logger.warning(f"  Failed to create features for game {game_id}: {e}")
                continue

        if not X_list:
            logger.error("No features created!")
            return np.array([]), np.array([])

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)

        logger.info(f"  Created features: X shape {X.shape}, y shape {y.shape}")

        return X, y

    @staticmethod
    def _get_lineup_from_df(
        lineup_df: pd.DataFrame,
        game_id: str,
        team: str
    ) -> List[str]:
        """
        Extract lineup for a team from lineup DataFrame

        Args:
            lineup_df: Lineup DataFrame
            game_id: Game identifier
            team: Team name

        Returns:
            List of player IDs (5 starters)
        """
        team_lineup = lineup_df[
            (lineup_df['game_id'] == game_id) &
            (lineup_df['team'] == team)
        ]

        if len(team_lineup) == 0:
            return []

        lineup_row = team_lineup.iloc[0]

        # Extract player columns (player1, player2, ..., player5)
        lineup = []
        for i in range(1, 6):
            player_col = f'player{i}'
            if player_col in lineup_row:
                player_id = lineup_row[player_col]
                if pd.notna(player_id):
                    lineup.append(str(player_id))

        return lineup

    # ========================================================================
    # FEATURE NORMALIZATION
    # ========================================================================

    @staticmethod
    def normalize_features(
        X: np.ndarray,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Normalize features (z-score normalization)

        Args:
            X: Feature matrix (n_samples, n_features)
            mean: Optional pre-computed mean (for test set)
            std: Optional pre-computed std (for test set)

        Returns:
            (X_normalized, mean, std)
        """
        if mean is None:
            mean = np.mean(X, axis=0)

        if std is None:
            std = np.std(X, axis=0)
            # Prevent division by zero
            std[std == 0] = 1.0

        X_normalized = (X - mean) / std

        return X_normalized, mean, std


# ============================================================================
# STANDALONE TESTING
# ============================================================================

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Create mock data for testing
    from .player_elo_system import PlayerEloSystem

    print("\n" + "="*60)
    print("FEATURE ENGINE TEST")
    print("="*60)

    # Initialize ELO system
    elo_system = PlayerEloSystem()

    # Mock player stats
    player_stats = pd.DataFrame({
        'player_id': [f'P{i}' for i in range(10)],
        'usage_pct': [20 + i for i in range(10)],
        'offensive_rating': [100 + i*2 for i in range(10)],
        'defensive_rating': [100 - i for i in range(10)],
        'minutes_per_game': [25 + i for i in range(10)],
    })

    # Create feature engine
    feature_engine = PlayerFeatureEngine(player_stats, elo_system)

    # Test lineup
    home_lineup = [f'P{i}' for i in range(5)]
    away_lineup = [f'P{i}' for i in range(5, 10)]

    # Create features
    features = feature_engine.create_matchup_features(
        home_lineup, away_lineup,
        datetime.now(),
        home_team='Team A',
        away_team='Team B'
    )

    print(f"\nFeature vector shape: {features.shape}")
    print(f"Feature vector (first 10): {features[:10]}")
    print(f"Feature vector (last 5): {features[-5:]}")

    assert features.shape == (TOTAL_FEATURES,), \
        f"Expected {TOTAL_FEATURES} features, got {features.shape[0]}"

    print("\n✓ Feature engine test passed!")
    print("="*60)
