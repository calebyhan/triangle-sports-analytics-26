"""
Feature Engineering Module for Triangle Sports Analytics
Creates features for point spread prediction models

Enhanced with:
- Dean Oliver's Four Factors
- Temporal/momentum features
- Multi-source ensemble ratings
- Elo integration
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta


# Dean Oliver's Four Factors weights
FOUR_FACTORS_WEIGHTS = {
    'efg': 0.40,    # Effective Field Goal %
    'tov': 0.25,    # Turnover %
    'orb': 0.20,    # Offensive Rebound %
    'ft_rate': 0.15  # Free Throw Rate
}


class FeatureEngine:
    """Create features for basketball point spread prediction"""
    
    # Average home court advantage in college basketball (points)
    HOME_COURT_ADVANTAGE = 3.5
    
    def __init__(self, team_stats_df: pd.DataFrame, games_df: Optional[pd.DataFrame] = None):
        """
        Initialize feature engine
        
        Args:
            team_stats_df: DataFrame with team statistics
            games_df: Optional DataFrame with historical game results
        """
        self.team_stats = team_stats_df.set_index('team') if 'team' in team_stats_df.columns else team_stats_df
        self.games_df = games_df
    
    def get_team_stat(self, team: str, stat: str, default: float = 0.0) -> float:
        """Safely get a team statistic"""
        try:
            return self.team_stats.loc[team, stat]
        except (KeyError, TypeError):
            return default
    
    def create_basic_features(self, home_team: str, away_team: str) -> Dict[str, float]:
        """
        Create basic differential features for a matchup
        
        Args:
            home_team: Home team name
            away_team: Away team name
            
        Returns:
            Dictionary of features
        """
        features = {}
        
        # Points per game differential
        home_ppg = self.get_team_stat(home_team, 'ppg', 75.0)
        away_ppg = self.get_team_stat(away_team, 'ppg', 75.0)
        features['ppg_diff'] = home_ppg - away_ppg
        
        # Opponent points per game differential (lower is better for defense)
        home_opp_ppg = self.get_team_stat(home_team, 'opp_ppg', 70.0)
        away_opp_ppg = self.get_team_stat(away_team, 'opp_ppg', 70.0)
        features['opp_ppg_diff'] = away_opp_ppg - home_opp_ppg  # Positive = home has better defense
        
        # Offensive efficiency differential
        home_off_eff = self.get_team_stat(home_team, 'off_efficiency', 100.0)
        away_off_eff = self.get_team_stat(away_team, 'off_efficiency', 100.0)
        features['off_efficiency_diff'] = home_off_eff - away_off_eff
        
        # Defensive efficiency differential (lower is better)
        home_def_eff = self.get_team_stat(home_team, 'def_efficiency', 100.0)
        away_def_eff = self.get_team_stat(away_team, 'def_efficiency', 100.0)
        features['def_efficiency_diff'] = away_def_eff - home_def_eff  # Positive = home has better defense
        
        # Net efficiency (adjusted efficiency margin)
        home_net = home_off_eff - home_def_eff
        away_net = away_off_eff - away_def_eff
        features['net_efficiency_diff'] = home_net - away_net
        
        # Shooting percentages
        features['fg_pct_diff'] = (
            self.get_team_stat(home_team, 'fg_pct', 0.45) - 
            self.get_team_stat(away_team, 'fg_pct', 0.45)
        )
        features['three_pct_diff'] = (
            self.get_team_stat(home_team, 'three_pct', 0.35) - 
            self.get_team_stat(away_team, 'three_pct', 0.35)
        )
        features['ft_pct_diff'] = (
            self.get_team_stat(home_team, 'ft_pct', 0.72) - 
            self.get_team_stat(away_team, 'ft_pct', 0.72)
        )
        
        # Rebounding differential
        home_rpg = self.get_team_stat(home_team, 'rpg', 35.0)
        away_rpg = self.get_team_stat(away_team, 'rpg', 35.0)
        features['rpg_diff'] = home_rpg - away_rpg
        
        # Turnover differential (lower is better)
        home_tpg = self.get_team_stat(home_team, 'tpg', 12.0)
        away_tpg = self.get_team_stat(away_team, 'tpg', 12.0)
        features['tpg_diff'] = away_tpg - home_tpg  # Positive = home commits fewer turnovers
        
        # Assists differential
        home_apg = self.get_team_stat(home_team, 'apg', 14.0)
        away_apg = self.get_team_stat(away_team, 'apg', 14.0)
        features['apg_diff'] = home_apg - away_apg
        
        # Home court advantage (binary)
        features['is_home'] = 1.0
        
        return features
    
    def create_advanced_features(self, home_team: str, away_team: str) -> Dict[str, float]:
        """
        Create advanced features for a matchup
        
        Args:
            home_team: Home team name
            away_team: Away team name
            
        Returns:
            Dictionary of advanced features
        """
        features = {}
        
        # Strength of schedule differential
        home_sos = self.get_team_stat(home_team, 'sos', 0.0)
        away_sos = self.get_team_stat(away_team, 'sos', 0.0)
        features['sos_diff'] = home_sos - away_sos
        
        # Win percentage differential
        home_win_pct = self.get_team_stat(home_team, 'win_pct', 0.5)
        away_win_pct = self.get_team_stat(away_team, 'win_pct', 0.5)
        features['win_pct_diff'] = home_win_pct - away_win_pct
        
        # Conference win percentage
        home_conf_win_pct = self.get_team_stat(home_team, 'conf_win_pct', 0.5)
        away_conf_win_pct = self.get_team_stat(away_team, 'conf_win_pct', 0.5)
        features['conf_win_pct_diff'] = home_conf_win_pct - away_conf_win_pct
        
        # Pace differential (possessions per game)
        home_pace = self.get_team_stat(home_team, 'pace', 70.0)
        away_pace = self.get_team_stat(away_team, 'pace', 70.0)
        features['pace_diff'] = home_pace - away_pace
        features['avg_pace'] = (home_pace + away_pace) / 2  # Expected game pace
        
        # Power rating differential (if available)
        home_rating = self.get_team_stat(home_team, 'power_rating', 0.0)
        away_rating = self.get_team_stat(away_team, 'power_rating', 0.0)
        features['power_rating_diff'] = home_rating - away_rating
        
        return features
    
    def create_contextual_features(
        self, 
        home_team: str, 
        away_team: str,
        game_date: datetime,
        home_last_game: Optional[datetime] = None,
        away_last_game: Optional[datetime] = None
    ) -> Dict[str, float]:
        """
        Create contextual features based on game situation
        
        Args:
            home_team: Home team name
            away_team: Away team name
            game_date: Date of the game
            home_last_game: Date of home team's last game
            away_last_game: Date of away team's last game
            
        Returns:
            Dictionary of contextual features
        """
        features = {}
        
        # Days of rest
        if home_last_game:
            home_rest = (game_date - home_last_game).days
            features['home_rest_days'] = min(home_rest, 7)  # Cap at 7 days
        else:
            features['home_rest_days'] = 3  # Default
        
        if away_last_game:
            away_rest = (game_date - away_last_game).days
            features['away_rest_days'] = min(away_rest, 7)
        else:
            features['away_rest_days'] = 3
        
        features['rest_diff'] = features['home_rest_days'] - features['away_rest_days']
        
        # Back-to-back indicator
        features['home_back_to_back'] = 1.0 if features['home_rest_days'] <= 1 else 0.0
        features['away_back_to_back'] = 1.0 if features['away_rest_days'] <= 1 else 0.0
        
        return features
    
    def create_rolling_features(
        self, 
        team: str, 
        as_of_date: datetime, 
        window: int = 5
    ) -> Dict[str, float]:
        """
        Create rolling average features for recent performance
        
        Args:
            team: Team name
            as_of_date: Calculate features as of this date
            window: Number of games for rolling average
            
        Returns:
            Dictionary of rolling features
        """
        features = {}
        
        if self.games_df is None:
            return features
        
        # Filter team's games before the given date
        team_games = self.games_df[
            ((self.games_df['home_team'] == team) | (self.games_df['away_team'] == team)) &
            (self.games_df['date'] < as_of_date)
        ].sort_values('date', ascending=False).head(window)
        
        if len(team_games) == 0:
            return features
        
        # Calculate rolling statistics
        # This needs to be adapted based on actual game data structure
        
        return features
    
    def create_matchup_features(
        self,
        home_team: str,
        away_team: str,
        game_date: Optional[datetime] = None
    ) -> Dict[str, float]:
        """
        Create all features for a matchup
        
        Args:
            home_team: Home team name
            away_team: Away team name
            game_date: Optional game date for contextual features
            
        Returns:
            Dictionary of all features
        """
        features = {}
        
        # Basic features
        features.update(self.create_basic_features(home_team, away_team))
        
        # Advanced features
        features.update(self.create_advanced_features(home_team, away_team))
        
        # Contextual features (if date provided)
        if game_date:
            features.update(self.create_contextual_features(
                home_team, away_team, game_date
            ))
        
        return features
    
    def create_prediction_dataset(
        self, 
        matchups_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create complete feature dataset for predictions
        
        Args:
            matchups_df: DataFrame with columns ['Date', 'Home', 'Away']
            
        Returns:
            DataFrame with features for each matchup
        """
        feature_list = []
        
        for idx, row in matchups_df.iterrows():
            game_date = pd.to_datetime(row['Date']) if pd.notna(row['Date']) else None
            
            features = self.create_matchup_features(
                home_team=row['Home'],
                away_team=row['Away'],
                game_date=game_date
            )
            
            # Add identifying information
            features['game_idx'] = idx
            features['home_team'] = row['Home']
            features['away_team'] = row['Away']
            features['date'] = row['Date']
            
            feature_list.append(features)
        
        return pd.DataFrame(feature_list)


def calculate_simple_spread(
    home_team_stats: Dict[str, float],
    away_team_stats: Dict[str, float],
    home_court_advantage: float = 3.5
) -> float:
    """
    Calculate a simple point spread prediction

    Uses the formula:
    Spread = (Home PPG - Home Opp PPG) - (Away PPG - Away Opp PPG) + HCA

    Args:
        home_team_stats: Dictionary with 'ppg' and 'opp_ppg' for home team
        away_team_stats: Dictionary with 'ppg' and 'opp_ppg' for away team
        home_court_advantage: Home court advantage in points

    Returns:
        Predicted point spread (positive = home favored)
    """
    home_margin = home_team_stats.get('ppg', 75) - home_team_stats.get('opp_ppg', 70)
    away_margin = away_team_stats.get('ppg', 75) - away_team_stats.get('opp_ppg', 70)

    spread = (home_margin - away_margin) / 2 + home_court_advantage

    return spread


class FourFactorsFeatures:
    """
    Dean Oliver's Four Factors feature engineering

    The Four Factors explain 96% of variance in team wins:
    1. eFG% (40%) - Effective Field Goal Percentage
    2. TOV% (25%) - Turnover Percentage
    3. ORB% (20%) - Offensive Rebound Percentage
    4. FT Rate (15%) - Free Throw Rate
    """

    WEIGHTS = FOUR_FACTORS_WEIGHTS

    def __init__(self, team_stats_df: pd.DataFrame):
        """
        Initialize with team statistics

        Args:
            team_stats_df: DataFrame with Four Factors columns per team
                Expected columns: team, efg_o, efg_d, tov_o, tov_d, orb, drb, ft_rate_o, ft_rate_d
        """
        self.team_stats = team_stats_df.set_index('team') if 'team' in team_stats_df.columns else team_stats_df

    def get_stat(self, team: str, stat: str, default: float = 0.0) -> float:
        """Safely get team statistic"""
        try:
            return float(self.team_stats.loc[team, stat])
        except (KeyError, TypeError, ValueError):
            return default

    def compute_four_factors_diff(
        self,
        home_team: str,
        away_team: str
    ) -> Dict[str, float]:
        """
        Compute Four Factors differential features

        Args:
            home_team: Home team name
            away_team: Away team name

        Returns:
            Dictionary of Four Factors differential features
        """
        features = {}

        # Effective Field Goal % (higher is better for offense)
        home_efg_o = self.get_stat(home_team, 'efg_o', 50.0)
        home_efg_d = self.get_stat(home_team, 'efg_d', 50.0)  # Opponent's eFG%
        away_efg_o = self.get_stat(away_team, 'efg_o', 50.0)
        away_efg_d = self.get_stat(away_team, 'efg_d', 50.0)

        features['efg_o_diff'] = home_efg_o - away_efg_o
        features['efg_d_diff'] = away_efg_d - home_efg_d  # Positive = home allows less

        # Turnover % (lower is better for offense)
        home_tov_o = self.get_stat(home_team, 'tov_o', 18.0)
        home_tov_d = self.get_stat(home_team, 'tov_d', 18.0)  # Forces on opponent
        away_tov_o = self.get_stat(away_team, 'tov_o', 18.0)
        away_tov_d = self.get_stat(away_team, 'tov_d', 18.0)

        features['tov_o_diff'] = away_tov_o - home_tov_o  # Positive = home commits fewer
        features['tov_d_diff'] = home_tov_d - away_tov_d  # Positive = home forces more

        # Offensive Rebound % (higher is better)
        home_orb = self.get_stat(home_team, 'orb_pct', 28.0)
        away_orb = self.get_stat(away_team, 'orb_pct', 28.0)

        features['orb_diff'] = home_orb - away_orb

        # Defensive Rebound % (higher is better for defense)
        home_drb = self.get_stat(home_team, 'drb_pct', 72.0)
        away_drb = self.get_stat(away_team, 'drb_pct', 72.0)

        features['drb_diff'] = home_drb - away_drb

        # Free Throw Rate (higher is better for offense)
        home_ftr_o = self.get_stat(home_team, 'ft_rate_o', 0.30)
        home_ftr_d = self.get_stat(home_team, 'ft_rate_d', 0.30)
        away_ftr_o = self.get_stat(away_team, 'ft_rate_o', 0.30)
        away_ftr_d = self.get_stat(away_team, 'ft_rate_d', 0.30)

        features['ftr_o_diff'] = home_ftr_o - away_ftr_o
        features['ftr_d_diff'] = away_ftr_d - home_ftr_d  # Positive = home allows less

        # Weighted composite Four Factors score
        features['four_factors_composite'] = (
            self.WEIGHTS['efg'] * (features['efg_o_diff'] + features['efg_d_diff']) +
            self.WEIGHTS['tov'] * (features['tov_o_diff'] + features['tov_d_diff']) +
            self.WEIGHTS['orb'] * features['orb_diff'] +
            self.WEIGHTS['ft_rate'] * (features['ftr_o_diff'] + features['ftr_d_diff'])
        )

        return features


class TemporalFeatures:
    """
    Temporal and momentum features for basketball prediction

    Includes:
    - Rest days
    - Back-to-back detection
    - Exponential weighted averages of recent performance
    - Road trip fatigue
    """

    def __init__(self, games_df: Optional[pd.DataFrame] = None):
        """
        Initialize with historical game data

        Args:
            games_df: DataFrame with game history for computing recent form
        """
        self.games_df = games_df

    def compute_rest_features(
        self,
        home_team: str,
        away_team: str,
        game_date: datetime,
        home_last_game: Optional[datetime] = None,
        away_last_game: Optional[datetime] = None
    ) -> Dict[str, float]:
        """
        Compute rest-related features

        Args:
            home_team: Home team name
            away_team: Away team name
            game_date: Date of game
            home_last_game: Date of home team's previous game
            away_last_game: Date of away team's previous game

        Returns:
            Dictionary of rest features
        """
        features = {}

        # Home team rest
        if home_last_game:
            home_rest = (game_date - home_last_game).days
        else:
            home_rest = 3  # Default
        features['home_rest_days'] = min(home_rest, 10)  # Cap at 10

        # Away team rest
        if away_last_game:
            away_rest = (game_date - away_last_game).days
        else:
            away_rest = 3
        features['away_rest_days'] = min(away_rest, 10)

        # Rest differential
        features['rest_diff'] = features['home_rest_days'] - features['away_rest_days']

        # Back-to-back indicators (1 day or less between games)
        features['home_b2b'] = 1.0 if features['home_rest_days'] <= 1 else 0.0
        features['away_b2b'] = 1.0 if features['away_rest_days'] <= 1 else 0.0
        features['b2b_advantage'] = features['away_b2b'] - features['home_b2b']

        # Well-rested advantage (3+ days)
        features['home_well_rested'] = 1.0 if features['home_rest_days'] >= 3 else 0.0
        features['away_well_rested'] = 1.0 if features['away_rest_days'] >= 3 else 0.0

        return features

    def compute_ewa_features(
        self,
        team: str,
        as_of_date: datetime,
        metrics: List[str],
        alpha: float = 0.3,
        min_games: int = 3
    ) -> Dict[str, float]:
        """
        Compute exponential weighted average of recent performance

        EWA weights more recent games more heavily than older ones.

        Args:
            team: Team name
            as_of_date: Compute features as of this date
            metrics: List of metric names to compute EWA for
            alpha: EWA decay factor (higher = more weight to recent)
            min_games: Minimum games required

        Returns:
            Dictionary of EWA features
        """
        features = {}

        if self.games_df is None:
            return features

        # Filter team's games before the given date
        mask = (
            ((self.games_df['home_team'] == team) | (self.games_df['away_team'] == team)) &
            (pd.to_datetime(self.games_df['date']) < as_of_date)
        )
        team_games = self.games_df[mask].sort_values('date', ascending=False)

        if len(team_games) < min_games:
            return features

        for metric in metrics:
            if metric not in team_games.columns:
                continue

            # Get metric values for this team (adjust for home/away)
            values = []
            for _, game in team_games.iterrows():
                if game['home_team'] == team:
                    values.append(game.get(f'home_{metric}', game.get(metric)))
                else:
                    values.append(game.get(f'away_{metric}', game.get(metric)))

            # Filter out None values
            values = [v for v in values if v is not None]

            if values:
                # Compute EWA
                weights = [(1 - alpha) ** i for i in range(len(values))]
                weight_sum = sum(weights)
                ewa = sum(v * w for v, w in zip(values, weights)) / weight_sum
                features[f'{team}_ewa_{metric}'] = ewa

        return features

    def compute_road_trip_features(
        self,
        team: str,
        as_of_date: datetime,
        current_location: str = 'home'
    ) -> Dict[str, float]:
        """
        Compute road trip fatigue features

        Args:
            team: Team name
            as_of_date: Compute features as of this date
            current_location: 'home' or 'away' for current game

        Returns:
            Dictionary of road trip features
        """
        features = {}

        if self.games_df is None:
            return features

        # Get team's recent games
        mask = (
            ((self.games_df['home_team'] == team) | (self.games_df['away_team'] == team)) &
            (pd.to_datetime(self.games_df['date']) < as_of_date)
        )
        team_games = self.games_df[mask].sort_values('date', ascending=False).head(5)

        # Count consecutive away games
        consecutive_away = 0
        for _, game in team_games.iterrows():
            if game['away_team'] == team:
                consecutive_away += 1
            else:
                break

        features['consecutive_away'] = consecutive_away
        features['on_road_trip'] = 1.0 if consecutive_away >= 2 else 0.0

        return features

    def compute_season_context(
        self,
        game_date: datetime,
        season_start: Optional[datetime] = None
    ) -> Dict[str, float]:
        """
        Compute season context features

        Args:
            game_date: Date of the game
            season_start: Start of season (defaults to November 1)

        Returns:
            Dictionary of season context features
        """
        features = {}

        if season_start is None:
            # Assume season starts November 1
            year = game_date.year if game_date.month >= 11 else game_date.year - 1
            season_start = datetime(year, 11, 1)

        # Days into season
        days_into_season = (game_date - season_start).days
        features['days_into_season'] = max(0, days_into_season)

        # Season phase (0 = early, 1 = mid, 2 = late/tournament)
        if days_into_season < 30:
            features['season_phase'] = 0  # Early season
        elif days_into_season < 100:
            features['season_phase'] = 1  # Conference play
        else:
            features['season_phase'] = 2  # Tournament time

        # Is conference play (roughly after Jan 1)
        features['is_conference_play'] = 1.0 if game_date.month >= 1 and days_into_season > 60 else 0.0

        return features


class EnsembleRatingsFeatures:
    """
    Multi-source ensemble rating features

    Combines ratings from multiple sources:
    - Barttorvik T-Rank
    - KenPom
    - ESPN BPI
    - Sagarin
    - Elo
    """

    def __init__(self, ratings_sources: Dict[str, pd.DataFrame]):
        """
        Initialize with multiple rating sources

        Args:
            ratings_sources: Dictionary of source_name -> DataFrame with team ratings
                Each DataFrame should have 'team' column and rating column
        """
        self.sources = ratings_sources
        self.normalized_ratings = {}
        self._normalize_all_sources()

    def _normalize_source(self, df: pd.DataFrame, rating_col: str) -> Dict[str, float]:
        """Normalize ratings to mean=0, std=1"""
        if df.empty or rating_col not in df.columns:
            return {}

        mean = df[rating_col].mean()
        std = df[rating_col].std()

        if std == 0:
            std = 1

        normalized = {}
        for _, row in df.iterrows():
            team = row['team']
            rating = (row[rating_col] - mean) / std
            normalized[team] = rating

        return normalized

    def _normalize_all_sources(self):
        """Normalize all rating sources"""
        # Define rating column for each source
        rating_cols = {
            'barttorvik': 'adj_em',
            'kenpom': 'adj_em',
            'bpi': 'bpi',
            'sagarin': 'rating',
            'elo': 'elo',
        }

        for source_name, df in self.sources.items():
            col = rating_cols.get(source_name, 'rating')
            self.normalized_ratings[source_name] = self._normalize_source(df, col)

    def get_normalized_rating(self, team: str, source: str) -> Optional[float]:
        """Get normalized rating for a team from a specific source"""
        return self.normalized_ratings.get(source, {}).get(team)

    def compute_ensemble_features(
        self,
        home_team: str,
        away_team: str,
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Compute ensemble rating features

        Args:
            home_team: Home team name
            away_team: Away team name
            weights: Optional weights for each source (default: equal)

        Returns:
            Dictionary of ensemble features
        """
        features = {}

        if weights is None:
            # Equal weights
            weights = {source: 1.0 / len(self.sources) for source in self.sources}

        # Compute weighted ensemble rating differential
        home_ensemble = 0.0
        away_ensemble = 0.0
        sources_used = 0

        for source, weight in weights.items():
            home_rating = self.get_normalized_rating(home_team, source)
            away_rating = self.get_normalized_rating(away_team, source)

            if home_rating is not None and away_rating is not None:
                home_ensemble += weight * home_rating
                away_ensemble += weight * away_rating
                sources_used += 1

                # Individual source differential
                features[f'{source}_diff'] = home_rating - away_rating

        if sources_used > 0:
            features['ensemble_diff'] = home_ensemble - away_ensemble
            features['ensemble_sources'] = sources_used

        return features


class EnhancedFeatureEngine(FeatureEngine):
    """
    Enhanced feature engine with all new features

    Combines:
    - Original basic/advanced features
    - Four Factors
    - Temporal features
    - Ensemble ratings
    - Elo integration
    """

    def __init__(
        self,
        team_stats_df: pd.DataFrame,
        games_df: Optional[pd.DataFrame] = None,
        elo_system: Optional[Any] = None,
        ratings_sources: Optional[Dict[str, pd.DataFrame]] = None
    ):
        """
        Initialize enhanced feature engine

        Args:
            team_stats_df: Team statistics DataFrame
            games_df: Historical games DataFrame
            elo_system: EloRatingSystem instance
            ratings_sources: Dictionary of rating sources for ensemble
        """
        super().__init__(team_stats_df, games_df)

        self.four_factors = FourFactorsFeatures(team_stats_df)
        self.temporal = TemporalFeatures(games_df)
        self.elo = elo_system

        if ratings_sources:
            self.ensemble = EnsembleRatingsFeatures(ratings_sources)
        else:
            self.ensemble = None

    def create_all_features(
        self,
        home_team: str,
        away_team: str,
        game_date: Optional[datetime] = None,
        home_last_game: Optional[datetime] = None,
        away_last_game: Optional[datetime] = None,
        neutral: bool = False
    ) -> Dict[str, float]:
        """
        Create comprehensive feature set for a matchup

        Args:
            home_team: Home team name
            away_team: Away team name
            game_date: Date of game
            home_last_game: Home team's previous game date
            away_last_game: Away team's previous game date
            neutral: Is neutral site

        Returns:
            Dictionary of all features
        """
        features = {}

        # Original basic features
        features.update(self.create_basic_features(home_team, away_team))

        # Original advanced features
        features.update(self.create_advanced_features(home_team, away_team))

        # Four Factors
        features.update(self.four_factors.compute_four_factors_diff(home_team, away_team))

        # Temporal features
        if game_date:
            features.update(self.temporal.compute_rest_features(
                home_team, away_team, game_date, home_last_game, away_last_game
            ))
            features.update(self.temporal.compute_season_context(game_date))

        # Elo features
        if self.elo:
            features['home_elo'] = self.elo.get_rating(home_team)
            features['away_elo'] = self.elo.get_rating(away_team)
            features['elo_diff'] = features['home_elo'] - features['away_elo']
            features['elo_spread'] = self.elo.predict_spread(home_team, away_team, neutral)
            features['elo_win_prob'] = self.elo.predict_win_probability(home_team, away_team, neutral)

        # Ensemble features
        if self.ensemble:
            features.update(self.ensemble.compute_ensemble_features(home_team, away_team))

        # Neutral site indicator
        features['neutral_site'] = 1.0 if neutral else 0.0

        return features

    def create_enhanced_prediction_dataset(
        self,
        matchups_df: pd.DataFrame,
        neutral_col: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Create complete enhanced feature dataset

        Args:
            matchups_df: DataFrame with columns ['Date', 'Home', 'Away']
            neutral_col: Optional column indicating neutral site

        Returns:
            DataFrame with all features for each matchup
        """
        feature_list = []

        for idx, row in matchups_df.iterrows():
            game_date = pd.to_datetime(row['Date']) if pd.notna(row.get('Date')) else None
            neutral = bool(row.get(neutral_col, False)) if neutral_col else False

            features = self.create_all_features(
                home_team=row['Home'],
                away_team=row['Away'],
                game_date=game_date,
                neutral=neutral
            )

            # Add identifying information
            features['game_idx'] = idx
            features['home_team'] = row['Home']
            features['away_team'] = row['Away']
            features['date'] = row.get('Date')

            feature_list.append(features)

        return pd.DataFrame(feature_list)
