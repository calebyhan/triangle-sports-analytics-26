"""
Feature Engineering Module for Triangle Sports Analytics
Creates features for point spread prediction models
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta


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
