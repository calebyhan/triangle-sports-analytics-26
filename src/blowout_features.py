"""
Blowout-Specific Feature Engineering

This module addresses the model's weakness in predicting blowouts (>15 point spreads).
Current error analysis shows:
- Close games (<5): 2.25 MAE ✓ Excellent
- Moderate (5-10): 3.11 MAE ✓ Good
- Large (10-15): 3.25 MAE ✓ Good
- Blowouts (>15): 4.31 MAE ⚠️ Needs improvement

Hypothesis: Blowouts occur when there's a combination of:
1. Large talent gap (already captured by efficiency differential)
2. Momentum/dominance (NOT currently captured)
3. Recent performance trajectory
4. Consistency/volatility in results

This module adds features to capture momentum and dominant performance patterns.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta


class BlowoutFeatureEngine:
    """
    Feature engineering specifically for predicting blowout games.

    Focuses on:
    - Recent run differential (how much teams win/lose by)
    - Performance momentum (improving vs declining)
    - Consistency (teams that consistently dominate vs teams with variable results)
    - Peak performance potential
    """

    def __init__(self, games_df: pd.DataFrame):
        """
        Initialize blowout feature engine.

        Args:
            games_df: DataFrame with historical game results
                Required columns: date, home_team, away_team, home_score, away_score
        """
        self.games_df = games_df.copy()
        self.games_df['date'] = pd.to_datetime(self.games_df['date'])
        self.games_df = self.games_df.sort_values('date')

        # Cache for computed statistics
        self._momentum_cache = {}
        self._run_diff_cache = {}

    def compute_run_differential_stats(
        self,
        team: str,
        as_of_date: datetime,
        lookback_games: int = 10
    ) -> Dict[str, float]:
        """
        Compute run differential statistics for a team.

        Run differential = margin of victory/defeat in recent games
        Strong predictor of dominant teams that blow out opponents.

        Args:
            team: Team name
            as_of_date: Compute stats using only games before this date
            lookback_games: Number of recent games to analyze

        Returns:
            Dictionary with:
            - avg_run_diff: Average margin in recent games
            - run_diff_std: Consistency of dominance (low std = consistent)
            - max_run_diff: Largest victory margin
            - min_run_diff: Largest defeat margin
            - blowout_rate: % of games won by 15+ points
            - blown_out_rate: % of games lost by 15+ points
        """
        cache_key = (team, as_of_date, lookback_games)
        if cache_key in self._run_diff_cache:
            return self._run_diff_cache[cache_key]

        # Get team's recent games
        team_games = self.games_df[
            (
                (self.games_df['home_team'] == team) |
                (self.games_df['away_team'] == team)
            ) &
            (self.games_df['date'] < as_of_date)
        ].tail(lookback_games)

        if len(team_games) == 0:
            # No historical data - return neutral values
            stats = {
                'avg_run_diff': 0.0,
                'run_diff_std': 10.0,  # Assume high variance
                'max_run_diff': 0.0,
                'min_run_diff': 0.0,
                'blowout_rate': 0.0,
                'blown_out_rate': 0.0,
                'games_analyzed': 0
            }
            self._run_diff_cache[cache_key] = stats
            return stats

        # Calculate run differentials (positive = win, negative = loss)
        run_diffs = []
        for _, game in team_games.iterrows():
            if game['home_team'] == team:
                diff = game['home_score'] - game['away_score']
            else:
                diff = game['away_score'] - game['home_score']
            run_diffs.append(diff)

        run_diffs = np.array(run_diffs)

        stats = {
            'avg_run_diff': float(np.mean(run_diffs)),
            'run_diff_std': float(np.std(run_diffs)),
            'max_run_diff': float(np.max(run_diffs)),
            'min_run_diff': float(np.min(run_diffs)),
            'blowout_rate': float(np.mean(run_diffs >= 15)),
            'blown_out_rate': float(np.mean(run_diffs <= -15)),
            'games_analyzed': len(run_diffs)
        }

        self._run_diff_cache[cache_key] = stats
        return stats

    def compute_momentum_features(
        self,
        team: str,
        as_of_date: datetime,
        short_window: int = 5,
        long_window: int = 15
    ) -> Dict[str, float]:
        """
        Compute momentum features comparing recent vs longer-term performance.

        Momentum captures whether a team is improving or declining.
        Teams on hot streaks tend to dominate opponents.

        Args:
            team: Team name
            as_of_date: Compute stats using only games before this date
            short_window: Recent games window
            long_window: Longer-term comparison window

        Returns:
            Dictionary with:
            - momentum_score: Difference between recent and long-term performance
            - recent_win_pct: Win % in last N games
            - trend_slope: Linear trend in run differential (improving/declining)
            - hot_streak: Current winning streak (0 if losing streak)
        """
        cache_key = (team, as_of_date, short_window, long_window)
        if cache_key in self._momentum_cache:
            return self._momentum_cache[cache_key]

        # Get team's games
        team_games = self.games_df[
            (
                (self.games_df['home_team'] == team) |
                (self.games_df['away_team'] == team)
            ) &
            (self.games_df['date'] < as_of_date)
        ]

        if len(team_games) < short_window:
            # Not enough data
            features = {
                'momentum_score': 0.0,
                'recent_win_pct': 0.5,
                'trend_slope': 0.0,
                'hot_streak': 0,
                'games_analyzed': len(team_games)
            }
            self._momentum_cache[cache_key] = features
            return features

        # Get run differentials for all games
        run_diffs = []
        wins = []
        for _, game in team_games.iterrows():
            if game['home_team'] == team:
                diff = game['home_score'] - game['away_score']
            else:
                diff = game['away_score'] - game['home_score']
            run_diffs.append(diff)
            wins.append(1 if diff > 0 else 0)

        run_diffs = np.array(run_diffs)
        wins = np.array(wins)

        # Recent vs long-term performance
        recent_avg = np.mean(run_diffs[-short_window:])
        if len(run_diffs) >= long_window:
            long_term_avg = np.mean(run_diffs[-long_window:-short_window])
        else:
            long_term_avg = np.mean(run_diffs[:-short_window]) if len(run_diffs) > short_window else 0.0

        momentum_score = recent_avg - long_term_avg

        # Recent win percentage
        recent_win_pct = np.mean(wins[-short_window:])

        # Trend slope (linear regression on run differential)
        if len(run_diffs) >= 5:
            x = np.arange(len(run_diffs[-15:]))  # Last 15 games
            y = run_diffs[-15:]
            trend_slope = np.polyfit(x, y, 1)[0]  # Slope of linear fit
        else:
            trend_slope = 0.0

        # Current streak (wins only, 0 if on losing streak)
        hot_streak = 0
        for win in reversed(wins[-10:]):  # Check last 10 games
            if win == 1:
                hot_streak += 1
            else:
                break

        features = {
            'momentum_score': float(momentum_score),
            'recent_win_pct': float(recent_win_pct),
            'trend_slope': float(trend_slope),
            'hot_streak': int(hot_streak),
            'games_analyzed': len(run_diffs)
        }

        self._momentum_cache[cache_key] = features
        return features

    def create_matchup_features(
        self,
        home_team: str,
        away_team: str,
        game_date: datetime,
        lookback_games: int = 10
    ) -> Dict[str, float]:
        """
        Create blowout-specific features for a matchup.

        Combines run differential and momentum features for both teams
        and computes differentials.

        Args:
            home_team: Home team name
            away_team: Away team name
            game_date: Date of the game
            lookback_games: Number of recent games to analyze

        Returns:
            Dictionary with differential features:
            - run_diff_differential: Difference in avg run differential
            - momentum_differential: Difference in momentum scores
            - blowout_tendency_diff: Difference in blowout rates
            - consistency_ratio: Ratio of standard deviations (who's more consistent)
            - hot_streak_advantage: Difference in winning streaks
        """
        # Get stats for both teams
        home_run_diff = self.compute_run_differential_stats(home_team, game_date, lookback_games)
        away_run_diff = self.compute_run_differential_stats(away_team, game_date, lookback_games)

        home_momentum = self.compute_momentum_features(home_team, game_date)
        away_momentum = self.compute_momentum_features(away_team, game_date)

        # Compute differentials
        features = {
            # Run differential features
            'run_diff_differential': home_run_diff['avg_run_diff'] - away_run_diff['avg_run_diff'],
            'max_run_diff_differential': home_run_diff['max_run_diff'] - away_run_diff['max_run_diff'],
            'blowout_tendency_diff': home_run_diff['blowout_rate'] - away_run_diff['blown_out_rate'],

            # Consistency features (lower std = more consistent dominance)
            'consistency_ratio': (
                away_run_diff['run_diff_std'] / home_run_diff['run_diff_std']
                if home_run_diff['run_diff_std'] > 0 else 1.0
            ),

            # Momentum features
            'momentum_differential': home_momentum['momentum_score'] - away_momentum['momentum_score'],
            'win_pct_differential': home_momentum['recent_win_pct'] - away_momentum['recent_win_pct'],
            'trend_slope_differential': home_momentum['trend_slope'] - away_momentum['trend_slope'],
            'hot_streak_advantage': home_momentum['hot_streak'] - away_momentum['hot_streak'],

            # Individual team features (for model to learn team-specific patterns)
            'home_avg_run_diff': home_run_diff['avg_run_diff'],
            'away_avg_run_diff': away_run_diff['avg_run_diff'],
            'home_blowout_rate': home_run_diff['blowout_rate'],
            'away_blown_out_rate': away_run_diff['blown_out_rate'],
            'home_momentum': home_momentum['momentum_score'],
            'away_momentum': away_momentum['momentum_score'],
        }

        return features

    def create_features_for_dataset(
        self,
        matchups_df: pd.DataFrame,
        lookback_games: int = 10
    ) -> pd.DataFrame:
        """
        Create blowout features for an entire dataset of matchups.

        Args:
            matchups_df: DataFrame with columns: date, home_team, away_team
            lookback_games: Number of recent games to analyze

        Returns:
            DataFrame with original columns plus new blowout features
        """
        feature_list = []

        for _, row in matchups_df.iterrows():
            game_date = pd.to_datetime(row['date'])
            features = self.create_matchup_features(
                row['home_team'],
                row['away_team'],
                game_date,
                lookback_games
            )
            feature_list.append(features)

        features_df = pd.DataFrame(feature_list)

        # Combine with original data
        result_df = pd.concat([
            matchups_df.reset_index(drop=True),
            features_df.reset_index(drop=True)
        ], axis=1)

        return result_df


def add_blowout_features_to_training_data(
    games_with_elo_df: pd.DataFrame,
    lookback_games: int = 10
) -> pd.DataFrame:
    """
    Convenience function to add blowout features to training data.

    Args:
        games_with_elo_df: DataFrame with historical games and Elo ratings
            Required columns: date, home_team, away_team, home_score, away_score
        lookback_games: Number of recent games to analyze

    Returns:
        DataFrame with blowout features added

    Example:
        >>> games = pd.read_csv('games.csv')
        >>> games_enhanced = add_blowout_features_to_training_data(games)
        >>> # Now use games_enhanced for training
    """
    engine = BlowoutFeatureEngine(games_with_elo_df)
    return engine.create_features_for_dataset(games_with_elo_df, lookback_games)
