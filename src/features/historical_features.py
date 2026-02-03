"""
Feature engineering from historical game data.

Extracts momentum, blowout tendency, rest days, and other features
from the historical games dataset.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional
from pathlib import Path


def calculate_team_momentum(games_df: pd.DataFrame, team: str, as_of_date: Optional[str] = None, window: int = 5) -> Dict[str, float]:
    """
    Calculate momentum features for a team based on recent games.

    Args:
        games_df: Historical games DataFrame (expects: date, home_team, away_team, home_score, away_score)
        team: Team name
        as_of_date: Calculate as of this date (None = use all games)
        window: Number of recent games to consider

    Returns:
        Dictionary with momentum features
    """
    # Filter to team's games
    team_games = games_df[
        (games_df['home_team'] == team) | (games_df['away_team'] == team)
    ].copy()

    if as_of_date:
        team_games = team_games[team_games['date'] <= as_of_date]

    # Sort by date
    team_games = team_games.sort_values('date')

    # Get last N games
    recent_games = team_games.tail(window)

    if len(recent_games) == 0:
        return {
            'win_streak': 0,
            'loss_streak': 0,
            'recent_win_pct': 0.5,
            'recent_avg_margin': 0.0,
            'games_played': 0,
        }

    # Calculate if team won each game
    recent_games['team_won'] = (
        ((recent_games['home_team'] == team) & (recent_games['home_score'] > recent_games['away_score'])) |
        ((recent_games['away_team'] == team) & (recent_games['away_score'] > recent_games['home_score']))
    )

    # Calculate margin from team's perspective
    recent_games['team_margin'] = recent_games.apply(
        lambda row: (row['home_score'] - row['away_score']) if row['home_team'] == team
                   else (row['away_score'] - row['home_score']),
        axis=1
    )

    # Calculate streaks
    win_streak = 0
    loss_streak = 0
    for won in reversed(recent_games['team_won'].tolist()):
        if won:
            if loss_streak > 0:
                break
            win_streak += 1
        else:
            if win_streak > 0:
                break
            loss_streak += 1

    return {
        'win_streak': win_streak,
        'loss_streak': loss_streak,
        'recent_win_pct': recent_games['team_won'].mean(),
        'recent_avg_margin': recent_games['team_margin'].mean(),
        'games_played': len(team_games),
    }


def calculate_blowout_tendency(games_df: pd.DataFrame, team: str, as_of_date: Optional[str] = None) -> Dict[str, float]:
    """
    Calculate blowout tendency - does this team win/lose by large margins?

    Args:
        games_df: Historical games DataFrame (expects: date, home_team, away_team, home_score, away_score)
        team: Team name
        as_of_date: Calculate as of this date (None = use all games)

    Returns:
        Dictionary with blowout tendency features
    """
    # Filter to team's games
    team_games = games_df[
        (games_df['home_team'] == team) | (games_df['away_team'] == team)
    ].copy()

    if as_of_date:
        team_games = team_games[team_games['date'] <= as_of_date]

    if len(team_games) == 0:
        return {
            'avg_margin_victory': 0.0,
            'avg_margin_defeat': 0.0,
            'blowout_win_rate': 0.0,  # % of wins by 15+
            'blowout_loss_rate': 0.0,  # % of losses by 15+
            'close_game_win_rate': 0.0,  # % of close games (within 5) won
        }

    # Calculate margin from team's perspective
    team_games['team_margin'] = team_games.apply(
        lambda row: (row['home_score'] - row['away_score']) if row['home_team'] == team
                   else (row['away_score'] - row['home_score']),
        axis=1
    )

    # Split into wins and losses
    wins = team_games[team_games['team_margin'] > 0]
    losses = team_games[team_games['team_margin'] < 0]
    close_games = team_games[team_games['team_margin'].abs() <= 5]

    return {
        'avg_margin_victory': wins['team_margin'].mean() if len(wins) > 0 else 0.0,
        'avg_margin_defeat': losses['team_margin'].mean() if len(losses) > 0 else 0.0,
        'blowout_win_rate': (wins['team_margin'] >= 15).sum() / len(team_games) if len(team_games) > 0 else 0.0,
        'blowout_loss_rate': (losses['team_margin'] <= -15).sum() / len(team_games) if len(team_games) > 0 else 0.0,
        'close_game_win_rate': (close_games['team_margin'] > 0).sum() / len(close_games) if len(close_games) > 0 else 0.5,
    }


def calculate_rest_advantage(games_df: pd.DataFrame, team1: str, team2: str, game_date: str) -> Dict[str, float]:
    """
    Calculate rest advantage between two teams.

    Args:
        games_df: Historical games DataFrame (expects: date, home_team, away_team, home_score, away_score)
        team1: First team
        team2: Second team
        game_date: Date of the matchup

    Returns:
        Dictionary with rest features
    """
    game_date_dt = pd.to_datetime(game_date)

    def get_days_rest(team):
        team_games = games_df[
            ((games_df['home_team'] == team) | (games_df['away_team'] == team)) &
            (games_df['date'] < game_date)
        ]
        if len(team_games) == 0:
            return 7  # Default to 1 week

        last_game = pd.to_datetime(team_games['date'].max())
        return (game_date_dt - last_game).days

    team1_rest = get_days_rest(team1)
    team2_rest = get_days_rest(team2)

    return {
        'team1_days_rest': team1_rest,
        'team2_days_rest': team2_rest,
        'rest_advantage': team1_rest - team2_rest,
    }


def calculate_home_court_strength(games_df: pd.DataFrame, team: str) -> float:
    """
    Calculate team-specific home court advantage.

    Args:
        games_df: Historical games DataFrame (expects: date, home_team, away_team, home_score, away_score, neutral_site)
        team: Team name

    Returns:
        Home court advantage in points
    """
    # Get team's home games (not neutral site)
    home_games = games_df[
        (games_df['home_team'] == team) & (games_df['neutral_site'] == False)
    ].copy()

    # Get team's away games (not neutral site)
    away_games = games_df[
        (games_df['away_team'] == team) & (games_df['neutral_site'] == False)
    ].copy()

    if len(home_games) == 0 or len(away_games) == 0:
        return 3.5  # Default HCA

    # Calculate average margin at home vs away
    home_games['margin'] = home_games['home_score'] - home_games['away_score']
    away_games['margin'] = away_games['away_score'] - away_games['home_score']

    home_margin = home_games['margin'].mean()
    away_margin = away_games['margin'].mean()

    # HCA is the difference
    hca = home_margin - away_margin

    # Cap at reasonable bounds
    return max(0, min(hca, 10))


def engineer_all_features(
    games_df: pd.DataFrame,
    teams: list,
    output_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Engineer all historical features for a list of teams.

    Args:
        games_df: Historical games DataFrame
        teams: List of team names
        output_path: Where to save the features CSV

    Returns:
        DataFrame with engineered features
    """
    features_list = []

    for team in teams:
        print(f"Engineering features for {team}...")

        # Calculate all features
        momentum = calculate_team_momentum(games_df, team)
        blowout = calculate_blowout_tendency(games_df, team)
        hca = calculate_home_court_strength(games_df, team)

        # Combine into single row
        team_features = {
            'team': team,
            **{f'momentum_{k}': v for k, v in momentum.items()},
            **{f'blowout_{k}': v for k, v in blowout.items()},
            'home_court_advantage': hca,
        }

        features_list.append(team_features)

    # Create DataFrame
    features_df = pd.DataFrame(features_list)

    # Save if output path provided
    if output_path:
        features_df.to_csv(output_path, index=False)
        print(f"\n✓ Saved features to {output_path}")

    return features_df


if __name__ == '__main__':
    # Test with sample data
    print("Historical Features Engineering Module")
    print("Run engineer_all_features() with your games data")
