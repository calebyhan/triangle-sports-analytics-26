"""
Player-based feature engineering from box score data.

Extracts star player metrics, offensive balance, and bench depth
to enhance team-level predictions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime


def calculate_star_player_power(box_scores: pd.DataFrame, team: str, top_n: int = 3) -> Dict[str, float]:
    """
    Calculate star player power metrics for a team.

    Args:
        box_scores: DataFrame with player box scores (columns: player, team, pts, fgm, fga, etc.)
        team: Team name
        top_n: Number of top players to consider

    Returns:
        Dictionary with star player metrics
    """
    team_players = box_scores[box_scores['team'] == team].copy()

    if len(team_players) == 0:
        return {
            'star_ppg_1': 0.0,
            'star_ppg_2': 0.0,
            'star_ppg_3': 0.0,
            'star_total_ppg': 0.0,
            'star_avg_efficiency': 0.0,
        }

    # Aggregate stats by player
    player_stats = team_players.groupby('player').agg({
        'pts': 'mean',
        'fgm': 'sum',
        'fga': 'sum',
        '3pm': 'sum',
        '3pa': 'sum',
        'ftm': 'sum',
        'fta': 'sum',
        'reb': 'mean',
        'ast': 'mean',
        'to': 'mean',  # Note: CBBpy uses 'to' not 'tov'
    }).reset_index()

    # Calculate true shooting percentage: TS% = PTS / (2 * (FGA + 0.44 * FTA))
    player_stats['ts_pct'] = player_stats.apply(
        lambda row: (row['fgm'] + row['ftm']) / (2 * (row['fga'] + 0.44 * row['fta']))
        if (row['fga'] + 0.44 * row['fta']) > 0 else 0,
        axis=1
    )

    # Sort by PPG to find top scorers
    top_players = player_stats.nlargest(top_n, 'pts')

    result = {
        'star_total_ppg': top_players['pts'].sum(),
        'star_avg_efficiency': top_players['ts_pct'].mean(),
    }

    # Add individual star PPG
    for i in range(top_n):
        if i < len(top_players):
            result[f'star_ppg_{i+1}'] = top_players.iloc[i]['pts']
        else:
            result[f'star_ppg_{i+1}'] = 0.0

    return result


def calculate_offensive_balance(box_scores: pd.DataFrame, team: str) -> Dict[str, float]:
    """
    Calculate offensive balance - how distributed is the scoring?

    Args:
        box_scores: DataFrame with player box scores
        team: Team name

    Returns:
        Dictionary with offensive balance metrics
    """
    team_players = box_scores[box_scores['team'] == team].copy()

    if len(team_players) == 0:
        return {
            'scoring_concentration': 0.5,  # Neutral
            'top_scorer_share': 0.3,  # Default
            'balanced_scoring': 0.5,  # Neutral
        }

    # Aggregate by player
    player_ppg = team_players.groupby('player')['pts'].mean().sort_values(ascending=False)
    total_ppg = player_ppg.sum()

    if total_ppg == 0:
        return {
            'scoring_concentration': 0.5,
            'top_scorer_share': 0.3,
            'balanced_scoring': 0.5,
        }

    # Herfindahl index for scoring concentration (lower = more balanced)
    scoring_shares = (player_ppg / total_ppg) ** 2
    herfindahl = scoring_shares.sum()

    # Top scorer's share of total points
    top_scorer_share = player_ppg.iloc[0] / total_ppg if len(player_ppg) > 0 else 0

    # Balanced scoring: inverse of concentration
    balanced_scoring = 1 - herfindahl

    return {
        'scoring_concentration': herfindahl,
        'top_scorer_share': top_scorer_share,
        'balanced_scoring': balanced_scoring,
    }


def calculate_bench_depth(box_scores: pd.DataFrame, team: str, starter_threshold: float = 25.0) -> Dict[str, float]:
    """
    Calculate bench depth score based on non-starter production.

    Args:
        box_scores: DataFrame with player box scores (must have 'min' column for minutes)
        team: Team name
        starter_threshold: Minutes per game threshold to be considered a starter

    Returns:
        Dictionary with bench depth metrics
    """
    team_players = box_scores[box_scores['team'] == team].copy()

    if len(team_players) == 0 or 'min' not in team_players.columns:
        return {
            'bench_ppg': 0.0,
            'bench_depth_score': 0.0,
            'bench_contributors': 0,
        }

    # Aggregate by player
    player_stats = team_players.groupby('player').agg({
        'pts': 'mean',
        'min': 'mean',
        'reb': 'mean',
        'ast': 'mean',
    }).reset_index()

    # Identify bench players (below starter threshold)
    bench_players = player_stats[player_stats['min'] < starter_threshold]

    if len(bench_players) == 0:
        return {
            'bench_ppg': 0.0,
            'bench_depth_score': 0.0,
            'bench_contributors': 0,
        }

    # Bench metrics
    bench_ppg = bench_players['pts'].sum()
    bench_contributors = len(bench_players[bench_players['pts'] >= 5.0])  # 5+ PPG

    # Depth score: weighted combination of production and number of contributors
    bench_depth_score = (bench_ppg * 0.7) + (bench_contributors * 2.0)

    return {
        'bench_ppg': bench_ppg,
        'bench_depth_score': bench_depth_score,
        'bench_contributors': bench_contributors,
    }


def calculate_key_player_efficiency(box_scores: pd.DataFrame, team: str, top_n: int = 3) -> Dict[str, float]:
    """
    Calculate efficiency metrics for key players.

    Args:
        box_scores: DataFrame with player box scores
        team: Team name
        top_n: Number of key players to analyze

    Returns:
        Dictionary with key player efficiency metrics
    """
    team_players = box_scores[box_scores['team'] == team].copy()

    if len(team_players) == 0:
        return {
            'key_player_ast_tov': 1.0,  # Neutral
            'key_player_reb_pg': 5.0,  # Average
            'key_player_usage': 0.0,
        }

    # Aggregate by player
    player_stats = team_players.groupby('player').agg({
        'pts': 'mean',
        'ast': ['mean', 'sum'],
        'to': ['mean', 'sum'],  # Note: CBBpy uses 'to' not 'tov'
        'reb': 'mean',
        'fga': 'sum',
    }).reset_index()

    # Flatten column names
    player_stats.columns = ['player', 'pts', 'ast_mean', 'ast_sum', 'to_mean', 'to_sum', 'reb', 'fga']

    # Get top scorers
    top_players = player_stats.nlargest(top_n, 'pts')

    if len(top_players) == 0:
        return {
            'key_player_ast_tov': 1.0,
            'key_player_reb_pg': 5.0,
            'key_player_usage': 0.0,
        }

    # Calculate metrics
    total_ast = top_players['ast_sum'].sum()
    total_to = top_players['to_sum'].sum()
    ast_tov_ratio = total_ast / total_to if total_to > 0 else 2.0

    avg_reb = top_players['reb'].mean()

    # Usage approximation: FGA share among key players
    total_fga = player_stats['fga'].sum()
    key_player_fga = top_players['fga'].sum()
    usage = key_player_fga / total_fga if total_fga > 0 else 0.5

    return {
        'key_player_ast_tov': ast_tov_ratio,
        'key_player_reb_pg': avg_reb,
        'key_player_usage': usage,
    }


def aggregate_all_player_features(
    box_scores: pd.DataFrame,
    teams: List[str],
    output_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Aggregate all player features for a list of teams.

    Args:
        box_scores: DataFrame with all player box scores
        teams: List of team names
        output_path: Where to save the features CSV

    Returns:
        DataFrame with player-based features
    """
    features_list = []

    for team in teams:
        print(f"Processing player features for {team}...")

        # Calculate all feature sets
        star_power = calculate_star_player_power(box_scores, team)
        balance = calculate_offensive_balance(box_scores, team)
        bench = calculate_bench_depth(box_scores, team)
        efficiency = calculate_key_player_efficiency(box_scores, team)

        # Combine into single row
        team_features = {
            'team': team,
            **{f'player_{k}': v for k, v in star_power.items()},
            **{f'player_{k}': v for k, v in balance.items()},
            **{f'player_{k}': v for k, v in bench.items()},
            **{f'player_{k}': v for k, v in efficiency.items()},
        }

        features_list.append(team_features)

    # Create DataFrame
    features_df = pd.DataFrame(features_list)

    # Save if output path provided
    if output_path:
        features_df.to_csv(output_path, index=False)
        print(f"\n✓ Saved player features to {output_path}")

    return features_df


if __name__ == '__main__':
    print("Player Features Engineering Module")
    print("Use aggregate_all_player_features() with your box score data")
