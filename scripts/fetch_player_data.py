"""
Fetch player box scores for ACC teams using CBBpy.

Collects 2025-26 season box scores for all 21 ACC teams
to enable player-based feature engineering.
"""

import sys
sys.path.append('..')

import pandas as pd
from pathlib import Path
from datetime import datetime
import time
from src.data_sources import cbbpy_enhanced
from src.config import PROCESSED_DATA_DIR, ACC_TEAMS
from src.logger import setup_logger

logger = setup_logger(__name__)

# Output path
OUTPUT_FILE = PROCESSED_DATA_DIR / 'player_box_scores_2026.csv'


def fetch_team_box_scores(team: str, season: int = 2026) -> pd.DataFrame:
    """
    Fetch all box scores for a team in a season.

    Args:
        team: Team name
        season: Season year

    Returns:
        DataFrame with player box scores
    """
    try:
        logger.info(f"Fetching box scores for {team}...")

        # Get games for team
        result = cbbpy_enhanced.fetch_games_team(team, season=season, include_all=True)

        if result is None:
            logger.warning(f"No data returned for {team}")
            return pd.DataFrame()

        schedule, box_scores, pbp = result

        if box_scores is None or len(box_scores) == 0:
            logger.warning(f"No box scores for {team}")
            return pd.DataFrame()

        # Box scores is a single DataFrame with all games
        # Just add game_date by merging with schedule
        if 'game_id' in box_scores.columns and 'date' in schedule.columns:
            box_scores = box_scores.merge(
                schedule[['game_id', 'date']],
                on='game_id',
                how='left'
            )
            box_scores = box_scores.rename(columns={'date': 'game_date'})

        logger.info(f"✓ {team}: {len(box_scores)} player-game records")

        return box_scores

    except Exception as e:
        logger.error(f"Error fetching {team}: {e}")
        return pd.DataFrame()


def main():
    """Fetch box scores for all ACC teams."""
    print("="*60)
    print("FETCHING PLAYER BOX SCORES FOR ACC TEAMS")
    print("="*60)
    print(f"Season: 2025-26")
    print(f"Teams: {len(ACC_TEAMS)}")
    print(f"Output: {OUTPUT_FILE}")
    print("="*60)

    all_box_scores = []

    for i, team in enumerate(ACC_TEAMS, 1):
        print(f"\n[{i}/{len(ACC_TEAMS)}] Processing {team}...")

        box_scores = fetch_team_box_scores(team, season=2026)

        if len(box_scores) > 0:
            all_box_scores.append(box_scores)

        # Rate limiting
        if i < len(ACC_TEAMS):
            time.sleep(3)  # 3 seconds between requests

    if len(all_box_scores) == 0:
        print("\n❌ No box scores collected!")
        return

    # Combine all data
    combined_df = pd.concat(all_box_scores, ignore_index=True)

    # Save
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(OUTPUT_FILE, index=False)

    print(f"\n{'='*60}")
    print(f"✓ COLLECTION COMPLETE")
    print(f"{'='*60}")
    print(f"Total records: {len(combined_df)}")
    print(f"Unique players: {combined_df['player'].nunique()}")
    print(f"Unique games: {combined_df['game_id'].nunique()}")
    print(f"Teams covered: {combined_df['team'].nunique()}")
    print(f"\nSaved to: {OUTPUT_FILE}")
    print("="*60)


if __name__ == '__main__':
    main()
