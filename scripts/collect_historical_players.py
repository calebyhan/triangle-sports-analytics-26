"""
Collect historical player box scores for D1 teams (2020-2024).

This script fetches 5 years of player-level data to enable enhanced
feature engineering in the training pipeline. Uses checkpointing for
recovery and handles rate limiting.

Expected runtime: ~2 hours for 350 D1 teams × 5 seasons
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import pickle
import time
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from src.data_sources import cbbpy_enhanced
from src.utils import fetch_barttorvik_year
from src import config
from src.logger import setup_logger

logger = setup_logger(__name__)

# Paths
CHECKPOINT_DIR = config.DATA_DIR / 'cache' / 'player_data'
FAILURES_LOG = config.DATA_DIR / 'logs' / 'player_data_failures.csv'
OUTPUT_FILE = config.PROCESSED_DATA_DIR / 'historical_player_box_scores_2020_2024.csv'


def get_d1_teams():
    """
    Get list of D1 teams from Barttorvik data.

    Returns:
        set: Unique D1 team names across training years
    """
    d1_teams = set()

    print("Identifying D1 teams from Barttorvik data...")
    for year in [2020, 2021, 2022, 2023, 2024]:
        try:
            df = fetch_barttorvik_year(year)
            teams_this_year = set(df['team'].unique())
            d1_teams.update(teams_this_year)
            print(f"  {year}: {len(teams_this_year)} teams")
        except Exception as e:
            logger.warning(f"Could not fetch {year} Barttorvik data: {e}")
            continue

    print(f"\n✓ Found {len(d1_teams)} unique D1 teams across all years")
    return sorted(d1_teams)


def save_checkpoint(data, checkpoint_dir, season, team_idx):
    """
    Save checkpoint for recovery.

    Args:
        data: List of DataFrames collected so far
        checkpoint_dir: Directory for checkpoints
        season: Current season
        team_idx: Current team index
    """
    checkpoint_path = checkpoint_dir / f"checkpoint_s{season}_t{team_idx}.pkl"

    with open(checkpoint_path, 'wb') as f:
        pickle.dump(data, f)

    logger.info(f"Checkpoint saved: {checkpoint_path.name}")


def load_completed_checkpoints(checkpoint_dir):
    """
    Load completed teams from checkpoints.

    Returns:
        set: Set of (team, season) tuples already completed
    """
    completed = set()

    if not checkpoint_dir.exists():
        return completed

    # Find all checkpoint files
    checkpoint_files = sorted(checkpoint_dir.glob("checkpoint_*.pkl"))

    if not checkpoint_files:
        return completed

    # Load latest checkpoint to see what's been done
    latest = checkpoint_files[-1]
    logger.info(f"Found checkpoint: {latest.name}")

    try:
        with open(latest, 'rb') as f:
            data = pickle.load(f)

        # Extract completed (team, season) pairs
        for df in data:
            if 'team' in df.columns and 'season' in df.columns:
                for team, season in zip(df['team'].unique(), df['season'].unique()):
                    completed.add((team, int(season)))

        logger.info(f"Resuming from checkpoint: {len(completed)} team-seasons completed")
    except Exception as e:
        logger.warning(f"Could not load checkpoint: {e}")

    return completed


def log_failure(team, season, error):
    """
    Log data collection failure.

    Args:
        team: Team name
        season: Season year
        error: Error message
    """
    FAILURES_LOG.parent.mkdir(parents=True, exist_ok=True)

    failure_df = pd.DataFrame([{
        'timestamp': datetime.now().isoformat(),
        'team': team,
        'season': season,
        'error': str(error)
    }])

    if FAILURES_LOG.exists():
        failure_df.to_csv(FAILURES_LOG, mode='a', header=False, index=False)
    else:
        failure_df.to_csv(FAILURES_LOG, index=False)


def fetch_team_season_data(team, season):
    """
    Fetch player box scores for a team in a season.

    Args:
        team: Team name
        season: Season year

    Returns:
        DataFrame with player box scores, or None if failed
    """
    try:
        result = cbbpy_enhanced.fetch_games_team(
            team, season=season, include_all=True
        )

        if result is None:
            return None

        schedule, box_scores, pbp = result

        if box_scores is None or len(box_scores) == 0:
            return None

        # Add metadata
        box_scores['team'] = team
        box_scores['season'] = season

        # Merge game dates from schedule
        if 'game_id' in box_scores.columns and 'date' in schedule.columns:
            box_scores = box_scores.merge(
                schedule[['game_id', 'date']],
                on='game_id',
                how='left'
            )
            box_scores.rename(columns={'date': 'game_date'}, inplace=True)

        # Filter out TEAM rows (aggregate statistics)
        if 'player' in box_scores.columns:
            box_scores = box_scores[box_scores['player'] != 'TEAM'].copy()

        return box_scores

    except Exception as e:
        raise e


def main():
    """Collect historical player data for all D1 teams."""
    print("="*60)
    print("HISTORICAL PLAYER DATA COLLECTION")
    print("="*60)
    print("Collecting player box scores for 2020-2024 seasons")
    print("This will take approximately 2 hours")
    print("="*60)

    # 1. Get D1 teams
    d1_teams = get_d1_teams()

    # 2. Setup checkpointing
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    completed = load_completed_checkpoints(CHECKPOINT_DIR)

    # 3. Collect data by season
    all_data = []

    for season in [2020, 2021, 2022, 2023, 2024]:
        print(f"\n{'='*60}")
        print(f"SEASON {season}")
        print(f"{'='*60}")

        season_start_time = time.time()
        season_success = 0
        season_failures = 0

        for team_idx, team in enumerate(tqdm(d1_teams, desc=f"Season {season}"), 1):
            # Skip if already completed
            if (team, season) in completed:
                season_success += 1
                continue

            try:
                box_scores = fetch_team_season_data(team, season)

                if box_scores is not None and len(box_scores) > 0:
                    all_data.append(box_scores)
                    season_success += 1
                else:
                    season_failures += 1
                    log_failure(team, season, "No data returned")

                # Checkpoint every 10 teams
                if team_idx % 10 == 0:
                    save_checkpoint(all_data, CHECKPOINT_DIR, season, team_idx)

            except Exception as e:
                season_failures += 1
                logger.error(f"Failed: {team} {season} - {e}")
                log_failure(team, season, str(e))

            # Rate limiting (3 seconds between requests)
            time.sleep(3)

        # Season summary
        season_elapsed = time.time() - season_start_time
        print(f"\nSeason {season} complete:")
        print(f"  Success: {season_success}/{len(d1_teams)} teams")
        print(f"  Failures: {season_failures}")
        print(f"  Time: {season_elapsed/60:.1f} minutes")

        # Save checkpoint at end of season
        save_checkpoint(all_data, CHECKPOINT_DIR, season, len(d1_teams))

    # 4. Combine and save final dataset
    if len(all_data) == 0:
        print("\n❌ No data collected!")
        return

    print(f"\n{'='*60}")
    print("COMBINING DATA")
    print(f"{'='*60}")

    combined = pd.concat(all_data, ignore_index=True)

    # Ensure game_date is datetime
    if 'game_date' in combined.columns:
        combined['game_date'] = pd.to_datetime(combined['game_date'])

    # Save final dataset
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT_FILE, index=False)

    # Final summary
    print(f"\n{'='*60}")
    print("✓ COLLECTION COMPLETE")
    print(f"{'='*60}")
    print(f"Total records: {len(combined):,}")
    print(f"Unique players: {combined['player'].nunique():,}")
    print(f"Unique teams: {combined['team'].nunique()}")
    print(f"Unique games: {combined['game_id'].nunique():,}")
    print(f"Seasons: {sorted(combined['season'].unique())}")
    print(f"\nSaved to: {OUTPUT_FILE}")
    print(f"File size: {OUTPUT_FILE.stat().st_size / (1024**2):.1f} MB")

    if FAILURES_LOG.exists():
        failures = pd.read_csv(FAILURES_LOG)
        print(f"\n⚠ {len(failures)} failures logged to: {FAILURES_LOG}")

    print("="*60)


if __name__ == '__main__':
    main()
