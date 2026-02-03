"""
Fetch game dates by fetching team schedules (faster than individual games).

Instead of fetching 28K individual games, fetch ~400 teams × 5 seasons = ~2K schedules.
Each schedule contains multiple games with dates.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import pickle
from tqdm import tqdm
import time
from src import config
from src.data_sources import cbbpy_enhanced

def main():
    print("="*60)
    print("FETCHING GAME DATES FROM TEAM SCHEDULES")
    print("="*60)

    # Load player data
    player_data_path = config.HISTORICAL_PLAYER_DATA
    print(f"\nLoading player data from {player_data_path}...")
    player_data = pd.read_csv(player_data_path)
    print(f"✓ Loaded {len(player_data):,} player-game records")
    print(f"  Unique game IDs: {player_data['game_id'].nunique():,}")

    # Check if game_date already exists with data
    if 'game_date' in player_data.columns:
        non_null_dates = player_data['game_date'].notna().sum()
        if non_null_dates > 0:
            match_rate = (non_null_dates / len(player_data)) * 100
            print(f"\n✓ game_date column already exists with {non_null_dates:,} dates ({match_rate:.1f}%)")
            if match_rate > 95:
                print("  Match rate is good, exiting.")
                return
            else:
                print("  Match rate is low, will try to fill in missing dates...")
        else:
            print("\n⚠ game_date column exists but has no data, will populate it...")
            player_data.drop(columns=['game_date'], inplace=True)

    # Get unique team-season combinations
    team_seasons = player_data[['team', 'season']].drop_duplicates()
    print(f"\nFound {len(team_seasons)} unique team-season combinations")

    # Setup cache
    cache_path = config.DATA_DIR / 'cache' / 'game_dates_map.pkl'
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing cache if exists
    game_date_map = {}
    if cache_path.exists():
        print(f"Loading existing cache from {cache_path}...")
        with open(cache_path, 'rb') as f:
            game_date_map = pickle.load(f)
        print(f"✓ Loaded {len(game_date_map):,} cached game dates")

    # Track which team-seasons we've already processed
    processed_cache_path = config.DATA_DIR / 'cache' / 'processed_team_seasons.pkl'
    if processed_cache_path.exists():
        with open(processed_cache_path, 'rb') as f:
            processed_team_seasons = pickle.load(f)
    else:
        processed_team_seasons = set()

    print(f"Already processed: {len(processed_team_seasons)} team-seasons")

    # Fetch schedules for remaining team-seasons
    failed_teams = []

    for idx, row in tqdm(team_seasons.iterrows(), total=len(team_seasons), desc="Fetching schedules"):
        team = row['team']
        season = int(row['season'])

        # Skip if already processed
        if (team, season) in processed_team_seasons:
            continue

        try:
            # Fetch schedule (include_all=False returns just schedule DataFrame, not tuple)
            schedule = cbbpy_enhanced.fetch_games_team(team, season=season, include_all=False)

            if schedule is None or (isinstance(schedule, pd.DataFrame) and len(schedule) == 0):
                failed_teams.append((team, season, "Empty schedule"))
                continue

            # Extract game_id -> date mapping (schedule uses 'game_day' not 'date')
            date_col = 'game_day' if 'game_day' in schedule.columns else 'date'

            if 'game_id' in schedule.columns and date_col in schedule.columns:
                for _, game in schedule.iterrows():
                    game_id = game['game_id']
                    game_date = game[date_col]
                    if pd.notna(game_id) and pd.notna(game_date):
                        game_date_map[str(game_id)] = pd.to_datetime(game_date)
            else:
                failed_teams.append((team, season, "Missing game_id or date columns"))
                continue

            # Mark as processed
            processed_team_seasons.add((team, season))

            # Rate limiting
            if (len(processed_team_seasons) % 10) == 0:
                time.sleep(0.3)

            # Save checkpoint every 50 team-seasons
            if (len(processed_team_seasons) % 50) == 0:
                with open(cache_path, 'wb') as f:
                    pickle.dump(game_date_map, f)
                with open(processed_cache_path, 'wb') as f:
                    pickle.dump(processed_team_seasons, f)
                tqdm.write(f"  Checkpoint: {len(game_date_map):,} game dates, {len(processed_team_seasons)} teams processed")

        except Exception as e:
            failed_teams.append((team, season, str(e)))
            time.sleep(1)  # Extra delay after error

    # Final save
    with open(cache_path, 'wb') as f:
        pickle.dump(game_date_map, f)
    with open(processed_cache_path, 'wb') as f:
        pickle.dump(processed_team_seasons, f)

    print(f"\n✓ Processed {len(processed_team_seasons)} team-seasons")
    print(f"✓ Collected {len(game_date_map):,} game dates")

    if failed_teams:
        print(f"\n⚠ Failed to fetch {len(failed_teams)} team-seasons")
        failed_df = pd.DataFrame(failed_teams, columns=['team', 'season', 'error'])
        failed_path = config.DATA_DIR / 'logs' / 'failed_schedule_fetches.csv'
        failed_path.parent.mkdir(parents=True, exist_ok=True)
        failed_df.to_csv(failed_path, index=False)
        print(f"  Saved failed fetches to: {failed_path}")

    # Convert game_ids in player data to string for matching
    print("\nAdding game_date column to player data...")
    player_data['game_id_str'] = player_data['game_id'].astype(str)
    player_data['game_date'] = player_data['game_id_str'].map(game_date_map)
    player_data.drop(columns=['game_id_str'], inplace=True)

    # Check match rate
    matched = player_data['game_date'].notna().sum()
    total = len(player_data)
    match_rate = (matched / total) * 100

    print(f"✓ Matched {matched:,} / {total:,} records ({match_rate:.1f}%)")

    if match_rate < 95:
        print(f"\n⚠ Warning: Only {match_rate:.1f}% of records have dates")
        unmatched = len(player_data[player_data['game_date'].isna()])
        print(f"  {unmatched:,} records missing dates")

        # Show sample of unmatched game_ids
        unmatched_ids = player_data[player_data['game_date'].isna()]['game_id'].unique()[:10]
        print(f"\n  Sample unmatched game_ids: {list(unmatched_ids)}")

    # Save updated player data
    output_path = player_data_path
    backup_path = player_data_path.parent / (player_data_path.stem + '_no_dates' + player_data_path.suffix)

    print(f"\nSaving backup (without dates) to {backup_path}...")
    pd.read_csv(player_data_path).to_csv(backup_path, index=False)

    print(f"Saving updated data to {output_path}...")
    player_data.to_csv(output_path, index=False)

    print("\n" + "="*60)
    print("✓ COMPLETE")
    print("="*60)
    print(f"Updated player data: {output_path}")
    print(f"Backup (no dates): {backup_path}")
    print(f"Game date cache: {cache_path}")
    print(f"\nMatch rate: {match_rate:.1f}%")
    print(f"Total game dates collected: {len(game_date_map):,}")
    print("="*60)


if __name__ == '__main__':
    main()
