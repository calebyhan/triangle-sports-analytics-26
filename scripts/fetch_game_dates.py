"""
Fetch game dates for player data by querying CBBpy for each unique game_id.

Much faster than re-collecting all player data (only ~10-20K unique games).
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
    print("FETCHING GAME DATES FOR PLAYER DATA")
    print("="*60)

    # Load player data
    player_data_path = config.HISTORICAL_PLAYER_DATA
    print(f"\nLoading player data from {player_data_path}...")
    player_data = pd.read_csv(player_data_path)
    print(f"✓ Loaded {len(player_data):,} player-game records")

    # Check if game_date already exists
    if 'game_date' in player_data.columns:
        print("\n✓ game_date column already exists!")
        return

    # Get unique game_ids
    unique_game_ids = player_data['game_id'].unique()
    print(f"\nFound {len(unique_game_ids):,} unique game IDs")

    # Setup cache
    cache_path = config.DATA_DIR / 'cache' / 'game_dates.pkl'
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing cache if exists
    game_date_map = {}
    if cache_path.exists():
        print(f"Loading existing cache from {cache_path}...")
        with open(cache_path, 'rb') as f:
            game_date_map = pickle.load(f)
        print(f"✓ Loaded {len(game_date_map):,} cached game dates")

    # Fetch dates for remaining games
    remaining_ids = [gid for gid in unique_game_ids if gid not in game_date_map]
    print(f"\nNeed to fetch {len(remaining_ids):,} game dates from CBBpy")

    if len(remaining_ids) > 0:
        print("This may take 10-30 minutes depending on rate limits...")

        failed_ids = []

        for i, game_id in enumerate(tqdm(remaining_ids, desc="Fetching dates")):
            try:
                # Fetch game info
                game_info = cbbpy_enhanced.get_game_info(game_id)

                if game_info is not None and 'date' in game_info:
                    game_date_map[game_id] = game_info['date']
                else:
                    failed_ids.append(game_id)

                # Rate limiting
                if (i + 1) % 10 == 0:
                    time.sleep(0.5)

                # Save checkpoint every 1000 games
                if (i + 1) % 1000 == 0:
                    with open(cache_path, 'wb') as f:
                        pickle.dump(game_date_map, f)
                    print(f"\n  Checkpoint: {len(game_date_map):,} dates cached")

            except Exception as e:
                print(f"\n  ⚠ Error fetching game {game_id}: {e}")
                failed_ids.append(game_id)
                time.sleep(1)  # Extra delay after error

        # Final save
        with open(cache_path, 'wb') as f:
            pickle.dump(game_date_map, f)

        if failed_ids:
            print(f"\n⚠ Failed to fetch {len(failed_ids)} game dates")
            failed_df = pd.DataFrame({'game_id': failed_ids})
            failed_path = config.DATA_DIR / 'logs' / 'failed_game_dates.csv'
            failed_path.parent.mkdir(parents=True, exist_ok=True)
            failed_df.to_csv(failed_path, index=False)
            print(f"  Saved failed IDs to: {failed_path}")

    # Add game_date to player data
    print(f"\nAdding game_date column to player data...")
    player_data['game_date'] = player_data['game_id'].map(game_date_map)

    # Check match rate
    matched = player_data['game_date'].notna().sum()
    total = len(player_data)
    match_rate = (matched / total) * 100

    print(f"✓ Matched {matched:,} / {total:,} records ({match_rate:.1f}%)")

    if match_rate < 95:
        print(f"\n⚠ Warning: Only {match_rate:.1f}% of records have dates")
        unmatched = len(player_data[player_data['game_date'].isna()])
        print(f"  {unmatched:,} records missing dates")

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
    print("="*60)


if __name__ == '__main__':
    main()
