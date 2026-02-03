"""
Extract game dates from existing collection checkpoints.

Much faster than re-fetching - just reads the pickle files we already have.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import pickle
from tqdm import tqdm
from src import config

def main():
    print("="*60)
    print("EXTRACTING GAME DATES FROM CHECKPOINTS")
    print("="*60)

    # Load player data
    player_data_path = config.HISTORICAL_PLAYER_DATA
    print(f"\nLoading player data from {player_data_path}...")
    player_data = pd.read_csv(player_data_path)
    print(f"✓ Loaded {len(player_data):,} player-game records")

    # Drop existing game_date column if it exists
    if 'game_date' in player_data.columns:
        player_data.drop(columns=['game_date'], inplace=True)

    # Find all checkpoint files
    checkpoint_dir = config.DATA_DIR / 'cache' / 'player_data'
    checkpoint_files = sorted(checkpoint_dir.glob("checkpoint_*.pkl"))

    print(f"\nFound {len(checkpoint_files)} checkpoint files")

    # Build game_id -> date mapping from checkpoints
    game_date_map = {}

    print("\nExtracting game dates from checkpoints...")
    for checkpoint_path in tqdm(checkpoint_files, desc="Processing checkpoints"):
        try:
            with open(checkpoint_path, 'rb') as f:
                data_list = pickle.load(f)

            # Each checkpoint contains a list of DataFrames (one per team-season)
            for df in data_list:
                # Check if this DataFrame has game_date column
                if 'game_date' in df.columns and 'game_id' in df.columns:
                    # Extract game_id -> date mapping
                    for idx, row in df.iterrows():
                        if pd.notna(row['game_id']) and pd.notna(row['game_date']):
                            game_date_map[str(row['game_id'])] = pd.to_datetime(row['game_date'])

        except Exception as e:
            print(f"\n  ⚠ Error reading {checkpoint_path.name}: {e}")
            continue

    print(f"\n✓ Extracted {len(game_date_map):,} unique game dates")

    # Map dates to player data
    print("\nAdding game_date column to player data...")
    player_data['game_id_str'] = player_data['game_id'].astype(str)
    player_data['game_date'] = player_data['game_id_str'].map(game_date_map)
    player_data.drop(columns=['game_id_str'], inplace=True)

    # Check match rate
    matched = player_data['game_date'].notna().sum()
    total = len(player_data)
    match_rate = (matched / total) * 100

    print(f"✓ Matched {matched:,} / {total:,} records ({match_rate:.1f}%)")

    if match_rate < 50:
        print(f"\n⚠ Warning: Low match rate ({match_rate:.1f}%)")
        print("  Checkpoints may not have game_date column")
        print("  Will need to use the slow fetch approach")
        return False

    # Save updated player data
    output_path = player_data_path
    backup_path = player_data_path.parent / (player_data_path.stem + '_backup' + player_data_path.suffix)

    print(f"\nSaving backup to {backup_path}...")
    pd.read_csv(player_data_path).to_csv(backup_path, index=False)

    print(f"Saving updated data to {output_path}...")
    player_data.to_csv(output_path, index=False)

    print("\n" + "="*60)
    print("✓ COMPLETE")
    print("="*60)
    print(f"Updated player data: {output_path}")
    print(f"Backup: {backup_path}")
    print(f"\nMatch rate: {match_rate:.1f}%")
    print(f"Game dates found: {len(game_date_map):,}")
    print("="*60)

    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
