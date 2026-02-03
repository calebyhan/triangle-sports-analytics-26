"""
Add game_date column to player data by joining with historical games.

This fixes the missing game_date column in the collected player data.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src import config

def main():
    print("="*60)
    print("ADDING GAME DATES TO PLAYER DATA")
    print("="*60)

    # Load player data
    player_data_path = config.HISTORICAL_PLAYER_DATA
    print(f"\nLoading player data from {player_data_path}...")
    player_data = pd.read_csv(player_data_path)
    print(f"✓ Loaded {len(player_data):,} player-game records")
    print(f"  Columns: {list(player_data.columns)}")

    # Check if game_date already exists
    if 'game_date' in player_data.columns:
        print("\n✓ game_date column already exists!")
        return

    # Load historical games
    print(f"\nLoading historical games from {config.HISTORICAL_GAMES_FILE}...")
    games = pd.read_csv(config.HISTORICAL_GAMES_FILE, parse_dates=['date'])
    print(f"✓ Loaded {len(games):,} games")

    # Check if games have game_id column
    if 'game_id' not in games.columns:
        print("\n⚠ Warning: games file doesn't have game_id column")
        print("  Will try to create game_id from other columns...")
        # Try to create a game_id if it doesn't exist
        # This is a fallback - ideally games should have game_id
        games['game_id'] = games.index.astype(str)

    # Create game_id -> date mapping
    game_date_map = games.set_index('game_id')['date'].to_dict()
    print(f"✓ Created mapping for {len(game_date_map):,} game IDs")

    # Add game_date to player data
    print("\nAdding game_date column...")
    player_data['game_date'] = player_data['game_id'].map(game_date_map)

    # Check how many matched
    matched = player_data['game_date'].notna().sum()
    total = len(player_data)
    match_rate = (matched / total) * 100

    print(f"✓ Matched {matched:,} / {total:,} records ({match_rate:.1f}%)")

    if match_rate < 90:
        print(f"\n⚠ Warning: Only {match_rate:.1f}% of records matched!")
        print("  Some game_ids in player data may not exist in games file")

        # Show some unmatched game_ids
        unmatched_ids = player_data[player_data['game_date'].isna()]['game_id'].unique()[:10]
        print(f"\n  Sample unmatched game_ids: {list(unmatched_ids)}")

    # Save updated player data
    output_path = player_data_path
    backup_path = player_data_path.parent / (player_data_path.stem + '_backup' + player_data_path.suffix)

    print(f"\nSaving backup to {backup_path}...")
    player_data_backup = pd.read_csv(player_data_path)  # Read original again for backup
    player_data_backup.to_csv(backup_path, index=False)

    print(f"Saving updated data to {output_path}...")
    player_data.to_csv(output_path, index=False)

    print("\n" + "="*60)
    print("✓ COMPLETE")
    print("="*60)
    print(f"Updated player data saved to: {output_path}")
    print(f"Backup saved to: {backup_path}")
    print(f"\nNew columns: {list(player_data.columns)}")
    print("="*60)


if __name__ == '__main__':
    main()
