"""
Fetch 2025-26 season games and update historical games file.

This script:
1. Fetches 2025-26 season games using CBBpy
2. Standardizes format to match historical_games_2019_2025.csv
3. Combines with historical data
4. Saves updated file for use in modeling

Usage:
    python scripts/fetch_2026_games.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from datetime import datetime
from src import config
from src.data_sources import cbbpy_enhanced

print("="*70)
print("FETCHING 2025-26 SEASON GAMES")
print("="*70)

# Fetch 2025-26 games using CBBpy
print(f"\nFetching {config.PREDICTION_YEAR} season games using CBBpy...")
try:
    games_2026_raw = cbbpy_enhanced.fetch_games_season(season=config.PREDICTION_YEAR)

    # Handle tuple return (games_df, metadata) or direct DataFrame
    if isinstance(games_2026_raw, tuple):
        games_2026 = games_2026_raw[0]  # Extract DataFrame from tuple
        print(f"✓ Fetched {len(games_2026)} games from CBBpy (tuple format)")
    else:
        games_2026 = games_2026_raw
        print(f"✓ Fetched {len(games_2026)} games from CBBpy")

    print(f"\nColumns available: {games_2026.columns.tolist()}")
    if len(games_2026) > 0:
        print(f"\nSample game:")
        print(games_2026.head(1).to_dict('records')[0])
except Exception as e:
    print(f"❌ Error fetching games: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Standardize column names
print("\nStandardizing format...")

# CBBpy column mappings (adjust based on actual columns)
column_mapping = {
    'game_date': 'date',
    'away': 'away_team',
    'home': 'home_team',
    'away_score': 'away_score',
    'home_score': 'home_score',
}

# Try common variations
for old_col in games_2026.columns:
    if 'date' in old_col.lower() and 'date' not in column_mapping.values():
        column_mapping[old_col] = 'date'
    elif 'away' in old_col.lower() and old_col != 'away_score' and 'away_team' not in column_mapping.values():
        column_mapping[old_col] = 'away_team'
    elif 'home' in old_col.lower() and old_col != 'home_score' and 'home_team' not in column_mapping.values():
        column_mapping[old_col] = 'home_team'

games_2026_clean = games_2026.rename(columns=column_mapping).copy()

# Ensure required columns exist
required = ['date', 'home_team', 'away_team', 'home_score', 'away_score']
missing = [col for col in required if col not in games_2026_clean.columns]

if missing:
    print(f"❌ Missing columns: {missing}")
    print(f"Available columns: {games_2026_clean.columns.tolist()}")
    sys.exit(1)

# Convert date to datetime
games_2026_clean['date'] = pd.to_datetime(games_2026_clean['date'])

# Add neutral site flag (if available)
if 'neutral_site' not in games_2026_clean.columns:
    if 'location' in games_2026_clean.columns:
        games_2026_clean['neutral_site'] = games_2026_clean['location'].str.upper().isin(['N', 'NEUTRAL'])
    else:
        games_2026_clean['neutral_site'] = False

# Calculate margin
games_2026_clean['margin'] = pd.to_numeric(games_2026_clean['home_score'], errors='coerce') - \
                              pd.to_numeric(games_2026_clean['away_score'], errors='coerce')

# Add season
games_2026_clean['season'] = config.PREDICTION_YEAR

# Select and order columns
output_cols = ['date', 'home_team', 'away_team', 'home_score', 'away_score',
               'neutral_site', 'season', 'margin']
games_2026_clean = games_2026_clean[output_cols].copy()

# Remove games without scores (future games)
before_filter = len(games_2026_clean)
games_2026_clean = games_2026_clean.dropna(subset=['home_score', 'away_score'])
print(f"✓ Filtered to {len(games_2026_clean)} completed games (removed {before_filter - len(games_2026_clean)} future games)")

if games_2026_clean.empty:
    print("❌ No completed games found for 2025-26 season yet")
    sys.exit(1)

# Load historical games
print(f"\nLoading historical games from {config.HISTORICAL_GAMES_FILE}...")
historical_games = pd.read_csv(config.HISTORICAL_GAMES_FILE, parse_dates=['date'])
print(f"✓ Loaded {len(historical_games):,} historical games")
print(f"  Date range: {historical_games['date'].min()} to {historical_games['date'].max()}")

# Combine with historical
print("\nCombining datasets...")
combined = pd.concat([historical_games, games_2026_clean], ignore_index=True)
combined = combined.sort_values('date').reset_index(drop=True)

# Remove duplicates
before_dedup = len(combined)
combined = combined.drop_duplicates(
    subset=['date', 'home_team', 'away_team'],
    keep='last'  # Keep the most recent data
)
if len(combined) < before_dedup:
    print(f"  Removed {before_dedup - len(combined)} duplicate games")

# Save combined file
output_path = config.RAW_DATA_DIR / "games" / "historical_games_with_2026.csv"
combined.to_csv(output_path, index=False)

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Historical games (2019-2025):  {len(historical_games):,}")
print(f"Current season games (2026):   {len(games_2026_clean):,}")
print(f"Total combined games:          {len(combined):,}")
print(f"Date range:                    {combined['date'].min()} to {combined['date'].max()}")
print(f"\nSaved to: {output_path}")
print("="*70)

# Show most recent games
print("\n10 Most recent games:")
recent = combined.tail(10)[['date', 'home_team', 'away_team', 'home_score', 'away_score', 'margin']]
for _, row in recent.iterrows():
    date_str = row['date'].strftime('%Y-%m-%d') if isinstance(row['date'], pd.Timestamp) else str(row['date'])
    print(f"  {date_str}: {row['away_team']:20} @ {row['home_team']:20} "
          f"({int(row['away_score'])}-{int(row['home_score'])}, margin: {int(row['margin']):+d})")

print("\n✓ Done! You can now use this file for training by updating config.HISTORICAL_GAMES_FILE")
