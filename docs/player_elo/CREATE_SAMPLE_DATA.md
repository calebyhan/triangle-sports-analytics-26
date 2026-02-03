# Barttorvik Data Collection Issue - Workaround

## Problem
The Barttorvik website is not returning CSV data in the expected format. The URL `https://barttorvik.com/playerstat.php?year={year}&csv=1` is returning malformed data.

## Immediate Solution: Use Team-Based System

While we debug the player data collection, use the existing team-based system which is already proven to work (MAE: 4.97):

```bash
# Train with team-based system (works perfectly)
python scripts/train_model.py
```

## Alternative: Manual Data Collection

1. **Visit Barttorvik directly:**
   - Go to: https://barttorvik.com/playerstat.php?year=2024
   - Click "Export to CSV" or download option
   - Save as `barttorvik_stats_2024.csv`

2. **Place file in correct location:**
   ```
   data/player_data/raw/player_stats/barttorvik_stats_2024_2024.csv
   ```

3. **Repeat for other years if needed** (2020-2025)

## Quick Test with Synthetic Data

For testing purposes, you can create sample player data:

```python
import pandas as pd
import numpy as np

# Create sample player data
n_players = 1000
players = []

for i in range(n_players):
    players.append({
        'player_id': f'PLAYER_{i:04d}',
        'player_name': f'Player {i}',
        'team': f'Team {i % 50}',  # 50 teams
        'season': 2024,
        'games_played': np.random.randint(20, 35),
        'usage_pct': np.random.uniform(15, 30),
        'offensive_rating': np.random.uniform(90, 120),
        'defensive_rating': np.random.uniform(90, 120),
        'minutes_per_game': np.random.uniform(15, 35),
        'efg_pct': np.random.uniform(0.4, 0.6),
        'tov_pct': np.random.uniform(0.1, 0.25),
        'orb_pct': np.random.uniform(0.05, 0.15),
        'ft_rate': np.random.uniform(0.2, 0.5),
    })

df = pd.DataFrame(players)
df.to_csv('data/player_data/raw/player_stats/barttorvik_stats_2024_2024.csv', index=False)
print(f"Created sample data with {len(df)} players")
```

## Debugging the URL Issue

The Barttorvik URL format may have changed. Try these URLs manually in your browser:

1. https://barttorvik.com/playerstat.php?year=2024&csv=1
2. https://barttorvik.com/playerstat.php?year=2024&export=csv
3. https://barttorvik.com/2024_player_stats.csv

If any of these work, update the URL in `src/player_elo/config.py`:
```python
BARTTORVIK_PLAYER_URL_TEMPLATE = "https://barttorvik.com/[working_url_pattern]"
```

## Recommendation

**For your competition deadline (Feb 6, 2026):**

Use the **team-based system** which is already working and validated (MAE: 4.97). The player-based system is an excellent research prototype but the data collection infrastructure needs more debugging.

The team-based predictions are already competitive and proven to work!
