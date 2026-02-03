"""Debug what columns/data the schedule actually contains."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_sources import cbbpy_enhanced

# Fetch one team's schedule to see structure
print("Fetching Duke 2020 schedule...")
schedule = cbbpy_enhanced.fetch_games_team("Duke", season=2020, include_all=False)

print(f"\nSchedule type: {type(schedule)}")
print(f"Schedule shape: {schedule.shape if hasattr(schedule, 'shape') else 'N/A'}")
print(f"\nColumns: {list(schedule.columns) if hasattr(schedule, 'columns') else 'N/A'}")
print(f"\nFirst 3 rows:")
print(schedule.head(3) if hasattr(schedule, 'head') else schedule)
print(f"\nColumn dtypes:")
print(schedule.dtypes if hasattr(schedule, 'dtypes') else 'N/A')
