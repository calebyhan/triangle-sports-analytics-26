"""Quick script to check what columns are in the checkpoint data."""
import pickle
from pathlib import Path

checkpoint_path = Path('data/cache/player_data/checkpoint_s2024_t90.pkl')

with open(checkpoint_path, 'rb') as f:
    data = pickle.load(f)

print(f"Loaded {len(data)} dataframes")
print(f"\nFirst dataframe shape: {data[0].shape}")
print(f"Columns: {list(data[0].columns)}")
print(f"\nFirst few rows:")
print(data[0].head(3))
