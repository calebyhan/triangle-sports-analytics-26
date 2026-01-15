"""
Download historical game data from lbenz730/NCAA_Hoops GitHub repository
Much faster than scraping - uses pre-compiled CSV files
"""

import pandas as pd
import requests
import os
import ssl
import urllib.request
from typing import List

# Disable SSL verification for pandas read_csv
ssl._create_default_https_context = ssl._create_unverified_context


def download_season_results(season: str, save_dir: str = "data/raw/games") -> pd.DataFrame:
    """
    Download game results for a specific season from NCAA_Hoops repository

    Args:
        season: Season string (e.g., "2023-24")
        save_dir: Directory to save downloaded files

    Returns:
        DataFrame with game results
    """
    base_url = f"https://raw.githubusercontent.com/lbenz730/NCAA_Hoops/master/3.0_Files/Results/{season}"

    # Try to find the most complete results file
    # Format: NCAA_Hoops_results_MM_DD_YYYY.csv (dated snapshots)
    # We want the latest file for each season

    # For completed seasons, try getting the final file
    # GitHub API to list files
    api_url = f"https://api.github.com/repos/lbenz730/NCAA_Hoops/contents/3.0_Files/Results/{season}"

    print(f"Fetching file list for {season}...")
    response = requests.get(api_url, timeout=30)

    if response.status_code != 200:
        print(f"  ✗ Failed to fetch file list: {response.status_code}")
        return pd.DataFrame()

    files = response.json()
    csv_files = [f['name'] for f in files if f['name'].endswith('.csv') and 'results' in f['name'].lower()]

    if not csv_files:
        print(f"  ✗ No results files found")
        return pd.DataFrame()

    # Get the latest (alphabetically last) file
    latest_file = sorted(csv_files)[-1]
    print(f"  → Downloading: {latest_file}")

    file_url = f"{base_url}/{latest_file}"

    try:
        df = pd.read_csv(file_url)
        print(f"  ✓ Loaded {len(df)} games")

        # Save locally
        os.makedirs(save_dir, exist_ok=True)
        local_path = os.path.join(save_dir, f"{season}_results.csv")
        df.to_csv(local_path, index=False)
        print(f"  ✓ Saved to: {local_path}")

        return df

    except Exception as e:
        print(f"  ✗ Error downloading: {e}")
        return pd.DataFrame()


def standardize_ncaa_hoops_data(df: pd.DataFrame, season: str) -> pd.DataFrame:
    """
    Standardize NCAA_Hoops data format to match our requirements

    Expected columns in NCAA_Hoops data:
    - date: Game date
    - team: Team name (usually the "away" team or first listed)
    - opponent: Opponent name
    - teamscore: Team's score
    - oppscore: Opponent's score
    - location: H (home), A (away), N (neutral)
    """
    if df is None or df.empty:
        return pd.DataFrame()

    print(f"\nStandardizing {len(df)} games from {season}...")
    print(f"Columns: {df.columns.tolist()}")

    standardized = pd.DataFrame()

    # Parse date from year/month/day columns (NCAA_Hoops format)
    if 'year' in df.columns and 'month' in df.columns and 'day' in df.columns:
        standardized['date'] = pd.to_datetime(
            df[['year', 'month', 'day']].rename(columns={'year': 'year', 'month': 'month', 'day': 'day'}),
            errors='coerce'
        )
    elif 'date' in df.columns:
        standardized['date'] = pd.to_datetime(df['date'], errors='coerce')
    elif 'Date' in df.columns:
        standardized['date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Determine home/away based on location column
    if 'location' in df.columns and 'team' in df.columns and 'opponent' in df.columns:
        # When location is 'H', team is home
        # When location is 'A', team is away
        # When location is 'N', treat as neutral

        def assign_teams(row):
            loc = str(row.get('location', 'N')).upper()
            team = str(row.get('team', ''))
            opp = str(row.get('opponent', ''))
            team_score = row.get('teamscore', 0)
            opp_score = row.get('oppscore', 0)

            if loc == 'H':
                # Team is at home
                return pd.Series({
                    'home_team': team,
                    'away_team': opp,
                    'home_score': team_score,
                    'away_score': opp_score,
                    'neutral_site': False
                })
            elif loc == 'A':
                # Team is away
                return pd.Series({
                    'home_team': opp,
                    'away_team': team,
                    'home_score': opp_score,
                    'away_score': team_score,
                    'neutral_site': False
                })
            else:  # Neutral
                # Treat team as "home" for consistency
                return pd.Series({
                    'home_team': team,
                    'away_team': opp,
                    'home_score': team_score,
                    'away_score': opp_score,
                    'neutral_site': True
                })

        team_data = df.apply(assign_teams, axis=1)
        standardized = pd.concat([standardized, team_data], axis=1)

    # Add season (parse from season string like "2019-20" -> 2020)
    season_end_year = int(season.split('-')[1])
    standardized['season'] = 2000 + season_end_year

    # Calculate margin
    standardized['margin'] = standardized['home_score'] - standardized['away_score']

    # Remove duplicates (each game appears twice in the data - once for each team)
    # Sort to keep consistent record
    standardized = standardized.sort_values(['date', 'home_team', 'away_team'])

    # Deduplicate by creating a game key
    def game_key(row):
        teams = sorted([str(row['home_team']), str(row['away_team'])])
        return f"{row['date']}_{teams[0]}_{teams[1]}"

    standardized['game_key'] = standardized.apply(game_key, axis=1)
    standardized = standardized.drop_duplicates(subset=['game_key'], keep='first')
    standardized = standardized.drop(columns=['game_key'])

    # Remove rows with missing data
    standardized = standardized.dropna(subset=['date', 'home_team', 'away_team', 'home_score', 'away_score'])

    print(f"✓ Standardized to {len(standardized)} unique games")

    return standardized


def download_all_seasons(
    seasons: List[str] = None,
    save_dir: str = "data/raw/games"
) -> pd.DataFrame:
    """
    Download multiple seasons of game data

    Args:
        seasons: List of season strings (e.g., ["2020-21", "2021-22"])
        save_dir: Directory to save files

    Returns:
        Combined DataFrame with all games
    """
    if seasons is None:
        seasons = ["2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25"]

    all_games = []

    for season in seasons:
        df = download_season_results(season, save_dir)
        if not df.empty:
            standardized = standardize_ncaa_hoops_data(df, season)
            if not standardized.empty:
                all_games.append(standardized)

    if not all_games:
        print("\n⚠️ No games downloaded!")
        return pd.DataFrame()

    # Combine all seasons
    combined = pd.concat(all_games, ignore_index=True)
    combined = combined.sort_values('date').reset_index(drop=True)

    print(f"\n{'='*60}")
    print(f"TOTAL GAMES DOWNLOADED: {len(combined)}")
    print(f"Date range: {combined['date'].min()} to {combined['date'].max()}")
    print(f"Seasons: {sorted(combined['season'].unique())}")
    print(f"Unique teams: {len(set(combined['home_team'].unique()) | set(combined['away_team'].unique()))}")
    print(f"{'='*60}")

    # Save combined file
    combined_path = os.path.join(save_dir, "historical_games_2019_2025.csv")
    combined.to_csv(combined_path, index=False)
    print(f"\n✓ Saved combined file: {combined_path}")

    return combined


if __name__ == "__main__":
    print("Downloading NCAA game data from lbenz730/NCAA_Hoops repository...")
    print("This is much faster than scraping!\n")

    games = download_all_seasons()

    if not games.empty:
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"\nGames per season:")
        print(games.groupby('season').size())
        print(f"\nNeutral site games: {games['neutral_site'].sum()}")
        print(f"Average margin: {games['margin'].abs().mean():.2f} points")
        print(f"Margin std dev: {games['margin'].std():.2f} points")
        print("\nSample games:")
        print(games[['date', 'home_team', 'away_team', 'home_score', 'away_score', 'margin']].head(10))
