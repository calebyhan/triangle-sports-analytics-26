"""
Historical Game Data Collection Module for Triangle Sports Analytics
Collects NCAA basketball game results from various sources
"""

import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import time
from datetime import datetime
import os
from typing import Dict, List, Optional, Tuple
import json
import re


class HistoricalDataCollector:
    """Collect historical NCAA basketball game results"""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, "raw")
        self.processed_dir = os.path.join(data_dir, "processed")

        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }

        # Conference mappings for season resets
        self.conferences = {
            'ACC': ['Duke', 'North Carolina', 'NC State', 'Virginia', 'Virginia Tech',
                   'Clemson', 'Florida State', 'Miami', 'Pitt', 'Syracuse', 'Louisville',
                   'Wake Forest', 'Georgia Tech', 'Boston College', 'Notre Dame',
                   'California', 'Stanford', 'SMU'],
            'SEC': ['Kentucky', 'Tennessee', 'Alabama', 'Auburn', 'Florida', 'Texas A&M',
                   'Arkansas', 'LSU', 'Mississippi State', 'Ole Miss', 'Missouri',
                   'South Carolina', 'Vanderbilt', 'Georgia', 'Oklahoma', 'Texas'],
            'Big Ten': ['Purdue', 'Michigan', 'Michigan State', 'Ohio State', 'Illinois',
                       'Indiana', 'Iowa', 'Wisconsin', 'Minnesota', 'Northwestern',
                       'Penn State', 'Rutgers', 'Maryland', 'Nebraska', 'Oregon', 'UCLA',
                       'USC', 'Washington'],
            'Big 12': ['Houston', 'Kansas', 'Baylor', 'Iowa State', 'BYU', 'Cincinnati',
                      'UCF', 'Colorado', 'Arizona', 'Arizona State', 'Utah', 'Kansas State',
                      'Oklahoma State', 'TCU', 'Texas Tech', 'West Virginia'],
            'Big East': ['UConn', 'Creighton', 'Marquette', 'Villanova', 'Xavier',
                        'Providence', 'Butler', 'St. Johns', 'Seton Hall', 'Georgetown', 'DePaul'],
        }

        # Build reverse mapping: team -> conference
        self.team_conference = {}
        for conf, teams in self.conferences.items():
            for team in teams:
                self.team_conference[team] = conf

    def get_team_conference(self, team: str) -> str:
        """Get conference for a team, default to 'Other' if unknown"""
        return self.team_conference.get(team, 'Other')

    def scrape_barttorvik_games(self, year: int, delay: float = 2.0) -> pd.DataFrame:
        """
        Scrape game results from Barttorvik for a given season

        Args:
            year: Season end year (e.g., 2024 for 2023-24 season)
            delay: Delay between requests

        Returns:
            DataFrame with game results
        """
        url = f"https://barttorvik.com/trank.php?year={year}&sort=&top=0&conlimit=All&type=conf"

        try:
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()

            # Barttorvik returns data in a specific format
            # This may need adjustment based on actual page structure

            print(f"Fetched Barttorvik page for {year}")
            return pd.DataFrame()

        except Exception as e:
            print(f"Error fetching Barttorvik data for {year}: {e}")
            return pd.DataFrame()

    def scrape_sports_reference_schedule(
        self,
        team_slug: str,
        year: int,
        delay: float = 3.0
    ) -> pd.DataFrame:
        """
        Scrape schedule/results for a team from Sports Reference

        Args:
            team_slug: URL slug for team (e.g., 'duke', 'north-carolina')
            year: Season end year
            delay: Delay after request

        Returns:
            DataFrame with game results
        """
        url = f"https://www.sports-reference.com/cbb/schools/{team_slug}/men/{year}-schedule.html"

        try:
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Find the schedule table
            table = soup.find('table', {'id': 'schedule'})
            if table is None:
                print(f"No schedule table found for {team_slug} {year}")
                return pd.DataFrame()

            # Parse table rows
            games = []
            rows = table.find('tbody').find_all('tr')

            for row in rows:
                # Skip header rows
                if row.get('class') and 'thead' in row.get('class'):
                    continue

                cells = row.find_all(['td', 'th'])
                if len(cells) < 10:
                    continue

                try:
                    game = {
                        'date': cells[0].get_text(strip=True),
                        'location': cells[2].get_text(strip=True),  # @, vs, or N
                        'opponent': cells[3].get_text(strip=True),
                        'result': cells[4].get_text(strip=True),  # W or L
                        'team_score': cells[5].get_text(strip=True),
                        'opp_score': cells[6].get_text(strip=True),
                    }
                    games.append(game)
                except (IndexError, AttributeError):
                    continue

            time.sleep(delay)

            df = pd.DataFrame(games)
            df['team_slug'] = team_slug
            df['season'] = year

            return df

        except Exception as e:
            print(f"Error fetching schedule for {team_slug} {year}: {e}")
            return pd.DataFrame()

    def load_kaggle_ncaa_data(self, filepath: str) -> pd.DataFrame:
        """
        Load NCAA game data from Kaggle dataset

        Expected format: CSV with columns like:
        Season, DayNum, WTeamID, WScore, LTeamID, LScore, WLoc, NumOT

        Args:
            filepath: Path to Kaggle NCAA data file

        Returns:
            DataFrame with standardized game data
        """
        if not os.path.exists(filepath):
            print(f"Kaggle data file not found: {filepath}")
            return pd.DataFrame()

        df = pd.read_csv(filepath)

        # Standardize column names based on common Kaggle NCAA formats
        # Adjust these mappings based on actual file structure

        games = []
        for _, row in df.iterrows():
            # Determine home/away based on WLoc (H=Home, A=Away, N=Neutral)
            wloc = row.get('WLoc', 'N')

            if wloc == 'H':
                home_team = row['WTeamID']
                away_team = row['LTeamID']
                home_score = row['WScore']
                away_score = row['LScore']
            elif wloc == 'A':
                home_team = row['LTeamID']
                away_team = row['WTeamID']
                home_score = row['LScore']
                away_score = row['WScore']
            else:  # Neutral site - treat winner as "home" for modeling
                home_team = row['WTeamID']
                away_team = row['LTeamID']
                home_score = row['WScore']
                away_score = row['LScore']

            games.append({
                'season': row['Season'],
                'day_num': row.get('DayNum', 0),
                'home_team': home_team,
                'away_team': away_team,
                'home_score': home_score,
                'away_score': away_score,
                'neutral_site': wloc == 'N',
                'margin': home_score - away_score,
            })

        return pd.DataFrame(games)

    def create_synthetic_training_data(
        self,
        team_stats: pd.DataFrame,
        seasons: List[int] = None,
        games_per_matchup: int = 1,
        margin_std: float = 11.0
    ) -> pd.DataFrame:
        """
        Create synthetic training data from team statistics
        This is a fallback when real game data isn't available

        Args:
            team_stats: DataFrame with team efficiency ratings by season
            seasons: List of seasons to include
            games_per_matchup: Number of synthetic games per matchup
            margin_std: Standard deviation for margin of victory

        Returns:
            DataFrame with synthetic game data
        """
        if seasons is None:
            seasons = team_stats['season'].unique().tolist()

        np.random.seed(42)
        games = []

        for season in seasons:
            season_stats = team_stats[team_stats['season'] == season]
            teams = season_stats['team'].tolist()

            for i, home_team in enumerate(teams):
                for away_team in teams:
                    if home_team == away_team:
                        continue

                    home_stats = season_stats[season_stats['team'] == home_team].iloc[0]
                    away_stats = season_stats[season_stats['team'] == away_team].iloc[0]

                    # Expected margin based on efficiency differential + HCA
                    home_net = home_stats.get('adj_oe', 100) - home_stats.get('adj_de', 100)
                    away_net = away_stats.get('adj_oe', 100) - away_stats.get('adj_de', 100)
                    expected_margin = (home_net - away_net) / 2 + 3.5  # HCA

                    for _ in range(games_per_matchup):
                        # Add random noise
                        actual_margin = expected_margin + np.random.normal(0, margin_std)

                        games.append({
                            'season': season,
                            'home_team': home_team,
                            'away_team': away_team,
                            'expected_margin': expected_margin,
                            'actual_margin': actual_margin,
                            'home_adj_oe': home_stats.get('adj_oe', 100),
                            'home_adj_de': home_stats.get('adj_de', 100),
                            'away_adj_oe': away_stats.get('adj_oe', 100),
                            'away_adj_de': away_stats.get('adj_de', 100),
                        })

        return pd.DataFrame(games)

    def fetch_barttorvik_team_data(self, year: int) -> pd.DataFrame:
        """
        Fetch team statistics from Barttorvik T-Rank

        Args:
            year: Season end year

        Returns:
            DataFrame with team ratings
        """
        url = f"https://barttorvik.com/trank.php?year={year}&sort=&top=0&conlimit=All"

        try:
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()

            # Parse HTML tables
            tables = pd.read_html(response.content)

            if tables:
                df = tables[0]
                df['season'] = year
                return df

            return pd.DataFrame()

        except Exception as e:
            print(f"Error fetching Barttorvik data for {year}: {e}")
            return pd.DataFrame()

    def collect_multi_year_ratings(
        self,
        start_year: int = 2019,
        end_year: int = 2025,
        delay: float = 3.0
    ) -> pd.DataFrame:
        """
        Collect team ratings from Barttorvik for multiple years

        Args:
            start_year: First season end year
            end_year: Last season end year
            delay: Delay between requests

        Returns:
            DataFrame with multi-year team ratings
        """
        all_data = []

        for year in range(start_year, end_year + 1):
            print(f"Fetching data for {year}...")
            df = self.fetch_barttorvik_team_data(year)

            if not df.empty:
                all_data.append(df)

            time.sleep(delay)

        if all_data:
            combined = pd.concat(all_data, ignore_index=True)

            # Save to file
            output_path = os.path.join(self.processed_dir, f"barttorvik_ratings_{start_year}_{end_year}.csv")
            combined.to_csv(output_path, index=False)
            print(f"Saved {len(combined)} team-seasons to {output_path}")

            return combined

        return pd.DataFrame()


class FourFactorsCollector:
    """Collect Dean Oliver's Four Factors data"""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }

    def fetch_four_factors_barttorvik(self, year: int) -> pd.DataFrame:
        """
        Fetch Four Factors data from Barttorvik

        Args:
            year: Season end year

        Returns:
            DataFrame with Four Factors for each team
        """
        # Barttorvik provides Four Factors in their team pages
        url = f"https://barttorvik.com/trank.php?year={year}&sort=&top=0&conlimit=All"

        try:
            response = requests.get(url, headers=self.headers, timeout=30)
            tables = pd.read_html(response.content)

            if tables:
                df = tables[0]

                # Map Barttorvik columns to Four Factors
                # Columns vary by year, but typically include:
                # EFG%, TO%, ORB%, FTR for offense and defense

                df['season'] = year
                return df

            return pd.DataFrame()

        except Exception as e:
            print(f"Error fetching Four Factors for {year}: {e}")
            return pd.DataFrame()

    def calculate_four_factors(
        self,
        fg: float, fga: float, fg3: float,
        to: float, poss: float,
        orb: float, drb_opp: float,
        fta: float
    ) -> Dict[str, float]:
        """
        Calculate Four Factors from raw stats

        Args:
            fg: Field goals made
            fga: Field goal attempts
            fg3: Three-pointers made
            to: Turnovers
            poss: Possessions
            orb: Offensive rebounds
            drb_opp: Opponent defensive rebounds
            fta: Free throw attempts

        Returns:
            Dictionary with Four Factors
        """
        # Effective Field Goal %
        efg_pct = (fg + 0.5 * fg3) / fga if fga > 0 else 0.0

        # Turnover %
        tov_pct = to / poss * 100 if poss > 0 else 0.0

        # Offensive Rebound %
        orb_pct = orb / (orb + drb_opp) * 100 if (orb + drb_opp) > 0 else 0.0

        # Free Throw Rate
        ft_rate = fta / fga if fga > 0 else 0.0

        return {
            'efg_pct': efg_pct,
            'tov_pct': tov_pct,
            'orb_pct': orb_pct,
            'ft_rate': ft_rate
        }


class ESPNBPICollector:
    """Collect ESPN BPI (Basketball Power Index) ratings"""

    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }

    def fetch_bpi_ratings(self) -> pd.DataFrame:
        """
        Fetch current BPI ratings from ESPN

        Returns:
            DataFrame with BPI ratings
        """
        url = "https://www.espn.com/mens-college-basketball/bpi"

        try:
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()

            # ESPN's BPI page uses dynamic loading, may need alternative approach
            soup = BeautifulSoup(response.content, 'html.parser')

            # Try to find BPI table
            tables = pd.read_html(response.content)

            if tables:
                return tables[0]

            return pd.DataFrame()

        except Exception as e:
            print(f"Error fetching ESPN BPI: {e}")
            return pd.DataFrame()


if __name__ == "__main__":
    # Test the collectors
    collector = HistoricalDataCollector(data_dir="data")

    # Example: Collect multi-year Barttorvik data
    # df = collector.collect_multi_year_ratings(start_year=2020, end_year=2025)

    print("Historical data collector initialized")
