"""
Data Collection Module for Triangle Sports Analytics
Collects NCAA basketball data from various sources
"""

import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
from datetime import datetime
import os
from typing import Dict, List, Optional
import json

class NCAADataCollector:
    """Collect NCAA basketball data from Sports Reference and other sources"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, "raw")
        self.processed_dir = os.path.join(data_dir, "processed")
        
        # Create directories if they don't exist
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        
        # Sports Reference team name mapping (URL slugs)
        self.team_slugs = {
            'Duke': 'duke',
            'North Carolina': 'north-carolina',
            'NC State': 'north-carolina-state',
            'Virginia': 'virginia',
            'Virginia Tech': 'virginia-tech',
            'Clemson': 'clemson',
            'Florida State': 'florida-state',
            'Miami': 'miami-fl',
            'Pitt': 'pittsburgh',
            'Pittsburgh': 'pittsburgh',
            'Syracuse': 'syracuse',
            'Louisville': 'louisville',
            'Wake Forest': 'wake-forest',
            'Georgia Tech': 'georgia-tech',
            'Boston College': 'boston-college',
            'Notre Dame': 'notre-dame',
            'California': 'california',
            'Stanford': 'stanford',
            'SMU': 'southern-methodist',
            # Non-ACC teams in schedule
            'Baylor': 'baylor',
            'Ohio State': 'ohio-state',
            'Michigan': 'michigan',
        }
    
    def get_acc_teams(self) -> List[str]:
        """Get list of ACC teams for 2025-26 season"""
        return [
            'Duke', 'North Carolina', 'NC State', 'Virginia',
            'Virginia Tech', 'Clemson', 'Florida State', 'Miami',
            'Pitt', 'Syracuse', 'Louisville', 'Wake Forest',
            'Georgia Tech', 'Boston College', 'Notre Dame',
            'California', 'Stanford', 'SMU'
        ]
    
    def get_all_teams_in_schedule(self) -> List[str]:
        """Get all teams that appear in the prediction schedule"""
        acc_teams = self.get_acc_teams()
        # Add non-ACC teams from schedule
        other_teams = ['Baylor', 'Ohio State', 'Michigan']
        return acc_teams + other_teams
    
    def get_team_stats_sports_ref(self, team: str, season: str = "2025-26") -> Optional[Dict]:
        """
        Scrape team statistics from Sports Reference
        
        Args:
            team: Team name
            season: Season string (e.g., "2025-26")
            
        Returns:
            Dictionary of team statistics
        """
        slug = self.team_slugs.get(team)
        if not slug:
            print(f"Warning: No slug found for team {team}")
            return None
        
        # Convert season to Sports Reference format (uses end year)
        year = int(season.split('-')[0]) + 1  # 2025-26 -> 2026
        
        url = f"https://www.sports-reference.com/cbb/schools/{slug}/men/{year}.html"
        
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'lxml')
            
            stats = {'team': team, 'season': season}
            
            # Parse team stats from the page
            # This will need to be adjusted based on actual page structure
            
            return stats
            
        except requests.RequestException as e:
            print(f"Error fetching data for {team}: {e}")
            return None
    
    def get_team_schedule_sports_ref(self, team: str, season: str = "2025-26") -> Optional[pd.DataFrame]:
        """
        Scrape team schedule and game results from Sports Reference
        
        Args:
            team: Team name
            season: Season string
            
        Returns:
            DataFrame with game results
        """
        slug = self.team_slugs.get(team)
        if not slug:
            print(f"Warning: No slug found for team {team}")
            return None
        
        year = int(season.split('-')[0]) + 1
        url = f"https://www.sports-reference.com/cbb/schools/{slug}/men/{year}-schedule.html"
        
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            # Parse tables from HTML
            tables = pd.read_html(response.content)
            
            if tables:
                schedule_df = tables[0]
                schedule_df['team'] = team
                return schedule_df
            
            return None
            
        except Exception as e:
            print(f"Error fetching schedule for {team}: {e}")
            return None
    
    def collect_all_team_data(self, season: str = "2025-26", delay: float = 3.0) -> pd.DataFrame:
        """
        Collect data for all teams
        
        Args:
            season: Season to collect
            delay: Delay between requests (be nice to servers)
            
        Returns:
            DataFrame with all team statistics
        """
        all_stats = []
        teams = self.get_all_teams_in_schedule()
        
        for team in teams:
            print(f"Collecting data for {team}...")
            stats = self.get_team_stats_sports_ref(team, season)
            
            if stats:
                all_stats.append(stats)
            
            time.sleep(delay)  # Rate limiting
        
        if all_stats:
            df = pd.DataFrame(all_stats)
            # Save to raw data
            output_path = os.path.join(self.raw_dir, f"team_stats_{season.replace('-', '_')}.csv")
            df.to_csv(output_path, index=False)
            print(f"Saved team stats to {output_path}")
            return df
        
        return pd.DataFrame()
    
    def load_prediction_template(self, template_path: str) -> pd.DataFrame:
        """
        Load the prediction template CSV
        
        Args:
            template_path: Path to template CSV
            
        Returns:
            DataFrame with games to predict
        """
        df = pd.read_csv(template_path)
        
        # Filter out empty rows
        df = df.dropna(subset=['Date', 'Away', 'Home'])
        
        # Parse dates
        df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
        
        print(f"Loaded {len(df)} games to predict")
        print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
        
        return df
    
    def get_unique_teams_from_template(self, template_df: pd.DataFrame) -> List[str]:
        """Get unique teams from the prediction template"""
        away_teams = template_df['Away'].unique().tolist()
        home_teams = template_df['Home'].unique().tolist()
        all_teams = list(set(away_teams + home_teams))
        return sorted(all_teams)


class BarttorviksCollector:
    """Collect data from Barttorvik (alternative data source for advanced metrics)"""
    
    def __init__(self):
        self.base_url = "https://barttorvik.com"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }
    
    def get_team_ratings(self, year: int = 2026) -> Optional[pd.DataFrame]:
        """
        Get team ratings from Barttorvik
        
        Args:
            year: Season end year
            
        Returns:
            DataFrame with team ratings
        """
        # This would need to be implemented based on Barttorvik's actual structure
        # They may have an API or require different scraping approach
        pass


if __name__ == "__main__":
    # Test the collector
    collector = NCAADataCollector(data_dir="data")
    
    # Load template
    template_path = "tsa_pt_spread_template_2026 - Sheet1.csv"
    if os.path.exists(template_path):
        template_df = collector.load_prediction_template(template_path)
        teams = collector.get_unique_teams_from_template(template_df)
        print(f"\nTeams in schedule: {teams}")
