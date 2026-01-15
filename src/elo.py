"""
Elo Rating System for NCAA Basketball
Based on FiveThirtyEight and Silver Bulletin methodologies

Key parameters:
- K-factor: 38 (governs update sensitivity)
- Home Court Advantage: ~4 points
- Season carryover: 0.64 * end_rating + 0.36 * conference_avg
- Regress to conference mean (not overall mean) due to skill gaps
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json
import os


class EloRatingSystem:
    """
    Elo rating system for NCAA basketball predictions

    Based on research from:
    - FiveThirtyEight NCAA methodology
    - Silver Bulletin SBCB ratings
    - GitHub: grdavis/college-basketball-elo
    """

    # Default parameters (tuned for NCAA basketball)
    DEFAULT_RATING = 1500
    K_FACTOR = 38
    HOME_COURT_ADVANTAGE = 4.0  # Points
    SEASON_CARRYOVER = 0.64  # How much of rating carries to next season

    def __init__(
        self,
        k_factor: float = K_FACTOR,
        hca: float = HOME_COURT_ADVANTAGE,
        carryover: float = SEASON_CARRYOVER,
        default_rating: float = DEFAULT_RATING
    ):
        """
        Initialize Elo rating system

        Args:
            k_factor: Update sensitivity (higher = more reactive to results)
            hca: Home court advantage in points
            carryover: Fraction of rating to carry to next season
            default_rating: Starting rating for new teams
        """
        self.k_factor = k_factor
        self.hca = hca
        self.carryover = carryover
        self.default_rating = default_rating

        # Current ratings: team_name -> rating
        self.ratings: Dict[str, float] = {}

        # Rating history for backtesting
        self.history: List[Dict] = []

        # Conference mappings for season resets
        self.team_conference: Dict[str, str] = {}
        self.conference_avg: Dict[str, float] = {}

    def get_rating(self, team: str) -> float:
        """Get current rating for a team, initializing if needed"""
        if team not in self.ratings:
            self.ratings[team] = self.default_rating
        return self.ratings[team]

    def set_team_conference(self, team: str, conference: str):
        """Set conference affiliation for a team"""
        self.team_conference[team] = conference

    def load_conference_mappings(self, mappings: Dict[str, List[str]]):
        """
        Load conference mappings from dictionary

        Args:
            mappings: Dict of conference_name -> list of team names
        """
        for conf, teams in mappings.items():
            for team in teams:
                self.team_conference[team] = conf

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """
        Calculate expected score (win probability) for team A

        Uses logistic curve with scale of 400 (standard Elo)

        Args:
            rating_a: Rating of team A
            rating_b: Rating of team B

        Returns:
            Expected probability of team A winning
        """
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))

    def margin_of_victory_multiplier(
        self,
        margin: float,
        winner_elo: float,
        loser_elo: float
    ) -> float:
        """
        Calculate margin of victory multiplier

        Adjusts update magnitude based on:
        - Larger margins get more weight
        - Diminishing returns for very large margins (log)
        - Upsets (beating higher-rated team) get bonus

        Based on FiveThirtyEight NBA formula adapted for college

        Args:
            margin: Absolute margin of victory
            winner_elo: Winner's Elo before game
            loser_elo: Loser's Elo before game

        Returns:
            Multiplier for K-factor
        """
        elo_diff = winner_elo - loser_elo

        # Log-based scaling with diminishing returns
        # +1 prevents log(0), creates smoother curve
        mov_factor = np.log(abs(margin) + 1)

        # Autocorrelation adjustment: reduces multiplier when favorite wins big
        # This prevents ratings from being too volatile
        autocorr = 2.2 / ((elo_diff * 0.001) + 2.2)

        return mov_factor * autocorr

    def update_ratings(
        self,
        home_team: str,
        away_team: str,
        home_score: float,
        away_score: float,
        neutral: bool = False
    ) -> Tuple[float, float]:
        """
        Update ratings after a game

        Args:
            home_team: Home team name
            away_team: Away team name
            home_score: Home team's score
            away_score: Away team's score
            neutral: If True, no home court advantage

        Returns:
            Tuple of (home_new_rating, away_new_rating)
        """
        home_elo = self.get_rating(home_team)
        away_elo = self.get_rating(away_team)

        # Adjust for home court advantage
        hca_adjustment = 0 if neutral else self.hca
        home_elo_adj = home_elo + (hca_adjustment * 400 / 28)  # Convert points to Elo

        # Calculate expected scores
        home_expected = self.expected_score(home_elo_adj, away_elo)
        away_expected = 1 - home_expected

        # Actual scores (1 for win, 0 for loss)
        margin = home_score - away_score
        if margin > 0:
            home_actual = 1
            away_actual = 0
            winner_elo = home_elo
            loser_elo = away_elo
        elif margin < 0:
            home_actual = 0
            away_actual = 1
            winner_elo = away_elo
            loser_elo = home_elo
        else:
            # Tie (rare in basketball)
            home_actual = 0.5
            away_actual = 0.5
            winner_elo = home_elo
            loser_elo = away_elo

        # Margin of victory multiplier
        mov_mult = self.margin_of_victory_multiplier(abs(margin), winner_elo, loser_elo)

        # Update ratings
        k = self.k_factor * mov_mult
        home_new = home_elo + k * (home_actual - home_expected)
        away_new = away_elo + k * (away_actual - away_expected)

        # Store new ratings
        self.ratings[home_team] = home_new
        self.ratings[away_team] = away_new

        # Record history
        self.history.append({
            'home_team': home_team,
            'away_team': away_team,
            'home_score': home_score,
            'away_score': away_score,
            'neutral': neutral,
            'home_elo_before': home_elo,
            'away_elo_before': away_elo,
            'home_elo_after': home_new,
            'away_elo_after': away_new,
            'home_expected': home_expected,
            'margin': margin,
        })

        return home_new, away_new

    def predict_spread(
        self,
        home_team: str,
        away_team: str,
        neutral: bool = False
    ) -> float:
        """
        Predict point spread for a game

        Args:
            home_team: Home team name
            away_team: Away team name
            neutral: If True, no home court advantage

        Returns:
            Predicted spread (positive = home favored)
        """
        home_elo = self.get_rating(home_team)
        away_elo = self.get_rating(away_team)

        # Elo difference to points conversion
        # Roughly 28 Elo points = 1 point spread
        elo_diff = home_elo - away_elo

        # Add home court advantage
        hca = 0 if neutral else self.hca

        spread = (elo_diff / 28) + hca

        return spread

    def predict_win_probability(
        self,
        home_team: str,
        away_team: str,
        neutral: bool = False
    ) -> float:
        """
        Predict home team's win probability

        Args:
            home_team: Home team name
            away_team: Away team name
            neutral: If True, no home court advantage

        Returns:
            Win probability for home team (0-1)
        """
        home_elo = self.get_rating(home_team)
        away_elo = self.get_rating(away_team)

        # Adjust for home court
        hca_adjustment = 0 if neutral else (self.hca * 400 / 28)
        home_elo_adj = home_elo + hca_adjustment

        return self.expected_score(home_elo_adj, away_elo)

    def calculate_conference_averages(self) -> Dict[str, float]:
        """Calculate average Elo rating by conference"""
        conf_ratings: Dict[str, List[float]] = {}

        for team, rating in self.ratings.items():
            conf = self.team_conference.get(team, 'Other')
            if conf not in conf_ratings:
                conf_ratings[conf] = []
            conf_ratings[conf].append(rating)

        self.conference_avg = {
            conf: np.mean(ratings) for conf, ratings in conf_ratings.items()
        }

        return self.conference_avg

    def season_reset(self):
        """
        Regress ratings to conference mean at start of new season

        Formula: new_rating = carryover * old_rating + (1 - carryover) * conf_avg
        """
        # Calculate conference averages first
        self.calculate_conference_averages()

        for team in list(self.ratings.keys()):
            old_rating = self.ratings[team]
            conf = self.team_conference.get(team, 'Other')
            conf_avg = self.conference_avg.get(conf, self.default_rating)

            # Regress to conference mean
            new_rating = (
                self.carryover * old_rating +
                (1 - self.carryover) * conf_avg
            )
            self.ratings[team] = new_rating

    def process_games(
        self,
        games_df: pd.DataFrame,
        date_col: str = 'date',
        home_col: str = 'home_team',
        away_col: str = 'away_team',
        home_score_col: str = 'home_score',
        away_score_col: str = 'away_score',
        neutral_col: Optional[str] = None,
        season_col: Optional[str] = None,
        save_snapshots: bool = True
    ) -> pd.DataFrame:
        """
        Process multiple games and update ratings

        Args:
            games_df: DataFrame with game results
            date_col: Column name for game date
            home_col: Column name for home team
            away_col: Column name for away team
            home_score_col: Column name for home score
            away_score_col: Column name for away score
            neutral_col: Optional column for neutral site flag
            season_col: Optional column for season (triggers reset between seasons)
            save_snapshots: If True, save Elo snapshots for each game

        Returns:
            DataFrame with Elo ratings at each game
        """
        # Sort by date
        games_df = games_df.sort_values(date_col)

        snapshots = []
        current_season = None

        for idx, row in games_df.iterrows():
            # Check for season change
            if season_col and season_col in row:
                season = row[season_col]
                if current_season is not None and season != current_season:
                    print(f"Season change: {current_season} -> {season}")
                    self.season_reset()
                current_season = season

            # Get game info
            home = row[home_col]
            away = row[away_col]
            home_score = row[home_score_col]
            away_score = row[away_score_col]
            neutral = row[neutral_col] if neutral_col and neutral_col in row else False

            # Record pre-game ratings
            home_elo_before = self.get_rating(home)
            away_elo_before = self.get_rating(away)

            # Update ratings
            home_elo_after, away_elo_after = self.update_ratings(
                home, away, home_score, away_score, neutral
            )

            if save_snapshots:
                snapshots.append({
                    'date': row.get(date_col),
                    'home_team': home,
                    'away_team': away,
                    'home_score': home_score,
                    'away_score': away_score,
                    'home_elo_before': home_elo_before,
                    'away_elo_before': away_elo_before,
                    'home_elo_after': home_elo_after,
                    'away_elo_after': away_elo_after,
                    'predicted_spread': self.predict_spread(home, away, neutral),
                    'actual_margin': home_score - away_score,
                })

        return pd.DataFrame(snapshots)

    def get_rankings(self, top_n: Optional[int] = None) -> pd.DataFrame:
        """
        Get current team rankings by Elo

        Args:
            top_n: Optional limit to top N teams

        Returns:
            DataFrame with rankings
        """
        rankings = [
            {
                'rank': i + 1,
                'team': team,
                'elo': rating,
                'conference': self.team_conference.get(team, 'Other')
            }
            for i, (team, rating) in enumerate(
                sorted(self.ratings.items(), key=lambda x: x[1], reverse=True)
            )
        ]

        df = pd.DataFrame(rankings)
        if top_n:
            df = df.head(top_n)

        return df

    def save_ratings(self, filepath: str):
        """Save current ratings to JSON file"""
        data = {
            'ratings': self.ratings,
            'conference_avg': self.conference_avg,
            'parameters': {
                'k_factor': self.k_factor,
                'hca': self.hca,
                'carryover': self.carryover,
                'default_rating': self.default_rating
            }
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def load_ratings(self, filepath: str):
        """Load ratings from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)

        self.ratings = data['ratings']
        self.conference_avg = data.get('conference_avg', {})

        params = data.get('parameters', {})
        self.k_factor = params.get('k_factor', self.K_FACTOR)
        self.hca = params.get('hca', self.HOME_COURT_ADVANTAGE)
        self.carryover = params.get('carryover', self.SEASON_CARRYOVER)

    def evaluate_predictions(self, test_df: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate Elo predictions on test data

        Args:
            test_df: DataFrame with actual game results and predictions

        Returns:
            Dictionary with evaluation metrics
        """
        if 'predicted_spread' not in test_df or 'actual_margin' not in test_df:
            raise ValueError("Need 'predicted_spread' and 'actual_margin' columns")

        errors = test_df['predicted_spread'] - test_df['actual_margin']

        # Mean Absolute Error
        mae = np.abs(errors).mean()

        # Root Mean Squared Error
        rmse = np.sqrt((errors ** 2).mean())

        # Correct side %
        correct_side = (
            (test_df['predicted_spread'] > 0) == (test_df['actual_margin'] > 0)
        ).mean()

        return {
            'mae': mae,
            'rmse': rmse,
            'correct_side_pct': correct_side,
            'n_games': len(test_df)
        }


def create_elo_from_efficiency(
    team_stats: pd.DataFrame,
    efficiency_col: str = 'adj_em',
    team_col: str = 'team'
) -> EloRatingSystem:
    """
    Initialize Elo ratings from efficiency metrics

    Converts adjusted efficiency margin to Elo scale

    Args:
        team_stats: DataFrame with team efficiency ratings
        efficiency_col: Column with adjusted efficiency margin
        team_col: Column with team names

    Returns:
        Initialized EloRatingSystem
    """
    elo = EloRatingSystem()

    # Convert efficiency to Elo (roughly 1 efficiency point = 28 Elo)
    for _, row in team_stats.iterrows():
        team = row[team_col]
        adj_em = row.get(efficiency_col, 0)

        # Scale to Elo (1500 base + efficiency * 28)
        rating = 1500 + (adj_em * 28)
        elo.ratings[team] = rating

    return elo


if __name__ == "__main__":
    # Example usage
    elo = EloRatingSystem()

    # Load conference mappings
    conferences = {
        'ACC': ['Duke', 'North Carolina', 'Virginia', 'NC State'],
        'SEC': ['Kentucky', 'Tennessee', 'Alabama', 'Auburn'],
        'Big Ten': ['Purdue', 'Michigan', 'Ohio State', 'Illinois'],
    }
    elo.load_conference_mappings(conferences)

    # Example: Simulate some games
    games = [
        ('Duke', 'North Carolina', 75, 72),
        ('Kentucky', 'Duke', 80, 78),
        ('Purdue', 'Duke', 70, 82),
    ]

    for home, away, h_score, a_score in games:
        elo.update_ratings(home, away, h_score, a_score)
        spread = elo.predict_spread(home, away)
        print(f"{home} vs {away}: {h_score}-{a_score}, Spread: {spread:.1f}")

    # Print rankings
    print("\nCurrent Rankings:")
    print(elo.get_rankings(top_n=10))
