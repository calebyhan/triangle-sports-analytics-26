"""
Unit tests for Elo rating system
"""

import sys
sys.path.insert(0, 'src')

import pytest
import pandas as pd
import numpy as np
from elo import EloRatingSystem


def test_elo_initialization():
    """Test Elo system initialization"""
    elo = EloRatingSystem(k_factor=38, hca=4.0, carryover=0.64)

    assert elo.k_factor == 38
    assert elo.home_court_advantage == 4.0
    assert elo.season_carryover == 0.64
    assert len(elo.ratings) == 0


def test_elo_get_rating():
    """Test getting Elo rating for teams"""
    elo = EloRatingSystem()

    # New team should get default rating
    rating = elo.get_rating('Duke')
    assert rating == 1500

    # Same team should return same rating
    rating2 = elo.get_rating('Duke')
    assert rating == rating2


def test_elo_update_ratings():
    """Test Elo rating updates after a game"""
    elo = EloRatingSystem(k_factor=38, hca=4.0)

    # Set initial ratings
    elo.ratings['Duke'] = 1600
    elo.ratings['UNC'] = 1500

    # Simulate Duke winning at home by 10
    elo.update_ratings('Duke', 'UNC', 80, 70, neutral=False)

    # Duke (home winner) should have higher rating than before
    assert elo.ratings['Duke'] > 1600
    # UNC (away loser) should have lower rating than before
    assert elo.ratings['UNC'] < 1500


def test_elo_predict_spread():
    """Test spread prediction"""
    elo = EloRatingSystem(hca=4.0)

    elo.ratings['Duke'] = 1600
    elo.ratings['UNC'] = 1500

    # Duke at home should be favored
    spread = elo.predict_spread('Duke', 'UNC', neutral=False)
    assert spread > 0

    # On neutral site, spread should be smaller
    neutral_spread = elo.predict_spread('Duke', 'UNC', neutral=True)
    assert neutral_spread < spread


def test_elo_season_reset():
    """Test season reset with carryover"""
    elo = EloRatingSystem(carryover=0.64)

    # Load conference mappings
    conferences = {
        'ACC': ['Duke', 'UNC'],
        'SEC': ['Kentucky']
    }
    elo.load_conference_mappings(conferences)

    # Set some ratings
    elo.ratings['Duke'] = 1700
    elo.ratings['UNC'] = 1300
    elo.ratings['Kentucky'] = 1600

    # Reset season
    elo.season_reset(2024)

    # Ratings should regress toward conference mean
    # Duke should be lower than 1700, UNC should be higher than 1300
    assert 1300 < elo.ratings['Duke'] < 1700
    assert 1300 < elo.ratings['UNC'] < 1700


def test_elo_process_games():
    """Test processing multiple games"""
    elo = EloRatingSystem()

    # Create sample games
    games_data = {
        'date': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03']),
        'home_team': ['Duke', 'UNC', 'Duke'],
        'away_team': ['UNC', 'NC State', 'Virginia'],
        'home_score': [80, 75, 85],
        'away_score': [70, 72, 80],
        'neutral_site': [False, False, False],
        'season': [2024, 2024, 2024]
    }
    games = pd.DataFrame(games_data)

    conferences = {'ACC': ['Duke', 'UNC', 'NC State', 'Virginia']}
    elo.load_conference_mappings(conferences)

    # Process games
    snapshots = elo.process_games(
        games,
        date_col='date',
        home_col='home_team',
        away_col='away_team',
        home_score_col='home_score',
        away_score_col='away_score',
        neutral_col='neutral_site',
        season_col='season',
        save_snapshots=True
    )

    # Should have snapshots for all games
    assert len(snapshots) == 3

    # All teams should have ratings
    assert 'Duke' in elo.ratings
    assert 'UNC' in elo.ratings
    assert 'NC State' in elo.ratings
    assert 'Virginia' in elo.ratings


def test_elo_neutral_site():
    """Test neutral site game prediction"""
    elo = EloRatingSystem(hca=4.0)

    elo.ratings['Team A'] = 1600
    elo.ratings['Team B'] = 1500

    # Home game
    home_spread = elo.predict_spread('Team A', 'Team B', neutral=False)

    # Neutral game
    neutral_spread = elo.predict_spread('Team A', 'Team B', neutral=True)

    # Home advantage should add ~4 Elo points worth of spread
    assert home_spread > neutral_spread
    assert abs(home_spread - neutral_spread) > 1.0  # Should be noticeable difference


def test_elo_margin_of_victory():
    """Test margin of victory adjustment"""
    elo = EloRatingSystem(k_factor=38)

    elo.ratings['Winner'] = 1500
    elo.ratings['Loser'] = 1500

    # Close game (1 point win)
    elo_close = EloRatingSystem(k_factor=38)
    elo_close.ratings['Winner'] = 1500
    elo_close.ratings['Loser'] = 1500
    elo_close.update_ratings('Winner', 'Loser', 70, 69, neutral=True)

    # Blowout (30 point win)
    elo_blowout = EloRatingSystem(k_factor=38)
    elo_blowout.ratings['Winner'] = 1500
    elo_blowout.ratings['Loser'] = 1500
    elo_blowout.update_ratings('Winner', 'Loser', 100, 70, neutral=True)

    # Blowout should result in larger rating change
    assert elo_blowout.ratings['Winner'] > elo_close.ratings['Winner']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
