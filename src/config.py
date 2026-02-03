"""
Configuration file for Triangle Sports Analytics project.

This module contains all configuration parameters used across the project,
including paths, model hyperparameters, Elo system parameters, and competition metadata.
"""
from pathlib import Path
from typing import List, Dict

# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PREDICTIONS_DIR = DATA_DIR / "predictions"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# ============================================================================
# DATA COLLECTION PARAMETERS
# ============================================================================

# Years to fetch Barttorvik data for training
TRAINING_YEARS = [2020, 2021, 2022, 2023, 2024, 2025]

# Current prediction year
PREDICTION_YEAR = 2026

# ACC Teams for 2025-26 season
ACC_TEAMS = [
    'Michigan', 'Duke', 'Virginia', 'Louisville', 'Clemson', 'NC State',
    'North Carolina', 'SMU', 'Ohio State', 'Miami', 'Virginia Tech',
    'California', 'Wake Forest', 'Syracuse', 'Baylor', 'Stanford',
    'Notre Dame', 'Florida State', 'Pitt', 'Georgia Tech', 'Boston College'
]

# Template file for competition submission
SUBMISSION_TEMPLATE = "tsa_pt_spread_template_2026 - Sheet1.csv"

# Historical games file
HISTORICAL_GAMES_FILE = RAW_DATA_DIR / "games" / "historical_games_2019_2025.csv"

# Barttorvik data fetch parameters
BARTTORVIK_MAX_RETRIES = 3
BARTTORVIK_RETRY_DELAY = 1.0  # seconds

# ============================================================================
# ELO RATING SYSTEM PARAMETERS
# ============================================================================

# Elo system configuration (based on FiveThirtyEight methodology)
ELO_CONFIG = {
    'k_factor': 38,                    # Update sensitivity (higher = more volatile)
    'default_rating': 1500,            # Starting rating for new teams
    'season_carryover': 0.64,          # Percentage of rating carried to next season (64%)
    'home_court_advantage': 4.0,       # Home court advantage in points
    'points_per_elo': 28,              # Point spread per 100 Elo points
}

# Conference-based seasonal regression targets
# Teams regress to their conference mean (not overall mean)
CONFERENCE_MAPPINGS: Dict[str, List[str]] = {
    'ACC': ['Duke', 'Miami (FL)', 'Clemson', 'Virginia', 'NC State', 'Syracuse',
            'North Carolina', 'Virginia Tech', 'Stanford', 'Louisville', 'SMU',
            'California', 'Wake Forest', 'Georgia Tech', 'Notre Dame', 'Pittsburgh',
            'Florida St.', 'Boston College'],
    'SEC': ['Vanderbilt', 'Texas A&M', 'Florida', 'Georgia', 'Arkansas', 'Missouri',
            'Alabama', 'Tennessee', 'Mississippi St.', 'Texas', 'Auburn', 'Kentucky',
            'South Carolina', 'Ole Miss', 'Oklahoma', 'LSU'],
    'Big 12': ['Arizona', 'Houston', 'BYU', 'Iowa St.', 'UCF', 'Colorado',
               'Texas Tech', 'Kansas', 'West Virginia', 'TCU', 'Arizona St.',
               'Oklahoma St.', 'Baylor', 'Kansas St.', 'Utah', 'Cincinnati'],
    'Big Ten': ['Nebraska', 'Purdue', 'Michigan St.', 'Michigan', 'Illinois',
                'Wisconsin', 'UCLA', 'Southern California', 'Indiana', 'Ohio St.',
                'Minnesota', 'Iowa', 'Washington', 'Rutgers', 'Oregon', 'Penn St.',
                'Northwestern', 'Maryland'],
    'Big East': ['UConn', 'Villanova', 'St. John\'s (NY)', 'Creighton', 'Seton Hall',
                 'Xavier', 'DePaul', 'Butler', 'Georgetown', 'Providence', 'Marquette'],
    'WCC': ['Gonzaga', 'Saint Mary\'s (CA)', 'Santa Clara', 'Washington St.',
            'San Francisco', 'Pacific', 'LMU (CA)', 'Portland', 'Oregon St.',
            'San Diego', 'Seattle U', 'Pepperdine'],
}

# ============================================================================
# MODEL HYPERPARAMETERS
# ============================================================================

# ImprovedSpreadModel (Ridge + LightGBM Ensemble) - Best configuration from tuning
MODEL_CONFIG = {
    # Ridge regression parameters
    'ridge_alpha': 1.0,                # L2 regularization strength

    # LightGBM parameters (optimized via hyperparameter tuning)
    'n_estimators': 100,               # Number of boosting rounds
    'max_depth': 8,                    # Maximum tree depth
    'learning_rate': 0.1,              # Learning rate
    'early_stopping_rounds': 10,       # Early stopping patience

    # Ensemble weights (optimized: Ridge 30%, LightGBM 70%)
    'ridge_weight': 0.3,
    'lgbm_weight': 0.7,
}

# Cross-validation configuration
CV_CONFIG = {
    'n_splits': 5,                     # Number of time-series splits
    'test_size': None,                 # Use default split sizes
}

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

# Core features used in baseline model (best performing - 11 features)
BASELINE_FEATURES = [
    'home_adj_oe',          # Home team adjusted offensive efficiency
    'home_adj_de',          # Home team adjusted defensive efficiency
    'home_adj_em',          # Home team adjusted efficiency margin
    'away_adj_oe',          # Away team adjusted offensive efficiency
    'away_adj_de',          # Away team adjusted defensive efficiency
    'away_adj_em',          # Away team adjusted efficiency margin
    'eff_diff',             # Efficiency differential (home - away)
    'home_elo_before',      # Home team Elo before game
    'away_elo_before',      # Away team Elo before game
    'elo_diff',             # Elo differential (home - away)
    'predicted_spread',     # Elo-based spread prediction
]

# Note: Experiments showed that adding Four Factors and temporal features
# actually HURT performance (see outputs/feature_experiments_results.csv)
# Baseline 11 features: 5.0012 MAE
# + Four Factors:       5.0172 MAE (worse by 0.016)
# + Temporal:           5.0253 MAE (worse by 0.024)

# ============================================================================
# COMPETITION METADATA
# ============================================================================

TEAM_INFO = {
    'team_name': 'CMMT',
    'members': [
        {
            'name': 'Caleb Han',
            'email': 'calebhan@unc.edu',
        },
        {
            'name': 'Mason Mines',
            'email': 'mmines@unc.edu',
        },
        {
            'name': 'Mason Wang',
            'email': 'mywang@unc.edu',
        },
        {
            'name': 'Tony Wang',
            'email': 'anwang@unc.edu',
        },
    ],
}

# Competition deadline
COMPETITION_DEADLINE = "February 6, 2026"

# ============================================================================
# ENHANCED FEATURES CONFIGURATION
# ============================================================================

# Enhanced features to add to baseline (13 features)
ENHANCED_FEATURES = [
    # Momentum features (3)
    'momentum_diff',           # Recent margin differential
    'win_streak_diff',         # Current streak differential
    'recent_win_pct_diff',     # Last 5 games win % differential

    # Blowout tendency (4)
    'run_diff_differential',   # Average margin differential
    'blowout_tendency_diff',   # Blowout rate differential
    'consistency_ratio',       # Consistency of dominance
    'hot_streak_advantage',    # Winning streak advantage

    # Player-based features (4)
    'star_power_diff',         # Top 3 scorers PPG differential
    'bench_depth_diff',        # Bench production differential
    'offensive_balance_diff',  # Scoring distribution differential
    'star_efficiency_diff',    # Star player TS% differential

    # Team-specific HCA (2)
    'home_team_hca',           # Home team's historical HCA
    'away_team_hca',           # Away team's historical HCA
]

# Feature flags (enable/disable groups)
ENHANCED_FEATURE_FLAGS = {
    'use_momentum': True,
    'use_blowout': True,
    'use_player_features': True,
    'use_team_hca': True,
    'use_haslametrics': False,  # Only if historical data available
}

# Historical player data path
HISTORICAL_PLAYER_DATA = PROCESSED_DATA_DIR / 'historical_player_box_scores_2020_2024.csv'

# ============================================================================
# OUTPUT FORMATTING
# ============================================================================

# Prediction output file
PREDICTION_OUTPUT_FILE = PREDICTIONS_DIR / "tsa_pt_spread_CMMT_2026.csv"

# Required columns for submission
SUBMISSION_COLUMNS = [
    'Date', 'Away', 'Home', 'pt_spread',
    'team_name', 'team_member', 'team_email'
]

# ============================================================================
# DATA SOURCE CONFIGURATION
# ============================================================================

# Data source directories (organized by source)
ESPN_DATA_DIR = RAW_DATA_DIR / "espn"
HASLAMETRICS_DATA_DIR = RAW_DATA_DIR / "haslametrics"
CBBPY_DATA_DIR = RAW_DATA_DIR / "cbbpy"
CACHE_DIR = DATA_DIR / "cache"

# ESPN API Configuration
ESPN_CONFIG = {
    'enabled': True,  # ✅ Working - team list available
    'base_url': 'https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball',
    'rate_limit': 100,  # requests per minute
    'cache_ttl': 3600,  # cache time-to-live in seconds (1 hour)
    'timeout': 10,  # request timeout in seconds
    'features': ['team_list', 'scoreboard'],  # Note: standings/BPI require web scraping
}

# Haslametrics Configuration
HASLAMETRICS_CONFIG = {
    'enabled': True,  # ✅ Working with Selenium for JavaScript rendering
    'base_url': 'https://www.haslametrics.com',
    'rate_limit': 30,  # requests per minute
    'cache_ttl': 86400,  # 24 hours
    'timeout': 15,
    'features': ['team_ratings', 'offensive_defensive_stats'],
    'note': 'Uses Selenium + ChromeDriver for JavaScript-rendered tables',
}

# CBBpy Configuration (Enhanced)
CBBPY_CONFIG = {
    'enabled': True,  # ✅ Patched locally - fixed NCAA.com API changes
    'rate_limit': 20,  # requests per minute (be respectful of NCAA.com)
    'cache_ttl': 3600,  # 1 hour
    'timeout': 30,
    'features': ['play_by_play', 'box_scores', 'player_stats', 'game_metadata'],
    'note': 'Patched .venv/lib/.../cbbpy/utils/cbbpy_utils.py for missing fields',
}

# NCAA API Configuration (Optional)
NCAA_API_CONFIG = {
    'enabled': False,  # Optional - self-hosted recommended
    'base_url': 'https://ncaa-api.henrygd.me',  # Public demo (rate limited)
    'rate_limit': 5,  # Public demo limit: 5 req/sec
    'cache_ttl': 3600,
    'timeout': 10,
    'features': ['scoreboard', 'team_stats', 'rankings'],
}

# Feature Flags (Enable/disable experimental features)
FEATURE_FLAGS = {
    'use_bpi_ratings': False,  # ESPN BPI ratings
    'use_player_stats': False,  # Player-level statistics
    'use_play_by_play': False,  # Play-by-play data for pace/tempo
    'use_momentum_metrics': False,  # Haslametrics momentum features
    'use_alternative_efficiency': False,  # Non-Barttorvik efficiency metrics
}

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

# Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOG_LEVEL = 'INFO'

# Log format
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Log date format
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
