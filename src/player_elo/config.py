"""
Configuration for Player-Based ELO System

This module contains all configuration parameters for the player-level ELO rating system,
including paths, data collection settings, model hyperparameters, and feature definitions.
"""

from pathlib import Path
from typing import Dict, List

# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "player_data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "player_elo"

# Raw data subdirectories
PLAYER_STATS_DIR = RAW_DATA_DIR / "player_stats"
BOXSCORES_DIR = RAW_DATA_DIR / "game_boxscores"
ROSTERS_DIR = RAW_DATA_DIR / "rosters"
LINEUPS_DIR = RAW_DATA_DIR / "lineups"

# Ensure directories exist
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, OUTPUTS_DIR,
                 PLAYER_STATS_DIR, BOXSCORES_DIR, ROSTERS_DIR, LINEUPS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATA COLLECTION PARAMETERS
# ============================================================================

# Years to collect player data for training
TRAINING_YEARS = [2020, 2021, 2022, 2023, 2024, 2025]

# Recent years for ELO initialization (more recent = more accurate)
ELO_INITIALIZATION_YEARS = [2024, 2025]

# Current prediction year
PREDICTION_YEAR = 2026

# Rate limiting for scraping (seconds between requests)
SCRAPING_DELAY = 2.0
BARTTORVIK_MAX_RETRIES = 3
BARTTORVIK_RETRY_DELAY = 1.0

# Data URLs
BARTTORVIK_PLAYER_URL_TEMPLATE = "https://barttorvik.com/playerstat.php?year={year}&csv=1"

# ============================================================================
# PLAYER ELO SYSTEM PARAMETERS
# ============================================================================

PLAYER_ELO_CONFIG = {
    # Rating initialization
    'default_rating': 1000,           # Starting rating for new players (lower than team's 1500)

    # Update parameters
    'k_factor': 20,                   # Update sensitivity (lower than team's 38 - less volatile)
    'minutes_threshold': 10,          # Minimum minutes played to update ELO

    # Seasonal parameters
    'season_carryover': 0.75,         # 75% carryover (higher than team's 64% - players more stable)
    'position_regression': {          # Regression targets by position
        'G': 1000,                    # Guards
        'F': 1000,                    # Forwards
        'C': 1000,                    # Centers
        'Unknown': 1000,              # Default for unknown positions
    },

    # Game context
    'home_court_advantage': 2.0,      # Home court advantage in ELO points
    'points_per_elo': 25,             # Conversion factor (ELO to points)

    # Team strength calculation
    'weighting_method': 'usage',      # Options: 'usage', 'minutes', 'equal'
    'min_players_for_team': 5,        # Minimum players required for team strength

    # Margin of victory scaling
    'mov_enabled': True,              # Whether to use margin of victory multiplier
    'mov_log_base': 2.2,             # Log base for MOV multiplier
    'mov_autocorr_factor': 0.001,    # Autocorrelation factor
}

# ============================================================================
# PYTORCH MODEL PARAMETERS
# ============================================================================

PYTORCH_CONFIG = {
    # Architecture
    'input_dim': 65,                  # Feature vector dimension
    'hidden_dims': [128, 64, 32],     # Hidden layer sizes
    'dropout': 0.2,                   # Dropout probability

    # Training
    'batch_size': 64,                 # Training batch size
    'learning_rate': 0.001,           # Initial learning rate
    'weight_decay': 1e-5,             # L2 regularization
    'epochs': 100,                    # Maximum training epochs

    # Early stopping
    'early_stopping_patience': 10,    # Early stopping patience
    'min_delta': 0.001,               # Minimum improvement for early stopping

    # Learning rate scheduling
    'lr_scheduler_patience': 5,       # LR reduction patience
    'lr_scheduler_factor': 0.5,       # LR reduction factor

    # Loss function
    'loss_function': 'huber',         # Options: 'huber', 'mse', 'mae'
    'huber_delta': 1.0,               # Delta parameter for Huber loss

    # Device
    'device': 'auto',                 # Options: 'auto', 'cuda', 'cpu'
}

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

# Player features (5 features × 10 players = 50)
PLAYER_FEATURES = [
    'player_elo',              # Player's ELO rating
    'usage_pct',               # Usage percentage
    'offensive_rating',        # Offensive rating (points per 100 possessions)
    'defensive_rating',        # Defensive rating (points allowed per 100 possessions)
    'minutes_per_game'         # Average minutes per game
]

# Lineup aggregate features (5 features × 2 teams = 10)
LINEUP_AGGREGATE_FEATURES = [
    'avg_elo',                 # Average ELO of lineup
    'elo_variance',            # Variance in ELO (measure of depth)
    'total_usage',             # Total usage percentage
    'avg_offensive_rating',    # Average offensive rating
    'avg_defensive_rating'     # Average defensive rating
]

# Contextual features (5 features)
CONTEXTUAL_FEATURES = [
    'home_court_advantage',    # Home court indicator (1.0 for home, 0.0 for neutral/away)
    'home_rest_days',          # Days since home team's last game
    'away_rest_days',          # Days since away team's last game
    'season_phase',            # Early/mid/late season indicator (0-1)
    'conference_game'          # Conference game indicator (1.0 or 0.0)
]

# Total feature count validation
TOTAL_FEATURES = len(PLAYER_FEATURES) * 10 + len(LINEUP_AGGREGATE_FEATURES) * 2 + len(CONTEXTUAL_FEATURES)
assert TOTAL_FEATURES == 65, f"Feature count mismatch: expected 65, got {TOTAL_FEATURES}"

# ============================================================================
# LINEUP PREDICTION
# ============================================================================

LINEUP_CONFIG = {
    'recent_games_window': 5,         # Look at last N games for lineup patterns
    'injury_weight': 0.5,             # Weight reduction for questionable players
    'doubtful_weight': 0.2,           # Weight reduction for doubtful players
    'out_weight': 0.0,                # Weight for players marked out (excluded)
    'matchup_adjustment': True,       # Adjust lineups based on opponent matchup
    'uncertainty_threshold': 0.3,     # Threshold for "uncertain" lineup prediction
    'min_confidence': 0.6,            # Minimum confidence for lineup prediction
}

# ============================================================================
# VALIDATION
# ============================================================================

VALIDATION_CONFIG = {
    'val_ratio': 0.2,                 # Validation set size (20%)
    'n_cv_splits': 5,                 # Time-series cross-validation splits
    'confidence_level': 0.80,         # Confidence interval level (80%)
    'random_seed': 42,                # Random seed for reproducibility
    'shuffle': False,                 # Never shuffle time-series data
}

# ============================================================================
# OUTPUT FILES
# ============================================================================

OUTPUT_CONFIG = {
    # Predictions
    'predictions_file': PROJECT_ROOT / "data" / "predictions" / "tsa_pt_spread_PLAYER_ELO_2026.csv",

    # Model artifacts
    'model_file': MODELS_DIR / "pytorch_model.pt",
    'model_state_dict': MODELS_DIR / "pytorch_model_state.pt",
    'elo_state_file': MODELS_DIR / "player_elo_state.json",
    'lineup_predictor_file': MODELS_DIR / "lineup_predictor.pkl",

    # Processed data
    'player_elo_ratings': PROCESSED_DATA_DIR / "player_elo_ratings.csv",
    'player_usage_stats': PROCESSED_DATA_DIR / "player_usage_stats.csv",
    'transfer_tracker': PROCESSED_DATA_DIR / "transfer_tracker.csv",
    'injury_status': PROCESSED_DATA_DIR / "injury_status.csv",

    # Evaluation outputs
    'training_log': OUTPUTS_DIR / "training_log.csv",
    'validation_results': OUTPUTS_DIR / "validation_results.csv",
    'backtest_results': OUTPUTS_DIR / "backtest_results.csv",
    'comparison_report': OUTPUTS_DIR / "team_vs_player_comparison.csv",
}

# ============================================================================
# LOGGING
# ============================================================================

LOGGING_CONFIG = {
    'level': 'INFO',                  # Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'date_format': '%Y-%m-%d %H:%M:%S',
    'log_dir': PROJECT_ROOT / "logs",
    'data_log': PROJECT_ROOT / "logs" / "player_elo_data.log",
    'elo_log': PROJECT_ROOT / "logs" / "player_elo_elo.log",
    'model_log': PROJECT_ROOT / "logs" / "player_elo_model.log",
}

# Ensure log directory exists
LOGGING_CONFIG['log_dir'].mkdir(parents=True, exist_ok=True)

# ============================================================================
# COMPETITION METADATA
# ============================================================================

TEAM_INFO = {
    'team_name': 'CMMT',
    'members': [
        'Caleb Han',
        'Mason Mines',
        'Mason Wang',
        'Tony Wang'
    ],
    'deadline': 'February 6, 2026',
    'model_type': 'Player-Based ELO with PyTorch Neural Network'
}

# ============================================================================
# CONFERENCE MAPPINGS (From existing config)
# ============================================================================

CONFERENCE_MAPPINGS = {
    'ACC': [
        'Boston College', 'Clemson', 'Duke', 'Florida St.', 'Georgia Tech',
        'Louisville', 'Miami', 'North Carolina', 'NC State', 'Notre Dame',
        'Pittsburgh', 'Syracuse', 'Virginia', 'Virginia Tech', 'Wake Forest',
        'California', 'SMU', 'Stanford'
    ],
    'SEC': [
        'Alabama', 'Arkansas', 'Auburn', 'Florida', 'Georgia', 'Kentucky',
        'LSU', 'Mississippi', 'Mississippi St.', 'Missouri', 'Oklahoma',
        'South Carolina', 'Tennessee', 'Texas', 'Texas A&M', 'Vanderbilt'
    ],
    'Big 12': [
        'Arizona', 'Arizona St.', 'Baylor', 'BYU', 'UCF', 'Cincinnati',
        'Colorado', 'Houston', 'Iowa St.', 'Kansas', 'Kansas St.',
        'Oklahoma St.', 'TCU', 'Texas Tech', 'Utah', 'West Virginia'
    ],
    'Big Ten': [
        'Illinois', 'Indiana', 'Iowa', 'Maryland', 'Michigan', 'Michigan St.',
        'Minnesota', 'Nebraska', 'Northwestern', 'Ohio St.', 'Oregon',
        'Penn St.', 'Purdue', 'Rutgers', 'UCLA', 'USC', 'Washington', 'Wisconsin'
    ],
    'Big East': [
        'Butler', 'Creighton', 'DePaul', 'Georgetown', 'Marquette',
        'Providence', 'Seton Hall', 'St. John\'s', 'UConn', 'Villanova', 'Xavier'
    ],
    'WCC': [
        'Gonzaga', 'Saint Mary\'s', 'BYU', 'San Francisco', 'Santa Clara',
        'Pepperdine', 'Loyola Marymount', 'Pacific', 'Portland', 'San Diego'
    ]
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_team_conference(team_name: str) -> str:
    """
    Get conference for a team

    Args:
        team_name: Team name

    Returns:
        Conference name or 'Unknown' if not found
    """
    for conference, teams in CONFERENCE_MAPPINGS.items():
        if team_name in teams:
            return conference
    return 'Unknown'


def get_conference_teams(conference: str) -> List[str]:
    """
    Get all teams in a conference

    Args:
        conference: Conference name

    Returns:
        List of team names
    """
    return CONFERENCE_MAPPINGS.get(conference, [])


def validate_config() -> bool:
    """
    Validate configuration parameters

    Returns:
        True if all validations pass
    """
    # Validate feature dimensions
    assert PYTORCH_CONFIG['input_dim'] == TOTAL_FEATURES, \
        f"Model input dim ({PYTORCH_CONFIG['input_dim']}) doesn't match feature count ({TOTAL_FEATURES})"

    # Validate ELO parameters
    assert 0 < PLAYER_ELO_CONFIG['season_carryover'] <= 1.0, \
        "Season carryover must be in (0, 1]"

    assert PLAYER_ELO_CONFIG['k_factor'] > 0, \
        "K-factor must be positive"

    # Validate training parameters
    assert PYTORCH_CONFIG['batch_size'] > 0, \
        "Batch size must be positive"

    assert 0 <= PYTORCH_CONFIG['dropout'] < 1.0, \
        "Dropout must be in [0, 1)"

    # Validate validation parameters
    assert 0 < VALIDATION_CONFIG['val_ratio'] < 1.0, \
        "Validation ratio must be in (0, 1)"

    return True


# Run validation on import
validate_config()
