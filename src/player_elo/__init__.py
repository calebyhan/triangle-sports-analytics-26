"""
Player-Based ELO System for NCAA Basketball Point Spread Prediction

This package implements a comprehensive player-level ELO rating system with PyTorch
neural networks for predicting point spreads in NCAA Division I basketball games.

Main Components:
- player_elo_system: Individual player ELO tracking and team strength aggregation
- lineup_predictor: Probabilistic starting lineup prediction
- player_data_collector: Data collection from CBBpy and Barttorvik
- roster_manager: Track player transfers, injuries, and eligibility
- features: Player-level feature engineering (65D vectors)
- pytorch_model: Neural network architecture and training
- training_pipeline: End-to-end training orchestration
- prediction_pipeline: 2026 prediction generation

Author: Team CMMT
Date: February 2026
"""

__version__ = '1.0.0'
__author__ = 'Team CMMT (Caleb Han, Mason Mines, Mason Wang, Tony Wang)'

# Configuration
from . import config

# Core modules will be imported as they are implemented
# from .player_elo_system import PlayerEloSystem
# from .lineup_predictor import LineupPredictor
# from .player_data_collector import PlayerDataCollector
# from .roster_manager import RosterManager
# from .features import PlayerFeatureEngine
# from .pytorch_model import PlayerELONet
# from .training_pipeline import train_player_model
# from .prediction_pipeline import generate_predictions

__all__ = [
    'config',
    # 'PlayerEloSystem',
    # 'LineupPredictor',
    # 'PlayerDataCollector',
    # 'RosterManager',
    # 'PlayerFeatureEngine',
    # 'PlayerELONet',
    # 'train_player_model',
    # 'generate_predictions',
]
