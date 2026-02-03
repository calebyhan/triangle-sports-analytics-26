"""
Hybrid Player-Team Prediction System

Combines player-based ELO ratings with team-level metrics to create
predictions that leverage both individual player skill and team synergy.

Key Innovation: Neural network learns how to map aggregated player ELOs
to team performance metrics, capturing coaching effects, team chemistry,
and opponent-specific matchups.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pickle
from sklearn.preprocessing import StandardScaler

from src.player_elo.player_elo_system import PlayerEloSystem
from src.player_elo.player_data_collector import PlayerDataCollector
from src.elo import EloRatingSystem


class HybridFeatureEngine:
    """
    Creates features combining player ELOs and team metrics
    """

    def __init__(
        self,
        player_elo_system: PlayerEloSystem,
        team_elo_system: EloRatingSystem,
        player_stats: pd.DataFrame,
        team_stats: pd.DataFrame
    ):
        self.player_elo_system = player_elo_system
        self.team_elo_system = team_elo_system
        self.player_stats = player_stats
        self.team_stats = team_stats

        # Team name mapping
        self.team_mapping = {
            'Florida State': 'Florida St.',
            'Miami': 'Miami FL',
            'NC State': 'N.C. State',
            'Pitt': 'Pittsburgh',
        }

    def get_team_lineup(self, team: str, top_n: int = 5) -> List[str]:
        """Get top N players for a team by minutes played"""
        mapped_team = self.team_mapping.get(team, team)
        team_players = self.player_stats[
            self.player_stats['team'] == mapped_team
        ].copy()

        if len(team_players) == 0:
            return []

        # Sort by minutes
        if 'minutes_per_game' in team_players.columns:
            team_players = team_players.sort_values(
                'minutes_per_game',
                ascending=False
            )

        return team_players['player_id'].head(top_n).tolist()

    def aggregate_player_elos(
        self,
        player_ids: List[str],
        weights: Optional[List[float]] = None
    ) -> Dict[str, float]:
        """
        Aggregate player ELOs with various statistics

        Returns dictionary with:
        - mean_elo: Average player ELO
        - max_elo: Best player ELO (star power)
        - min_elo: Weakest player ELO (depth)
        - std_elo: ELO variance (consistency)
        - weighted_elo: Usage-weighted ELO
        """
        if not player_ids:
            return {
                'mean_elo': 1000.0,
                'max_elo': 1000.0,
                'min_elo': 1000.0,
                'std_elo': 0.0,
                'weighted_elo': 1000.0
            }

        elos = [self.player_elo_system.get_player_elo(pid) for pid in player_ids]

        if weights is None:
            weights = [1.0] * len(elos)

        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize

        return {
            'mean_elo': np.mean(elos),
            'max_elo': np.max(elos),
            'min_elo': np.min(elos),
            'std_elo': np.std(elos),
            'weighted_elo': np.sum(elos * weights)
        }

    def get_player_weights(self, player_ids: List[str]) -> List[float]:
        """Get usage weights for players"""
        weights = []
        for pid in player_ids:
            player_row = self.player_stats[
                self.player_stats['player_id'] == pid
            ]
            if len(player_row) > 0 and 'minutes_per_game' in player_row.columns:
                weights.append(player_row['minutes_per_game'].iloc[0])
            else:
                weights.append(20.0)  # Default

        return weights

    def get_team_metrics(self, team: str) -> Dict[str, float]:
        """Get team-level statistics"""
        mapped_team = self.team_mapping.get(team, team)

        team_row = self.team_stats[self.team_stats['team'] == mapped_team]

        if len(team_row) == 0:
            # Defaults
            return {
                'team_elo': self.team_elo_system.get_rating(team),
                'off_efficiency': 100.0,
                'def_efficiency': 100.0,
                'pace': 70.0,
                'power_rating': 0.0
            }

        row = team_row.iloc[0]
        return {
            'team_elo': self.team_elo_system.get_rating(team),
            'off_efficiency': row.get('off_efficiency', 100.0),
            'def_efficiency': row.get('def_efficiency', 100.0),
            'pace': row.get('pace', 70.0),
            'power_rating': row.get('power_rating', 0.0)
        }

    def create_hybrid_features(
        self,
        home_team: str,
        away_team: str,
        neutral: bool = False
    ) -> np.ndarray:
        """
        Create comprehensive feature vector combining player and team data

        Feature groups (38 features total):
        1. Player ELO aggregates (5 × 2 teams = 10)
        2. Team metrics (5 × 2 teams = 10)
        3. Player-Team interactions (4 × 2 teams = 8)
        4. Matchup features (10)
        """
        # Get lineups
        home_lineup = self.get_team_lineup(home_team)
        away_lineup = self.get_team_lineup(away_team)

        # Player ELO weights
        home_weights = self.get_player_weights(home_lineup)
        away_weights = self.get_player_weights(away_lineup)

        # Player ELO aggregates
        home_player_elo = self.aggregate_player_elos(home_lineup, home_weights)
        away_player_elo = self.aggregate_player_elos(away_lineup, away_weights)

        # Team metrics
        home_team_metrics = self.get_team_metrics(home_team)
        away_team_metrics = self.get_team_metrics(away_team)

        # Player-Team interaction features
        # How well does player ELO predict team ELO?
        home_elo_residual = home_player_elo['weighted_elo'] - home_team_metrics['team_elo']
        away_elo_residual = away_player_elo['weighted_elo'] - away_team_metrics['team_elo']

        # Team chemistry indicators
        home_chemistry = home_team_metrics['team_elo'] - home_player_elo['mean_elo']
        away_chemistry = away_team_metrics['team_elo'] - away_player_elo['mean_elo']

        # Depth indicators
        home_depth = home_player_elo['min_elo'] / home_player_elo['max_elo']
        away_depth = away_player_elo['min_elo'] / away_player_elo['max_elo']

        # Star power
        home_star = home_player_elo['max_elo'] - home_player_elo['mean_elo']
        away_star = away_player_elo['max_elo'] - away_player_elo['mean_elo']

        # Matchup features
        team_elo_diff = home_team_metrics['team_elo'] - away_team_metrics['team_elo']
        player_elo_diff = home_player_elo['weighted_elo'] - away_player_elo['weighted_elo']

        off_vs_def = home_team_metrics['off_efficiency'] - away_team_metrics['def_efficiency']
        def_vs_off = home_team_metrics['def_efficiency'] - away_team_metrics['off_efficiency']

        pace_diff = home_team_metrics['pace'] - away_team_metrics['pace']
        power_diff = home_team_metrics['power_rating'] - away_team_metrics['power_rating']

        consistency_diff = home_player_elo['std_elo'] - away_player_elo['std_elo']
        chemistry_diff = home_chemistry - away_chemistry
        depth_diff = home_depth - away_depth
        star_diff = home_star - away_star

        # Assemble feature vector
        features = [
            # Home player ELO aggregates (5)
            home_player_elo['mean_elo'],
            home_player_elo['max_elo'],
            home_player_elo['min_elo'],
            home_player_elo['std_elo'],
            home_player_elo['weighted_elo'],

            # Away player ELO aggregates (5)
            away_player_elo['mean_elo'],
            away_player_elo['max_elo'],
            away_player_elo['min_elo'],
            away_player_elo['std_elo'],
            away_player_elo['weighted_elo'],

            # Home team metrics (5)
            home_team_metrics['team_elo'],
            home_team_metrics['off_efficiency'],
            home_team_metrics['def_efficiency'],
            home_team_metrics['pace'],
            home_team_metrics['power_rating'],

            # Away team metrics (5)
            away_team_metrics['team_elo'],
            away_team_metrics['off_efficiency'],
            away_team_metrics['def_efficiency'],
            away_team_metrics['pace'],
            away_team_metrics['power_rating'],

            # Home player-team interactions (4)
            home_elo_residual,
            home_chemistry,
            home_depth,
            home_star,

            # Away player-team interactions (4)
            away_elo_residual,
            away_chemistry,
            away_depth,
            away_star,

            # Matchup features (10)
            team_elo_diff,
            player_elo_diff,
            off_vs_def,
            def_vs_off,
            pace_diff,
            power_diff,
            consistency_diff,
            chemistry_diff,
            depth_diff,
            star_diff
        ]

        return np.array(features, dtype=np.float32)


class HybridNet(nn.Module):
    """
    Neural network that learns player-team correlations

    Architecture designed to capture:
    - Individual player contributions
    - Team synergy effects
    - Opponent-specific matchups
    """

    def __init__(self, input_dim: int = 38, hidden_dims: List[int] = [128, 64, 32]):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for better convergence"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.network(x).squeeze(-1)


class HybridPredictionSystem:
    """
    Complete hybrid prediction system combining player and team data
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        scaler_path: Optional[Path] = None,
        device: str = 'cpu'
    ):
        self.device = torch.device(device)
        self.model = None
        self.scaler = None
        self.feature_engine = None

        if model_path and model_path.exists():
            self.load_model(model_path)

        if scaler_path and scaler_path.exists():
            self.load_scaler(scaler_path)

    def load_model(self, model_path: Path):
        """Load trained hybrid model"""
        checkpoint = torch.load(model_path, map_location=self.device)

        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            input_dim = checkpoint.get('input_dim', 38)
            hidden_dims = checkpoint.get('hidden_dims', [128, 64, 32])
        else:
            state_dict = checkpoint
            input_dim = 38
            hidden_dims = [128, 64, 32]

        self.model = HybridNet(input_dim, hidden_dims)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()

    def load_scaler(self, scaler_path: Path):
        """Load feature scaler"""
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)

    def initialize_feature_engine(
        self,
        player_elo_system: PlayerEloSystem,
        team_elo_system: EloRatingSystem,
        player_stats: pd.DataFrame,
        team_stats: pd.DataFrame
    ):
        """Initialize feature engine with data"""
        self.feature_engine = HybridFeatureEngine(
            player_elo_system,
            team_elo_system,
            player_stats,
            team_stats
        )

    def predict(
        self,
        home_team: str,
        away_team: str,
        neutral: bool = False
    ) -> float:
        """Predict point spread for a game"""
        if self.feature_engine is None:
            raise ValueError("Feature engine not initialized")

        # Create features
        features = self.feature_engine.create_hybrid_features(
            home_team, away_team, neutral
        )

        # Scale features
        if self.scaler is not None:
            features = self.scaler.transform(features.reshape(1, -1)).flatten()

        # Predict
        if self.model is not None:
            with torch.no_grad():
                features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
                prediction = self.model(features_tensor).item()
        else:
            # Fallback: simple ELO difference
            prediction = features[28]  # team_elo_diff

        return prediction

    def predict_batch(
        self,
        games: pd.DataFrame
    ) -> np.ndarray:
        """Predict multiple games at once"""
        predictions = []

        for _, row in games.iterrows():
            pred = self.predict(
                row['Home'],
                row['Away'],
                neutral=False
            )
            predictions.append(pred)

        return np.array(predictions)
