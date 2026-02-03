"""
Prediction Pipeline for Player-Based ELO System

Generates 2026 point spread predictions using trained player ELO model.

Usage:
    from src.player_elo.prediction_pipeline import generate_predictions

    predictions_df = generate_predictions(
        model_path='data/player_data/models/pytorch_model.pt',
        elo_state_path='data/player_data/models/player_elo_state.json',
        games_file='data/processed/acc_games_2026.csv',
        output_file='data/predictions/tsa_pt_spread_PLAYER_ELO_2026.csv'
    )
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import torch
import numpy as np
import pandas as pd

from .config import (
    PROJECT_ROOT, PLAYER_ELO_CONFIG, PYTORCH_CONFIG,
    PLAYER_STATS_DIR, MODELS_DIR
)
from .player_elo_system import PlayerEloSystem
from .player_data_collector import PlayerDataCollector
from .features import PlayerFeatureEngine
from .pytorch_model import PlayerELONet

# Set up logging
logger = logging.getLogger(__name__)

# Team name mapping (games file → player data)
TEAM_NAME_MAPPING = {
    'Florida State': 'Florida St.',
    'Miami': 'Miami FL',
    'NC State': 'N.C. State',
    'Pitt': 'Pittsburgh',
    # Add identity mappings for other teams
    'Boston College': 'Boston College',
    'California': 'California',
    'Clemson': 'Clemson',
    'Duke': 'Duke',
    'Georgia Tech': 'Georgia Tech',
    'Louisville': 'Louisville',
    'North Carolina': 'North Carolina',
    'Notre Dame': 'Notre Dame',
    'SMU': 'SMU',
    'Stanford': 'Stanford',
    'Syracuse': 'Syracuse',
    'Virginia': 'Virginia',
    'Virginia Tech': 'Virginia Tech',
    'Wake Forest': 'Wake Forest',
    # Additional teams that might appear
    'Baylor': 'Baylor',
    'Michigan': 'Michigan',
    'Ohio State': 'Ohio St.',
}


class PredictionPipeline:
    """
    Pipeline for generating predictions with trained player ELO model
    """

    def __init__(
        self,
        model_path: Path,
        elo_state_path: Path,
        device: str = 'cpu'
    ):
        """
        Initialize prediction pipeline

        Args:
            model_path: Path to trained PyTorch model
            elo_state_path: Path to player ELO state JSON
            device: Device for PyTorch ('cpu' or 'cuda')
        """
        self.model_path = Path(model_path)
        self.elo_state_path = Path(elo_state_path)
        self.device = device

        # Load model
        self.model = self._load_model()

        # Load ELO state
        self.elo_system = self._load_elo_state()

        # Initialize data collector
        self.data_collector = PlayerDataCollector()

        # Feature engine will be initialized when we have player stats
        self.feature_engine = None

        logger.info("PredictionPipeline initialized")
        logger.info(f"  Model: {self.model_path}")
        logger.info(f"  ELO state: {self.elo_state_path}")
        logger.info(f"  Device: {self.device}")

    def _load_model(self) -> PlayerELONet:
        """Load trained PyTorch model"""
        logger.info(f"Loading model from: {self.model_path}")

        # Load state dict
        state_dict = torch.load(self.model_path, map_location=self.device)

        # Create model with same architecture
        model = PlayerELONet(
            input_dim=65,
            hidden_dims=PYTORCH_CONFIG['hidden_dims'],
            dropout=PYTORCH_CONFIG['dropout']
        )

        # Load weights
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()

        logger.info("  Model loaded successfully")
        return model

    def _load_elo_state(self) -> PlayerEloSystem:
        """Load player ELO state from JSON"""
        logger.info(f"Loading ELO state from: {self.elo_state_path}")

        with open(self.elo_state_path, 'r') as f:
            state = json.load(f)

        # Create new ELO system
        elo_system = PlayerEloSystem()

        # Load player ratings
        elo_system.player_elos = {
            pid: float(rating)
            for pid, rating in state['player_elos'].items()
        }

        # Load player usage
        elo_system.player_usage = {
            pid: float(usage)
            for pid, usage in state.get('player_usage', {}).items()
        }

        # Load player minutes
        elo_system.player_minutes = {
            pid: float(mins)
            for pid, mins in state.get('player_minutes', {}).items()
        }

        logger.info(f"  Loaded {len(elo_system.player_elos)} player ratings")
        return elo_system

    def predict_lineup(
        self,
        team: str,
        player_stats: pd.DataFrame,
        top_n: int = 5
    ) -> List[str]:
        """
        Predict starting lineup for a team

        Uses heuristic: top 5 players by minutes played

        Args:
            team: Team name
            player_stats: Player statistics DataFrame
            top_n: Number of players in lineup (default: 5)

        Returns:
            List of player IDs for predicted lineup
        """
        # Map team name if needed
        mapped_team = TEAM_NAME_MAPPING.get(team, team)

        # Get team's players
        team_players = player_stats[player_stats['team'] == mapped_team].copy()

        if len(team_players) == 0:
            logger.warning(f"  No players found for team: {team}")
            return []

        # Sort by minutes per game (descending)
        if 'minutes_per_game' in team_players.columns:
            team_players = team_players.sort_values('minutes_per_game', ascending=False)
        elif 'mpg' in team_players.columns:
            team_players = team_players.sort_values('mpg', ascending=False)
        else:
            # If no minutes data, sort by games played
            team_players = team_players.sort_values('games_played', ascending=False)

        # Get top N players
        lineup = team_players.head(top_n)['player_id'].tolist()

        return lineup

    def predict_game(
        self,
        home_team: str,
        away_team: str,
        game_date: datetime,
        player_stats: pd.DataFrame
    ) -> Tuple[float, Dict]:
        """
        Predict point spread for a single game

        Args:
            home_team: Home team name
            away_team: Away team name
            game_date: Game date
            player_stats: Player statistics DataFrame

        Returns:
            (predicted_spread, metadata_dict)
        """
        # Initialize feature engine if needed
        if self.feature_engine is None:
            self.feature_engine = PlayerFeatureEngine(player_stats, self.elo_system)

        # Predict lineups
        home_lineup = self.predict_lineup(home_team, player_stats)
        away_lineup = self.predict_lineup(away_team, player_stats)

        # Check if we have lineups
        if not home_lineup or not away_lineup:
            logger.warning(f"  Missing lineup data for {home_team} vs {away_team}")
            # Fallback: predict 0 (neutral)
            return 0.0, {
                'home_lineup_size': len(home_lineup),
                'away_lineup_size': len(away_lineup),
                'prediction_method': 'fallback'
            }

        # Create feature vector
        features = self.feature_engine.create_matchup_features(
            home_lineup=home_lineup,
            away_lineup=away_lineup,
            game_date=game_date,
            home_team=home_team,
            away_team=away_team
        )

        # Convert to tensor
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)

        # Predict
        with torch.no_grad():
            prediction = self.model(features_tensor).item()

        # Metadata
        metadata = {
            'home_lineup_size': len(home_lineup),
            'away_lineup_size': len(away_lineup),
            'home_avg_elo': np.mean([self.elo_system.get_player_elo(pid) for pid in home_lineup]),
            'away_avg_elo': np.mean([self.elo_system.get_player_elo(pid) for pid in away_lineup]),
            'prediction_method': 'neural_network'
        }

        return prediction, metadata

    def generate_predictions(
        self,
        games_df: pd.DataFrame,
        player_stats_df: pd.DataFrame,
        output_file: Optional[Path] = None,
        team_name: str = "CMMT"
    ) -> pd.DataFrame:
        """
        Generate predictions for all games

        Args:
            games_df: DataFrame with columns: Date, Home, Away
            player_stats_df: Player statistics DataFrame
            output_file: Path to save predictions (optional)
            team_name: Team identifier for submission

        Returns:
            DataFrame with predictions
        """
        logger.info(f"Generating predictions for {len(games_df)} games...")

        predictions = []

        for idx, row in games_df.iterrows():
            game_date = pd.to_datetime(row['Date'])
            home_team = row['Home']
            away_team = row['Away']

            # Predict
            spread, metadata = self.predict_game(
                home_team, away_team, game_date, player_stats_df
            )

            # Store prediction
            predictions.append({
                'Date': row['Date'],
                'Home': home_team,
                'Away': away_team,
                'pt_spread': round(spread, 2),
                'team_name': team_name,
                **metadata
            })

            if (idx + 1) % 10 == 0:
                logger.info(f"  Processed {idx + 1}/{len(games_df)} games")

        # Create DataFrame
        predictions_df = pd.DataFrame(predictions)

        logger.info(f"Generated {len(predictions_df)} predictions")
        logger.info(f"  Mean spread: {predictions_df['pt_spread'].mean():.2f}")
        logger.info(f"  Std spread: {predictions_df['pt_spread'].std():.2f}")
        logger.info(f"  Range: [{predictions_df['pt_spread'].min():.2f}, {predictions_df['pt_spread'].max():.2f}]")

        # Save if output file specified
        if output_file:
            # Save submission file (required columns only)
            submission_df = predictions_df[['Date', 'Home', 'Away', 'pt_spread', 'team_name']]
            submission_df.to_csv(output_file, index=False)
            logger.info(f"Saved predictions to: {output_file}")

            # Save detailed file with metadata
            detailed_file = output_file.parent / f"{output_file.stem}_detailed.csv"
            predictions_df.to_csv(detailed_file, index=False)
            logger.info(f"Saved detailed predictions to: {detailed_file}")

        return predictions_df


def generate_predictions(
    model_path: str = None,
    elo_state_path: str = None,
    games_file: str = None,
    player_stats_year: int = 2025,
    output_file: str = None,
    team_name: str = "CMMT"
) -> pd.DataFrame:
    """
    Main function to generate predictions

    Args:
        model_path: Path to trained model (default: latest in models dir)
        elo_state_path: Path to ELO state (default: latest in models dir)
        games_file: Path to games CSV (default: ACC 2026 games)
        player_stats_year: Year for player stats (default: 2025 - most recent available)
        output_file: Output CSV path (default: data/predictions/)
        team_name: Team identifier (default: CMMT)

    Returns:
        DataFrame with predictions
    """
    # Set defaults
    if model_path is None:
        model_path = MODELS_DIR / 'pytorch_model.pt'
    if elo_state_path is None:
        elo_state_path = MODELS_DIR / 'player_elo_state.json'
    if games_file is None:
        games_file = PROJECT_ROOT / 'data' / 'processed' / 'games_to_predict.csv'
    if output_file is None:
        output_file = PROJECT_ROOT / 'data' / 'predictions' / 'tsa_pt_spread_PLAYER_ELO_2026.csv'

    # Convert to Path objects
    model_path = Path(model_path)
    elo_state_path = Path(elo_state_path)
    games_file = Path(games_file)
    output_file = Path(output_file)

    logger.info("="*70)
    logger.info("  PLAYER-BASED ELO PREDICTION PIPELINE")
    logger.info("="*70)
    logger.info(f"\nConfiguration:")
    logger.info(f"  Model: {model_path}")
    logger.info(f"  ELO state: {elo_state_path}")
    logger.info(f"  Games: {games_file}")
    logger.info(f"  Player stats year: {player_stats_year}")
    logger.info(f"  Output: {output_file}")
    logger.info("="*70 + "\n")

    # Load games
    if not games_file.exists():
        raise FileNotFoundError(f"Games file not found: {games_file}")

    logger.info(f"Loading games from: {games_file}")
    games_df = pd.read_csv(games_file)
    logger.info(f"  Loaded {len(games_df)} games\n")

    # Load player stats
    logger.info(f"Loading player statistics for {player_stats_year}...")
    collector = PlayerDataCollector()
    player_stats_df = collector.collect_player_stats_from_local([player_stats_year])

    if player_stats_df.empty:
        raise ValueError(f"No player statistics found for {player_stats_year}")

    logger.info(f"  Loaded {len(player_stats_df)} player records\n")

    # Initialize pipeline
    pipeline = PredictionPipeline(
        model_path=model_path,
        elo_state_path=elo_state_path
    )

    # Generate predictions
    predictions_df = pipeline.generate_predictions(
        games_df=games_df,
        player_stats_df=player_stats_df,
        output_file=output_file,
        team_name=team_name
    )

    logger.info("\n" + "="*70)
    logger.info("  PREDICTION COMPLETE!")
    logger.info("="*70)
    logger.info(f"\nGenerated {len(predictions_df)} predictions")
    logger.info(f"Saved to: {output_file}")
    logger.info("="*70 + "\n")

    return predictions_df


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Generate predictions
    predictions = generate_predictions()

    print("\nFirst 5 predictions:")
    print(predictions[['Date', 'Home', 'Away', 'pt_spread']].head())
