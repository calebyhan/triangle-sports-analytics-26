"""
Enhanced Feature Engineering Pipeline

Integrates momentum, blowout, player-based, and team-specific HCA features
for training data. Maintains temporal integrity and uses caching for efficiency.

Key Features:
- Reuses existing BlowoutFeatureEngine, historical_features, player_features
- Batch processing with checkpointing
- Strict temporal boundaries (< as_of_date) to prevent data leakage
- Month-based caching for better hit rates
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime
from tqdm import tqdm

from src.blowout_features import BlowoutFeatureEngine
from src.features import historical_features, player_features
from src import config


class EnhancedFeatureEngine:
    """
    Unified feature engineering with temporal safety.

    Integrates:
    - Momentum (from historical_features.py)
    - Blowout (from blowout_features.py - reuses existing engine)
    - Player stats (from player_features.py)
    - Team HCA (from historical_features.py)
    """

    def __init__(
        self,
        games_df: pd.DataFrame,
        player_data_df: Optional[pd.DataFrame] = None,
        use_player_features: bool = True,
        checkpoint_dir: Optional[Path] = None
    ):
        """
        Initialize enhanced feature engine.

        Args:
            games_df: Historical games DataFrame
            player_data_df: Historical player box scores (optional)
            use_player_features: Whether to compute player features
            checkpoint_dir: Directory for checkpoints
        """
        # Reuse existing blowout engine (already has caching!)
        self.blowout_engine = BlowoutFeatureEngine(games_df)

        # Store data
        self.games_df = games_df.copy()
        self.games_df['date'] = pd.to_datetime(self.games_df['date'])
        self.games_df = self.games_df.sort_values('date')

        self.player_data = player_data_df
        if player_data_df is not None:
            self.player_data['game_date'] = pd.to_datetime(self.player_data['game_date'])

        self.use_player_features = use_player_features and (player_data_df is not None)

        # Caches for performance (cache by month for better hit rate)
        self.momentum_cache = {}
        self.player_cache = {}
        self.hca_cache = {}

        # Checkpoint directory
        self.checkpoint_dir = checkpoint_dir or (config.DATA_DIR / 'cache' / 'enhanced_features')
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _get_momentum_features(self, team: str, as_of_date: datetime) -> Dict[str, float]:
        """
        Get momentum features with caching.

        IMPORTANT: Uses strict < as_of_date boundary (not <=)

        Args:
            team: Team name
            as_of_date: Temporal boundary

        Returns:
            Dictionary with momentum features
        """
        # Cache by month for better hit rate
        month_key = as_of_date.strftime('%Y-%m')
        cache_key = f"{team}_{month_key}"

        if cache_key in self.momentum_cache:
            return self.momentum_cache[cache_key]

        # Filter to games BEFORE as_of_date (strict boundary)
        filtered_games = self.games_df[self.games_df['date'] < as_of_date]

        # Use existing function from historical_features.py
        # Note: We use the filtered dataframe and pass None for as_of_date
        # since we already filtered
        momentum = historical_features.calculate_team_momentum(
            filtered_games, team, as_of_date=None, window=5
        )

        self.momentum_cache[cache_key] = momentum
        return momentum

    def _get_player_features(self, team: str, as_of_date: datetime) -> Dict[str, float]:
        """
        Calculate player-based features using only past data.

        CRITICAL: Filters player_data to game_date < as_of_date

        Args:
            team: Team name
            as_of_date: Temporal boundary

        Returns:
            Dictionary with player features
        """
        if not self.use_player_features:
            return {
                'star_total_ppg': 0.0,
                'star_avg_efficiency': 0.5,
                'bench_depth_score': 0.0,
                'balanced_scoring': 0.5,
            }

        # Cache by month
        month_key = as_of_date.strftime('%Y-%m')
        cache_key = f"{team}_{month_key}"

        if cache_key in self.player_cache:
            return self.player_cache[cache_key]

        # CRITICAL: Filter to only past games
        past_player_data = self.player_data[
            self.player_data['game_date'] < as_of_date
        ]

        # Filter to team
        team_player_data = past_player_data[past_player_data['team'] == team]

        if len(team_player_data) == 0:
            # No data - return defaults
            features = {
                'star_total_ppg': 0.0,
                'star_avg_efficiency': 0.5,
                'bench_depth_score': 0.0,
                'balanced_scoring': 0.5,
            }
            self.player_cache[cache_key] = features
            return features

        # Calculate features using existing functions
        star = player_features.calculate_star_player_power(team_player_data, team, top_n=3)
        balance = player_features.calculate_offensive_balance(team_player_data, team)
        bench = player_features.calculate_bench_depth(team_player_data, team)

        features = {
            'star_total_ppg': star.get('star_total_ppg', 0.0),
            'star_avg_efficiency': star.get('star_avg_efficiency', 0.5),
            'bench_depth_score': bench.get('bench_depth_score', 0.0),
            'balanced_scoring': balance.get('balanced_scoring', 0.5),
        }

        self.player_cache[cache_key] = features
        return features

    def _get_team_hca(self, team: str, as_of_date: datetime) -> float:
        """
        Get team-specific home court advantage.

        Args:
            team: Team name
            as_of_date: Temporal boundary

        Returns:
            Home court advantage in points
        """
        # Cache by month
        month_key = as_of_date.strftime('%Y-%m')
        cache_key = f"{team}_{month_key}"

        if cache_key in self.hca_cache:
            return self.hca_cache[cache_key]

        # Filter to games before date
        filtered_games = self.games_df[self.games_df['date'] < as_of_date]

        # Use existing function
        hca = historical_features.calculate_home_court_strength(filtered_games, team)

        self.hca_cache[cache_key] = hca
        return hca

    def _compute_game_features(
        self,
        game_row: pd.Series,
        as_of_date: datetime
    ) -> Dict[str, float]:
        """
        Compute all enhanced features for a single game.

        Args:
            game_row: Row from games DataFrame
            as_of_date: Temporal boundary (game date)

        Returns:
            Dictionary with all enhanced features
        """
        home = game_row['home_team']
        away = game_row['away_team']

        features = {}

        # 1. Momentum features (from historical_features.py)
        home_momentum = self._get_momentum_features(home, as_of_date)
        away_momentum = self._get_momentum_features(away, as_of_date)

        features['momentum_diff'] = (
            home_momentum['recent_avg_margin'] - away_momentum['recent_avg_margin']
        )
        features['win_streak_diff'] = (
            home_momentum['win_streak'] - away_momentum['win_streak']
        )
        features['recent_win_pct_diff'] = (
            home_momentum['recent_win_pct'] - away_momentum['recent_win_pct']
        )

        # 2. Blowout features (from BlowoutFeatureEngine)
        blowout_features = self.blowout_engine.create_matchup_features(
            home, away, as_of_date, lookback_games=10
        )
        features['run_diff_differential'] = blowout_features.get('run_diff_differential', 0.0)
        features['blowout_tendency_diff'] = blowout_features.get('blowout_tendency_diff', 0.0)
        features['consistency_ratio'] = blowout_features.get('consistency_ratio', 1.0)
        features['hot_streak_advantage'] = blowout_features.get('hot_streak_advantage', 0.0)

        # 3. Player features (if enabled)
        if self.use_player_features:
            home_player = self._get_player_features(home, as_of_date)
            away_player = self._get_player_features(away, as_of_date)

            features['star_power_diff'] = (
                home_player['star_total_ppg'] - away_player['star_total_ppg']
            )
            features['bench_depth_diff'] = (
                home_player['bench_depth_score'] - away_player['bench_depth_score']
            )
            features['offensive_balance_diff'] = (
                home_player['balanced_scoring'] - away_player['balanced_scoring']
            )
            features['star_efficiency_diff'] = (
                home_player['star_avg_efficiency'] - away_player['star_avg_efficiency']
            )
        else:
            # Default values if player features disabled
            features['star_power_diff'] = 0.0
            features['bench_depth_diff'] = 0.0
            features['offensive_balance_diff'] = 0.0
            features['star_efficiency_diff'] = 0.0

        # 4. Team-specific HCA
        features['home_team_hca'] = self._get_team_hca(home, as_of_date)
        features['away_team_hca'] = self._get_team_hca(away, as_of_date)

        return features

    def _save_checkpoint(self, batch_idx: int, features: List[Dict]):
        """
        Save checkpoint for recovery.

        Args:
            batch_idx: Current batch index
            features: List of feature dictionaries computed so far
        """
        checkpoint_path = self.checkpoint_dir / f"checkpoint_batch_{batch_idx}.pkl"

        with open(checkpoint_path, 'wb') as f:
            pickle.dump(features, f)

        print(f"  Checkpoint saved: batch {batch_idx}")

    def _load_checkpoint(self) -> Optional[List[Dict]]:
        """
        Load latest checkpoint if exists.

        Returns:
            List of feature dictionaries, or None if no checkpoint
        """
        checkpoint_files = sorted(self.checkpoint_dir.glob("checkpoint_batch_*.pkl"))

        if not checkpoint_files:
            return None

        latest = checkpoint_files[-1]
        print(f"Resuming from checkpoint: {latest.name}")

        with open(latest, 'rb') as f:
            features = pickle.load(f)

        return features

    def compute_all_features(
        self,
        games_df: pd.DataFrame,
        batch_size: int = 1000
    ) -> pd.DataFrame:
        """
        Compute enhanced features for all games.

        Main entry point. Processes games chronologically in batches
        with checkpointing.

        Args:
            games_df: DataFrame with games to process
            batch_size: Batch size for processing

        Returns:
            DataFrame with enhanced features (one row per game)
        """
        print("\n" + "="*60)
        print("COMPUTING ENHANCED FEATURES")
        print("="*60)
        print(f"Games to process: {len(games_df):,}")
        print(f"Batch size: {batch_size}")
        print(f"Player features: {'enabled' if self.use_player_features else 'disabled'}")
        print("="*60)

        # Sort chronologically (CRITICAL for temporal integrity)
        games_df = games_df.sort_values('date').reset_index(drop=True)

        # Check for checkpoint
        all_features = self._load_checkpoint()
        start_idx = len(all_features) if all_features else 0

        if all_features is None:
            all_features = []

        # Process in batches
        for batch_start in range(start_idx, len(games_df), batch_size):
            batch_end = min(batch_start + batch_size, len(games_df))
            batch = games_df.iloc[batch_start:batch_end]

            print(f"\nBatch {batch_start//batch_size + 1}: Games {batch_start}-{batch_end}")

            batch_features = []
            for idx, game_row in tqdm(batch.iterrows(), total=len(batch), desc="Computing features"):
                as_of_date = pd.to_datetime(game_row['date'])

                try:
                    features = self._compute_game_features(game_row, as_of_date)
                    batch_features.append(features)
                except Exception as e:
                    print(f"\n⚠ Error computing features for game {idx}: {e}")
                    # Add default features
                    batch_features.append(self._get_default_features())

            all_features.extend(batch_features)

            # Checkpoint every 5 batches (5K games)
            if ((batch_start // batch_size) + 1) % 5 == 0:
                self._save_checkpoint(batch_start // batch_size, all_features)

        # Final checkpoint
        self._save_checkpoint(len(games_df) // batch_size, all_features)

        # Convert to DataFrame
        features_df = pd.DataFrame(all_features)

        print("\n" + "="*60)
        print("✓ FEATURE COMPUTATION COMPLETE")
        print("="*60)
        print(f"Total games processed: {len(features_df):,}")
        print(f"Features per game: {len(features_df.columns)}")
        print("\nFeature categories:")
        print("  - Momentum (3 features)")
        print("  - Blowout tendency (4 features)")
        if self.use_player_features:
            print("  - Player-based (4 features)")
        else:
            print("  - Player-based (disabled)")
        print("  - Team-specific HCA (2 features)")
        print(f"\nTotal: {len(features_df.columns)} enhanced features")
        print("="*60)

        return features_df

    def _get_default_features(self) -> Dict[str, float]:
        """
        Get default feature values for error cases.

        Returns:
            Dictionary with default feature values
        """
        return {
            'momentum_diff': 0.0,
            'win_streak_diff': 0.0,
            'recent_win_pct_diff': 0.0,
            'run_diff_differential': 0.0,
            'blowout_tendency_diff': 0.0,
            'consistency_ratio': 1.0,
            'hot_streak_advantage': 0.0,
            'star_power_diff': 0.0,
            'bench_depth_diff': 0.0,
            'offensive_balance_diff': 0.0,
            'star_efficiency_diff': 0.0,
            'home_team_hca': 3.5,
            'away_team_hca': 3.5,
        }


if __name__ == '__main__':
    print("Enhanced Feature Engineering Pipeline")
    print("Use EnhancedFeatureEngine.compute_all_features() with your data")
