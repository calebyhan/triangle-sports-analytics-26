"""
Training Pipeline for Player-Based ELO System

Orchestrates the complete training workflow:
1. Load historical games (2020-2025)
2. Collect player statistics from Barttorvik
3. Process games chronologically through player ELO system
4. Create training features (prevent data leakage)
5. Train PyTorch neural network
6. Validate and save model

Usage:
    from src.player_elo.training_pipeline import train_player_model
    train_player_model(years=[2020, 2021, 2022, 2023, 2024, 2025])
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import logging
from sklearn.model_selection import TimeSeriesSplit

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from .config import (
    PROJECT_ROOT, TRAINING_YEARS, MODELS_DIR, OUTPUTS_DIR,
    PLAYER_ELO_CONFIG, PYTORCH_CONFIG, VALIDATION_CONFIG,
    OUTPUT_CONFIG
)
from .player_data_collector import PlayerDataCollector
from .roster_manager import RosterManager
from .player_elo_system import PlayerEloSystem
from .features import PlayerFeatureEngine
from .pytorch_model import (
    PlayerELONet, create_data_loaders, train_player_elo_net,
    evaluate_model
)

# Set up logging
logger = logging.getLogger(__name__)


class PlayerModelTrainer:
    """
    Orchestrates complete player-based model training pipeline
    """

    def __init__(self, training_years: List[int] = None):
        """
        Initialize trainer

        Args:
            training_years: Years to include in training (defaults to config)
        """
        self.training_years = training_years or TRAINING_YEARS
        self.collector = PlayerDataCollector()
        self.roster_manager = RosterManager()
        self.elo_system = PlayerEloSystem()
        self.feature_engine = None  # Created after player stats loaded
        self.model = None

        logger.info(f"PlayerModelTrainer initialized for years: {self.training_years}")

    # ========================================================================
    # STEP 1: DATA COLLECTION
    # ========================================================================

    def collect_data(self, use_cached: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Collect historical games and player statistics

        Args:
            use_cached: If True, use existing data files; if False, re-download

        Returns:
            (games_df, player_stats_df)
        """
        logger.info("="*70)
        logger.info("STEP 1: DATA COLLECTION")
        logger.info("="*70)

        # Load historical games from existing team-based system
        games_file = PROJECT_ROOT / "data" / "raw" / "games" / "historical_games_2019_2025.csv"

        if games_file.exists():
            logger.info(f"Loading historical games from: {games_file}")
            games_df = pd.read_csv(games_file)
            logger.info(f"  ✓ Loaded {len(games_df)} games")
        else:
            logger.error(f"Historical games file not found: {games_file}")
            raise FileNotFoundError(f"Cannot find historical games at {games_file}")

        # Collect player statistics
        logger.info("\nCollecting player statistics from local files...")

        if use_cached:
            # Try to load cached data
            cached_file = self.collector.player_stats_dir / f"barttorvik_stats_{min(self.training_years)}_{max(self.training_years)}.csv"
            if cached_file.exists():
                logger.info(f"  Using cached player stats: {cached_file}")
                player_stats_df = pd.read_csv(cached_file)
                logger.info(f"  [OK] Loaded {len(player_stats_df)} player records")
            else:
                logger.info(f"  No cached data found, loading from local CSV files...")
                player_stats_df = self.collector.collect_player_stats_from_local(self.training_years)
        else:
            logger.info(f"  Loading fresh data from local CSV files...")
            player_stats_df = self.collector.collect_player_stats_from_local(self.training_years)

        return games_df, player_stats_df

    # ========================================================================
    # STEP 2: PREPARE ROSTERS
    # ========================================================================

    def prepare_rosters(self, player_stats_df: pd.DataFrame):
        """
        Create rosters from player statistics

        Args:
            player_stats_df: Player statistics DataFrame
        """
        logger.info("\n" + "="*70)
        logger.info("STEP 2: ROSTER PREPARATION")
        logger.info("="*70)

        # Check if we have player data
        if player_stats_df.empty or 'season' not in player_stats_df.columns:
            logger.warning("  ⚠ No player statistics available - skipping roster preparation")
            logger.warning("  ⚠ Training will fail without player data")
            logger.warning("  ⚠ Please check Barttorvik data collection or use sample data")
            return

        for year in self.training_years:
            year_stats = player_stats_df[player_stats_df['season'] == year]

            if len(year_stats) > 0:
                self.roster_manager.create_roster_from_stats(year_stats, year, min_games=5)
                logger.info(f"  ✓ Created roster for {year}: {len(self.roster_manager.rosters.get(year, {}))} teams")

    # ========================================================================
    # STEP 3: PROCESS GAMES THROUGH ELO SYSTEM
    # ========================================================================

    def process_games(
        self,
        games_df: pd.DataFrame,
        player_stats_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Process games chronologically through player ELO system

        Args:
            games_df: Historical games
            player_stats_df: Player statistics

        Returns:
            DataFrame with game data and ELO snapshots
        """
        logger.info("\n" + "="*70)
        logger.info("STEP 3: PROCESS GAMES THROUGH PLAYER ELO SYSTEM")
        logger.info("="*70)

        # Filter to training years
        games_df['season'] = pd.to_datetime(games_df['date']).dt.year
        games_df = games_df[games_df['season'].isin(self.training_years)]

        # Sort chronologically (critical for ELO)
        games_df = games_df.sort_values('date').reset_index(drop=True)

        logger.info(f"Processing {len(games_df)} games chronologically...")

        # Set up player metadata from stats
        for _, player in player_stats_df.iterrows():
            self.elo_system.set_player_metadata(
                player['player_id'],
                usage=player.get('usage_pct', 20.0),
                minutes=player.get('minutes_per_game', 20.0),
                position=player.get('position', 'Unknown'),
                team=player.get('team')
            )

        # Process each game
        game_records = []
        games_processed = 0

        for idx, game in games_df.iterrows():
            # Get lineups (use top 5 players by minutes from each team)
            home_lineup = self._get_lineup_for_team(
                game['home_team'],
                game['season'],
                player_stats_df
            )
            away_lineup = self._get_lineup_for_team(
                game['away_team'],
                game['season'],
                player_stats_df
            )

            if not home_lineup or not away_lineup:
                continue  # Skip if can't get lineups

            # Get pre-game team strengths
            home_elo_before = self.elo_system.calculate_team_strength(home_lineup)
            away_elo_before = self.elo_system.calculate_team_strength(away_lineup)

            # Predict spread
            predicted_spread = self.elo_system.predict_spread(home_lineup, away_lineup)

            # Update ELO system
            self.elo_system.update_from_game(
                home_lineup,
                away_lineup,
                game['home_score'],
                game['away_score'],
                neutral=game.get('neutral', False)
            )

            # Record game with ELO snapshots
            game_records.append({
                'game_id': idx,
                'date': game['date'],
                'season': game['season'],
                'home_team': game['home_team'],
                'away_team': game['away_team'],
                'home_score': game['home_score'],
                'away_score': game['away_score'],
                'actual_margin': game['home_score'] - game['away_score'],
                'home_elo_before': home_elo_before,
                'away_elo_before': away_elo_before,
                'predicted_spread': predicted_spread,
                'home_lineup': ','.join(home_lineup),
                'away_lineup': ','.join(away_lineup),
                'neutral': game.get('neutral', False),
                'conference_game': self._is_conference_game(game['home_team'], game['away_team']),
            })

            games_processed += 1

            if games_processed % 1000 == 0:
                logger.info(f"  Processed {games_processed} games...")

        logger.info(f"  ✓ Processed {games_processed} games with lineups")

        return pd.DataFrame(game_records)

    def _get_lineup_for_team(
        self,
        team: str,
        season: int,
        player_stats_df: pd.DataFrame
    ) -> List[str]:
        """
        Get lineup for a team (top 5 by minutes)

        Args:
            team: Team name
            season: Season year
            player_stats_df: Player statistics

        Returns:
            List of 5 player IDs
        """
        # Get team's players for this season
        team_players = player_stats_df[
            (player_stats_df['team'] == team) &
            (player_stats_df['season'] == season)
        ].copy()

        if len(team_players) == 0:
            return []

        # Sort by minutes and take top 5
        team_players = team_players.sort_values('minutes_per_game', ascending=False)
        lineup = team_players.head(5)['player_id'].tolist()

        return lineup if len(lineup) == 5 else []

    def _is_conference_game(self, team1: str, team2: str) -> bool:
        """Check if game is conference matchup"""
        conf1 = self.elo_system.team_conference.get(team1)
        conf2 = self.elo_system.team_conference.get(team2)
        return conf1 == conf2 if conf1 and conf2 else False

    # ========================================================================
    # STEP 4: CREATE FEATURES
    # ========================================================================

    def create_features(
        self,
        game_records_df: pd.DataFrame,
        player_stats_df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create feature matrix from game records

        Args:
            game_records_df: Processed game records with ELO snapshots
            player_stats_df: Player statistics

        Returns:
            (X, y) feature matrix and targets
        """
        logger.info("\n" + "="*70)
        logger.info("STEP 4: FEATURE ENGINEERING")
        logger.info("="*70)

        # Initialize feature engine
        self.feature_engine = PlayerFeatureEngine(player_stats_df, self.elo_system)

        X_list = []
        y_list = []

        logger.info(f"Creating features for {len(game_records_df)} games...")

        for idx, game in game_records_df.iterrows():
            # Parse lineups
            home_lineup = game['home_lineup'].split(',')
            away_lineup = game['away_lineup'].split(',')

            # Create features
            try:
                features = self.feature_engine.create_matchup_features(
                    home_lineup,
                    away_lineup,
                    pd.to_datetime(game['date']),
                    game['home_team'],
                    game['away_team'],
                    game['neutral'],
                    game['conference_game'],
                    game_records_df  # For rest days calculation
                )

                X_list.append(features)
                y_list.append(game['actual_margin'])

            except Exception as e:
                logger.warning(f"  Failed to create features for game {idx}: {e}")
                continue

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)

        logger.info(f"  ✓ Created features: X shape {X.shape}, y shape {y.shape}")

        return X, y

    # ========================================================================
    # STEP 5: TRAIN MODEL
    # ========================================================================

    def train_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int = None
    ) -> Dict:
        """
        Train PyTorch model with cross-validation

        Args:
            X: Feature matrix
            y: Target vector
            n_splits: Number of CV splits (defaults to config)

        Returns:
            Training results dictionary
        """
        logger.info("\n" + "="*70)
        logger.info("STEP 5: TRAIN PYTORCH MODEL")
        logger.info("="*70)

        if n_splits is None:
            n_splits = VALIDATION_CONFIG['n_cv_splits']

        # Time-series cross-validation
        tscv = TimeSeriesSplit(n_splits=n_splits)

        cv_results = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            logger.info(f"\nFold {fold + 1}/{n_splits}")
            logger.info(f"  Train: {len(train_idx)} samples, Val: {len(val_idx)} samples")

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Create data loaders
            train_loader, val_loader = create_data_loaders(
                X_train, y_train, X_val, y_val
            )

            # Create model
            model = PlayerELONet()

            # Train
            save_path = MODELS_DIR / f"pytorch_model_fold{fold+1}.pt"
            model, history = train_player_elo_net(
                model, train_loader, val_loader,
                save_path=save_path
            )

            # Evaluate
            metrics = evaluate_model(model, val_loader)

            cv_results.append({
                'fold': fold + 1,
                'train_samples': len(train_idx),
                'val_samples': len(val_idx),
                'val_mae': metrics['mae'],
                'val_rmse': metrics['rmse'],
                'direction_accuracy': metrics['direction_accuracy']
            })

            logger.info(f"  Fold {fold+1} Results:")
            logger.info(f"    MAE: {metrics['mae']:.4f}")
            logger.info(f"    RMSE: {metrics['rmse']:.4f}")
            logger.info(f"    Direction Accuracy: {metrics['direction_accuracy']:.2%}")

        # Summary
        cv_df = pd.DataFrame(cv_results)
        logger.info("\n" + "="*70)
        logger.info("CROSS-VALIDATION SUMMARY")
        logger.info("="*70)
        logger.info(f"  Mean MAE: {cv_df['val_mae'].mean():.4f} ± {cv_df['val_mae'].std():.4f}")
        logger.info(f"  Mean RMSE: {cv_df['val_rmse'].mean():.4f} ± {cv_df['val_rmse'].std():.4f}")
        logger.info(f"  Mean Direction Acc: {cv_df['direction_accuracy'].mean():.2%}")

        # Train final model on all data
        logger.info("\nTraining final model on all data...")
        self.model = PlayerELONet()
        train_loader, val_loader = create_data_loaders(
            X[:-len(X)//5], y[:-len(y)//5],  # 80% train
            X[-len(X)//5:], y[-len(y)//5:]   # 20% val
        )

        save_path = OUTPUT_CONFIG['model_file']
        self.model, history = train_player_elo_net(
            self.model, train_loader, val_loader,
            save_path=save_path
        )

        logger.info(f"  ✓ Final model saved to: {save_path}")

        return {
            'cv_results': cv_df,
            'final_model': self.model,
            'mean_mae': cv_df['val_mae'].mean(),
            'std_mae': cv_df['val_mae'].std()
        }

    # ========================================================================
    # STEP 6: SAVE ARTIFACTS
    # ========================================================================

    def save_artifacts(self):
        """Save trained model and ELO state"""
        logger.info("\n" + "="*70)
        logger.info("STEP 6: SAVE ARTIFACTS")
        logger.info("="*70)

        # Save ELO state
        elo_state_file = OUTPUT_CONFIG['elo_state_file']
        self.elo_system.save_state(elo_state_file)
        logger.info(f"  ✓ Saved ELO state to: {elo_state_file}")

        # Save roster data
        self.roster_manager.save_transfers()
        logger.info(f"  ✓ Saved transfer data")

        logger.info("\n  All artifacts saved successfully!")


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_player_model(
    years: List[int] = None,
    use_cached_data: bool = True,
    n_cv_splits: int = 5
) -> Dict:
    """
    Main training function - orchestrates complete pipeline

    Args:
        years: Training years (defaults to config)
        use_cached_data: Use cached player stats if available
        n_cv_splits: Number of cross-validation splits

    Returns:
        Training results dictionary
    """
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    logger.info("\n" + "="*70)
    logger.info("  PLAYER-BASED ELO MODEL TRAINING PIPELINE")
    logger.info("="*70 + "\n")

    # Initialize trainer
    trainer = PlayerModelTrainer(years)

    # Step 1: Collect data
    games_df, player_stats_df = trainer.collect_data(use_cached=use_cached_data)

    # Check if we have player data
    if player_stats_df.empty:
        logger.error("\n" + "="*70)
        logger.error("  TRAINING FAILED: No Player Data Available")
        logger.error("="*70)
        logger.error("\nBarttorvik data collection failed.")
        logger.error("\nPossible solutions:")
        logger.error("1. Check internet connection")
        logger.error("2. Barttorvik URL may have changed")
        logger.error("3. Try manual data collection:")
        logger.error("   - Visit https://barttorvik.com/playerstat.php?year=2024&csv=1")
        logger.error("   - Save as CSV and place in data/player_data/raw/player_stats/")
        logger.error("\nFor now, you can use the team-based system instead:")
        logger.error("   python scripts/train_model.py")
        raise ValueError("No player statistics available. Cannot train player-based model.")

    # Step 2: Prepare rosters
    trainer.prepare_rosters(player_stats_df)

    # Step 3: Process games through ELO
    game_records_df = trainer.process_games(games_df, player_stats_df)

    # Step 4: Create features
    X, y = trainer.create_features(game_records_df, player_stats_df)

    # Step 5: Train model
    results = trainer.train_model(X, y, n_splits=n_cv_splits)

    # Step 6: Save artifacts
    trainer.save_artifacts()

    logger.info("\n" + "="*70)
    logger.info("  TRAINING COMPLETE!")
    logger.info("="*70)
    logger.info(f"\n  Final Performance:")
    logger.info(f"    MAE: {results['mean_mae']:.4f} ± {results['std_mae']:.4f}")
    logger.info(f"    Saved model: {OUTPUT_CONFIG['model_file']}")
    logger.info(f"    Saved ELO state: {OUTPUT_CONFIG['elo_state_file']}")
    logger.info("\n" + "="*70 + "\n")

    return results


# ============================================================================
# STANDALONE EXECUTION
# ============================================================================

if __name__ == "__main__":
    results = train_player_model()
