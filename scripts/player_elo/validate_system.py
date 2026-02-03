"""
System Validation Script for Player-Based ELO System

Tests all implemented components end-to-end:
1. Data collection (sample)
2. Player ELO system
3. Roster management
4. Feature engineering
5. PyTorch model training
6. End-to-end prediction

Usage:
    python scripts/player_elo/validate_system.py
"""

import sys
from pathlib import Path
import logging
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.player_elo.config import (
    PLAYER_STATS_DIR, MODELS_DIR, OUTPUTS_DIR,
    PLAYER_ELO_CONFIG, PYTORCH_CONFIG
)
from src.player_elo.player_data_collector import PlayerDataCollector
from src.player_elo.player_elo_system import PlayerEloSystem
from src.player_elo.roster_manager import RosterManager
from src.player_elo.features import PlayerFeatureEngine
from src.player_elo.pytorch_model import (
    PlayerELONet, create_data_loaders, train_player_elo_net,
    evaluate_model, predict
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_section(title: str):
    """Print section header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")


def test_data_collector():
    """Test 1: Data Collection (Barttorvik only - faster)"""
    print_section("TEST 1: Data Collection")

    try:
        collector = PlayerDataCollector()

        # Test Barttorvik collection for single year (2024)
        logger.info("Testing Barttorvik player stats collection for 2024...")
        stats = collector.collect_player_stats_barttorvik([2024])

        if not stats.empty:
            print(f"✓ Collected {len(stats)} player records for 2024")
            print(f"✓ Unique players: {stats['player_id'].nunique()}")
            print(f"✓ Unique teams: {stats['team'].nunique()}")
            print(f"\nSample data:")
            print(stats[['player_name', 'team', 'usage_pct', 'offensive_rating']].head(10))
            return True, stats
        else:
            print("✗ No data collected")
            return False, pd.DataFrame()

    except Exception as e:
        logger.error(f"Data collection failed: {e}")
        return False, pd.DataFrame()


def test_player_elo_system(player_stats: pd.DataFrame):
    """Test 2: Player ELO System"""
    print_section("TEST 2: Player ELO System")

    try:
        elo_system = PlayerEloSystem()

        # Create sample lineups from real players
        if not player_stats.empty:
            # Get two teams with most players
            top_teams = player_stats['team'].value_counts().head(2).index.tolist()

            team1 = top_teams[0] if len(top_teams) > 0 else 'Duke'
            team2 = top_teams[1] if len(top_teams) > 1 else 'UNC'

            team1_players = player_stats[player_stats['team'] == team1]['player_id'].head(5).tolist()
            team2_players = player_stats[player_stats['team'] == team2]['player_id'].head(5).tolist()
        else:
            # Use synthetic data
            team1_players = [f"TEAM1_P{i}" for i in range(5)]
            team2_players = [f"TEAM2_P{i}" for i in range(5)]
            team1, team2 = "Team1", "Team2"

        # Set player metadata
        for player_id in team1_players + team2_players:
            elo_system.set_player_metadata(
                player_id,
                usage=20.0,
                minutes=25.0,
                position='G'
            )

        print(f"Testing with {team1} vs {team2}")
        print(f"  {team1} lineup: {len(team1_players)} players")
        print(f"  {team2} lineup: {len(team2_players)} players")

        # Calculate initial team strengths
        team1_elo = elo_system.calculate_team_strength(team1_players)
        team2_elo = elo_system.calculate_team_strength(team2_players)

        print(f"\nInitial ELO ratings:")
        print(f"  {team1}: {team1_elo:.1f}")
        print(f"  {team2}: {team2_elo:.1f}")

        # Predict spread
        spread = elo_system.predict_spread(team1_players, team2_players)
        win_prob = elo_system.predict_win_probability(team1_players, team2_players)

        print(f"\nPredictions (neutral site):")
        print(f"  Spread: {spread:+.1f} (favors {team1 if spread > 0 else team2})")
        print(f"  {team1} win probability: {win_prob:.1%}")

        # Simulate game and update
        print(f"\nSimulating game: {team1} 85, {team2} 78")
        updates = elo_system.update_from_game(
            team1_players, team2_players,
            85, 78,
            neutral=True
        )

        print(f"✓ Updated {len(updates)} player ratings")

        # Show updated team strengths
        team1_elo_new = elo_system.calculate_team_strength(team1_players)
        team2_elo_new = elo_system.calculate_team_strength(team2_players)

        print(f"\nUpdated ELO ratings:")
        print(f"  {team1}: {team1_elo_new:.1f} ({team1_elo_new - team1_elo:+.1f})")
        print(f"  {team2}: {team2_elo_new:.1f} ({team2_elo_new - team2_elo:+.1f})")

        return True, elo_system

    except Exception as e:
        logger.error(f"Player ELO system test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_roster_manager(player_stats: pd.DataFrame):
    """Test 3: Roster Manager"""
    print_section("TEST 3: Roster Manager")

    try:
        manager = RosterManager()

        # Create roster from stats
        if not player_stats.empty:
            manager.create_roster_from_stats(player_stats, 2024, min_games=1)

            # Get summary for a team
            teams = list(manager.rosters.get(2024, {}).keys())
            if teams:
                team = teams[0]
                summary = manager.get_roster_summary(team, 2024)

                print(f"✓ Loaded rosters for {len(teams)} teams")
                print(f"\nSample roster ({team}):")
                print(f"  Size: {summary['roster_size']} players")
                print(f"  Positions: {summary.get('position_breakdown', {})}")

                return True, manager

        print("✓ Roster manager initialized (no data)")
        return True, manager

    except Exception as e:
        logger.error(f"Roster manager test failed: {e}")
        return False, None


def test_feature_engineering(player_stats: pd.DataFrame, elo_system: PlayerEloSystem):
    """Test 4: Feature Engineering"""
    print_section("TEST 4: Feature Engineering")

    try:
        feature_engine = PlayerFeatureEngine(player_stats, elo_system)

        # Create sample lineups
        if not player_stats.empty:
            teams = player_stats['team'].value_counts().head(2).index.tolist()
            home_lineup = player_stats[player_stats['team'] == teams[0]]['player_id'].head(5).tolist()
            away_lineup = player_stats[player_stats['team'] == teams[1]]['player_id'].head(5).tolist()
        else:
            home_lineup = [f"HOME_P{i}" for i in range(5)]
            away_lineup = [f"AWAY_P{i}" for i in range(5)]

        # Create feature vector
        features = feature_engine.create_matchup_features(
            home_lineup,
            away_lineup,
            datetime.now(),
            home_team='Home',
            away_team='Away'
        )

        print(f"✓ Created feature vector")
        print(f"  Shape: {features.shape}")
        print(f"  Expected: (65,)")
        print(f"  Match: {'✓' if features.shape == (65,) else '✗'}")

        print(f"\nFeature breakdown:")
        print(f"  Player features (0-49): {features[:50].mean():.2f} ± {features[:50].std():.2f}")
        print(f"  Lineup features (50-59): {features[50:60].mean():.2f} ± {features[50:60].std():.2f}")
        print(f"  Contextual features (60-64): {features[60:].mean():.2f}")

        return True, feature_engine

    except Exception as e:
        logger.error(f"Feature engineering test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_pytorch_model():
    """Test 5: PyTorch Model Training"""
    print_section("TEST 5: PyTorch Model Training")

    try:
        # Create synthetic training data
        n_samples = 500
        n_features = 65

        X = np.random.randn(n_samples, n_features).astype(np.float32)
        y = (X[:, 0] * 2 + X[:, 1] * 1.5 + np.random.randn(n_samples) * 3).astype(np.float32)

        # Split
        split_idx = int(0.8 * n_samples)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        print(f"Training data: {X_train.shape[0]} samples")
        print(f"Validation data: {X_val.shape[0]} samples")

        # Create model
        model = PlayerELONet()
        print(f"\n✓ Model created: {model.count_parameters():,} parameters")

        # Create data loaders
        train_loader, val_loader = create_data_loaders(
            X_train, y_train, X_val, y_val,
            batch_size=32
        )

        print(f"✓ Data loaders created")

        # Train model (quick test - 5 epochs)
        print(f"\nTraining model (5 epochs)...")
        save_path = MODELS_DIR / "test_model.pt"

        model, history = train_player_elo_net(
            model, train_loader, val_loader,
            epochs=5,
            lr=0.001,
            save_path=save_path
        )

        # Evaluate
        metrics = evaluate_model(model, val_loader)

        print(f"\n✓ Training complete")
        print(f"  Final train loss: {history['train_loss'][-1]:.4f}")
        print(f"  Final val loss: {history['val_loss'][-1]:.4f}")
        print(f"  Val MAE: {metrics['mae']:.4f}")
        print(f"  Val RMSE: {metrics['rmse']:.4f}")

        # Test prediction
        test_sample = X_val[:5]
        predictions = predict(model, test_sample)

        print(f"\nSample predictions:")
        for i, (pred, actual) in enumerate(zip(predictions, y_val[:5])):
            print(f"  Sample {i+1}: pred={pred:+.2f}, actual={actual:+.2f}, error={abs(pred-actual):.2f}")

        return True, model

    except Exception as e:
        logger.error(f"PyTorch model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_end_to_end(player_stats: pd.DataFrame):
    """Test 6: End-to-End Pipeline"""
    print_section("TEST 6: End-to-End Integration")

    try:
        print("Initializing all components...")

        # Initialize components
        elo_system = PlayerEloSystem()
        feature_engine = PlayerFeatureEngine(player_stats, elo_system)
        model = PlayerELONet()

        # Create sample game
        if not player_stats.empty:
            teams = player_stats['team'].value_counts().head(2).index.tolist()
            home_lineup = player_stats[player_stats['team'] == teams[0]]['player_id'].head(5).tolist()
            away_lineup = player_stats[player_stats['team'] == teams[1]]['player_id'].head(5).tolist()
            home_team, away_team = teams[0], teams[1]
        else:
            home_lineup = [f"HOME_P{i}" for i in range(5)]
            away_lineup = [f"AWAY_P{i}" for i in range(5)]
            home_team, away_team = "Home", "Away"

        print(f"\nSimulating prediction: {home_team} vs {away_team}")

        # 1. Calculate ELO-based spread
        elo_spread = elo_system.predict_spread(home_lineup, away_lineup)
        print(f"  1. ELO spread: {elo_spread:+.2f}")

        # 2. Create features
        features = feature_engine.create_matchup_features(
            home_lineup, away_lineup,
            datetime.now(),
            home_team, away_team
        )
        print(f"  2. Features created: shape {features.shape}")

        # 3. Neural network prediction (untrained - random)
        nn_pred = predict(model, features.reshape(1, -1))[0]
        print(f"  3. NN prediction: {nn_pred:+.2f} (untrained - random)")

        print(f"\n✓ End-to-end pipeline working!")
        print(f"  All components integrated successfully")

        return True

    except Exception as e:
        logger.error(f"End-to-end test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all validation tests"""
    print("\n" + "="*70)
    print("  PLAYER-BASED ELO SYSTEM VALIDATION")
    print("  Testing all implemented components")
    print("="*70)

    results = {}

    # Test 1: Data Collection
    success, player_stats = test_data_collector()
    results['Data Collection'] = success

    # Test 2: Player ELO System
    success, elo_system = test_player_elo_system(player_stats)
    results['Player ELO System'] = success

    # Test 3: Roster Manager
    success, roster_mgr = test_roster_manager(player_stats)
    results['Roster Manager'] = success

    # Test 4: Feature Engineering
    if elo_system:
        success, feature_engine = test_feature_engineering(player_stats, elo_system)
        results['Feature Engineering'] = success
    else:
        results['Feature Engineering'] = False

    # Test 5: PyTorch Model
    success, model = test_pytorch_model()
    results['PyTorch Model'] = success

    # Test 6: End-to-End
    success = test_end_to_end(player_stats)
    results['End-to-End Integration'] = success

    # Summary
    print_section("VALIDATION SUMMARY")

    passed = sum(results.values())
    total = len(results)

    print("Test Results:")
    for test_name, passed_test in results.items():
        status = "✓ PASS" if passed_test else "✗ FAIL"
        print(f"  {test_name:.<50} {status}")

    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.0f}%)")

    if passed == total:
        print("\n🎉 All tests passed! System is ready for full implementation.")
    else:
        print(f"\n⚠ {total - passed} test(s) failed. Review errors above.")

    print("\n" + "="*70)

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
