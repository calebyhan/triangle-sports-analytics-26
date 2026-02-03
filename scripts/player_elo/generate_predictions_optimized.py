"""
Generate Optimized 2026 Predictions

Uses all improvements for minimum MAE
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import logging
from src.player_elo.prediction_pipeline_optimized import generate_predictions_optimized

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    """Main execution"""
    print("\n" + "="*70)
    print("  OPTIMIZED PLAYER-BASED ELO PREDICTIONS")
    print("="*70)
    print("\nEnhancements applied:")
    print("  [x] Prediction clipping (+/-30 points)")
    print("  [x] Feature normalization (StandardScaler)")
    print("  [x] Improved lineup prediction (recency + ELO)")
    print("  [x] Ensemble averaging (5 CV models)")
    print("  [x] Confidence-based adjustments")
    print("  [x] Better team name mapping")
    print("\n" + "="*70 + "\n")

    # Generate predictions
    try:
        predictions = generate_predictions_optimized(
            player_stats_year=2025,
            use_ensemble=True,
            team_name="CMMT"
        )

        print("\n[SUCCESS] Optimized predictions generated!")
        print(f"\nPrediction Statistics:")
        print(f"  Total games: {len(predictions)}")
        print(f"  Mean spread: {predictions['pt_spread'].mean():.2f} points")
        print(f"  Std spread: {predictions['pt_spread'].std():.2f} points")
        print(f"  Min spread: {predictions['pt_spread'].min():.2f} points")
        print(f"  Max spread: {predictions['pt_spread'].max():.2f} points")
        print(f"  Mean confidence: {predictions['confidence'].mean():.3f}")

        print(f"\nFirst 10 predictions:")
        display_cols = ['Date', 'Home', 'Away', 'pt_spread', 'confidence']
        print(predictions[display_cols].head(10).to_string(index=False))

        print(f"\nPredictions ready for competition!")
        print(f"File: data/predictions/tsa_pt_spread_PLAYER_ELO_OPTIMIZED_2026.csv")

    except Exception as e:
        print(f"\n[ERROR] Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
