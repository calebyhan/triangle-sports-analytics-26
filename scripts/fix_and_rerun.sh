#!/bin/bash
# Fix and Re-run Framework Comparison and Ensemble
#
# This script re-runs only the steps that need fixing:
# 1. Framework comparison (with fixed model loading)
# 2. Ensemble creation (with corrected predictions)

set -e

echo "=========================================================================="
echo "  FIXING PIPELINE ISSUES"
echo "=========================================================================="
echo ""
echo "Issues being fixed:"
echo "  1. Model architecture mismatch in framework comparison"
echo "  2. Ensemble using incorrect player predictions"
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Check if we're in SLURM environment
if [ -n "$SLURM_JOB_ID" ]; then
    echo "Running in SLURM environment"
    # Activate conda environment
    module load anaconda 2>/dev/null || true
    eval "$(conda shell.bash hook)"
    conda activate "$PROJECT_ROOT/.venv"
fi

# Set Python path
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

echo "=========================================================================="
echo "  STEP 1: Validate Model Loading Fix"
echo "=========================================================================="
echo ""
python scripts/validate_model_loading.py

echo ""
echo "=========================================================================="
echo "  STEP 2: Re-run Framework Comparison"
echo "=========================================================================="
echo ""
python scripts/compare_frameworks.py

echo ""
echo "=========================================================================="
echo "  STEP 3: Regenerate Ensemble"
echo "=========================================================================="
echo ""
python scripts/create_ensemble.py

echo ""
echo "=========================================================================="
echo "  FIX COMPLETE"
echo "=========================================================================="
echo ""
echo "Updated files:"
echo "  - data/test_data/framework_comparison_results.csv"
echo "  - data/predictions/tsa_pt_spread_ENSEMBLE_2026.csv"
echo "  - data/predictions/tsa_pt_spread_ENSEMBLE_2026_detailed.csv"
echo ""
echo "Next steps:"
echo "  1. Review the new framework comparison results"
echo "  2. Check if player-based direction accuracy improved (was 43.26%)"
echo "  3. Verify ensemble weights are more reasonable"
echo ""
