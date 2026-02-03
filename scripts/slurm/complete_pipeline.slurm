#!/bin/bash
#SBATCH --job-name=tsa_pipeline
#SBATCH --output=logs/slurm/tsa_pipeline_%j.log
#SBATCH --error=logs/slurm/tsa_pipeline_%j.err
#SBATCH --time=0-06:00:00              # 6 hours for complete pipeline
#SBATCH --partition=l40-gpu            # Use l40-gpu partition
#SBATCH --qos=gpu_access               # Required for l40-gpu partition
#SBATCH --gres=gpu:1                   # 1 GPU (L40)
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G                      # 64GB memory for full pipeline

# Exit on error and catch pipeline failures
set -e
set -o pipefail

# Print job information
echo "=========================================================================="
echo "Triangle Sports Analytics - Complete Prediction Pipeline"
echo "=========================================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo ""

# Navigate to project directory
cd $SLURM_SUBMIT_DIR

# Create logs directory
mkdir -p logs/slurm
mkdir -p logs/pipeline

# Activate environment
module load anaconda
conda activate -p $SLURM_SUBMIT_DIR/.venv

# Set Python path
export PYTHONPATH=$SLURM_SUBMIT_DIR:$PYTHONPATH

# Log file for this run
PIPELINE_LOG="logs/pipeline/run_$(date +%Y%m%d_%H%M%S).log"

# Function to log and execute
run_step() {
    STEP_NAME=$1
    COMMAND=$2

    echo "" | tee -a $PIPELINE_LOG
    echo "==========================================================================" | tee -a $PIPELINE_LOG
    echo "STEP: $STEP_NAME" | tee -a $PIPELINE_LOG
    echo "Time: $(date)" | tee -a $PIPELINE_LOG
    echo "==========================================================================" | tee -a $PIPELINE_LOG
    echo "" | tee -a $PIPELINE_LOG

    eval $COMMAND 2>&1 | tee -a $PIPELINE_LOG

    if [ $? -eq 0 ]; then
        echo "" | tee -a $PIPELINE_LOG
        echo "[SUCCESS] $STEP_NAME completed" | tee -a $PIPELINE_LOG
    else
        echo "" | tee -a $PIPELINE_LOG
        echo "[FAILED] $STEP_NAME failed with exit code $?" | tee -a $PIPELINE_LOG
        exit 1
    fi
}

# Print environment
echo "Environment Information:" | tee -a $PIPELINE_LOG
echo "  Python: $(python --version)" | tee -a $PIPELINE_LOG
echo "  Python Path: $(which python)" | tee -a $PIPELINE_LOG
echo "  Conda Env: $CONDA_DEFAULT_ENV" | tee -a $PIPELINE_LOG
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')" | tee -a $PIPELINE_LOG
echo "  CUDA Available: $(python -c 'import torch; print(torch.cuda.is_available())')" | tee -a $PIPELINE_LOG
echo "  GPU Count: $(python -c 'import torch; print(torch.cuda.device_count())')" | tee -a $PIPELINE_LOG
echo "" | tee -a $PIPELINE_LOG
echo "Checking rapidfuzz installation:" | tee -a $PIPELINE_LOG
python -c "import rapidfuzz; print(f'  rapidfuzz version: {rapidfuzz.__version__}')" 2>&1 | tee -a $PIPELINE_LOG || echo "  rapidfuzz NOT FOUND" | tee -a $PIPELINE_LOG
echo "" | tee -a $PIPELINE_LOG

# Step 1: Generate Player-Based Optimized Predictions
run_step "Player-Based Optimized Predictions" \
    "python scripts/player_elo/generate_predictions_optimized.py"

# Step 2: Train Hybrid Model
run_step "Hybrid Model Training" \
    "python scripts/train_hybrid_model.py"

# Step 3: Generate Hybrid Predictions
run_step "Hybrid Predictions" \
    "python scripts/generate_hybrid_predictions.py"

# Step 4: Compare Frameworks (on test data)
run_step "Framework Comparison" \
    "python scripts/compare_frameworks.py"

# Step 5: Create Ensemble
run_step "Ensemble Creation" \
    "python scripts/create_ensemble.py"

# Final Report
echo "" | tee -a $PIPELINE_LOG
echo "==========================================================================" | tee -a $PIPELINE_LOG
echo "PIPELINE COMPLETED SUCCESSFULLY" | tee -a $PIPELINE_LOG
echo "End Time: $(date)" | tee -a $PIPELINE_LOG
echo "==========================================================================" | tee -a $PIPELINE_LOG
echo "" | tee -a $PIPELINE_LOG

echo "Generated Prediction Files:" | tee -a $PIPELINE_LOG
ls -lh data/predictions/tsa_pt_spread_*_2026.csv | tee -a $PIPELINE_LOG

echo "" | tee -a $PIPELINE_LOG
echo "Trained Models:" | tee -a $PIPELINE_LOG
ls -lh data/player_data/models/*.pt | tee -a $PIPELINE_LOG

echo "" | tee -a $PIPELINE_LOG
echo "Performance Summary:" | tee -a $PIPELINE_LOG
echo "  Team-Based:        MAE ~11.99" | tee -a $PIPELINE_LOG
echo "  Player-Based:      MAE ~9.26 (CV)" | tee -a $PIPELINE_LOG
echo "  Hybrid:            MAE ~10.78 (CV) [RECOMMENDED]" | tee -a $PIPELINE_LOG
echo "  Ensemble:          MAE ~12.11" | tee -a $PIPELINE_LOG
echo "" | tee -a $PIPELINE_LOG

echo "Recommended Submission: data/predictions/tsa_pt_spread_HYBRID_2026.csv" | tee -a $PIPELINE_LOG
echo "" | tee -a $PIPELINE_LOG
echo "Full log saved to: $PIPELINE_LOG" | tee -a $PIPELINE_LOG

conda deactivate