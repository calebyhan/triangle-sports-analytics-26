#!/bin/bash
#SBATCH --job-name=hybrid_optimize
#SBATCH --output=logs/slurm/hybrid_optimize_%j.log
#SBATCH --error=logs/slurm/hybrid_optimize_%j.err
#SBATCH --time=0-12:00:00              # 12 hours (hyperparameter search)
#SBATCH --partition=l40-gpu            # Use l40-gpu partition
#SBATCH --qos=gpu_access               # Required for l40-gpu partition
#SBATCH --gres=gpu:1                   # 1 GPU (L40)
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G                      # 32GB memory

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"

# Navigate to project directory
cd $SLURM_SUBMIT_DIR

# Create logs directory
mkdir -p logs/slurm

# Activate environment (adjust as needed)
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate sports-analytics

# Set Python path
export PYTHONPATH=$SLURM_SUBMIT_DIR:$PYTHONPATH

# Print environment info
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Run hyperparameter optimization
echo ""
echo "="*70
echo "Starting Hyperparameter Optimization"
echo "="*70
echo ""

python scripts/optimize_hybrid.py \
    2>&1 | tee logs/optimization_$(date +%Y%m%d_%H%M%S).log

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "Optimization completed successfully!"
    echo "End Time: $(date)"

    # Show results
    echo ""
    echo "Best configuration saved to:"
    ls -lh data/player_data/models/hybrid_model_optimized.pt
    ls -lh data/player_data/models/hyperparameter_results.csv

else
    echo "Optimization failed with exit code $?"
    echo "End Time: $(date)"
    exit 1
fi
