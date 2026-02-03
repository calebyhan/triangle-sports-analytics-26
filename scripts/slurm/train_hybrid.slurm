#!/bin/bash
#SBATCH --job-name=hybrid_train
#SBATCH --output=logs/slurm/hybrid_train_%j.log
#SBATCH --error=logs/slurm/hybrid_train_%j.err
#SBATCH --time=0-02:00:00              # 2 hours (hybrid model training)
#SBATCH --partition=l40-gpu            # Use l40-gpu partition
#SBATCH --qos=gpu_access               # Required for l40-gpu partition
#SBATCH --gres=gpu:1                   # 1 GPU (L40)
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G                      # 32GB memory for training

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $SLURM_SUBMIT_DIR"

# Navigate to project directory
cd $SLURM_SUBMIT_DIR

# Create logs directory if it doesn't exist
mkdir -p logs/slurm

# Activate conda environment (adjust environment name as needed)
# Uncomment and modify if using conda:
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate sports-analytics

# Or activate virtualenv:
# source venv/bin/activate

# Set Python path
export PYTHONPATH=$SLURM_SUBMIT_DIR:$PYTHONPATH

# Print environment info
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"

# Train hybrid model
echo ""
echo "="*70
echo "Starting Hybrid Model Training"
echo "="*70
echo ""

python scripts/train_hybrid_model.py \
    2>&1 | tee logs/hybrid_training_$(date +%Y%m%d_%H%M%S).log

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "="*70
    echo "Training completed successfully!"
    echo "="*70
    echo "End Time: $(date)"

    # List generated model files
    echo ""
    echo "Generated model files:"
    ls -lh data/player_data/models/hybrid_model*.pt
    ls -lh data/player_data/models/hybrid_scaler*.pkl

else
    echo ""
    echo "="*70
    echo "Training failed with exit code $?"
    echo "="*70
    echo "End Time: $(date)"
    exit 1
fi
