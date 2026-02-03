# SLURM Scripts for Triangle Sports Analytics

These SLURM scripts are designed for running the prediction pipeline on GPU clusters (e.g., university HPC systems).

## Available Scripts

### 1. `train_hybrid.slurm` - Train Hybrid Model Only
Trains the hybrid player-team model (fastest single model).

**Resources**: 1 GPU, 8 CPUs, 32GB RAM, ~2 hours

**Usage**:
```bash
sbatch scripts/slurm/train_hybrid.slurm
```

**Outputs**:
- `data/player_data/models/hybrid_model.pt`
- `data/player_data/models/hybrid_model_fold{1-5}.pt`
- `data/player_data/models/hybrid_scaler.pkl`

---

### 2. `optimize_hybrid.slurm` - Hyperparameter Optimization
Runs grid search over 72 hyperparameter configurations to find optimal settings.

**Resources**: 1 GPU, 8 CPUs, 32GB RAM, ~12 hours

**Usage**:
```bash
# First, ensure training cache exists
sbatch scripts/slurm/train_hybrid.slurm

# Then run optimization
sbatch scripts/slurm/optimize_hybrid.slurm
```

**Outputs**:
- `data/player_data/models/hybrid_model_optimized.pt`
- `data/player_data/models/hybrid_scaler_optimized.pkl`
- `data/player_data/models/hyperparameter_results.csv`

---

### 3. `complete_pipeline.slurm` - Full Pipeline
Runs the entire prediction pipeline from start to finish:
1. Train player-based model
2. Generate player predictions
3. Train player-based optimized
4. Generate optimized predictions
5. Train hybrid model
6. Generate hybrid predictions
7. Compare all frameworks
8. Create ensemble

**Resources**: 1 GPU, 8 CPUs, 64GB RAM, ~6 hours

**Usage**:
```bash
sbatch scripts/slurm/complete_pipeline.slurm
```

**Outputs**: All prediction files for all 4 systems

---

## Setup Instructions

### 1. Before First Run

Create the logs directory:
```bash
mkdir -p logs/slurm
mkdir -p logs/pipeline
```

### 2. Configure Python Environment

Edit the SLURM scripts to activate your environment. Choose one:

**Option A: Conda**
```bash
# Uncomment in SLURM script:
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sports-analytics
```

**Option B: Virtualenv**
```bash
# Uncomment in SLURM script:
source venv/bin/activate
```

**Option C: Module System**
```bash
# Add to SLURM script:
module load python/3.11
module load cuda/12.1
```

### 3. Install Dependencies

Ensure PyTorch with CUDA support is installed:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

---

## Monitoring Jobs

### Check job status
```bash
squeue -u $USER
```

### View live output
```bash
tail -f logs/slurm/hybrid_train_JOBID.log
```

### Check GPU usage
```bash
ssh NODE
nvidia-smi
```

### Cancel a job
```bash
scancel JOBID
```

---

## Resource Requirements

| Script | GPUs | CPUs | Memory | Time | Priority |
|--------|------|------|--------|------|----------|
| `train_hybrid.slurm` | 1 | 8 | 32GB | 2h | High |
| `optimize_hybrid.slurm` | 1 | 8 | 32GB | 12h | Medium |
| `complete_pipeline.slurm` | 1 | 8 | 64GB | 6h | Low |

---

## Troubleshooting

### "Out of Memory" Error

Reduce batch size in training scripts:
```python
# In train_hybrid_model.py
batch_size = 32  # Instead of 64
```

### "CUDA Out of Memory"

Use smaller hidden dimensions:
```python
# In hybrid_model.py
hidden_dims = [96, 48, 24]  # Instead of [128, 64, 32]
```

### "Module not found"

Ensure PYTHONPATH is set:
```bash
export PYTHONPATH=/path/to/triangle-sports-analytics-26:$PYTHONPATH
```

### Job Stuck in Queue

Check partition availability:
```bash
sinfo -p l40-gpu
```

Or try a different partition:
```bash
#SBATCH --partition=gpu  # Instead of l40-gpu
```

---

## Expected Results

### After `train_hybrid.slurm`
```
Hybrid Model Performance:
  MAE: 10.78 ± 1.25
  RMSE: 13.21
  Direction Accuracy: 61.02%
```

### After `optimize_hybrid.slurm`
```
Optimized Performance:
  MAE: 10.5-10.8 (depends on best config found)
  Expected improvement: 0.2-0.5 points
```

### After `complete_pipeline.slurm`
```
All Systems Ready:
  ✓ Team-based predictions
  ✓ Player-based predictions (2 variants)
  ✓ Hybrid predictions (RECOMMENDED)
  ✓ Ensemble predictions
  ✓ Comparison analysis
```

---

## Quick Start

**For immediate results** (fastest):
```bash
sbatch scripts/slurm/train_hybrid.slurm
```

**For best accuracy** (slower):
```bash
sbatch scripts/slurm/optimize_hybrid.slurm
```

**For complete analysis** (slowest):
```bash
sbatch scripts/slurm/complete_pipeline.slurm
```

---

## Output Files

After successful run, you'll have:

**Predictions** (ready for competition):
- `data/predictions/tsa_pt_spread_HYBRID_2026.csv` ⭐ **RECOMMENDED**
- `data/predictions/tsa_pt_spread_ENSEMBLE_2026.csv`
- `data/predictions/tsa_pt_spread_PLAYER_ELO_OPTIMIZED_2026.csv`
- `data/predictions/tsa_pt_spread_CMMT_2026.csv`

**Models**:
- `data/player_data/models/hybrid_model.pt`
- `data/player_data/models/*.pkl` (scalers)

**Logs**:
- `logs/slurm/hybrid_train_JOBID.log`
- `logs/pipeline/run_TIMESTAMP.log`

---

## Notes

1. **GPU Requirement**: All scripts require GPU access for PyTorch training
2. **Memory**: 32GB minimum, 64GB recommended for full pipeline
3. **Time Estimates**: Based on L40 GPU, may vary by system
4. **Data**: Ensure `data/raw_pd/` contains player statistics files
5. **Submission**: Use hybrid model predictions for best results

---

## Support

If you encounter issues:
1. Check logs in `logs/slurm/`
2. Verify GPU access: `python -c "import torch; print(torch.cuda.is_available())"`
3. Ensure all dependencies installed: `pip list | grep torch`
4. Review README.md in project root

---

**Last Updated**: 2026-02-03
**Tested On**: L40 GPU cluster with SLURM 23.02
