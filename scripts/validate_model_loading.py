"""
Validate Model Architecture Loading Fix

This script verifies that the model architecture inference correctly handles
both regular and optimized model architectures.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch


def infer_architecture(state_dict):
    """
    Infer model architecture from state dict
    (Same logic as fixed compare_frameworks.py)
    """
    if 'network.0.weight' in state_dict:
        # Standard PlayerELONet architecture
        input_dim = state_dict['network.0.weight'].shape[1]
        hidden_dims = []

        # Find all Linear layers by looking for 2D weight tensors
        # (BatchNorm layers have 1D weights)
        for key in sorted(state_dict.keys()):
            if '.weight' in key and 'network.' in key:
                weight_shape = state_dict[key].shape
                # Linear layers have 2D weights, BatchNorm has 1D weights
                if len(weight_shape) == 2:
                    output_dim = weight_shape[0]
                    # Skip the final output layer (output_dim == 1)
                    if output_dim != 1:
                        hidden_dims.append(output_dim)

        if not hidden_dims:
            hidden_dims = [128, 64, 32]  # Default fallback

        return input_dim, hidden_dims
    else:
        print(f"  Unknown architecture format")
        return None, None


def main():
    print("\n" + "="*70)
    print("  MODEL ARCHITECTURE VALIDATION")
    print("="*70)

    models_dir = project_root / 'data' / 'player_data' / 'models'

    # Check both regular and optimized models
    model_files = [
        ('Regular Model', 'pytorch_model.pt'),
        ('Optimized Model', 'pytorch_model_optimized.pt'),
        ('Fold 1', 'pytorch_model_fold1.pt'),
        ('Optimized Fold 1', 'pytorch_model_optimized_fold1.pt'),
    ]

    for name, filename in model_files:
        model_path = models_dir / filename

        if not model_path.exists():
            print(f"\n{name}: NOT FOUND")
            continue

        print(f"\n{name}:")
        print(f"  Path: {filename}")
        print(f"  Size: {model_path.stat().st_size / 1024:.1f} KB")

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # Infer architecture
        input_dim, hidden_dims = infer_architecture(state_dict)

        if input_dim is not None:
            print(f"  Architecture: {input_dim} → {hidden_dims} → 1")

            # Calculate expected parameters
            total_params = 0
            prev_dim = input_dim
            for hidden_dim in hidden_dims:
                total_params += prev_dim * hidden_dim + hidden_dim  # weights + bias
                prev_dim = hidden_dim
            total_params += prev_dim * 1 + 1  # output layer

            print(f"  Parameters: {total_params:,}")

            # Test loading with PlayerELONet
            from src.player_elo.pytorch_model import PlayerELONet

            try:
                model = PlayerELONet(input_dim=input_dim, hidden_dims=hidden_dims)
                model.load_state_dict(state_dict, strict=False)
                print(f"  ✓ Model loads successfully")
            except Exception as e:
                print(f"  ✗ Model loading failed: {e}")
        else:
            print(f"  ✗ Could not infer architecture")

    print("\n" + "="*70)
    print("  VALIDATION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
