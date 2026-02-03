"""
Inspect Model Structure - Deep Dive

This script inspects the actual layer structure of the optimized model
to understand why architecture inference is failing.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch


def inspect_model(model_path, model_name):
    """Inspect a model's structure in detail"""
    print(f"\n{'='*70}")
    print(f"  {model_name}")
    print('='*70)

    checkpoint = torch.load(model_path, map_location='cpu')

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        print("Checkpoint is a dict with 'model_state_dict' key")
        state_dict = checkpoint['model_state_dict']
        print(f"Other keys: {[k for k in checkpoint.keys() if k != 'model_state_dict']}")
    else:
        print("Checkpoint is raw state dict")
        state_dict = checkpoint

    print(f"\nAll state dict keys:")
    for i, key in enumerate(sorted(state_dict.keys())):
        shape = state_dict[key].shape
        print(f"  {i:2d}. {key:40s} {str(shape):20s}")

    # Analyze structure
    print(f"\nArchitecture Analysis:")

    # Find all Linear layers
    linear_layers = {}
    for key in state_dict.keys():
        if '.weight' in key and 'network.' in key:
            layer_num = key.split('.')[1]
            if layer_num not in linear_layers:
                linear_layers[layer_num] = {}
            linear_layers[layer_num]['weight'] = state_dict[key].shape
        if '.bias' in key and 'network.' in key:
            layer_num = key.split('.')[1]
            if layer_num not in linear_layers:
                linear_layers[layer_num] = {}
            linear_layers[layer_num]['bias'] = state_dict[key].shape

    print(f"\nLinear layers found:")
    for layer_num in sorted(linear_layers.keys(), key=int):
        info = linear_layers[layer_num]
        if 'weight' in info:
            w_shape = info['weight']
            # Linear layers have 2D weights, BatchNorm has 1D
            if len(w_shape) == 2:
                print(f"  Layer {layer_num}: {w_shape[1]} → {w_shape[0]} (Linear)")
            else:
                print(f"  Layer {layer_num}: {w_shape[0]} (BatchNorm)")

    # Calculate total params
    total_params = sum(p.numel() for p in state_dict.values())
    print(f"\nTotal parameters: {total_params:,}")
    print(f"File size: {model_path.stat().st_size / 1024:.1f} KB")
    print(f"Theoretical size (params * 4 bytes): {total_params * 4 / 1024:.1f} KB")


def main():
    print("\n" + "="*70)
    print("  DETAILED MODEL STRUCTURE INSPECTION")
    print("="*70)

    models_dir = project_root / 'data' / 'player_data' / 'models'

    # Inspect both models
    models = [
        ('pytorch_model.pt', 'REGULAR MODEL'),
        ('pytorch_model_optimized.pt', 'OPTIMIZED MODEL'),
    ]

    for filename, name in models:
        model_path = models_dir / filename
        if model_path.exists():
            inspect_model(model_path, name)
        else:
            print(f"\n{name}: NOT FOUND at {model_path}")


if __name__ == "__main__":
    main()
