"""
Test script to validate STG-NF model implementation
This script tests basic model functionality without requiring full dataset
"""
import sys
import os

# Use utility function for path setup
sys.path.insert(0, os.path.dirname(__file__))
from utils import setup_model_path

# Setup model path
setup_model_path()

def test_model_import():
    """Test that STG-NF model can be imported"""
    print("Testing STG-NF model import...")
    try:
        from stg_nf import STG_NF
        print("✓ STG-NF model imported successfully")
        return True, STG_NF
    except ImportError as e:
        print(f"✗ Failed to import STG-NF model: {e}")
        return False, None

def test_model_instantiation(STG_NF):
    """Test that STG-NF model can be instantiated"""
    print("\nTesting STG-NF model instantiation...")
    try:
        model = STG_NF(
            input_size=(2, 12, 6),  # (C, T, V) - arms only
            K=3,
            L=4,
            hidden_channels=512,
            device='cpu'
        )
        print(f"✓ Model instantiated successfully")
        print(f"  - Input size: (2, 12, 6)")
        print(f"  - Feature dimension: {model.feature_dim}")
        return True, model
    except Exception as e:
        print(f"✗ Failed to instantiate model: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_model_forward(model):
    """Test that model forward pass works"""
    print("\nTesting model forward pass...")
    try:
        import torch
        import numpy as np
        
        # Create dummy input
        batch_size = 4
        x = torch.randn(batch_size, 2, 12, 6)  # (B, C, T, V)
        
        # Forward pass
        z, log_det = model(x)
        
        print(f"✓ Forward pass successful")
        print(f"  - Input shape: {x.shape}")
        print(f"  - Output (z) shape: {z.shape}")
        print(f"  - Log det shape: {log_det.shape}")
        
        # Test loss computation
        loss = model.loss(z, log_det)
        print(f"  - Loss: {loss.item():.4f}")
        
        return True
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_pipeline_imports():
    """Test that training pipeline can import required modules"""
    print("\nTesting training pipeline imports...")
    try:
        sys.path.insert(0, os.path.dirname(__file__))
        
        # Test individual imports
        print("  - Testing train_pipeline module...")
        import train_pipeline
        print("    ✓ train_pipeline imported")
        
        print("  - Testing JOINT_SUBSET_MAP...")
        from train_pipeline import JOINT_SUBSET_MAP
        print(f"    ✓ JOINT_SUBSET_MAP loaded with {len(JOINT_SUBSET_MAP)} subsets")
        
        print("  - Testing set_seed function...")
        from train_pipeline import set_seed
        set_seed(42)
        print("    ✓ set_seed function works")
        
        return True
    except Exception as e:
        print(f"✗ Training pipeline import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("="*80)
    print("STG-NF Model Validation Tests")
    print("="*80)
    
    results = []
    
    # Test 1: Import
    success, STG_NF = test_model_import()
    results.append(("Model Import", success))
    
    if not success:
        print("\n✗ Cannot proceed without successful model import")
        return False
    
    # Test 2: Instantiation
    success, model = test_model_instantiation(STG_NF)
    results.append(("Model Instantiation", success))
    
    if not success:
        print("\n✗ Cannot proceed without successful model instantiation")
        return False
    
    # Test 3: Forward pass
    success = test_model_forward(model)
    results.append(("Forward Pass", success))
    
    # Test 4: Training pipeline
    success = test_training_pipeline_imports()
    results.append(("Training Pipeline Imports", success))
    
    # Summary
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(passed for _, passed in results)
    print("\n" + "="*80)
    if all_passed:
        print("✓ All tests passed!")
        print("="*80)
        return True
    else:
        print("✗ Some tests failed")
        print("="*80)
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
