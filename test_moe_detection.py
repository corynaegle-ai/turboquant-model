#!/usr/bin/env python3
"""Test MoE detection and offloading on GX10."""

import torch
import sys

def test_imports():
    """Test that all MoE modules import correctly."""
    print("Test 1: Importing MoE modules...")
    try:
        from turboquant_model.moe import detect_moe_layers, MoELayerInfo, TurboQuantMoEExpert
        from turboquant_model.offload import ExpertOffloadManager
        from turboquant_model import TurboQuantConfig
        print("  ✓ All imports successful")
        return True
    except Exception as e:
        print(f"  ✗ Import failed: {e}")
        return False


def test_config():
    """Test MoE config creation."""
    print("\nTest 2: Creating MoE config...")
    try:
        from turboquant_model import TurboQuantConfig
        config = TurboQuantConfig(
            bit_width=4,
            moe_offload=True,
            expert_cache_size=8,
            offload_path="/tmp/tq-experts/"
        )
        print(f"  ✓ Config created")
        print(f"    bit_width: {config.bit_width}")
        print(f"    moe_offload: {config.moe_offload}")
        print(f"    expert_cache_size: {config.expert_cache_size}")
        return True
    except Exception as e:
        print(f"  ✗ Config failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_moe_detection_synthetic():
    """Test MoE detection with a synthetic MoE-like module."""
    print("\nTest 3: Synthetic MoE detection...")
    try:
        import torch.nn as nn
        from turboquant_model.moe import detect_moe_layers
        
        # Create a fake MoE model
        class FakeMoE(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([
                    self._make_layer() for _ in range(2)
                ])
            
            def _make_layer(self):
                layer = nn.Module()
                layer.block_sparse_moe = nn.Module()
                layer.block_sparse_moe.gate = nn.Linear(256, 8)
                layer.block_sparse_moe.experts = nn.ModuleList([
                    self._make_expert() for _ in range(8)
                ])
                return layer
            
            def _make_expert(self):
                expert = nn.Module()
                expert.gate_proj = nn.Linear(256, 512)
                expert.up_proj = nn.Linear(256, 512)
                expert.down_proj = nn.Linear(512, 256)
                return expert
        
        model = FakeMoE()
        moe_layers = detect_moe_layers(model)
        
        print(f"  ✓ Detected {len(moe_layers)} MoE layers")
        for layer in moe_layers:
            print(f"    - {layer.layer_name}: {layer.num_experts} experts")
        
        return len(moe_layers) > 0
    except Exception as e:
        print(f"  ✗ Detection failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_real_model():
    """Test with a real MoE model (Qwen1.5-MoE-A2.7B)."""
    print("\nTest 4: Real MoE model detection...")
    try:
        from transformers import AutoConfig
        from turboquant_model.moe import detect_moe_layers
        
        # Just check config first
        model_name = "Qwen/Qwen1.5-MoE-A2.7B"
        print(f"  Loading config for {model_name}...")
        config = AutoConfig.from_pretrained(model_name)
        
        # Print MoE-related config
        moe_attrs = ['num_experts', 'num_experts_per_tok', 'num_local_experts', 
                     'num_selected_experts', 'moe_intermediate_size']
        for attr in moe_attrs:
            val = getattr(config, attr, None)
            if val is not None:
                print(f"    {attr}: {val}")
        
        print("  ✓ Config loaded successfully")
        return True
    except Exception as e:
        print(f"  ✗ Real model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gpu():
    """Test GPU availability."""
    print("\nTest 5: GPU status...")
    if torch.cuda.is_available():
        print(f"  ✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"    Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        return True
    else:
        print("  ✗ No GPU available")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("TurboQuant MoE Detection Tests")
    print("=" * 60)
    
    results = []
    results.append(("Imports", test_imports()))
    results.append(("Config", test_config()))
    results.append(("Synthetic MoE", test_moe_detection_synthetic()))
    results.append(("Real Model", test_real_model()))
    results.append(("GPU", test_gpu()))
    
    print("\n" + "=" * 60)
    print("Results:")
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
    
    all_passed = all(r[1] for r in results)
    sys.exit(0 if all_passed else 1)
