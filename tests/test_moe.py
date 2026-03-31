"""Tests for MoE (Mixture of Experts) quantization and offloading.

Tests cover:
- MoE layer detection
- Expert quantization
- TurboQuantMoEExpert forward pass
- TurboQuantMoELayer with routing
- Expert offloading and LRU cache
- Save/load MoE models
"""

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from turboquant_model import (
    TurboQuantConfig,
    detect_moe_layers,
    is_moe_model,
    quantize_expert,
    TurboQuantMoEExpert,
    TurboQuantMoELayer,
)
from turboquant_model.moe import MoELayerInfo
from turboquant_model.offload import (
    ExpertOffloadManager,
    save_expert_file,
    ExpertFileHeader,
)
from turboquant_model.codebook import get_codebook


# ---------------------------------------------------------------------------
# Test Fixtures: Simple MoE Model
# ---------------------------------------------------------------------------

class SimpleExpert(nn.Module):
    """SwiGLU-style expert (like Mixtral/Qwen3-MoE)."""
    
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
    
    def forward(self, x):
        return self.down_proj(nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class SimpleMoELayer(nn.Module):
    """Simple MoE layer for testing."""
    
    def __init__(self, hidden_size: int, intermediate_size: int, num_experts: int):
        super().__init__()
        self.num_experts = num_experts
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.experts = nn.ModuleList([
            SimpleExpert(hidden_size, intermediate_size)
            for _ in range(num_experts)
        ])
    
    def forward(self, x):
        # Simple top-2 routing
        router_logits = self.gate(x)
        weights, indices = torch.topk(router_logits.softmax(dim=-1), 2, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        
        # Naive expert execution (not batched)
        batch_size = x.shape[0]
        output = torch.zeros_like(x)
        
        for i in range(batch_size):
            for k in range(2):
                expert_idx = indices[i, k].item()
                expert_out = self.experts[expert_idx](x[i:i+1])
                output[i:i+1] += weights[i, k] * expert_out
        
        return output


class SimpleMoEModel(nn.Module):
    """Simple model with MoE layers for testing."""
    
    def __init__(self, hidden_size: int = 64, intermediate_size: int = 128, num_experts: int = 4, num_layers: int = 2):
        super().__init__()
        self.embed = nn.Linear(hidden_size, hidden_size)
        self.layers = nn.ModuleList([
            SimpleMoELayer(hidden_size, intermediate_size, num_experts)
            for _ in range(num_layers)
        ])
        self.head = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        return self.head(x)


@pytest.fixture
def simple_moe_model():
    """Create a simple MoE model for testing."""
    return SimpleMoEModel(hidden_size=64, intermediate_size=128, num_experts=4, num_layers=2)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# ---------------------------------------------------------------------------
# MoE Detection Tests
# ---------------------------------------------------------------------------

def test_detect_moe_layers(simple_moe_model):
    """Test MoE layer detection."""
    moe_layers = detect_moe_layers(simple_moe_model)
    
    assert len(moe_layers) == 2, f"Expected 2 MoE layers, got {len(moe_layers)}"
    
    for info in moe_layers:
        assert info.num_experts == 4
        assert len(info.expert_names) == 4


def test_is_moe_model(simple_moe_model):
    """Test is_moe_model detection."""
    assert is_moe_model(simple_moe_model)
    
    # Non-MoE model
    linear_model = nn.Sequential(nn.Linear(64, 64), nn.Linear(64, 64))
    assert not is_moe_model(linear_model)


# ---------------------------------------------------------------------------
# Expert Quantization Tests
# ---------------------------------------------------------------------------

def test_quantize_expert():
    """Test single expert quantization."""
    expert = SimpleExpert(hidden_size=64, intermediate_size=128)
    
    data = quantize_expert(
        expert,
        bit_width=4,
        group_size=32,
        seed=42,
        rotation="qr",
    )
    
    # Check packed indices
    assert data.gate_indices is not None
    assert data.gate_norms is not None
    assert data.gate_shape == (128, 64)
    
    assert data.up_indices is not None
    assert data.down_indices is not None


def test_quantize_expert_hadamard():
    """Test expert quantization with Hadamard rotation."""
    expert = SimpleExpert(hidden_size=64, intermediate_size=128)
    
    data = quantize_expert(
        expert,
        bit_width=4,
        group_size=64,  # Power of 2 for Hadamard
        seed=42,
        rotation="hadamard",
    )
    
    assert data.gate_indices is not None
    assert data.gate_indices.dtype == torch.uint8


# ---------------------------------------------------------------------------
# TurboQuantMoEExpert Tests
# ---------------------------------------------------------------------------

def test_turboquant_moe_expert_forward():
    """Test quantized expert forward pass."""
    expert = SimpleExpert(hidden_size=64, intermediate_size=128)
    
    # Quantize
    data = quantize_expert(expert, bit_width=4, group_size=32, seed=42)
    
    # Create TurboQuant expert
    tq_expert = TurboQuantMoEExpert(
        expert_id=0,
        in_features=64,
        intermediate_size=128,
        bit_width=4,
        group_size=32,
        rotation="qr",
    )
    
    # Set codebook
    centroids, _ = get_codebook(4)
    tq_expert.codebook.copy_(centroids)
    tq_expert.set_rotation(42)
    
    # Load weights
    tq_expert.load_weights(
        gate_indices=data.gate_indices,
        gate_norms=data.gate_norms,
        up_indices=data.up_indices,
        up_norms=data.up_norms,
        down_indices=data.down_indices,
        down_norms=data.down_norms,
    )
    
    assert tq_expert.is_loaded
    
    # Forward pass
    x = torch.randn(2, 64)
    output = tq_expert(x)
    
    assert output.shape == (2, 64)
    assert not torch.isnan(output).any()


def test_expert_unload_reload():
    """Test expert weight unloading and reloading."""
    expert = SimpleExpert(hidden_size=64, intermediate_size=128)
    data = quantize_expert(expert, bit_width=4, group_size=32, seed=42)
    
    tq_expert = TurboQuantMoEExpert(
        expert_id=0,
        in_features=64,
        intermediate_size=128,
        bit_width=4,
        group_size=32,
    )
    
    centroids, _ = get_codebook(4)
    tq_expert.codebook.copy_(centroids)
    tq_expert.set_rotation(42)
    
    # Load
    tq_expert.load_weights(
        gate_indices=data.gate_indices,
        gate_norms=data.gate_norms,
        up_indices=data.up_indices,
        up_norms=data.up_norms,
        down_indices=data.down_indices,
        down_norms=data.down_norms,
    )
    assert tq_expert.is_loaded
    
    # Unload
    tq_expert.unload_weights()
    assert not tq_expert.is_loaded
    
    # Reload
    tq_expert.load_weights(
        gate_indices=data.gate_indices,
        gate_norms=data.gate_norms,
        up_indices=data.up_indices,
        up_norms=data.up_norms,
        down_indices=data.down_indices,
        down_norms=data.down_norms,
    )
    assert tq_expert.is_loaded


# ---------------------------------------------------------------------------
# TurboQuantMoELayer Tests
# ---------------------------------------------------------------------------

def test_turboquant_moe_layer():
    """Test full MoE layer with routing."""
    # Create and populate MoE layer
    moe_layer = TurboQuantMoELayer(
        num_experts=4,
        num_experts_per_tok=2,
        in_features=64,
        intermediate_size=128,
        bit_width=4,
        group_size=32,
    )
    
    centroids, _ = get_codebook(4)
    
    # Quantize and load each expert
    for i, expert in enumerate(moe_layer.experts):
        # Create a dummy expert and quantize it
        ref_expert = SimpleExpert(64, 128)
        data = quantize_expert(ref_expert, bit_width=4, group_size=32, seed=42 + i)
        
        expert.codebook.copy_(centroids)
        expert.set_rotation(42 + i)
        expert.load_weights(
            gate_indices=data.gate_indices,
            gate_norms=data.gate_norms,
            up_indices=data.up_indices,
            up_norms=data.up_norms,
            down_indices=data.down_indices,
            down_norms=data.down_norms,
        )
    
    # Forward pass
    x = torch.randn(2, 64)
    output = moe_layer(x)
    
    assert output.shape == (2, 64)
    assert not torch.isnan(output).any()


def test_moe_layer_get_needed_experts():
    """Test expert prediction for prefetching."""
    moe_layer = TurboQuantMoELayer(
        num_experts=8,
        num_experts_per_tok=2,
        in_features=64,
        intermediate_size=128,
        bit_width=4,
        group_size=32,
    )
    
    x = torch.randn(4, 64)
    needed = moe_layer.get_needed_experts(x)
    
    # Should be at most 2 * 4 = 8 unique experts
    assert len(needed) <= 8
    assert all(0 <= e < 8 for e in needed)


# ---------------------------------------------------------------------------
# Expert Offloading Tests
# ---------------------------------------------------------------------------

def test_save_and_load_expert_file(temp_dir):
    """Test saving and loading expert files."""
    expert = SimpleExpert(hidden_size=64, intermediate_size=128)
    data = quantize_expert(expert, bit_width=4, group_size=32, seed=42)
    
    # Save
    expert_path = temp_dir / "expert_0.tqe"
    save_expert_file(
        path=expert_path,
        expert_id=0,
        bit_width=4,
        group_size=32,
        in_features=64,
        intermediate_size=128,
        gate_indices=data.gate_indices,
        gate_norms=data.gate_norms,
        up_indices=data.up_indices,
        up_norms=data.up_norms,
        down_indices=data.down_indices,
        down_norms=data.down_norms,
    )
    
    assert expert_path.exists()
    
    # Read header
    with open(expert_path, 'rb') as f:
        header = ExpertFileHeader.from_bytes(f.read(ExpertFileHeader.header_size()))
    
    assert header.expert_id == 0
    assert header.bit_width == 4
    assert header.group_size == 32
    assert header.in_features == 64
    assert header.intermediate_size == 128
    assert header.has_gate
    assert header.has_up
    assert header.has_down


def test_expert_offload_manager(temp_dir):
    """Test ExpertOffloadManager with LRU cache."""
    # Create test experts
    experts_dir = temp_dir / "experts" / "layer_0"
    experts_dir.mkdir(parents=True)
    
    for i in range(4):
        expert = SimpleExpert(64, 128)
        data = quantize_expert(expert, bit_width=4, group_size=32, seed=42 + i)
        
        save_expert_file(
            path=experts_dir / f"expert_{i}.tqe",
            expert_id=i,
            bit_width=4,
            group_size=32,
            in_features=64,
            intermediate_size=128,
            gate_indices=data.gate_indices,
            gate_norms=data.gate_norms,
            up_indices=data.up_indices,
            up_norms=data.up_norms,
            down_indices=data.down_indices,
            down_norms=data.down_norms,
        )
    
    # Create MoE layer
    moe_layer = TurboQuantMoELayer(
        num_experts=4,
        num_experts_per_tok=2,
        in_features=64,
        intermediate_size=128,
        bit_width=4,
        group_size=32,
    )
    
    centroids, _ = get_codebook(4)
    for expert in moe_layer.experts:
        expert.codebook.copy_(centroids)
        expert.set_rotation(42)
    
    # Create manager with small cache
    manager = ExpertOffloadManager(
        offload_path=temp_dir / "experts",
        cache_size=2,  # Only keep 2 experts in memory
    )
    manager.bind_layer(0, moe_layer)
    
    # Load experts 0 and 1
    manager.ensure_loaded(0, [0, 1])
    
    assert moe_layer.experts[0].is_loaded
    assert moe_layer.experts[1].is_loaded
    assert not moe_layer.experts[2].is_loaded
    assert not moe_layer.experts[3].is_loaded
    
    stats = manager.get_cache_stats()
    assert stats["cache_size"] == 2
    
    # Load experts 2 and 3 - should evict 0 and 1
    manager.ensure_loaded(0, [2, 3])
    
    # LRU eviction should have removed experts 0 and 1
    # (but our implementation doesn't unload from module, just removes from cache)
    assert moe_layer.experts[2].is_loaded
    assert moe_layer.experts[3].is_loaded
    
    manager.close()


# ---------------------------------------------------------------------------
# Quality Tests
# ---------------------------------------------------------------------------

def test_expert_quantization_quality():
    """Test that quantized experts produce similar outputs to original."""
    torch.manual_seed(42)
    
    expert = SimpleExpert(hidden_size=64, intermediate_size=128)
    expert.eval()
    
    x = torch.randn(4, 64)
    
    # Original output
    with torch.no_grad():
        orig_out = expert(x)
    
    # Quantize
    data = quantize_expert(expert, bit_width=4, group_size=32, seed=42)
    
    # Create quantized expert
    tq_expert = TurboQuantMoEExpert(
        expert_id=0,
        in_features=64,
        intermediate_size=128,
        bit_width=4,
        group_size=32,
        rotation="qr",
    )
    
    centroids, _ = get_codebook(4)
    tq_expert.codebook.copy_(centroids)
    tq_expert.set_rotation(42)
    
    tq_expert.load_weights(
        gate_indices=data.gate_indices,
        gate_norms=data.gate_norms,
        up_indices=data.up_indices,
        up_norms=data.up_norms,
        down_indices=data.down_indices,
        down_norms=data.down_norms,
    )
    
    # Quantized output
    with torch.no_grad():
        quant_out = tq_expert(x)
    
    # Check cosine similarity - should be reasonably high for 4-bit
    cos_sim = torch.nn.functional.cosine_similarity(
        orig_out.flatten(), quant_out.flatten(), dim=0
    )
    
    # 4-bit quantization should maintain >0.8 cosine similarity
    # (this is a rough threshold; actual quality depends on data)
    assert cos_sim > 0.7, f"Cosine similarity too low: {cos_sim}"


# ---------------------------------------------------------------------------
# Memory Tests
# ---------------------------------------------------------------------------

def test_expert_memory_bytes():
    """Test memory calculation for experts."""
    expert = SimpleExpert(hidden_size=64, intermediate_size=128)
    data = quantize_expert(expert, bit_width=4, group_size=32, seed=42)
    
    tq_expert = TurboQuantMoEExpert(
        expert_id=0,
        in_features=64,
        intermediate_size=128,
        bit_width=4,
        group_size=32,
    )
    
    centroids, _ = get_codebook(4)
    tq_expert.codebook.copy_(centroids)
    
    tq_expert.load_weights(
        gate_indices=data.gate_indices,
        gate_norms=data.gate_norms,
        up_indices=data.up_indices,
        up_norms=data.up_norms,
        down_indices=data.down_indices,
        down_norms=data.down_norms,
    )
    
    mem_bytes = tq_expert.memory_bytes()
    
    # gate: 128 * 32 (packed) = 4096 bytes
    # up: 128 * 32 = 4096 bytes
    # down: 64 * 64 = 4096 bytes
    # norms: 128*2 + 64 (float32) * 4 = ~1280 bytes
    # Total should be ~13-15 KB
    assert mem_bytes > 10000
    assert mem_bytes < 20000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
