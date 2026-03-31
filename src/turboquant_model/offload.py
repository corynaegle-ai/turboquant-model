"""Expert offloading for MoE models via memory-mapped files.

Enables running 1T+ parameter MoE models on systems with limited GPU memory
by keeping most experts on NVMe and loading them on-demand.

Key features:
- mmap-based expert storage on NVMe
- LRU cache for frequently-used experts in GPU memory
- Async prefetching for next-token prediction
- Thread-safe loading/unloading

Memory layout:
  GPU (30-40GB):
    - Embeddings, attention layers (always loaded)
    - Shared expert (always loaded)
    - Active expert cache (k experts + LRU buffer)
  
  NVMe (mmap):
    - All N expert weights (4-bit TurboQuant packed)
"""

from __future__ import annotations

import logging
import mmap
import os
import struct
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from concurrent.futures import ThreadPoolExecutor

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Expert File Format
# ---------------------------------------------------------------------------

EXPERT_FILE_MAGIC = b"TQEX"  # TurboQuant Expert
EXPERT_FILE_VERSION = 1

@dataclass
class ExpertFileHeader:
    """Header for an expert file."""
    expert_id: int
    bit_width: int
    group_size: int
    in_features: int
    intermediate_size: int
    has_gate: bool
    has_up: bool
    has_down: bool
    # Offsets within file
    gate_indices_offset: int
    gate_indices_size: int
    gate_norms_offset: int
    gate_norms_size: int
    up_indices_offset: int
    up_indices_size: int
    up_norms_offset: int
    up_norms_size: int
    down_indices_offset: int
    down_indices_size: int
    down_norms_offset: int
    down_norms_size: int
    
    def to_bytes(self) -> bytes:
        """Serialize header to bytes."""
        flags = (self.has_gate << 0) | (self.has_up << 1) | (self.has_down << 2)
        return struct.pack(
            "<4sIIIIIIBIIIIIIIIIIII",
            EXPERT_FILE_MAGIC,
            EXPERT_FILE_VERSION,
            self.expert_id,
            self.bit_width,
            self.group_size,
            self.in_features,
            self.intermediate_size,
            flags,
            self.gate_indices_offset,
            self.gate_indices_size,
            self.gate_norms_offset,
            self.gate_norms_size,
            self.up_indices_offset,
            self.up_indices_size,
            self.up_norms_offset,
            self.up_norms_size,
            self.down_indices_offset,
            self.down_indices_size,
            self.down_norms_offset,
            self.down_norms_size,
        )
    
    @classmethod
    def from_bytes(cls, data: bytes) -> "ExpertFileHeader":
        """Deserialize header from bytes."""
        unpacked = struct.unpack("<4sIIIIIIBIIIIIIIIIIII", data[:84])
        magic, version, expert_id, bit_width, group_size, in_features, intermediate_size, flags = unpacked[:8]
        
        if magic != EXPERT_FILE_MAGIC:
            raise ValueError(f"Invalid expert file magic: {magic}")
        if version != EXPERT_FILE_VERSION:
            raise ValueError(f"Unsupported expert file version: {version}")
        
        return cls(
            expert_id=expert_id,
            bit_width=bit_width,
            group_size=group_size,
            in_features=in_features,
            intermediate_size=intermediate_size,
            has_gate=bool(flags & 1),
            has_up=bool(flags & 2),
            has_down=bool(flags & 4),
            gate_indices_offset=unpacked[8],
            gate_indices_size=unpacked[9],
            gate_norms_offset=unpacked[10],
            gate_norms_size=unpacked[11],
            up_indices_offset=unpacked[12],
            up_indices_size=unpacked[13],
            up_norms_offset=unpacked[14],
            up_norms_size=unpacked[15],
            down_indices_offset=unpacked[16],
            down_indices_size=unpacked[17],
            down_norms_offset=unpacked[18],
            down_norms_size=unpacked[19],
        )
    
    @staticmethod
    def header_size() -> int:
        return 84


# ---------------------------------------------------------------------------
# Expert Storage
# ---------------------------------------------------------------------------

def save_expert_file(
    path: Path,
    expert_id: int,
    bit_width: int,
    group_size: int,
    in_features: int,
    intermediate_size: int,
    gate_indices: Optional[torch.Tensor],
    gate_norms: Optional[torch.Tensor],
    up_indices: Optional[torch.Tensor],
    up_norms: Optional[torch.Tensor],
    down_indices: Optional[torch.Tensor],
    down_norms: Optional[torch.Tensor],
):
    """Save a single expert to a binary file for mmap access.
    
    File format:
        [Header: 84 bytes]
        [gate_indices: M * N//2 bytes]
        [gate_norms: M * n_groups * 4 bytes]
        [up_indices: M * N//2 bytes]
        [up_norms: M * n_groups * 4 bytes]
        [down_indices: inter * hidden//2 bytes]
        [down_norms: inter * n_groups * 4 bytes]
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    header_size = ExpertFileHeader.header_size()
    current_offset = header_size
    
    # Calculate sizes and offsets
    gate_indices_offset = current_offset
    gate_indices_size = gate_indices.numel() if gate_indices is not None else 0
    current_offset += gate_indices_size
    
    gate_norms_offset = current_offset
    gate_norms_size = gate_norms.numel() * 4 if gate_norms is not None else 0
    current_offset += gate_norms_size
    
    up_indices_offset = current_offset
    up_indices_size = up_indices.numel() if up_indices is not None else 0
    current_offset += up_indices_size
    
    up_norms_offset = current_offset
    up_norms_size = up_norms.numel() * 4 if up_norms is not None else 0
    current_offset += up_norms_size
    
    down_indices_offset = current_offset
    down_indices_size = down_indices.numel() if down_indices is not None else 0
    current_offset += down_indices_size
    
    down_norms_offset = current_offset
    down_norms_size = down_norms.numel() * 4 if down_norms is not None else 0
    current_offset += down_norms_size
    
    header = ExpertFileHeader(
        expert_id=expert_id,
        bit_width=bit_width,
        group_size=group_size,
        in_features=in_features,
        intermediate_size=intermediate_size,
        has_gate=gate_indices is not None,
        has_up=up_indices is not None,
        has_down=down_indices is not None,
        gate_indices_offset=gate_indices_offset,
        gate_indices_size=gate_indices_size,
        gate_norms_offset=gate_norms_offset,
        gate_norms_size=gate_norms_size,
        up_indices_offset=up_indices_offset,
        up_indices_size=up_indices_size,
        up_norms_offset=up_norms_offset,
        up_norms_size=up_norms_size,
        down_indices_offset=down_indices_offset,
        down_indices_size=down_indices_size,
        down_norms_offset=down_norms_offset,
        down_norms_size=down_norms_size,
    )
    
    with open(path, 'wb') as f:
        f.write(header.to_bytes())
        
        if gate_indices is not None:
            f.write(gate_indices.contiguous().numpy().tobytes())
        if gate_norms is not None:
            f.write(gate_norms.contiguous().numpy().tobytes())
        if up_indices is not None:
            f.write(up_indices.contiguous().numpy().tobytes())
        if up_norms is not None:
            f.write(up_norms.contiguous().numpy().tobytes())
        if down_indices is not None:
            f.write(down_indices.contiguous().numpy().tobytes())
        if down_norms is not None:
            f.write(down_norms.contiguous().numpy().tobytes())


# ---------------------------------------------------------------------------
# Expert Offload Manager
# ---------------------------------------------------------------------------

@dataclass
class LoadedExpertInfo:
    """Tracks a loaded expert in GPU memory."""
    expert_id: int
    layer_idx: int
    memory_bytes: int
    last_used: int  # Monotonic counter for LRU


class ExpertOffloadManager:
    """Manages expert loading/unloading with LRU cache.
    
    Keeps frequently-used experts in GPU memory and pages others
    from NVMe via mmap.
    
    Thread-safe for concurrent loading requests.
    
    Usage:
        manager = ExpertOffloadManager(
            model=model,
            offload_path="/mnt/nvme/experts/",
            cache_size=16,  # Keep 16 experts in GPU
            device="cuda"
        )
        
        # During forward pass:
        needed_experts = moe_layer.get_needed_experts(x)
        manager.ensure_loaded(layer_idx, needed_experts)
        output = moe_layer(x)
    """
    
    def __init__(
        self,
        offload_path: str | Path,
        cache_size: int = 16,
        device: str = "cuda",
        prefetch_threads: int = 2,
    ):
        self.offload_path = Path(offload_path)
        self.cache_size = cache_size
        self.device = device
        self.prefetch_threads = prefetch_threads
        
        # LRU cache: (layer_idx, expert_id) -> LoadedExpertInfo
        self._cache: OrderedDict[Tuple[int, int], LoadedExpertInfo] = OrderedDict()
        self._lock = threading.Lock()
        self._access_counter = 0
        
        # Memory tracking
        self._current_memory = 0
        self._max_memory = 0  # Set from config
        
        # Expert file handles (mmap)
        self._expert_mmaps: Dict[Tuple[int, int], Tuple[mmap.mmap, ExpertFileHeader]] = {}
        self._mmap_lock = threading.Lock()
        
        # Prefetch executor
        self._executor = ThreadPoolExecutor(max_workers=prefetch_threads)
        self._prefetch_futures: Dict[Tuple[int, int], Any] = {}
        
        # Model reference (set when binding)
        self._moe_layers: Dict[int, Any] = {}  # layer_idx -> TurboQuantMoELayer
    
    def bind_layer(self, layer_idx: int, moe_layer):
        """Bind an MoE layer to this manager."""
        self._moe_layers[layer_idx] = moe_layer
        moe_layer.set_offload_manager(self)
    
    def _get_expert_path(self, layer_idx: int, expert_id: int) -> Path:
        """Get the path to an expert file."""
        return self.offload_path / f"layer_{layer_idx}" / f"expert_{expert_id}.tqe"
    
    def _get_mmap(self, layer_idx: int, expert_id: int) -> Tuple[mmap.mmap, ExpertFileHeader]:
        """Get or create mmap handle for an expert file."""
        key = (layer_idx, expert_id)
        
        with self._mmap_lock:
            if key not in self._expert_mmaps:
                path = self._get_expert_path(layer_idx, expert_id)
                if not path.exists():
                    raise FileNotFoundError(f"Expert file not found: {path}")
                
                f = open(path, 'rb')
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                header = ExpertFileHeader.from_bytes(mm[:ExpertFileHeader.header_size()])
                self._expert_mmaps[key] = (mm, header)
            
            return self._expert_mmaps[key]
    
    def _load_tensor_from_mmap(
        self,
        mm: mmap.mmap,
        offset: int,
        size: int,
        dtype: torch.dtype,
        shape: Tuple[int, ...],
    ) -> torch.Tensor:
        """Load a tensor from mmap into GPU memory."""
        if size == 0:
            return None
        
        # Read from mmap
        data = mm[offset:offset + size]
        
        # Convert to tensor
        if dtype == torch.uint8:
            tensor = torch.frombuffer(bytearray(data), dtype=torch.uint8).reshape(shape)
        else:
            tensor = torch.frombuffer(bytearray(data), dtype=torch.float32).reshape(shape)
        
        # Move to GPU
        return tensor.to(self.device)
    
    def _load_expert(self, layer_idx: int, expert_id: int):
        """Load an expert from mmap into GPU memory."""
        key = (layer_idx, expert_id)
        
        with self._lock:
            if key in self._cache:
                # Already loaded, update LRU
                self._cache.move_to_end(key)
                self._cache[key].last_used = self._access_counter
                self._access_counter += 1
                return
        
        # Get mmap handle
        mm, header = self._get_mmap(layer_idx, expert_id)
        
        # Get the MoE layer and expert
        if layer_idx not in self._moe_layers:
            raise ValueError(f"Layer {layer_idx} not bound to manager")
        
        moe_layer = self._moe_layers[layer_idx]
        expert = moe_layer.experts[expert_id]
        
        # Calculate shapes
        n_groups_in = (header.in_features + header.group_size - 1) // header.group_size
        n_groups_inter = (header.intermediate_size + header.group_size - 1) // header.group_size
        
        gate_out = header.intermediate_size
        packed_in = (header.in_features + 1) // 2
        packed_inter = (header.intermediate_size + 1) // 2
        
        # Load tensors from mmap
        gate_indices = None
        gate_norms = None
        up_indices = None
        up_norms = None
        down_indices = None
        down_norms = None
        
        if header.has_gate and header.gate_indices_size > 0:
            gate_indices = self._load_tensor_from_mmap(
                mm, header.gate_indices_offset, header.gate_indices_size,
                torch.uint8, (gate_out, packed_in)
            )
            gate_norms = self._load_tensor_from_mmap(
                mm, header.gate_norms_offset, header.gate_norms_size,
                torch.float32, (gate_out, n_groups_in) if n_groups_in > 1 else (gate_out,)
            )
        
        if header.has_up and header.up_indices_size > 0:
            up_indices = self._load_tensor_from_mmap(
                mm, header.up_indices_offset, header.up_indices_size,
                torch.uint8, (gate_out, packed_in)
            )
            up_norms = self._load_tensor_from_mmap(
                mm, header.up_norms_offset, header.up_norms_size,
                torch.float32, (gate_out, n_groups_in) if n_groups_in > 1 else (gate_out,)
            )
        
        if header.has_down and header.down_indices_size > 0:
            down_indices = self._load_tensor_from_mmap(
                mm, header.down_indices_offset, header.down_indices_size,
                torch.uint8, (header.in_features, packed_inter)
            )
            down_norms = self._load_tensor_from_mmap(
                mm, header.down_norms_offset, header.down_norms_size,
                torch.float32, (header.in_features, n_groups_inter) if n_groups_inter > 1 else (header.in_features,)
            )
        
        # Load into expert module
        expert.load_weights(
            gate_indices=gate_indices,
            gate_norms=gate_norms,
            up_indices=up_indices,
            up_norms=up_norms,
            down_indices=down_indices,
            down_norms=down_norms,
            device=self.device,
        )
        
        # Update cache
        memory_bytes = expert.memory_bytes()
        
        with self._lock:
            self._cache[key] = LoadedExpertInfo(
                expert_id=expert_id,
                layer_idx=layer_idx,
                memory_bytes=memory_bytes,
                last_used=self._access_counter,
            )
            self._access_counter += 1
            self._current_memory += memory_bytes
            
            # Evict if over cache size
            self._evict_if_needed()
    
    def _evict_if_needed(self):
        """Evict LRU experts if cache is over size."""
        while len(self._cache) > self.cache_size:
            # Get LRU entry
            key, info = next(iter(self._cache.items()))
            
            # Unload expert
            if key[0] in self._moe_layers:
                expert = self._moe_layers[key[0]].experts[key[1]]
                expert.unload_weights()
            
            # Remove from cache
            del self._cache[key]
            self._current_memory -= info.memory_bytes
            
            logger.debug(f"Evicted expert {key[1]} from layer {key[0]}")
    
    def ensure_loaded(self, layer_idx: int, expert_ids: List[int]):
        """Ensure the specified experts are loaded for a layer.
        
        Blocks until all requested experts are in GPU memory.
        Uses LRU eviction if cache is full.
        """
        for expert_id in expert_ids:
            self._load_expert(layer_idx, expert_id)
    
    def prefetch(self, layer_idx: int, expert_ids: List[int]):
        """Async prefetch experts (non-blocking).
        
        Call ahead of forward pass to hide loading latency.
        """
        for expert_id in expert_ids:
            key = (layer_idx, expert_id)
            
            with self._lock:
                if key in self._cache or key in self._prefetch_futures:
                    continue
            
            future = self._executor.submit(self._load_expert, layer_idx, expert_id)
            self._prefetch_futures[key] = future
    
    def wait_prefetch(self, layer_idx: int, expert_ids: List[int]):
        """Wait for prefetch to complete."""
        for expert_id in expert_ids:
            key = (layer_idx, expert_id)
            if key in self._prefetch_futures:
                self._prefetch_futures[key].result()
                del self._prefetch_futures[key]
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                "cache_size": len(self._cache),
                "max_cache_size": self.cache_size,
                "current_memory_mb": self._current_memory / (1024 * 1024),
                "loaded_experts": list(self._cache.keys()),
                "access_count": self._access_counter,
            }
    
    def close(self):
        """Close all mmap handles and shutdown executor."""
        self._executor.shutdown(wait=False)
        
        with self._mmap_lock:
            for mm, _ in self._expert_mmaps.values():
                mm.close()
            self._expert_mmaps.clear()
        
        with self._lock:
            self._cache.clear()
            self._current_memory = 0


# ---------------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------------

def save_experts_to_offload_dir(
    moe_layers: Dict[int, Any],  # layer_idx -> MoE layer with quantized experts
    offload_path: str | Path,
    bit_width: int = 4,
    group_size: int = 128,
):
    """Save all quantized experts to the offload directory.
    
    Creates the directory structure:
        offload_path/
        ├── layer_0/
        │   ├── expert_0.tqe
        │   ├── expert_1.tqe
        │   └── ...
        ├── layer_1/
        │   └── ...
        └── config.json
    """
    import json
    
    offload_path = Path(offload_path)
    offload_path.mkdir(parents=True, exist_ok=True)
    
    config = {
        "bit_width": bit_width,
        "group_size": group_size,
        "layers": {},
    }
    
    total_experts = 0
    total_bytes = 0
    
    for layer_idx, moe_layer in moe_layers.items():
        layer_dir = offload_path / f"layer_{layer_idx}"
        layer_dir.mkdir(exist_ok=True)
        
        config["layers"][layer_idx] = {
            "num_experts": len(moe_layer.experts),
            "in_features": moe_layer.in_features,
            "intermediate_size": moe_layer.intermediate_size,
        }
        
        for expert in moe_layer.experts:
            expert_path = layer_dir / f"expert_{expert.expert_id}.tqe"
            
            save_expert_file(
                path=expert_path,
                expert_id=expert.expert_id,
                bit_width=bit_width,
                group_size=group_size,
                in_features=moe_layer.in_features,
                intermediate_size=moe_layer.intermediate_size,
                gate_indices=expert._gate_indices,
                gate_norms=expert._gate_norms,
                up_indices=expert._up_indices,
                up_norms=expert._up_norms,
                down_indices=expert._down_indices,
                down_norms=expert._down_norms,
            )
            
            total_experts += 1
            total_bytes += expert_path.stat().st_size
    
    # Save config
    with open(offload_path / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(
        f"Saved {total_experts} experts to {offload_path} "
        f"({total_bytes / 1024**2:.1f} MB)"
    )


def create_offload_manager(
    model,
    offload_path: str | Path,
    cache_size: int = 16,
    device: str = "cuda",
) -> ExpertOffloadManager:
    """Create an offload manager and bind it to all MoE layers in a model.
    
    Args:
        model: Model with TurboQuantMoELayer instances
        offload_path: Path to expert files
        cache_size: Number of experts to keep in GPU
        device: Target device
    
    Returns:
        Configured ExpertOffloadManager
    """
    from turboquant_model.moe import TurboQuantMoELayer
    
    manager = ExpertOffloadManager(
        offload_path=offload_path,
        cache_size=cache_size,
        device=device,
    )
    
    # Find and bind MoE layers
    layer_idx = 0
    for name, module in model.named_modules():
        if isinstance(module, TurboQuantMoELayer):
            manager.bind_layer(layer_idx, module)
            layer_idx += 1
    
    logger.info(f"Bound {layer_idx} MoE layers to offload manager (cache_size={cache_size})")
    
    return manager
