import math
import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from hbdk4.compiler import leap
from torch import nn

from leap_llm.nn.modules import ConstFakeQuant, FakeQuantLinear
from leap_llm.nn.modules.ops import FakeQuantReduceMean
from leap_llm.nn.modules.rms_norm import FakeQuantRMSNorm
from leap_llm.nn.utils import Model, Module

from .blocks import Gemma4VisionAttention, Gemma4VisionEncoderLayer, Gemma4VisionMLP


# ============================================================================
# Config
# ============================================================================


@dataclass
class Gemma4VisionConfig:
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_hidden_layers: int = 16
    num_attention_heads: int = 12
    num_key_value_heads: int = 12
    head_dim: int = 64
    rms_norm_eps: float = 1e-6
    patch_size: int = 16
    pooling_kernel_size: int = 3
    position_embedding_size: int = 10240
    rope_theta: float = 100.0
    # Image grid: h_patches * w_patches = num_patches
    # Default: 768x768 image -> 48x48 patches = 2304
    h_patches: int = 48
    w_patches: int = 48
    # embed_vision projector output dim (text hidden_size)
    text_hidden_size: int = 1536


# ============================================================================
# Patch Embedding
# ============================================================================


class PatchEmbedding(Module):
    """Linear patch projection + 2D learned position embedding.

    Input: pre-patchified pixel_values [num_patches, 3*patch_size^2]
    Output: [num_patches, hidden_size]
    """

    def __init__(self, config: Gemma4VisionConfig):
        super().__init__()
        self.config = config
        patch_dim = 3 * config.patch_size ** 2
        self.input_proj = FakeQuantLinear(patch_dim, config.hidden_size, bias=False)
        self.input_norm = ConstFakeQuant(8)

        # Position embedding table: [2, position_embedding_size, hidden_size]
        # Will be loaded from checkpoint
        self.position_embedding_table = nn.Parameter(
            torch.ones(2, config.position_embedding_size, config.hidden_size)
        )

        # Pre-computed position embeddings buffer (set in load_model)
        self.register_buffer("position_embeddings", None)

    def _compute_position_embeddings(self, pixel_position_ids):
        """Pre-compute position embeddings for fixed resolution.

        pixel_position_ids: [num_patches, 2] (x, y coordinates)
        Returns: [num_patches, hidden_size]
        """
        clamped = pixel_position_ids.clamp(min=0)
        one_hot = F.one_hot(
            clamped, num_classes=self.config.position_embedding_size
        )  # [num_patches, 2, pos_emb_size]
        one_hot = one_hot.permute(1, 0, 2).float()  # [2, num_patches, pos_emb_size]
        # [2, num_patches, pos_emb_size] @ [2, pos_emb_size, hidden_size]
        pos_emb = torch.bmm(one_hot, self.position_embedding_table.data.float())
        pos_emb = pos_emb.sum(dim=0)  # [num_patches, hidden_size]
        return pos_emb

    def build(self, pixel_values):
        """pixel_values: [num_patches, patch_dim] already normalized to [0, 1]"""
        # Normalize: 2*(x - 0.5) = 2x - 1, maps [0,1] to [-1,1]
        x = leap.mul(pixel_values, 2.0)
        x = leap.add(x, -1.0)
        x = self.input_norm(x)
        hidden_states = self.input_proj(x)
        # Add pre-computed position embeddings
        pos_emb = self.position_embeddings.data.to(torch.float32)
        hidden_states = leap.add(hidden_states, pos_emb)
        return hidden_states

    def forward(self, pixel_values):
        """pixel_values: [1, num_patches, patch_dim]"""
        x = 2 * (pixel_values - 0.5)
        x = self.input_norm(x)
        hidden_states = self.input_proj(x)
        if self.position_embeddings is not None:
            hidden_states = hidden_states + self.position_embeddings.unsqueeze(0)
        return hidden_states


# ============================================================================
# 2D Rotary Embedding
# ============================================================================


class VisionRotaryEmbedding:
    """Pre-computes 2D rotary position embeddings for the vision encoder.

    Not a Module - just a utility to compute cos/sin constants.
    """

    @staticmethod
    def compute(config: Gemma4VisionConfig):
        """Compute cos/sin for all patch positions.

        Returns: cos [num_patches, head_dim], sin [num_patches, head_dim]
        """
        h, w = config.h_patches, config.w_patches
        head_dim = config.head_dim
        spatial_dim = head_dim // 2  # 32
        base = config.rope_theta

        # inv_freq: [spatial_dim // 2] = [16]
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, spatial_dim, 2, dtype=torch.float32) / spatial_dim
            )
        )

        # Position IDs: [num_patches, 2] as (x, y)
        positions_x = torch.arange(w).repeat(h)  # [num_patches]
        positions_y = torch.arange(h).repeat_interleave(w)  # [num_patches]

        all_cos, all_sin = [], []
        for positions in [positions_x, positions_y]:
            # [num_patches, 1] @ [1, spatial_dim//2] -> [num_patches, spatial_dim//2]
            freqs = positions.float().unsqueeze(1) * inv_freq.unsqueeze(0)
            emb = torch.cat([freqs, freqs], dim=-1)  # [num_patches, spatial_dim]
            all_cos.append(emb.cos())
            all_sin.append(emb.sin())

        cos = torch.cat(all_cos, dim=-1)  # [num_patches, head_dim]
        sin = torch.cat(all_sin, dim=-1)  # [num_patches, head_dim]
        return cos, sin

    @staticmethod
    def compute_pixel_position_ids(config: Gemma4VisionConfig):
        """Compute pixel_position_ids for fixed resolution.

        Returns: [num_patches, 2] as (x, y) coordinates.
        """
        h, w = config.h_patches, config.w_patches
        xs = torch.arange(w).repeat(h)
        ys = torch.arange(h).repeat_interleave(w)
        return torch.stack([xs, ys], dim=-1)


# ============================================================================
# Pooler
# ============================================================================


class VisionPooler(Module):
    """Spatial average pooling + scaling.

    For fixed resolution: uniform k x k average pooling.
    output_length = num_patches / (k * k)
    """

    def __init__(self, config: Gemma4VisionConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.root_hidden_size = config.hidden_size ** 0.5
        self.k = config.pooling_kernel_size
        k2 = self.k * self.k

        h, w = config.h_patches, config.w_patches
        self.num_patches = h * w
        self.output_length = self.num_patches // k2
        self.h_out = h // self.k
        self.w_out = w // self.k
        self.pool = FakeQuantReduceMean(quantized=False)

        # Pre-compute pooling weight matrix: [output_length, num_patches]
        # For a regular grid, each output token averages k*k input patches
        self.register_buffer("pool_weights", None)
        self._precompute_pool_weights(config)

    def _precompute_pool_weights(self, config):
        h, w = config.h_patches, config.w_patches
        k = self.k
        k2 = k * k
        h_out = h // k
        w_out = w // k
        output_length = h_out * w_out

        weights = torch.zeros(output_length, h * w)
        for oy in range(h_out):
            for ox in range(w_out):
                out_idx = oy * w_out + ox
                for dy in range(k):
                    for dx in range(k):
                        iy = oy * k + dy
                        ix = ox * k + dx
                        in_idx = iy * w + ix
                        weights[out_idx, in_idx] = 1.0 / k2
        self.pool_weights = weights

    def build(self, hidden_states):
        """hidden_states: [num_patches, hidden_size]"""
        hidden_states = leap.reshape(
            hidden_states,
            [self.h_out, self.k, self.w_out, self.k, self.hidden_size],
        )
        pooled = self.pool(hidden_states, [1, 3])
        pooled = leap.reshape(pooled, [self.output_length, self.hidden_size])
        pooled = leap.mul(pooled, self.root_hidden_size)
        return pooled

    def forward(self, hidden_states):
        """hidden_states: [1, num_patches, hidden_size]"""
        h = hidden_states.squeeze(0).float()
        pooled = h.reshape(
            self.h_out, self.k, self.w_out, self.k, self.hidden_size
        ).mean(dim=(1, 3))
        pooled = pooled.reshape(1, self.output_length, self.hidden_size)
        pooled = pooled * self.root_hidden_size
        return pooled.to(hidden_states.dtype)


# ============================================================================
# Vision Projector (embed_vision)
# ============================================================================


class VisionProjector(Module):
    """Projects vision hidden states to text hidden space.

    RMSNorm (no scale) + Linear projection.
    """

    def __init__(self, vision_hidden_size, text_hidden_size, rms_norm_eps):
        super().__init__()
        self.norm = FakeQuantRMSNorm(
            vision_hidden_size, eps=rms_norm_eps, fuse_norm=True
        )
        self.projection = FakeQuantLinear(
            vision_hidden_size, text_hidden_size, bias=False
        )

    def build(self, hidden_states):
        hidden_states = self.norm(hidden_states)
        return self.projection(hidden_states)

    def forward(self, hidden_states):
        hidden_states = self.norm(hidden_states)
        return self.projection(hidden_states)


# ============================================================================
# Full Vision Model
# ============================================================================


class Gemma4VisionModel(Model):
    """Gemma4 Vision Encoder: PatchEmbedder + Encoder + Pooler + Projector.

    Compiled model input: pixel_values [num_patches, patch_dim]
    Compiled model output: [output_length, text_hidden_size]
    """

    def __init__(self, config: Gemma4VisionConfig):
        super().__init__()
        self.config = config
        self.num_patches = config.h_patches * config.w_patches

        self.patch_embedder = PatchEmbedding(config)
        self.layers = nn.ModuleList(
            [
                Gemma4VisionEncoderLayer(
                    layer_id=i,
                    hidden_size=config.hidden_size,
                    num_attention_heads=config.num_attention_heads,
                    head_dim=config.head_dim,
                    intermediate_size=config.intermediate_size,
                    rms_norm_eps=config.rms_norm_eps,
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.pooler = VisionPooler(config)
        self.projector = VisionProjector(
            config.hidden_size, config.text_hidden_size, config.rms_norm_eps
        )

        # Pre-computed RoPE buffers
        self.register_buffer("rope_cos", None)
        self.register_buffer("rope_sin", None)

    def _init_constants(self):
        """Pre-compute position embeddings and RoPE for fixed resolution."""
        config = self.config

        # Position embeddings for patch embedder
        pos_ids = VisionRotaryEmbedding.compute_pixel_position_ids(config)
        pos_emb = self.patch_embedder._compute_position_embeddings(pos_ids)
        self.patch_embedder.position_embeddings = pos_emb

        # RoPE cos/sin
        cos, sin = VisionRotaryEmbedding.compute(config)
        self.rope_cos = cos
        self.rope_sin = sin

    def build(self, pixel_values):
        """
        pixel_values: [num_patches, patch_dim]
        Returns: [output_length, text_hidden_size]
        """
        # Cast to float32 to match FakeQuantLinear weight dtype (avoids si8:f16 vs si8:f32)
        pixel_values = leap.cast_type(pixel_values, output_type=leap.float32)

        # Patch embedding + position embedding
        hidden_states = self.patch_embedder(pixel_values)

        # Encoder layers
        cos = self.rope_cos.data.to(torch.float32)
        sin = self.rope_sin.data.to(torch.float32)
        for layer in self.layers:
            hidden_states = layer(hidden_states, cos, sin)

        # Pooler
        hidden_states = self.pooler(hidden_states)

        # Vision projector
        hidden_states = self.projector(hidden_states)

        return hidden_states

    def forward(self, pixel_values):
        """
        pixel_values: [1, num_patches, patch_dim]
        Returns: [1, output_length, text_hidden_size]
        """
        hidden_states = self.patch_embedder(pixel_values)

        cos = self.rope_cos.unsqueeze(0).to(pixel_values.device)
        sin = self.rope_sin.unsqueeze(0).to(pixel_values.device)
        for layer in self.layers:
            hidden_states = layer(hidden_states, cos, sin)

        hidden_states = self.pooler(hidden_states)
        hidden_states = self.projector(hidden_states)

        return hidden_states


# ============================================================================
# Wrapper (load_model / compile / etc.)
# ============================================================================


class Gemma4Vision:
    """Wrapper for Gemma4 Vision model: loading, calibration, compilation."""

    def __init__(self, model: Gemma4VisionModel, config: Gemma4VisionConfig):
        self.model = model
        self.config = config

    @staticmethod
    def load_model(model_path, checkpoint=None, config=None):
        """Load Gemma4 vision model from checkpoint.

        Args:
            model_path: path to the model directory
            checkpoint: pre-loaded state dict (optional)
            config: Gemma4VisionConfig (optional, auto-detected from checkpoint)
        """
        import json

        if config is None:
            config_path = os.path.join(model_path, "config.json")
            with open(config_path) as f:
                raw = json.load(f)
            vc = raw.get("vision_config", {})
            config = Gemma4VisionConfig(
                hidden_size=vc.get("hidden_size", 768),
                intermediate_size=vc.get("intermediate_size", 3072),
                num_hidden_layers=vc.get("num_hidden_layers", 16),
                num_attention_heads=vc.get("num_attention_heads", 12),
                num_key_value_heads=vc.get("num_key_value_heads", 12),
                head_dim=vc.get("head_dim", 64),
                rms_norm_eps=vc.get("rms_norm_eps", 1e-6),
                patch_size=vc.get("patch_size", 16),
                pooling_kernel_size=vc.get("pooling_kernel_size", 3),
                position_embedding_size=vc.get("position_embedding_size", 10240),
                rope_theta=vc.get("rope_parameters", {}).get("rope_theta", 100.0),
                text_hidden_size=raw.get("text_config", {}).get("hidden_size", 1536),
            )

        model = Gemma4VisionModel(config)

        if checkpoint is None:
            checkpoint = Gemma4Vision._load_checkpoint(model_path)

        # Map weights
        mapped = Gemma4Vision._map_weights(checkpoint, config)
        missing, unexpected = model.load_state_dict(mapped, strict=False)

        # Filter out expected missing keys (pre-computed buffers + no-scale norms)
        expected_missing = {
            "patch_embedder.position_embeddings",
            "rope_cos",
            "rope_sin",
            "pooler.pool_weights",
            "projector.norm.weight",  # HF with_scale=False, our FakeQuantRMSNorm has weight=ones
        }
        # v_norm weights: HF with_scale=False, our FakeQuantRMSNorm keeps weight=ones
        for i in range(config.num_hidden_layers):
            expected_missing.add(f"layers.{i}.self_attn.v_norm.weight")
        real_missing = [k for k in missing if k not in expected_missing]
        if real_missing:
            print(f"[Gemma4Vision] Warning: missing keys: {real_missing}")
        if unexpected:
            print(f"[Gemma4Vision] Warning: unexpected keys: {unexpected[:10]}...")

        # Initialize pre-computed constants
        model._init_constants()

        wrapper = Gemma4Vision(model, config)
        return wrapper

    @staticmethod
    def _load_checkpoint(model_path):
        """Load from safetensors or pytorch_model.bin."""
        from safetensors import safe_open

        st_path = os.path.join(model_path, "model.safetensors")
        if os.path.exists(st_path):
            checkpoint = {}
            with safe_open(st_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    checkpoint[key] = f.get_tensor(key)
            return checkpoint

        pt_path = os.path.join(model_path, "pytorch_model.bin")
        if os.path.exists(pt_path):
            return torch.load(pt_path, map_location="cpu")

        raise FileNotFoundError(
            f"No model file found at {model_path} "
            "(expected model.safetensors or pytorch_model.bin)"
        )

    @staticmethod
    def _map_weights(checkpoint, config):
        """Map HF checkpoint keys to leap_llm model keys."""
        mapped = {}

        # Prefixes to strip
        vt_prefix = "model.vision_tower."
        ev_prefix = "model.embed_vision."

        for key, tensor in checkpoint.items():
            new_key = None

            if key.startswith(vt_prefix):
                new_key = key[len(vt_prefix) :]
            elif key.startswith(ev_prefix):
                suffix = key[len(ev_prefix) :]
                # embed_vision.embedding_projection.weight -> projector.projection.weight
                if "embedding_projection.weight" in suffix:
                    new_key = "projector.projection.weight"
                # embedding_pre_projection_norm has no weight (with_scale=False)
                else:
                    continue
            else:
                continue

            if new_key is None:
                continue

            # Skip ClippableLinear clipping buffers
            if any(
                s in new_key
                for s in ["input_min", "input_max", "output_min", "output_max"]
            ):
                continue

            # Map .linear.weight -> .weight (ClippableLinear -> FakeQuantLinear)
            new_key = new_key.replace(".linear.weight", ".weight")

            # Strip encoder. prefix: encoder.layers.X -> layers.X
            if new_key.startswith("encoder."):
                new_key = new_key[len("encoder.") :]

            # Convert bfloat16 to float32 for loading
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.float()

            mapped[new_key] = tensor

        return mapped

    def set_compile_mode(self, compiled: bool):
        self.model.compile_mode(compiled)

    def set_model_device(self, device, dtype=None):
        if dtype is not None:
            self.model = self.model.to(device=device, dtype=dtype)
        else:
            self.model = self.model.to(device=device)

    def forward(self, pixel_values):
        """Run PyTorch forward pass for calibration."""
        with torch.no_grad():
            return self.model(pixel_values)

    def get_leap_inputs(self, dtype):
        """Return leap input types for compilation."""
        patch_dim = 3 * self.config.patch_size ** 2
        num_patches = self.config.h_patches * self.config.w_patches
        return [leap.TensorType([num_patches, patch_dim], dtype)]

    def compile(self, dtype, output_model_path, vit_core_num=1, **kwargs):
        """Export -> convert_mlir -> compile_hbo -> link."""
        from pathlib import Path

        from leap_llm.nn.utils import statistics

        model = self.model
        assert model.is_compiled, "Model must be in compile mode before compiling."

        kwargs["core_num"] = vit_core_num
        if vit_core_num > 1:
            kwargs["max_l2m_size"] = 25165824

        inputs = self.get_leap_inputs(dtype)

        # Step 1: Export
        bc_path = str(Path(output_model_path).with_suffix(".bc"))
        print("[Gemma4Vision] Exporting...")
        bc_module = model.export_module(inputs, "Gemma4VisionModel", bc_path)

        # Step 2: Convert MLIR
        convert_bc_path = str(Path(output_model_path).with_suffix(".convert.bc"))
        print("[Gemma4Vision] Converting MLIR...")
        mlir_module = model.convert_mlir(
            bc_module,
            save_path=convert_bc_path,
            march=kwargs["march"],
        )
        statistics(mlir_module)

        # Step 3: Compile HBO
        hbo_path = str(Path(output_model_path).with_suffix(".hbo"))
        print(f"[Gemma4Vision] Compiling HBO (core_num={vit_core_num})...")
        hbo_model = model.compile_hbo(mlir_module, hbo_path, **kwargs)

        # Step 4: Link
        hbm_path = output_model_path
        if not hbm_path.endswith(".hbm"):
            hbm_path = str(Path(output_model_path).with_suffix(".hbm"))
        print("[Gemma4Vision] Linking HBM...")
        model.link_models([hbo_model], hbm_path)
        print(f"[Gemma4Vision] Done: {hbm_path}")
