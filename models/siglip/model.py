import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from hbdk4.compiler import leap, save, statistics
from torch import nn

from leap_llm.models.siglip.blocks import SiglipEncoderLayer
from leap_llm.nn.modules import ConstFakeQuant, LayerNormSplit
from leap_llm.nn.utils import Model, Module, timeit


# ---------------------------------------------------------------------------
# PatchEmbedding: Conv2d patch projection with fake quantization
# ---------------------------------------------------------------------------
class PatchEmbedding(Module):
    def __init__(self, embed_dim: int, num_channels: int, patch_size: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_channels = num_channels
        self.patch_size = patch_size

        self.weight = nn.Parameter(
            torch.empty(embed_dim, num_channels, patch_size, patch_size)
        )
        self.bias = nn.Parameter(torch.empty(embed_dim))
        self.x_fake_quant = ConstFakeQuant(8)
        self.absmax_weight = None

    def forward(self, x: torch.Tensor):
        x = self.x_fake_quant(x)
        if self.absmax_weight is None:
            last_dim = self.patch_size * self.patch_size * self.num_channels
            weight = torch.reshape(self.weight.data, [self.embed_dim, last_dim])
            per_channel_max, _ = torch.max(weight.abs(), dim=1)
            self.absmax_weight = per_channel_max

        return F.conv2d(
            input=x,
            weight=self.weight,
            bias=self.bias,
            stride=(self.patch_size, self.patch_size),
        )

    def build(self, x):
        x_cast = leap.cast_type(x, output_type=leap.float32)
        x_quant = self.x_fake_quant(x_cast)

        if self.absmax_weight is None:
            last_dim = self.patch_size * self.patch_size * self.num_channels
            weight = torch.reshape(self.weight.data, [self.embed_dim, last_dim])
            per_channel_max, _ = torch.max(weight.abs(), dim=1)
            self.absmax_weight = per_channel_max

        weight_min = (-self.absmax_weight).tolist()
        weight_max = self.absmax_weight.tolist()
        # NCHW -> NHWC for leap.conv2d
        weight = self.weight.data.permute(0, 2, 3, 1).contiguous()
        weight_quant = leap.const_fake_quant(
            weight, weight_min, weight_max, 8, True, axis=0,
        )
        conv_res = leap.conv2d(
            input=x_quant,
            weight=weight_quant,
            bias=self.bias.data,
            stride=(self.patch_size, self.patch_size),
        )
        return leap.cast_type(conv_res, output_type=x.type.element_type)


# ---------------------------------------------------------------------------
# PositionEmbedding: learned absolute position embedding
# ---------------------------------------------------------------------------
class PositionEmbedding(Module):
    def __init__(self, num_positions: int, embed_dim: int):
        super().__init__()
        self.num_positions = num_positions
        self.embed_dim = embed_dim
        self.weight = nn.Parameter(torch.empty(num_positions, embed_dim))

    def forward(self, x: torch.Tensor):
        return F.embedding(x, self.weight)

    def build(self, x):
        return leap.reshape(
            self.weight.data, shape=[1, self.num_positions, self.embed_dim]
        )


# ---------------------------------------------------------------------------
# SiglipVisionEmbeddings: patch embed + position embed
# ---------------------------------------------------------------------------
class SiglipVisionEmbeddings(Module):
    def __init__(
        self,
        hidden_size: int,
        num_channels: int,
        patch_size: int,
        image_size: int,
    ):
        super().__init__()
        self.embed_dim = hidden_size
        self.patch_size = patch_size
        self.image_size = image_size
        self.num_patches = (image_size // patch_size) ** 2
        self.num_positions = self.num_patches

        self.patch_embedding = PatchEmbedding(hidden_size, num_channels, patch_size)
        self.position_embedding = PositionEmbedding(self.num_positions, hidden_size)
        self.position_ids = (
            torch.arange(self.num_positions).unsqueeze(0).contiguous()
        )

    def forward(self, pixel_values: torch.Tensor):
        patch_embeds = self.patch_embedding(pixel_values)
        embeddings = patch_embeds.flatten(2).transpose(1, 2)
        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings

    def build(self, pixel_values):
        # Ensure patch_embedding weights are f32 so conv2d input/weight types match
        self.patch_embedding.to("cpu", dtype=torch.float32)
        patch_embeds = self.patch_embedding(pixel_values)
        batch = pixel_values.type.shape[0]
        embeddings = leap.reshape(
            patch_embeds, shape=[batch, self.num_positions, self.embed_dim]
        )
        embeddings = leap.add(
            embeddings, self.position_embedding(self.position_ids)
        )
        return embeddings


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class SiglipVisionConfig:
    hidden_size: int = 1152
    image_size: int = 384
    intermediate_size: int = 4304
    layer_norm_eps: float = 1e-06
    num_attention_heads: int = 16
    num_channels: int = 3
    num_hidden_layers: int = 27
    patch_size: int = 14


# ---------------------------------------------------------------------------
# SiglipVisionModel: full vision transformer (Model subclass for compilation)
# ---------------------------------------------------------------------------
class SiglipVisionModel(Model):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config

        self.embeddings = SiglipVisionEmbeddings(
            config.hidden_size,
            config.num_channels,
            config.patch_size,
            config.image_size,
        )

        self.layers = nn.ModuleList()
        for layer_id in range(config.num_hidden_layers):
            self.layers.append(
                SiglipEncoderLayer(
                    layer_id=layer_id,
                    hidden_size=config.hidden_size,
                    num_attention_heads=config.num_attention_heads,
                    intermediate_size=config.intermediate_size,
                    layer_norm_eps=config.layer_norm_eps,
                )
            )

        self.post_layernorm = LayerNormSplit(
            config.hidden_size, eps=config.layer_norm_eps
        )

    def build(self, pixel_values):
        # NCHW -> NHWC for leap conv2d
        pixel_values = leap.transpose(pixel_values, dims=[0, 2, 3, 1])
        hidden_states = self.embeddings(pixel_values)

        for layer in self.layers:
            hidden_states = layer(hidden_states)

        hidden_states = self.post_layernorm(hidden_states)
        return hidden_states

    def forward(self, pixel_values: torch.Tensor):
        hidden_states = self.embeddings(pixel_values)

        for layer in self.layers:
            hidden_states = layer(hidden_states)

        hidden_states = self.post_layernorm(hidden_states)
        return hidden_states


# ---------------------------------------------------------------------------
# SiglipVision: wrapper with weight loading and compilation pipeline
# ---------------------------------------------------------------------------
class SiglipVision:
    @staticmethod
    @timeit
    def load_model(
        input_model_path: str,
        checkpoint: dict,
    ) -> "SiglipVision":
        config_path = os.path.join(input_model_path, "config.json")
        assert os.path.exists(config_path), (
            f"config.json not found in {input_model_path}"
        )
        with open(config_path, "r", encoding="utf-8") as f:
            raw_config = json.load(f)

        # Support both top-level keys and nested "vision_config"
        if "vision_config" in raw_config:
            cfg_dict = raw_config["vision_config"]
        else:
            cfg_dict = raw_config

        model_args_dict = {
            field.name: cfg_dict.get(field.name, field.default)
            for field in fields(SiglipVisionConfig)
        }
        config = SiglipVisionConfig(**model_args_dict)
        model = SiglipVisionModel(config)

        # ---- weight mapping ----
        # Transformers SiglipVisionModel stores weights under:
        #   vision_model.embeddings.patch_embedding.weight/bias
        #   vision_model.embeddings.position_embedding.weight
        #   vision_model.encoder.layers.{i}.layer_norm1.weight/bias
        #   vision_model.encoder.layers.{i}.self_attn.{q,k,v,out}_proj.weight/bias
        #   vision_model.encoder.layers.{i}.layer_norm2.weight/bias
        #   vision_model.encoder.layers.{i}.mlp.fc1.weight/bias
        #   vision_model.encoder.layers.{i}.mlp.fc2.weight/bias
        #   vision_model.post_layernorm.weight/bias

        new_state_dict = {}
        for key, value in checkpoint.items():
            new_key = key

            # Strip "vision_model." prefix if present
            if new_key.startswith("vision_model."):
                new_key = new_key[len("vision_model."):]

            # Strip "encoder." prefix for layer weights
            if new_key.startswith("encoder."):
                new_key = new_key[len("encoder."):]

            # Map self_attn -> self_attn (names already match)
            # Map embeddings.patch_embedding -> embeddings.patch_embedding
            # Map embeddings.position_embedding.weight -> embeddings.position_embedding.weight

            new_state_dict[new_key] = value

        # Convert patch_embedding weight from NCHW to NHWC for our PatchEmbedding
        patch_weight_key = "embeddings.patch_embedding.weight"
        if patch_weight_key in new_state_dict:
            w = new_state_dict[patch_weight_key]
            # Already OIHW (out_channels, in_channels, H, W), keep as-is
            # Our PatchEmbedding stores weight as (embed_dim, num_channels, H, W)
            # and handles permutation internally in forward/build
            new_state_dict[patch_weight_key] = w

        # Position embedding: transformers stores as nn.Embedding(num_positions, dim)
        # We store as nn.Parameter(num_positions, dim) — same shape
        pos_key = "embeddings.position_embedding.weight"
        if pos_key not in new_state_dict:
            # Some checkpoints use "embeddings.position_embedding.weight" directly
            pass

        model.load_state_dict(new_state_dict, strict=False)
        return SiglipVision(model, config)

    def __init__(self, model: SiglipVisionModel, config: SiglipVisionConfig):
        self.model = model
        self.config = config

    def get_leap_inputs(self, dtype) -> list:
        pixel_shape = (
            1,
            self.config.num_channels,
            self.config.image_size,
            self.config.image_size,
        )
        return [leap.TensorType(pixel_shape, dtype)]

    def compile(
        self,
        dtype,
        output_model_path: str,
        core_num: int = 1,
        **kwargs,
    ):
        assert self.model.is_compiled, (
            "Model must be in compile mode before compiling."
        )
        kwargs["core_num"] = core_num
        if core_num > 1:
            kwargs["max_l2m_size"] = 25165824

        inputs = self.get_leap_inputs(dtype)
        bc_path = str(Path(output_model_path).with_suffix(".bc"))
        bc_module = self.model.export_module(inputs, "SiglipVisionModel", bc_path)

        convert_bc_path = str(
            Path(output_model_path).with_suffix(".convert.bc")
        )
        mlir_module = self.model.convert_mlir(
            bc_module,
            save_path=convert_bc_path,
            march=kwargs["march"],
        )
        statistics(mlir_module)

        hbo_path = str(Path(output_model_path).with_suffix(".hbo"))
        hbo_model = self.model.compile_hbo(
            mlir_module,
            hbo_path,
            **kwargs,
        )

        hbm_path = output_model_path
        if not hbm_path.endswith(".hbm"):
            hbm_path = str(Path(output_model_path).with_suffix(".hbm"))
        return self.model.link_models([hbo_model], hbm_path)

    def forward(self, pixel_values: torch.Tensor):
        return self.model(pixel_values)

    def set_compile_mode(self, mode: bool):
        self.model.compile_mode(mode)

    def set_model_device(self, device, dtype):
        self.model.to(device, dtype=dtype)
