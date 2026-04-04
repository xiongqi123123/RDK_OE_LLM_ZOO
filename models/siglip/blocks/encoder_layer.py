import torch
from hbdk4.compiler import leap

from leap_llm.nn.modules import LayerNormSplit
from leap_llm.nn.utils import Module

from .attention import SiglipAttention
from .mlp import SiglipMLP


class SiglipEncoderLayer(Module):
    def __init__(
        self,
        layer_id: int,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
        layer_norm_eps: float,
    ):
        super().__init__()
        self.layer_id = layer_id

        self.self_attn = SiglipAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
        )
        self.layer_norm1 = LayerNormSplit(hidden_size, eps=layer_norm_eps)
        self.mlp = SiglipMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )
        self.layer_norm2 = LayerNormSplit(hidden_size, eps=layer_norm_eps)

    def build(self, hidden_states):
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = leap.add(residual, hidden_states)

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = leap.add(residual, hidden_states)

        return hidden_states

    def forward(self, hidden_states: torch.Tensor):
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states
