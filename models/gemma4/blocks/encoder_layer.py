import torch
from hbdk4.compiler import leap

from leap_llm.nn.modules.rms_norm import FakeQuantRMSNorm
from leap_llm.nn.utils import Module

from .attention import Gemma4VisionAttention
from .mlp import Gemma4VisionMLP


class Gemma4VisionEncoderLayer(Module):
    """Single encoder layer with 4 RMSNorms (pre/post for both attn and ffn)."""

    def __init__(
        self,
        layer_id,
        hidden_size,
        num_attention_heads,
        head_dim,
        intermediate_size,
        rms_norm_eps,
    ):
        super().__init__()
        self.layer_id = layer_id

        self.self_attn = Gemma4VisionAttention(
            hidden_size, num_attention_heads, head_dim, rms_norm_eps
        )
        self.mlp = Gemma4VisionMLP(hidden_size, intermediate_size)

        self.input_layernorm = FakeQuantRMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = FakeQuantRMSNorm(
            hidden_size, eps=rms_norm_eps
        )
        self.pre_feedforward_layernorm = FakeQuantRMSNorm(
            hidden_size, eps=rms_norm_eps
        )
        self.post_feedforward_layernorm = FakeQuantRMSNorm(
            hidden_size, eps=rms_norm_eps
        )

    def build(self, hidden_states, cos, sin):
        # Attention block
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, cos, sin)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = leap.add(residual, hidden_states)

        # FFN block
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = leap.add(residual, hidden_states)

        return hidden_states

    def forward(self, hidden_states, cos, sin):
        # Attention block
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, cos, sin)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # FFN block
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states
