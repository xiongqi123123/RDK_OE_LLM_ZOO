import math
from typing import Optional, Tuple

import torch
from hbdk4.compiler import leap

from leap_llm.nn.modules import (
    ConstFakeQuant,
    FakeQuantLinear,
    FakeQuantMatmul,
    FakeQuantMul,
    FakeQuantSoftmax,
)
from leap_llm.nn.utils import Module


class SiglipAttention(Module):
    """Multi-headed attention for SigLIP vision encoder."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
    ):
        super().__init__()
        self.embed_dim = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = FakeQuantLinear(self.embed_dim, self.embed_dim, bias=True)
        self.k_proj = FakeQuantLinear(self.embed_dim, self.embed_dim, bias=True)
        self.v_proj = FakeQuantLinear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = FakeQuantLinear(self.embed_dim, self.embed_dim, bias=True)

        self.qk = FakeQuantMatmul(8, 16)
        self.sv = FakeQuantMatmul(None, 8)
        self.mul_attn_weight = FakeQuantMul(quantized=False)
        self.softmax = FakeQuantSoftmax(quant_bits=16, quantized=True)

    def build(self, hidden_states):
        batch_size = hidden_states.type.shape[0]
        q_len = hidden_states.type.shape[1]

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = leap.reshape(
            query_states, [batch_size, q_len, self.num_heads, self.head_dim]
        )
        query_states = leap.transpose(query_states, [0, 2, 1, 3])
        key_states = leap.reshape(
            key_states, [batch_size, q_len, self.num_heads, self.head_dim]
        )
        key_states = leap.transpose(key_states, [0, 2, 3, 1])
        value_states = leap.reshape(
            value_states, [batch_size, q_len, self.num_heads, self.head_dim]
        )
        value_states = leap.transpose(value_states, [0, 2, 1, 3])

        attn_weights = self.qk(query_states, key_states)
        attn_weights = leap.mul(attn_weights, self.scale)
        attn_weights = self.softmax(attn_weights)
        attn_output = self.sv(attn_weights, value_states)

        attn_output = leap.transpose(attn_output, [0, 2, 1, 3])
        attn_output = leap.reshape(attn_output, [batch_size, q_len, self.embed_dim])
        attn_output = self.out_proj(attn_output)

        return attn_output

    def forward(self, hidden_states: torch.Tensor):
        batch_size, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            batch_size, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            batch_size, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            batch_size, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        attn_weights = self.qk(query_states, key_states.transpose(2, 3)) * self.scale
        attn_weights = self.softmax(attn_weights)
        attn_output = self.sv(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, q_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output
