import math

import torch
from hbdk4.compiler import leap

from leap_llm.nn.modules import (
    ConstFakeQuant,
    FakeQuantAdd,
    FakeQuantLinear,
    FakeQuantMatmul,
    FakeQuantMul,
    FakeQuantSoftmax,
)
from leap_llm.nn.modules.rms_norm import FakeQuantRMSNorm
from leap_llm.nn.utils import Module


class RotateHalf(Module):
    """Rotates half the hidden dims of the input."""

    def __init__(self):
        super().__init__()
        self.mul = FakeQuantMul(quantized=False)

    def build(self, x):
        d0, d1, head_dim = x.type.shape
        half = head_dim // 2
        x1 = leap.slice(x, [0, 0, 0], [d0, d1, half], [1, 1, 1])
        x2 = leap.slice(x, [0, 0, half], [d0, d1, head_dim], [1, 1, 1])
        x2 = self.mul(-1, x2)
        return leap.concat([x2, x1], 2)

    def forward(self, x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        x2 = self.mul(-1, x2)
        return torch.cat((x2, x1), dim=-1)


class Gemma4VisionAttention(Module):
    """Gemma4 Vision encoder attention with 2D multidimensional RoPE.

    - MHA (num_kv_heads == num_heads)
    - QKV norms (RMSNorm on q/k with scale, on v without scale)
    - 2D RoPE: split head_dim into 2 halves for (x, y) spatial dims
    - No KV cache, no causal mask (bidirectional encoder)
    - scaling = 1.0 (norms replace 1/sqrt(d))
    """

    def __init__(self, hidden_size, num_attention_heads, head_dim, rms_norm_eps):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = head_dim
        self.half_dim = head_dim // 2  # per spatial dimension

        # Projections (no bias, matching HF Gemma4ClippableLinear)
        self.q_proj = FakeQuantLinear(
            hidden_size, num_attention_heads * head_dim, bias=False
        )
        self.k_proj = FakeQuantLinear(
            hidden_size, num_attention_heads * head_dim, bias=False
        )
        self.v_proj = FakeQuantLinear(
            hidden_size, num_attention_heads * head_dim, bias=False, quant_bits=8
        )
        self.o_proj = FakeQuantLinear(
            num_attention_heads * head_dim, hidden_size, bias=False
        )

        # QKV norms
        self.q_norm = FakeQuantRMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = FakeQuantRMSNorm(head_dim, eps=rms_norm_eps)
        # v_norm: no learnable weight (with_scale=False)
        self.v_norm = FakeQuantRMSNorm(head_dim, eps=rms_norm_eps, fuse_norm=True)

        # 2D RoPE components: one set per spatial dimension (x, y)
        # Dim 0 (x)
        self.rotate_half_q0 = RotateHalf()
        self.rotate_half_k0 = RotateHalf()
        self.mul_q_cos0 = FakeQuantMul(quantized=False)
        self.mul_q_sin0 = FakeQuantMul(quantized=False)
        self.add_q0 = FakeQuantAdd(quantized=False)
        self.mul_k_cos0 = FakeQuantMul(quantized=False)
        self.mul_k_sin0 = FakeQuantMul(quantized=False)
        self.add_k0 = FakeQuantAdd(quantized=True)

        # Dim 1 (y)
        self.rotate_half_q1 = RotateHalf()
        self.rotate_half_k1 = RotateHalf()
        self.mul_q_cos1 = FakeQuantMul(quantized=False)
        self.mul_q_sin1 = FakeQuantMul(quantized=False)
        self.add_q1 = FakeQuantAdd(quantized=False)
        self.mul_k_cos1 = FakeQuantMul(quantized=False)
        self.mul_k_sin1 = FakeQuantMul(quantized=False)
        self.add_k1 = FakeQuantAdd(quantized=True)

        self.q_fq = ConstFakeQuant(16, quantized=False)
        self.k_fq = ConstFakeQuant(16, quantized=False)

        # Attention
        self.qk = FakeQuantMatmul(8, 16)
        self.sv = FakeQuantMatmul(None, 8)
        self.softmax = FakeQuantSoftmax(quant_bits=16, quantized=True)

    # ----- 2D RoPE helpers -----

    def _apply_rope_2d_build(self, q, k, cos, sin):
        """Apply 2D multidimensional RoPE in build() mode.

        q, k: [num_heads, num_patches, head_dim]
        cos, sin: [num_patches, head_dim]  (concatenated [cos_x, cos_y])
        """
        num_heads, num_patches, head_dim = q.type.shape
        half = head_dim // 2

        # Split cos/sin into per-dimension parts
        cos0 = leap.slice(cos, [0, 0], [num_patches, half], [1, 1])
        sin0 = leap.slice(sin, [0, 0], [num_patches, half], [1, 1])
        cos1 = leap.slice(cos, [0, half], [num_patches, head_dim], [1, 1])
        sin1 = leap.slice(sin, [0, half], [num_patches, head_dim], [1, 1])

        # Split q, k into per-dimension parts
        q0 = leap.slice(q, [0, 0, 0], [num_heads, num_patches, half], [1, 1, 1])
        q1 = leap.slice(
            q, [0, 0, half], [num_heads, num_patches, head_dim], [1, 1, 1]
        )
        k0 = leap.slice(k, [0, 0, 0], [num_heads, num_patches, half], [1, 1, 1])
        k1 = leap.slice(
            k, [0, 0, half], [num_heads, num_patches, head_dim], [1, 1, 1]
        )

        q0 = self.q_fq(q0)
        k0 = self.k_fq(k0)

        # Apply RoPE to dim 0
        q0_rope = self.add_q0(
            self.mul_q_cos0(q0, cos0), self.mul_q_sin0(self.rotate_half_q0(q0), sin0)
        )
        k0_rope = self.add_k0(
            self.mul_k_cos0(k0, cos0), self.mul_k_sin0(self.rotate_half_k0(k0), sin0)
        )

        # Apply RoPE to dim 1
        q1_rope = self.add_q1(
            self.mul_q_cos1(q1, cos1), self.mul_q_sin1(self.rotate_half_q1(q1), sin1)
        )
        k1_rope = self.add_k1(
            self.mul_k_cos1(k1, cos1), self.mul_k_sin1(self.rotate_half_k1(k1), sin1)
        )

        # Concatenate back
        q_out = leap.concat([q0_rope, q1_rope], 2)
        k_out = leap.concat([k0_rope, k1_rope], 2)
        return q_out, k_out

    def _apply_rope_2d_forward(self, q, k, cos, sin):
        """Apply 2D multidimensional RoPE in forward() mode.

        q, k: [num_heads, num_patches, head_dim]
        cos, sin: [num_patches, head_dim]
        """
        half = self.half_dim

        cos0, cos1 = cos[..., :half], cos[..., half:]
        sin0, sin1 = sin[..., :half], sin[..., half:]
        q0, q1 = q[..., :half], q[..., half:]
        k0, k1 = k[..., :half], k[..., half:]

        q0 = self.q_fq(q0)
        k0 = self.k_fq(k0)

        q0_rope = self.add_q0(
            self.mul_q_cos0(q0, cos0), self.mul_q_sin0(self.rotate_half_q0(q0), sin0)
        )
        k0_rope = self.add_k0(
            self.mul_k_cos0(k0, cos0), self.mul_k_sin0(self.rotate_half_k0(k0), sin0)
        )

        q1_rope = self.add_q1(
            self.mul_q_cos1(q1, cos1), self.mul_q_sin1(self.rotate_half_q1(q1), sin1)
        )
        k1_rope = self.add_k1(
            self.mul_k_cos1(k1, cos1), self.mul_k_sin1(self.rotate_half_k1(k1), sin1)
        )

        q_out = torch.cat([q0_rope, q1_rope], dim=-1)
        k_out = torch.cat([k0_rope, k1_rope], dim=-1)
        return q_out, k_out

    # ----- main methods -----

    def build(self, hidden_states, cos, sin):
        """
        hidden_states: [num_patches, hidden_size]
        cos, sin: [num_patches, head_dim]  (pre-computed 2D RoPE)
        """
        num_patches = hidden_states.type.shape[0]

        # Project
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape: [num_patches, num_heads, head_dim]
        q = leap.reshape(q, [num_patches, self.num_heads, self.head_dim])
        k = leap.reshape(k, [num_patches, self.num_heads, self.head_dim])
        v = leap.reshape(v, [num_patches, self.num_heads, self.head_dim])

        # Transpose to [num_heads, num_patches, head_dim]
        q = leap.transpose(q, [1, 0, 2])
        k = leap.transpose(k, [1, 0, 2])
        v = leap.transpose(v, [1, 0, 2])

        # QKV norms
        q = self.q_norm(q)
        k = self.k_norm(k)
        v = self.v_norm(v)

        # 2D RoPE
        q, k = self._apply_rope_2d_build(q, k, cos, sin)

        # Attention: Q @ K^T (no scaling, norms handle magnitude)
        k_t = leap.transpose(k, [0, 2, 1])
        attn_weights = self.qk(q, k_t)

        # Softmax (bidirectional, no mask)
        attn_weights = self.softmax(attn_weights)

        # Score @ V
        attn_output = self.sv(attn_weights, v)

        # Transpose back: [num_patches, num_heads, head_dim]
        attn_output = leap.transpose(attn_output, [1, 0, 2])
        attn_output = leap.reshape(attn_output, [num_patches, self.hidden_size])

        # Output projection
        return self.o_proj(attn_output)

    def forward(self, hidden_states, cos, sin):
        """
        hidden_states: [1, num_patches, hidden_size]
        cos, sin: [1, num_patches, head_dim]
        """
        batch_size, num_patches, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # [1, num_patches, num_heads, head_dim] -> squeeze batch -> [num_patches, num_heads, head_dim]
        q = q.view(num_patches, self.num_heads, self.head_dim)
        k = k.view(num_patches, self.num_heads, self.head_dim)
        v = v.view(num_patches, self.num_heads, self.head_dim)

        # [num_heads, num_patches, head_dim]
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)

        # QKV norms
        q = self.q_norm(q)
        k = self.k_norm(k)
        v = self.v_norm(v)

        # Squeeze cos/sin batch dim for broadcast: [num_patches, head_dim]
        cos_sq = cos.squeeze(0)
        sin_sq = sin.squeeze(0)

        # 2D RoPE
        q, k = self._apply_rope_2d_forward(q, k, cos_sq, sin_sq)

        # Attention
        k_t = k.transpose(-1, -2)
        attn_weights = self.qk(q, k_t)

        attn_weights = self.softmax(attn_weights)

        attn_output = self.sv(attn_weights, v)

        # [num_heads, num_patches, head_dim] -> [num_patches, num_heads, head_dim]
        attn_output = attn_output.transpose(0, 1)
        attn_output = attn_output.reshape(batch_size, num_patches, self.hidden_size)

        return self.o_proj(attn_output)
