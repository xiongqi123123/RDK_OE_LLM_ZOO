import math

import torch
from hbdk4.compiler import leap
from torch import nn

from leap_llm.nn.utils import Module

from .const_fake_quant import ConstFakeQuant
from .ops import (
    FakeQuantAdd,
    FakeQuantMul,
    FakeQuantPow,
    FakeQuantReduceMean,
    FakeQuantRsqrt,
)


class FakeQuantRMSNorm(Module):
    def __init__(
        self, dim: int, eps: float = 1e-6, preserve_precision=False, fuse_norm=False
    ):
        super().__init__()
        self.variance_epsilon = eps
        self.weight = nn.Parameter(torch.ones(dim))

        self.pow = FakeQuantPow(quantized=False)

        quantized = False if preserve_precision else True
        self.reduce_mean = FakeQuantReduceMean(quantized=quantized)
        self.add_eps = FakeQuantAdd(quantized=quantized)
        self.rsqrt = FakeQuantRsqrt(quantized=quantized)

        self.mul_inv_sqrt = FakeQuantMul(quantized=quantized)
        self.mul_weight = FakeQuantMul(quantized=quantized)
        self.weight_fake_quant = ConstFakeQuant(16, quantized=quantized)
        self.absmax_weight = None
        self.fuse_norm = fuse_norm

    def build(self, hidden_states):
        """
        # weight = leap.reshape(self.weight.data, [1, self.weight.shape[-1]])
        # return leap.rms_norm(x, [-1], self.variance_epsilon, weight=weight)
        """

        squared = self.pow(hidden_states, 2)
        variance = self.reduce_mean(squared, [-1])

        adjusted_variance = self.add_eps(variance, self.variance_epsilon)
        inv_sqrt = self.rsqrt(adjusted_variance)
        hidden_states = self.mul_inv_sqrt(hidden_states, inv_sqrt)

        if self.fuse_norm:
            return hidden_states
        else:
            weight_data = self.weight_fake_quant(self.weight.data)
            output = self.mul_weight(weight_data, hidden_states)

            return output

    def forward(self, hidden_states: torch.Tensor):
        """
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)
        """
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)

        squared = self.pow(hidden_states, 2)

        variance = self.reduce_mean(squared, -1, keepdim=True)

        adjusted_variance = self.add_eps(variance, self.variance_epsilon)
        inv_sqrt = self.rsqrt(adjusted_variance)

        hidden_states = self.mul_inv_sqrt(hidden_states, inv_sqrt)

        if self.fuse_norm:
            return hidden_states
        else:
            if self.absmax_weight is None:
                weight_data = self.weight_fake_quant(self.weight.data)

            output = self.mul_weight(weight_data, hidden_states)
            output = output.to(input_dtype)
            return output


class RMSNorm(Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(dim))

        self.scale = 1.0
        self.i_scale = 1.0
        self.i_scale_pow = 1.0
        self.summax_hidden = None
        # max float16 sqrt
        self.max_float16 = 65504.0

    def build(self, x):
        x = leap.mul(x, self.i_scale)
        eps = self.eps * self.i_scale_pow
        weight = leap.reshape(self.weight.data, [1, self.weight.shape[-1]])
        return leap.rms_norm(x, [-1], eps, weight=weight)

    def forward(self, hidden_states: torch.Tensor):
        # for caculate scale
        hidden_states = hidden_states
        h_pow = torch.sum(hidden_states**2, dim=-1)
        curr_absmax = h_pow.max()
        # 更新全局峰值
        if self.summax_hidden is None or curr_absmax > self.summax_hidden:
            self.summax_hidden = curr_absmax
        # 2) 动态算出 raw scale, mul 2 for more robust
        raw_scale = math.sqrt(self.summax_hidden / self.max_float16) * 2
        self.scale = raw_scale if raw_scale > 1.0 else 1.0
        self.i_scale = 1 / self.scale
        self.i_scale_pow = 1 / (self.scale * self.scale)
        # print(
        #     f"\n[Calib] summax={self.summax_hidden:.4f},",
        #     f"scale={self.scale:.6f}",
        #     f"inverse_scale={self.i_scale:.6f}",
        # )
        # pow_in_min = hidden_states.min().item()
        # pow_in_max = hidden_states.max().item()
        # print(f"=== pow in: {pow_in_min:.11f}, pow_in_max:{pow_in_max:.6f}")

        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)

        return self.weight * hidden_states


class Qwen2RMSNorm(Module):
    def __init__(
        self, dim: int, eps: float = 1e-6, preserve_precision=False, fp16_tiny=True
    ):
        super().__init__()
        self.variance_epsilon = eps
        self.fp16_tiny = fp16_tiny
        self.weight = nn.Parameter(torch.ones(dim))
        self.hidden_states_fq = ConstFakeQuant(16)
        self.mul_scale = FakeQuantMul(quantized=True)
        self.pow = FakeQuantPow(quantized=False)
        self.reduce_mean = FakeQuantReduceMean(quantized=True)

        self.preserve_precision = preserve_precision

        quantized = True
        self.add_eps = FakeQuantAdd(quantized=quantized)
        self.rsqrt = FakeQuantRsqrt(quantized=quantized)

        self.mul_inv_sqrt = FakeQuantMul(quantized=quantized)
        self.mul_weight = FakeQuantMul(quantized=quantized)
        self.weight_fake_quant = ConstFakeQuant(16, quantized=quantized)
        self.absmax_weight = None

        self.summax_hidden = None
        self.max_float16 = 65504.0
        self.scale = 1.0
        self.i_scale = 1.0

    def build(self, hidden_states):

        eps_scaled = self.variance_epsilon * (self.i_scale * self.i_scale)

        if self.preserve_precision:
            hidden_states = leap.cast_type(hidden_states, output_type=leap.float32)
            hidden_states = leap.mul(hidden_states, self.i_scale)
            squared = leap.pow(hidden_states, 2)
            variance = leap.reduce_mean(squared, [-1])
            adjusted_variance = leap.add(variance, eps_scaled)
            inv_sqrt = leap.rsqrt(adjusted_variance)
        else:
            hidden_states = self.hidden_states_fq(hidden_states)
            hidden_states = self.mul_scale(hidden_states, self.i_scale)
            squared = self.pow(hidden_states, 2)
            variance = self.reduce_mean(squared, [-1])

            if self.fp16_tiny:
                if eps_scaled < torch.finfo(torch.float16).tiny:
                    eps_scaled = torch.finfo(torch.float16).tiny

            adjusted_variance = self.add_eps(variance, eps_scaled)
            inv_sqrt = self.rsqrt(adjusted_variance)

        hidden_states = self.mul_inv_sqrt(hidden_states, inv_sqrt)
        weight = self.weight_fake_quant(self.weight.data)
        output = self.mul_weight(weight, hidden_states)

        return output

    def forward(self, hidden_states: torch.Tensor):

        hidden_fp32_for_pow = hidden_states.to(torch.float32)
        h_pow = torch.sum(hidden_fp32_for_pow**2, dim=-1)  # float32
        curr_absmax = h_pow.max().item()

        if (self.summax_hidden is None) or (curr_absmax > self.summax_hidden):
            self.summax_hidden = curr_absmax

        raw_scale = math.sqrt(self.summax_hidden / self.max_float16) * 2.0
        self.scale = raw_scale if (raw_scale > 1.0) else 1.0
        self.i_scale = 1.0 / self.scale

        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        hidden_states = self.hidden_states_fq(hidden_states)
        hidden_states = self.mul_scale(hidden_states, self.i_scale)
        squared = self.pow(hidden_states, 2)
        variance = self.reduce_mean(squared, -1, keepdim=True)

        eps_scaled = self.variance_epsilon * (self.i_scale * self.i_scale)

        if self.fp16_tiny:
            if eps_scaled < torch.finfo(torch.float16).tiny:
                eps_scaled = torch.finfo(torch.float16).tiny

        adjusted_variance = self.add_eps(variance, eps_scaled)
        inv_sqrt = self.rsqrt(adjusted_variance)
        hidden_states = self.mul_inv_sqrt(hidden_states, inv_sqrt)

        if self.absmax_weight is None:
            weight_data = self.weight_fake_quant(self.weight.data)

        output = self.mul_weight(weight_data, hidden_states)
        output = output.to(input_dtype)
        return output
