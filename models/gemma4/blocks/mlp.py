import torch
from hbdk4.compiler import leap

from leap_llm.nn.modules import FakeQuantGELU, FakeQuantLinear, FakeQuantMul
from leap_llm.nn.utils import Module


class Gemma4VisionMLP(Module):
    """SwiGLU MLP: down_proj(gelu(gate_proj(x)) * up_proj(x))"""

    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = FakeQuantLinear(hidden_size, intermediate_size, bias=False)
        self.up_proj = FakeQuantLinear(hidden_size, intermediate_size, bias=False)
        self.down_proj = FakeQuantLinear(intermediate_size, hidden_size, bias=False)
        self.act_fn = FakeQuantGELU(quantized=True, quant_bits=16)
        self.gate_mul = FakeQuantMul(quantized=False)

    def build(self, x):
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        hidden = self.gate_mul(gate, up)
        return self.down_proj(hidden)

    def forward(self, x):
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        hidden = self.gate_mul(gate, up)
        return self.down_proj(hidden)
