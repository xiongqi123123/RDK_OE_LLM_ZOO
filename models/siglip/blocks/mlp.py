import torch
from hbdk4.compiler import leap

from leap_llm.nn.modules import FakeQuantLinear, FakeQuantGELU
from leap_llm.nn.utils import Module


class SiglipMLP(Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
    ):
        super().__init__()
        self.fc1 = FakeQuantLinear(hidden_size, intermediate_size, bias=True)
        self.fc2 = FakeQuantLinear(intermediate_size, hidden_size, bias=True)
        self.activation_fn = FakeQuantGELU(quantized=True, quant_bits=16)

    def build(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states
