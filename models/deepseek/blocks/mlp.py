from leap_llm.nn.modules import FakeQuantLinear, FakeQuantMul, FakeQuantSwish
from leap_llm.nn.utils import Module


class MLP(Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        w_bits: int,
        has_scale: bool,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.gate_proj = FakeQuantLinear(
            self.hidden_size,
            self.intermediate_size,
            bias=False,
            w_bits=w_bits,
            has_scale=has_scale,
        )
        self.up_proj = FakeQuantLinear(
            self.hidden_size,
            self.intermediate_size,
            bias=False,
            w_bits=w_bits,
            has_scale=has_scale,
        )
        self.down_proj = FakeQuantLinear(
            self.intermediate_size,
            self.hidden_size,
            bias=False,
            w_bits=w_bits,
            has_scale=has_scale,
        )
        # self.act_fn = ACT2FN[config.hidden_act]
        self.act_fn = FakeQuantSwish(True, 16)

        self.mul = FakeQuantMul(quantized=False)

    def build(self, hidden_state):

        x = self.gate_proj(hidden_state)
        x = self.act_fn(x)
        up_proj_h = self.up_proj(hidden_state)
        x = self.mul(x, up_proj_h)
        return self.down_proj(x)

    def forward(self, hidden_state):
        x = self.gate_proj(hidden_state)
        x = self.act_fn(x)
        up_proj_h = self.up_proj(hidden_state)
        x = self.mul(x, up_proj_h)
        return self.down_proj(x)

        # return self.down_proj(
        #     self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state)
        # )
