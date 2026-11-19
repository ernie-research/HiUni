import math

import torch
import torch.nn as nn
from megatron.core import tensor_parallel
from megatron.core.transformer import MegatronModule

try:
    import megatron.core.parallel_state as mpu
    MEGATRON_INITIALIZED = mpu.is_initialized() if hasattr(mpu, 'is_initialized') else False
except Exception:
    MEGATRON_INITIALIZED = False


class HiGate(MegatronModule):
    """HiGate module in Megatron format with tensor parallel support.

    Gate MLP: Linear(embed_dim -> embed_dim) -> SiLU -> Linear(embed_dim -> 1)
    Output: LayerNorm(low_info + gate * high_info)
    """

    def __init__(self, config, embed_dim):
        super().__init__(config=config)

        tp_size = getattr(config, 'tensor_model_parallel_size', 1)
        use_megatron_parallel = MEGATRON_INITIALIZED and tp_size > 1

        if use_megatron_parallel:
            self.gate_fc1 = tensor_parallel.ColumnParallelLinear(
                embed_dim,
                embed_dim,
                config=config,
                init_method=config.init_method,
                bias=True,
                gather_output=False,
            )
            self.gate_fc2 = tensor_parallel.RowParallelLinear(
                embed_dim,
                1,
                config=config,
                init_method=config.init_method,
                bias=True,
                input_is_parallel=True,
                skip_bias_add=False,
            )
        else:
            self.gate_fc1 = nn.Linear(embed_dim, embed_dim, bias=True)
            self.gate_fc2 = nn.Linear(embed_dim, 1, bias=True)

        self.use_megatron_parallel = use_megatron_parallel
        self.activation = nn.SiLU()
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.hi_info_norm = nn.LayerNorm(embed_dim)
        # self.gate_activation = nn.Tanh()

        # Zero-init: gate starts at tanh(0)=0, gradients bounded by tanh derivative (max=1)
        nn.init.normal_(self.gate_fc1.weight, std=0.02)
        nn.init.normal_(self.gate_fc2.weight, std=0.02)
        nn.init.zeros_(self.gate_fc1.bias)
        nn.init.zeros_(self.gate_fc2.bias)

    def forward(self, low_info, high_info, heat_map=False):
        if self.use_megatron_parallel:
            gate_hidden, _ = self.gate_fc1(low_info)
            gate_hidden = self.activation(gate_hidden)
            hi_gate, _ = self.gate_fc2(gate_hidden)
        else:
            gate_hidden = self.gate_fc1(low_info)
            gate_hidden = self.activation(gate_hidden)
            hi_gate = self.gate_fc2(gate_hidden)

        high_info = self.hi_info_norm(high_info)
        # hi_gate = self.gate_activation(hi_gate)
        output = low_info + hi_gate * high_info
        output = self.layer_norm(output)

        if heat_map:
            hi_gate_map = hi_gate[0, :, 0]
            seq_len = hi_gate_map.shape[0]
            H = W = int(math.sqrt(seq_len))
            heat_map = hi_gate_map.view(H, W).detach().float().cpu().numpy()
            return output, heat_map
        return output