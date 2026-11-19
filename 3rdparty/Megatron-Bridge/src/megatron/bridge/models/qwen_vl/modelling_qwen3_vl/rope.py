# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import List, Optional

import torch
from megatron.core.packed_seq_params import PackedSeqParams
from torch import Tensor
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding
from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeTextRotaryEmbedding


class Qwen3VLMoETextRotaryEmbedding(Qwen3VLMoeTextRotaryEmbedding):
    """Qwen3-VL MoE text rotary position embedding."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Keep a fp32 copy of inv_freq to avoid bf16 precision loss
        self._inv_freq_fp32 = self.inv_freq.float().clone()

    def forward(
        self,
        position_ids: torch.Tensor,
        mrope_section: List[int],
        packed_seq_params: Optional[PackedSeqParams] = None,
        **kwargs,
    ) -> Tensor:
        """Forward pass of multimodal RoPE embedding.

        Args:
            position_ids (torch.Tensor): A postion_id tensor with shape [3, batchsize, seqlens]
            mrope_section (list[int]): Multimodal rope section is for channel dimension of temporal,
                height and width in rope calculation.

        Returns:
            Tensor: Raw frequency embeddings for Megatron Core (shape: [seq_length, bs, 1, dim]).
                    Megatron Core will compute cos/sin internally and apply attention_scaling.
        """
        # Use fp32 inv_freq to avoid bf16 quantization error accumulating with position
        inv_freq = self._inv_freq_fp32.to(device=position_ids.device)

        # shape (3, bs, dim, 1)
        inv_freq_expanded = inv_freq[None, None, :, None].expand(3, position_ids.shape[1], -1, 1)
        # shape (3, bs, 1, seq_length)
        seq_expanded = position_ids[:, :, None, :].float()
        # shape (3, bs, seq_length, dim)
        if isinstance(position_ids.device.type, str) and position_ids.device.type != "mps":
            device_type = position_ids.device.type
        else:
            device_type = "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded @ seq_expanded).transpose(2, 3)
        freqs = self.apply_interleaved_mrope(freqs, mrope_section)
        emb = torch.cat((freqs, freqs), dim=-1)
        emb = emb[..., None, :].transpose(0, 1).contiguous()
        _ = packed_seq_params  # packed sequences not supported yet
        return emb


class Qwen3VLTextRotaryEmbedding(Qwen3VLTextRotaryEmbedding):
    """Qwen3-VL text rotary position embedding for non-MoE models."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Keep a fp32 copy of inv_freq to avoid bf16 precision loss
        self._inv_freq_fp32 = self.inv_freq.float().clone()

    def forward(
        self,
        position_ids: torch.Tensor,
        mrope_section: List[int] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        **kwargs,
    ) -> Tensor:
        """Forward pass for non-MoE Qwen3-VL RoPE.

        Args:
            position_ids: Position IDs tensor
            mrope_section: Optional mrope section (if not provided, uses self.mrope_section)
        """
        if mrope_section is None:
            mrope_section = self.mrope_section

        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        # Use fp32 inv_freq to avoid bf16 quantization error accumulating with position
        inv_freq = self._inv_freq_fp32.to(device=position_ids.device)

        inv_freq_expanded = inv_freq[None, None, :, None].expand(3, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[:, :, None, :].float()  # shape (3, bs, 1, positions)
        if isinstance(position_ids.device.type, str) and position_ids.device.type != "mps":
            device_type = position_ids.device.type
        else:
            device_type = "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded @ position_ids_expanded).transpose(2, 3)
        freqs = self.apply_interleaved_mrope(freqs, mrope_section)
        emb = torch.cat((freqs, freqs), dim=-1)
        emb = emb[..., None, :].transpose(0, 1).contiguous()
        _ = packed_seq_params  # packed sequences not supported yet
        return emb
