"""Opt-in forward prototype: project final hidden states only at requested slots."""
import torch
import torch.nn.functional as F


class SlotProjectionForward(torch.nn.Module):
    """Return [slot_capacity, options]; caller supplies valid flat row/token indices.

    Input indices are fixed-size CUDA int64 buffers for graph replay. Unused
    entries can point to row/token zero; the caller discards their output rows.
    This leaves the Engine's serving and cache interfaces unchanged.
    """
    def __init__(self, core, option_weight):
        super().__init__()
        self.core = core
        self.register_buffer('option_weight', option_weight, persistent=False)

    def forward(self, input_ids, slot_rows, slot_positions):
        hidden = self.core(input_ids=input_ids, use_cache=False).last_hidden_state
        selected = hidden[slot_rows, slot_positions]
        return F.linear(selected, self.option_weight).float()
