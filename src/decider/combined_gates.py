"""Opt-in combined gate projection for inference experiments."""
import torch
from decider.unpack_gate_transform_kernel import unpack_gate


@torch.library.custom_op('fastdecisions::unpack_gate', mutates_args=(), device_types='cuda')
def unpack_gate_op(combined: torch.Tensor, a_log: torch.Tensor,
                   dt_bias: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return unpack_gate(combined, a_log, dt_bias)


@unpack_gate_op.register_fake
def _fake(combined, a_log, dt_bias):
    shape = (*combined.shape[:-1], 16)
    return (combined.new_empty(shape, dtype=torch.float32),
            combined.new_empty(shape, dtype=torch.bfloat16))


def install_combined_gates(core):
    """Snapshot weights after loading; activate only in the packed inference branch.

    Original projections remain intact for fallback. Reinvoke after modifying
    projection weights to refresh the nonpersistent combined-weight buffers.
    """
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet
    from decider.packed_conv import install_packed_conv
    install_packed_conv(core)
    count = 0
    for module in core.modules():
        if isinstance(module, Qwen3_5GatedDeltaNet):
            if (module.in_proj_a.weight.shape != (16, 2048)
                    or module.in_proj_b.weight.shape != (16, 2048)
                    or module.in_proj_a.bias is not None
                    or module.in_proj_b.bias is not None):
                raise ValueError('Combined gates require two bias-free [16, 2048] projections')
            with torch.no_grad():
                weight = torch.cat((module.in_proj_a.weight, module.in_proj_b.weight), dim=0).detach()
            if '_combined_gate_weight' in module._buffers:
                module._combined_gate_weight = weight
            else:
                module.register_buffer('_combined_gate_weight', weight, persistent=False)
            count += 1
    return count
