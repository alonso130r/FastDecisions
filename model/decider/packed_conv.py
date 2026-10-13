"""Inference-only packed QKV preparation for the measured L4/2048-token path."""
from types import MethodType

import torch
import torch.nn.functional as F

from decider.packed_conv_kernel import prep_qkv


# Keep the private Inductor launcher outside Dynamo tracing. Fake outputs expose
# the exact allocation contract; the custom op is a node, not a graph break.
@torch.library.custom_op('fastdecisions::packed_conv', mutates_args=(), device_types='cuda')
def packed_conv(x: torch.Tensor, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return prep_qkv(x, weight)


@packed_conv.register_fake
def _packed_conv_fake(x, weight):
    batch, seq, _ = x.shape
    shape = (batch, seq, 16, 128)
    return tuple(x.new_empty(shape) for _ in range(3))


def _forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
    # The original method owns all cache updates, alternate geometries, training,
    # and lengths not yet validated by the prototype's correctness suite.
    if (cache_params is not None or self.training or torch.is_grad_enabled()
            or not hidden_states.is_cuda or hidden_states.dtype != torch.bfloat16
            or hidden_states.shape[1] != 2048
            or self.key_dim != 2048 or self.value_dim != 2048
            or self.num_k_heads != 16 or self.num_v_heads != 16
            or self.head_k_dim != 128 or self.head_v_dim != 128
            or self.conv1d.weight.shape != (6144, 1, 4)
            or self.conv1d.bias is not None or self.activation != 'silu'):
        return self._packed_original_forward(
            hidden_states, cache_params=cache_params,
            attention_mask=attention_mask, **kwargs
        )

    from transformers.models.qwen3_5 import modeling_qwen3_5 as mq
    hidden_states = mq.apply_mask_to_padding_states(hidden_states, attention_mask)
    batch, seq, _ = hidden_states.shape
    mixed = self.in_proj_qkv(hidden_states)
    query, key, value = packed_conv(mixed, self.conv1d.weight)
    z = self.in_proj_z(hidden_states).reshape(batch, seq, -1, self.head_v_dim)
    b = self.in_proj_b(hidden_states)
    a = self.in_proj_a(hidden_states)
    beta = b.sigmoid()
    g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
    core_attn_out, _ = mq.torch_chunk_gated_delta_rule(
        query, key, value, g=g, beta=beta, initial_state=None,
        output_final_state=False, use_qk_l2norm_in_kernel=True,
        cu_seqlens=kwargs.pop('cu_seq_lens_q', None), **kwargs,
    )
    core_attn_out = self.norm(core_attn_out.reshape(-1, self.head_v_dim),
                              z.reshape(-1, self.head_v_dim))
    return self.out_proj(core_attn_out.reshape(batch, seq, -1))


def install_packed_conv(core):
    """Patch this model's delta modules only; return the number installed."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet
    count = 0
    for module in core.modules():
        if isinstance(module, Qwen3_5GatedDeltaNet):
            if not hasattr(module, '_packed_original_forward'):
                module._packed_original_forward = module.forward
                module.forward = MethodType(_forward, module)
            count += 1
    return count
