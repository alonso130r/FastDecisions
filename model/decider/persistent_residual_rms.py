"""Opt-in decoder-boundary patch for persistent residual RMSNorm experiments."""
from types import MethodType
import torch
from decider.persistent_residual_rms_kernel import residual_rms


@torch.library.custom_op('fastdecisions::residual_rms', mutates_args=(), device_types='cuda')
def residual_rms_op(a: torch.Tensor, b: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return residual_rms(a, b, weight)


@residual_rms_op.register_fake
def _fake(a, b, weight):
    return a.new_empty(a.shape)


def _forward(self, hidden_states, position_embeddings, attention_mask=None,
             position_ids=None, past_key_values=None, **kwargs):
    if (past_key_values is not None or self.training or torch.is_grad_enabled()
            or not hidden_states.is_cuda or hidden_states.dtype != torch.bfloat16
            or hidden_states.shape[-1] != 2048 or not hidden_states.is_contiguous()):
        return self._persistent_original_forward(
            hidden_states, position_embeddings, attention_mask=attention_mask,
            position_ids=position_ids, past_key_values=past_key_values, **kwargs)
    original = hidden_states
    normalized = self.input_layernorm(hidden_states)
    attention = self.linear_attn(hidden_states=normalized, cache_params=None,
                                 attention_mask=attention_mask, **kwargs)
    normalized = residual_rms_op(original, attention, self.post_attention_layernorm.weight)
    mlp_output = self.mlp(normalized)
    # Keep the final residual expression visible to Inductor, as in the original
    # compiled sequence. No BF16 residual is fed into the custom normalization.
    return (original + attention) + mlp_output


def install_persistent_residual_rms(core):
    """Patch only delta decoder layers on this instance; cached calls fall back."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer
    count = 0
    for module in core.modules():
        if isinstance(module, Qwen3_5DecoderLayer) and module.block_type == 'linear_attention':
            norm = module.post_attention_layernorm
            if norm.weight.shape != (2048,) or norm.eps != 1e-6:
                raise ValueError('Persistent RMSNorm requires width 2048 and epsilon 1e-6')
            if not hasattr(module, '_persistent_original_forward'):
                module._persistent_original_forward = module.forward
                module.forward = MethodType(_forward, module)
            count += 1
    return count
