from torch._inductor.runtime.triton_compat import libdevice
from torch._inductor.runtime import triton_heuristics
import triton
import triton.language as tl
import torch
from copy import deepcopy
from functools import lru_cache

from .packed_conv_kernel import BASE_TRITON_META, BASE_INDUCTOR_META


@lru_cache(maxsize=32)
def get_gate_transform_kernel(n_tok, a_log_dtype, bias_dtype):
    pointer_types = {
        torch.bfloat16: "*bf16",
        torch.float32: "*fp32",
    }

    meta = deepcopy(BASE_TRITON_META)
    meta["signature"] = {
        "comb_ptr": "*bf16",
        "a_log_ptr": pointer_types[a_log_dtype],
        "dt_bias_ptr": pointer_types[bias_dtype],
        "decay_out_ptr": "*fp32",
        "beta_out_ptr": "*bf16",
        "xnumel": "i32",
        "XBLOCK": "constexpr",
    }
    meta["constants"] = {}
    # Avoid inheriting alignment assumptions from the other kernel.
    meta["configs"] = [{}]

    ind = deepcopy(BASE_INDUCTOR_META)
    ind.update(
        grid_type="Grid1D",
        kernel_name=f"unpack_gate_transform_n{n_tok}",
        num_load=4,
        num_store=2,
        num_reduction=0,
    )
    ind.pop("tiling_scores", None)

    return triton_heuristics.pointwise(
        size_hints={"x": triton.next_power_of_2(n_tok * 16)},
        filename=__file__,
        triton_meta=meta,
        inductor_meta=ind,
        min_elem_per_thread=0,
    )(unpack_gate_transform)


@triton.jit
def unpack_gate_transform(
    comb_ptr,
    a_log_ptr,
    dt_bias_ptr,
    decay_out_ptr,
    beta_out_ptr,
    xnumel,
    XBLOCK: tl.constexpr,
):
    offset = tl.program_id(0) * XBLOCK
    index = offset + tl.arange(0, XBLOCK)
    tok_idx = index // 16
    head_idx = index % 16
    # 3rd dim: 0 <= c < 16 => a, 16 <= c < 32 => b

    valid = index < xnumel

    a = tl.load(comb_ptr + tok_idx * 32 + head_idx, mask=valid, other=0.0).to(
        tl.float32
    )
    b = tl.load(comb_ptr + tok_idx * 32 + head_idx + 16, mask=valid, other=0.0).to(
        tl.float32
    )

    a_log = tl.load(a_log_ptr + head_idx, mask=valid, other=0.0).to(tl.float32)
    bias = tl.load(dt_bias_ptr + head_idx, mask=valid, other=0.0).to(tl.float32)

    # compute decay transform for a
    x = a + bias
    softplus = tl.where(x > 20, x, libdevice.log1p(libdevice.exp(tl.minimum(x, 20))))
    g = -libdevice.exp(a_log) * softplus

    # sigmoid on b
    b_sig = tl.sigmoid(b).to(tl.bfloat16)

    tl.store(decay_out_ptr + index, g, mask=valid)
    tl.store(beta_out_ptr + index, b_sig, mask=valid)


def unpack_gate(combined, a_log, dt_bias):
    # combined: contiguous BF16 [B, T, 32]
    if combined.ndim != 3 or combined.shape[-1] != 32:
        raise ValueError("combined must have shape [B, T, 32]")
    if not combined.is_cuda or combined.dtype != torch.bfloat16:
        raise ValueError("combined must be CUDA BF16")
    if not combined.is_contiguous():
        raise ValueError("combined must be contiguous")

    B, T, _ = combined.shape
    if B < 1 or T < 1:
        raise ValueError("B and T must be positive")

    for name, parameter in (("a_log", a_log), ("dt_bias", dt_bias)):
        if parameter.shape != (16,):
            raise ValueError(f"{name} must have shape [16]")
        if parameter.device != combined.device:
            raise ValueError(f"{name} must share combined's device")
        if parameter.dtype not in (torch.bfloat16, torch.float32):
            raise ValueError(f"{name} must be BF16 or FP32")
        if not parameter.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    with torch.cuda.device(combined.device):
        decay = torch.empty((B, T, 16), device=combined.device, dtype=torch.float32)
        beta = torch.empty((B, T, 16), device=combined.device, dtype=torch.bfloat16)

        kernel = get_gate_transform_kernel(B * T, a_log.dtype, dt_bias.dtype)
        kernel.run(
            combined,
            a_log,
            dt_bias,
            decay,
            beta,
            B * T * 16,
            stream=torch.cuda.current_stream(combined.device).cuda_stream,
        )

    return decay, beta
