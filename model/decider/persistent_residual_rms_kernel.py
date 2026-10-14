from copy import deepcopy
from functools import lru_cache

import torch
import triton
import triton.language as tl
from torch._inductor.runtime import triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice
from torch._inductor.runtime.hints import ReductionHint

from .packed_conv_kernel import BASE_TRITON_META, BASE_INDUCTOR_META


@lru_cache(maxsize=32)
def get_persistent_residual_rms_kernel(n_rows, weight_dtype):
    if n_rows < 1:
        raise ValueError("n_rows must be positive")
    pointer_types = {torch.bfloat16: "*bf16", torch.float32: "*fp32"}
    meta = deepcopy(BASE_TRITON_META)
    meta["signature"] = {
        "in_ptr0": "*bf16",
        "in_ptr1": "*bf16",
        "in_ptr2": pointer_types[weight_dtype],
        "out_ptr1": "*bf16",
        "xnumel": "i32",
        "r0_numel": "i32",
        "XBLOCK": "constexpr",
    }
    meta["constants"] = {}
    meta["configs"] = [{}]
    ind = deepcopy(BASE_INDUCTOR_META)
    ind.update(
        grid_type="Grid1D",
        kernel_name=f"persistent_residual_rms_rows{n_rows}",
        num_load=3,
        num_store=1,
        num_reduction=1,
    )
    ind.pop("tiling_scores", None)
    return triton_heuristics.persistent_reduction(
        size_hints={"x": triton.next_power_of_2(n_rows), "r0_": 2048},
        reduction_hint=ReductionHint.INNER,
        filename=__file__,
        triton_meta=meta,
        inductor_meta=ind,
    )(persistent_residual_rms)


@triton.jit
def persistent_residual_rms(
    in_ptr0,
    in_ptr1,
    in_ptr2,
    out_ptr1,
    xnumel,
    r0_numel,
    XBLOCK: tl.constexpr,
):
    row = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)[:, None]
    feat = tl.arange(0, 2048)[None, :]
    valid = row < xnumel
    offset = row * 2048 + feat

    a = tl.load(in_ptr0 + offset, mask=valid, other=0.0).to(tl.float32)
    b = tl.load(in_ptr1 + offset, mask=valid, other=0.0).to(tl.float32)
    residual = a + b
    sum_squares = tl.sum(residual * residual, axis=1)[:, None]
    inv_rms = libdevice.rsqrt(sum_squares / 2048.0 + 1e-6)
    weight = tl.load(in_ptr2 + feat).to(tl.float32)
    normalized = residual * inv_rms
    output = normalized * (weight + 1.0)
    tl.store(out_ptr1 + offset, output, mask=valid)


def residual_rms(a: torch.Tensor, b: torch.Tensor, weight: torch.Tensor):
    if a.ndim < 2 or a.shape[-1] != 2048 or a.numel() == 0:
        raise ValueError(
            "Expected nonempty [..., 2048] input with at least two dimensions"
        )
    if a.shape != b.shape:
        raise ValueError("Residual inputs must have identical shapes")
    if not a.is_cuda or b.device != a.device or weight.device != a.device:
        raise ValueError("All tensors must share a CUDA device")
    if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
        raise ValueError("Residual inputs must be BF16")
    if weight.shape != (2048,) or weight.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("weight must be a BF16 or FP32 vector of length 2048")
    if not all(t.is_contiguous() for t in (a, b, weight)):
        raise ValueError("All inputs must be contiguous")

    n_rows = a.numel() // 2048
    with torch.cuda.device(a.device):
        output = torch.empty(a.shape, device=a.device, dtype=torch.bfloat16)
        kernel = get_persistent_residual_rms_kernel(n_rows, weight.dtype)
        kernel.run(
            a,
            b,
            weight,
            output,
            n_rows,
            2048,
            stream=torch.cuda.current_stream(a.device).cuda_stream,
        )
    return output
