"""Benchmark baseline copied from the generated two-pass kernel.

Row count is parameterized; arithmetic and two-loop structure are preserved.
"""
from copy import deepcopy
from functools import lru_cache
import triton
import triton.language as tl
from torch._inductor.runtime import triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice
from torch._inductor.runtime.hints import ReductionHint
from .packed_conv_kernel import BASE_TRITON_META, BASE_INDUCTOR_META


@lru_cache(maxsize=32)
def get_two_pass_kernel(n_rows):
    meta = deepcopy(BASE_TRITON_META)
    meta['signature'] = {
        'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16',
        'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32',
        'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr',
    }
    meta['constants'] = {}
    meta['configs'] = [{}]
    ind = deepcopy(BASE_INDUCTOR_META)
    ind.update(grid_type='Grid1D', kernel_name=f'two_pass_residual_rms_rows{n_rows}',
               num_load=5, num_store=1, num_reduction=1)
    ind.pop('tiling_scores', None)
    return triton_heuristics.reduction(
        size_hints={'x': triton.next_power_of_2(n_rows), 'r0_': 2048},
        reduction_hint=ReductionHint.INNER, filename=__file__,
        triton_meta=meta, inductor_meta=ind,
    )(two_pass_residual_rms)


@triton.jit
def two_pass_residual_rms(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 2048
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp6 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 =r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tmp3 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(r0_mask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp8 = tl.load(in_ptr0 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp9 = tl.load(in_ptr1 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp18 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp10 = tmp8 + tmp9
        tmp11 = tmp10.to(tl.float32)
        tmp12 = tl.full([1, 1], 2048.0, tl.float32)
        tmp13 = (tmp6 / tmp12)
        tmp14 = tl.full([1, 1], 1e-06, tl.float32)
        tmp15 = tmp13 + tmp14
        tmp16 = libdevice.rsqrt(tmp15)
        tmp17 = tmp11 * tmp16
        tmp19 = tmp18.to(tl.float32)
        tmp20 = tl.full([1, 1], 1.0, tl.float32)
        tmp21 = tmp19 + tmp20
        tmp22 = tmp17 * tmp21
        tmp23 = tmp22.to(tl.float32)
        tl.store(out_ptr1 + (r0_1 + 2048*x0), tmp23, r0_mask & xmask)
