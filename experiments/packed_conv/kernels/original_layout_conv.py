"""Baseline arithmetic and channel-major stores from the generated producer.

Row masking and sequence length are parameterized identically to the candidate.
"""
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice
from torch._inductor.runtime import triton_heuristics
from torch._inductor.runtime.hints import TileHint
from copy import deepcopy
from functools import lru_cache
from .packed_output_conv import BASE_TRITON_META, BASE_INDUCTOR_META


@lru_cache(maxsize=32)
def get_original_conv_kernel(batch_size, seq_len):
    meta = deepcopy(BASE_TRITON_META)
    meta["signature"] = {
        "in_ptr0": "*bf16", "in_ptr1": "*bf16", "out_ptr": "*bf16",
        "ynumel": "i32", "xnumel": "i32", "YBLOCK": "constexpr",
        "XBLOCK": "constexpr", "SEQ_LEN": "constexpr",
    }
    meta["constants"] = {"SEQ_LEN": seq_len}
    meta["configs"] = [{(i,): [["tt.divisibility", 16]] for i in range(3)}]
    ind = deepcopy(BASE_INDUCTOR_META)
    ind["num_store"] = 1
    ind["kernel_name"] = f"original_layout_conv_b{batch_size}_t{seq_len}"
    return triton_heuristics.pointwise(
        size_hints={"y": triton.next_power_of_2(batch_size * seq_len), "x": 8192},
        tile_hint=TileHint.DEFAULT, filename=__file__,
        triton_meta=meta, inductor_meta=ind, min_elem_per_thread=0,
    )(original_layout_conv)


@triton.jit
def original_layout_conv(
    in_ptr0,
    in_ptr1,
    out_ptr,
    ynumel,
    xnumel,
    YBLOCK: tl.constexpr,
    XBLOCK: tl.constexpr,
    SEQ_LEN: tl.constexpr,
):
    xnumel = 6144
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    ymask = yindex < ynumel
    valid = xmask & ymask
    y0 = yindex % SEQ_LEN
    x2 = xindex
    y3 = yindex
    tmp4 = tl.load(in_ptr1 + (3 + 4 * x2), xmask, eviction_policy="evict_last").to(
        tl.float32
    )
    tmp9 = tl.load(in_ptr1 + (4 * x2), xmask, eviction_policy="evict_last").to(
        tl.float32
    )
    tmp15 = tl.load(in_ptr1 + (1 + 4 * x2), xmask, eviction_policy="evict_last").to(
        tl.float32
    )
    tmp21 = tl.load(in_ptr1 + (2 + 4 * x2), xmask, eviction_policy="evict_last").to(
        tl.float32
    )
    tmp0 = (y0).to(tl.int32)
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + (x2 + 6144 * y3), tmp2 & valid, other=0.0).to(tl.float32)
    tmp5 = tmp3 * tmp4
    tmp6 = ((-3) + y0).to(tl.int32)
    tmp7 = tmp6 >= tmp1
    tmp8 = tl.load(in_ptr0 + ((-18432) + x2 + 6144 * y3), tmp7 & valid, other=0.0).to(
        tl.float32
    )
    tmp10 = tmp8 * tmp9
    tmp11 = tmp5 + tmp10
    tmp12 = ((-2) + y0).to(tl.int32)
    tmp13 = tmp12 >= tmp1
    tmp14 = tl.load(in_ptr0 + ((-12288) + x2 + 6144 * y3), tmp13 & valid, other=0.0).to(
        tl.float32
    )
    tmp16 = tmp14 * tmp15
    tmp17 = tmp11 + tmp16
    tmp18 = ((-1) + y0).to(tl.int32)
    tmp19 = tmp18 >= tmp1
    tmp20 = tl.load(in_ptr0 + ((-6144) + x2 + 6144 * y3), tmp19 & valid, other=0.0).to(
        tl.float32
    )
    tmp22 = tmp20 * tmp21
    tmp23 = tmp17 + tmp22
    tmp24 = tmp23.to(tl.float32)
    tmp25 = -tmp24
    tmp26 = libdevice.exp(tmp25)
    tmp27 = tl.full([1, 1], 1.0, tl.float32)
    tmp28 = tmp26 + tmp27
    tmp29 = tmp24 / tmp28
    tmp30 = tmp29.to(tl.float32)

    batch = yindex // SEQ_LEN
    tl.store(out_ptr + batch * (6144 * SEQ_LEN) + x2 * SEQ_LEN + y0,
             tmp30, mask=valid)
