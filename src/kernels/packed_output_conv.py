import triton
import triton.language as tl
import torch

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import (
    AutotuneHint,
    ReductionHint,
    TileHint,
    DeviceProperties,
)


@triton_heuristics.pointwise(
    size_hints={"y": 2048, "x": 8192},
    tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={
        "signature": {
            "in_ptr0": "*bf16",
            "in_ptr1": "*bf16",
            "q_ptr": "*bf16",
            "k_ptr": "*bf16",
            "v_ptr": "*bf16",
            "ynumel": "i32",
            "xnumel": "i32",
            "YBLOCK": "constexpr",
            "XBLOCK": "constexpr",
            "SEQ_LEN": "constexpr",
        },
        "device": DeviceProperties(
            type="cuda",
            index=0,
            multi_processor_count=58,
            cc=89,
            major=8,
            regs_per_multiprocessor=65536,
            max_threads_per_multi_processor=1536,
            max_threads_per_block=1024,
            warp_size=32,
        ),
        "constants": {"SEQ_LEN": 2048},
        "native_matmul": False,
        "enable_fp_fusion": True,
        "launch_pdl": False,
        "disable_ftz": False,
        "configs": [
            {
                (0,): [["tt.divisibility", 16]],
                (1,): [["tt.divisibility", 16]],
                (2,): [["tt.divisibility", 16]],
                (3,): [["tt.divisibility", 16]],
                (4,): [["tt.divisibility", 16]],
            }
        ],
    },
    inductor_meta={
        "grid_type": "Grid2D",
        "kernel_name": "mod_triton_poi_fused__unsafe_view_add_constant_pad_nd_mul_select_silu_slice_squeeze_transpose_unsqueeze_0",
        "mutated_arg_names": [],
        "optimize_mem": True,
        "no_x_dim": False,
        "atomic_add_found": False,
        "num_load": 8,
        "num_store": 3,
        "num_reduction": 0,
        "autotune_hints": set(),
        "backend_hash": "0EB81230D0E8E9A69D6E23E8DC64F284B8D62060645A5659C0934B0E14DC7AE4",
        "assert_indirect_indexing": True,
        "autotune_local_cache": True,
        "autotune_pointwise": True,
        "autotune_remote_cache": None,
        "force_disable_caches": False,
        "dynamic_scale_rblock": True,
        "incremental_autotune": False,
        "max_autotune": False,
        "max_autotune_pointwise": False,
        "min_split_scan_rblock": 256,
        "spill_threshold": 16,
        "store_cubin": False,
        "deterministic": False,
        "batch_invariant": False,
        "force_filter_reduction_configs": False,
        "mix_order_reduction_allow_multi_stages": True,
        "dynamic_disable_pipelining": True,
        "are_deterministic_algorithms_enabled": False,
    },
    min_elem_per_thread=0,
)
@triton.jit
def mod_triton_poi_fused__unsafe_view_add_constant_pad_nd_mul_select_silu_slice_squeeze_transpose_unsqueeze_0(
    in_ptr0,
    in_ptr1,
    q_ptr,
    k_ptr,
    v_ptr,
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

    q_mask = valid & (x2 < 2048)
    k_mask = valid & (2048 <= x2) & (x2 < 4096)
    v_mask = valid & (4096 <= x2) & (x2 < 6144)

    tl.store(q_ptr + 2048 * yindex + x2, tmp30, mask=q_mask)
    tl.store(k_ptr + 2048 * yindex + (x2 - 2048), tmp30, mask=k_mask)
    tl.store(v_ptr + 2048 * yindex + (x2 - 4096), tmp30, mask=v_mask)


def prep_qkv(input_qkv: torch.Tensor, weight: torch.Tensor):
    if not input_qkv.is_cuda:
        raise ValueError("input_qkv must be on CUDA")
    if input_qkv.ndim != 3:
        raise ValueError("Expected input_qkv shape [B, 2048, 6144]")

    B, T, C = input_qkv.shape
    if B < 1 or T != 2048 or C != 6144:
        raise ValueError("Expected input_qkv shape [B, 2048, 6144], B >= 1")
    if input_qkv.dtype != torch.bfloat16 or not input_qkv.is_contiguous():
        raise ValueError("input_qkv must be contiguous BF16")
    if weight.device != input_qkv.device:
        raise ValueError("weight and input_qkv must share a device")
    if weight.dtype != torch.bfloat16 or not weight.is_contiguous():
        raise ValueError("weight must be contiguous BF16")
    if tuple(weight.shape) not in ((6144, 4), (6144, 1, 4)):
        raise ValueError("Expected weight shape [6144, 4] or [6144, 1, 4]")
    if input_qkv.data_ptr() % 16 or weight.data_ptr() % 16:
        raise ValueError("Inputs must be aligned to 16 bytes")

    with torch.cuda.device(input_qkv.device):
        shape = (B, T, 16, 128)
        q = torch.empty(shape, device=input_qkv.device, dtype=input_qkv.dtype)
        k = torch.empty_like(q)
        v = torch.empty_like(q)

        kernel = mod_triton_poi_fused__unsafe_view_add_constant_pad_nd_mul_select_silu_slice_squeeze_transpose_unsqueeze_0
        kernel.run(
            input_qkv,
            weight,
            q,
            k,
            v,
            B * T,
            C,
            stream=torch.cuda.current_stream(input_qkv.device).cuda_stream,
        )

    return q, k, v
