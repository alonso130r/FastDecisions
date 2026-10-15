import torch
import triton
import triton.language as tl


@triton.jit
def native_gqa(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    stride_qb,
    stride_qh,
    stride_qt,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kt,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vt,
    stride_vd,
    stride_ob,
    stride_oh,
    stride_ot,
    stride_od,
    SEQ_LEN: tl.constexpr,
    N_Q_HEADS: tl.constexpr,  # 8
    N_KV_HEADS: tl.constexpr,  # 2
    HEAD_DIM: tl.constexpr,  # 256
    SCALE: tl.constexpr,  # 1 / 16
    BLOCK_M: tl.constexpr,  # query tokens per tile
    BLOCK_N: tl.constexpr,  # key tokens per tile
):
    query_tile = tl.program_id(0)
    batch_head = tl.program_id(1)

    batch = batch_head // N_Q_HEADS
    q_head = batch_head % N_Q_HEADS
    kv_head = q_head // (N_Q_HEADS // N_KV_HEADS)

    query_pos = query_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    key_offset = tl.arange(0, BLOCK_N)
    feature = tl.arange(0, HEAD_DIM)

    q_base = q_ptr + batch * stride_qb + q_head * stride_qh
    k_base = k_ptr + batch * stride_kb + kv_head * stride_kh
    v_base = v_ptr + batch * stride_vb + kv_head * stride_vh
    out_base = out_ptr + batch * stride_ob + q_head * stride_oh

    q = tl.load(
        q_base + query_pos[:, None] * stride_qt + feature[None, :] * stride_qd,
        mask=query_pos[:, None] < SEQ_LEN,
        other=0.0,
    )

    running_max = tl.full((BLOCK_M,), float("-inf"), tl.float32)
    denominator = tl.zeros((BLOCK_M,), tl.float32)
    accumulator = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)

    # enforce causality
    key_limit = tl.minimum((query_tile + 1) * BLOCK_M, SEQ_LEN)

    for key_start in range(0, key_limit, BLOCK_N):
        key_pos = key_start + key_offset

        k = tl.load(
            k_base + feature[:, None] * stride_kd + key_pos[None, :] * stride_kt,
            mask=key_pos[None, :] < SEQ_LEN,
            other=0.0,
        )

        scores = tl.dot(q, k) * SCALE

        allowed = (
            (query_pos[:, None] < SEQ_LEN)
            & (key_pos[None, :] < SEQ_LEN)
            & (key_pos[None, :] <= query_pos[:, None])
        )
        scores = tl.where(allowed, scores, float("-inf"))

        # safe max to avoid infs
        tile_max = tl.max(scores, axis=1)
        new_max = tl.maximum(running_max, tile_max)
        new_max = tl.where(query_pos < SEQ_LEN, new_max, 0.0)

        correction = tl.exp(running_max - new_max)
        probabilities = tl.exp(scores - new_max[:, None])

        v = tl.load(
            v_base + key_pos[:, None] * stride_vt + feature[None, :] * stride_vd,
            mask=key_pos[:, None] < SEQ_LEN,
            other=0.0,
        )

        accumulator = accumulator * correction[:, None]
        accumulator = tl.dot(
            probabilities.to(v.dtype),
            v,
            accumulator,
        )
        denominator = denominator * correction + tl.sum(probabilities, axis=1)
        running_max = new_max

    safe_denominator = tl.where(query_pos < SEQ_LEN, denominator, 1.0)
    output = accumulator / safe_denominator[:, None]

    tl.store(
        out_base + query_pos[:, None] * stride_ot + feature[None, :] * stride_od,
        output.to(out_ptr.dtype.element_ty),
        mask=query_pos[:, None] < SEQ_LEN,
    )


def native_gqa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    # Logical shapes: Q [B, Hq, T, D], K/V [B, Hkv, T, D].
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("Q, K and V must be four-dimensional")

    batch, q_heads, seq_len, head_dim = q.shape
    kv_heads = k.shape[1]

    if min(batch, q_heads, kv_heads, seq_len) < 1:
        raise ValueError("Batch, head counts and sequence length must be positive")
    if q_heads % kv_heads:
        raise ValueError("Query heads must be divisible by KV heads")
    if head_dim != 256:
        raise ValueError("This launcher supports head dimension 256")
    if k.shape != (batch, kv_heads, seq_len, head_dim) or v.shape != k.shape:
        raise ValueError("Incompatible Q/K/V shapes")

    for tensor in (q, k, v):
        if not tensor.is_cuda or tensor.device != q.device:
            raise ValueError("Q, K and V must share a CUDA device")
        if tensor.dtype != torch.bfloat16:
            raise ValueError("Q, K and V must be BF16")
        if tensor.stride(-1) != 1:
            raise ValueError("The feature dimension must be contiguous")

    with torch.cuda.device(q.device):
        # Physical layout [B, T, Hq, D], logical layout [B, Hq, T, D].
        output = torch.empty(
            (batch, seq_len, q_heads, head_dim),
            device=q.device,
            dtype=q.dtype,
        ).permute(0, 2, 1, 3)

        native_gqa[(triton.cdiv(seq_len, 64), batch * q_heads)](
            q,
            k,
            v,
            output,
            *q.stride(),
            *k.stride(),
            *v.stride(),
            *output.stride(),
            SEQ_LEN=seq_len,
            N_Q_HEADS=q_heads,
            N_KV_HEADS=kv_heads,
            HEAD_DIM=head_dim,
            SCALE=head_dim**-0.5,
            BLOCK_M=64,
            BLOCK_N=32,
            num_warps=4,
            num_stages=2,
        )

    return output
