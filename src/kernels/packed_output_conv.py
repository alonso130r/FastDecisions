"""Prototype import path; the shared implementation ships with the model runtime."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'model'))
from decider.packed_conv_kernel import (  # noqa: F401,E402
    BASE_TRITON_META,
    BASE_INDUCTOR_META,
    get_packed_conv_kernel,
    mod_triton_poi_fused__unsafe_view_add_constant_pad_nd_mul_select_silu_slice_squeeze_transpose_unsqueeze_0,
    prep_qkv,
)
