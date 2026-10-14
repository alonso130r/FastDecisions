# Persistent residual RMSNorm: removing repeated row reads

This experiment follows the [combined-gate results](compiled-structure-and-fusion-targets.md#combined-gate-benchmark-results). The gate experiment passed numerical comparisons and improved its isolated projection/transform sequence, but did not demonstrate an integrated model speedup. Its results are already recorded in that note.

## Previous experiment and motivation for changing direction

Combined gates reduced the batch-one isolated sequence from 35.333 to 18.929 microseconds. Across eighteen layers, that saving extrapolated to only about 0.295 ms. Integrated medians were 133.316 ms for packed convolution and 134.075 ms with combined gates, a 0.760 ms increase. The configurations ran sequentially, so this does not establish a definite regression, but it provides no evidence to enable the experiment by default. Tested slot probabilities were unchanged.

Rather than pursue launch-count savings alone, the next experiment targets explicit repeated work visible in generated code: residual-plus-RMSNorm reads both input rows twice and computes their sum twice.

## What the current generated kernel does

Baseline: [partition 06](../data/l4-source-b1-packed/compiled_forward_06.py), function `triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_0`, beginning around line 87.

For each hidden-state row of width 2048, its first loop loads both BF16 inputs, converts them to FP32, adds them, and accumulates squares. After reducing the sum of squares, its second loop reloads both inputs and repeats their addition. It uses the resulting RMS factor and learned scale to write the normalized BF16 output.

The source therefore contains four activation loads per element across the two inputs and two passes. A weight load makes five load expressions in the generated metadata. This is a source-level count, not a measured DRAM count: second-pass accesses may hit cache, and the chosen reduction block can cover the entire row in one iteration.

Streaming is not intrinsically incorrect. It trades retention of row values for additional loads, potentially reducing register pressure. The fixed width of 2048 makes a persistent alternative a contained experiment. No speedup is established merely by noticing the two loops.

## Proposed and implemented computation

Implementation: [persistent_residual_rms_kernel.py](../model/decider/persistent_residual_rms_kernel.py).

The replacement loads both input rows once and keeps their FP32 sums available through the reduction. It reduces their squares to one factor per row, scales the retained values and stores the output. It does not allocate or return a residual tensor.

For residual value $s_{i,d}$ and learned parameter $w_d$:

$$
s_{i,d}=\operatorname{FP32}(a_{i,d})+\operatorname{FP32}(b_{i,d}),
$$

$$
r_i=\operatorname{rsqrt}\left(\frac{1}{2048}\sum_{d=0}^{2047}s_{i,d}^2+10^{-6}\right),
$$

$$
y_{i,d}=\operatorname{BF16}\left((s_{i,d}r_i)(1+\operatorname{FP32}(w_d))\right).
$$

The row shares one reciprocal RMS factor. Each feature still needs multiplication by that factor and its learned scale; this is not literally one divide for the entire output.

The FP32 addition before normalization matches the inspected compiled kernel. A reference that first adds BF16 tensors and materializes their sum may round differently. A different reduction arrangement can also change floating-point accumulation order. Numerical comparisons should use the original generated computation and explicitly stated tolerances.

Retained values are compiler-managed register values, not a manually allocated cache. Register pressure or spills could offset eliminated loads. Compiler output and measurement must establish the actual behavior.

## Launch-mapping correctness fix

The initial draft used one row per program but retained the old streaming-reduction decorator and its `XBLOCK` parameter. Inductor's `Grid1D` divides the row count by the selected `XBLOCK`. If the body ignores that block size and computes only the row given by the program ID, selecting a block size above one leaves output rows unprocessed.

The corrected body handles a tile shaped as `XBLOCK` rows by 2048 features. Row indices start at the program ID times `XBLOCK`; a row-validity mask covers the last partial tile. Each row's reduction spans all 2048 features, with no feature streaming loop. Thus the launch grid and body agree.

## Autotuning cleanup

The cached factory now uses `triton_heuristics.persistent_reduction` with an inner reduction hint. Its row-count size hint is the next power of two of the actual flattened row count; its reduction-width hint stays fixed at 2048.

Removed the unused `R0_BLOCK` parameter and the copied tiling scores. Metadata describes three load expressions, one store and one reduction. It declares the correct pointers, runtime counts and compile-time row block size. Learned weights may be BF16 or FP32; residual inputs and output are BF16. Metadata dictionaries are copied before modification, and no inherited pointer/count alignment assertions are imposed.

The signature retains a runtime `r0_numel` argument supplied as 2048, matching the reduction metadata convention; the body itself fixes the feature width at 2048. This is not support for arbitrary hidden widths.

The factory caches launchers by row count and weight dtype. Hardware metadata still comes from the L4 prototype; this implementation has not been validated as a general-device kernel.

## Launcher contract

`residual_rms(a, b, weight)` accepts matching, nonempty contiguous BF16 inputs ending in dimension 2048, with at least two dimensions. All inputs must share a CUDA device. The contiguous learned-weight vector must have length 2048 and BF16 or FP32 dtype.

The launcher flattens leading dimensions into a row count, allocates a contiguous BF16 output with the same shape, obtains the cached launcher, and calls `.run` on the current device/stream. The persistent heuristic supplies the row block size and `Grid1D` supplies the grid. There is no explicit synchronization or host copy.

## Pseudocode

```text
WRAPPER(a, b, weight):
    validate matching contiguous CUDA BF16 inputs of width 2048
    validate contiguous learned weight and shared device
    flatten leading dimensions into row_count
    allocate BF16 output with the input shape
    obtain cached persistent launcher for row_count and weight dtype
    launch on the current stream with row_count and feature count 2048
    return output

KERNEL(a, b, weight, output, row_count, XBLOCK):
    identify XBLOCK rows starting at program_id times XBLOCK
    identify all 2048 features for each row
    mark rows beyond row_count invalid

    load each input value once and convert to FP32
    residual = first input plus second input
    retain residual values for the full tile

    for each row:
        sum_squares = reduce residual squared across features
        inverse_rms = reciprocal square root of mean square plus epsilon

    load the learned feature weights and convert to FP32
    normalized = retained residual times its row's inverse_rms
    result = normalized times one plus learned weight
    store result as BF16, masking invalid rows
```

## Validation status and next benchmark

The rewritten file passed Python syntax and diff checks at implementation time. No GPU compilation, numerical test, isolated timing or model integration result has been established for this kernel.

The next benchmark should compare against the exact generated two-pass residual-plus-RMSNorm sequence, using matching input layouts and dtypes. Include multiple row counts and partial final row tiles to exercise block coverage. Preserve compilation/autotuning outside timing, compare warmed CUDA graph replay, and inspect selected row block sizes and register/spill behavior where available.

Only after a standalone gain should the operation replace the corresponding model boundary. It needs compiler-compatible integration and full-model probability/latency validation. Do not replace a differently rounded eager residual operation merely because its mathematical formula looks equivalent.
