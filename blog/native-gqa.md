# Native GQA: implementation, numerical diagnostics and model benchmarks

This records the work after the persistent residual RMSNorm experiment, through the combined packed-convolution/native-GQA benchmark. The next proposed experiment, projecting only decision-slot hidden states, has not been implemented.

## Motivation and scope

The six full-attention layers use eight query heads, two KV heads and head dimension 256. They are separate from the eighteen FLA delta layers. This experiment does not modify FLA internals.

The existing masked SDPA path expands each KV head four times before attention. Native grouped-query attention can read the original two KV heads directly. Combining native grouping with implicit causality also avoids explicit causal-mask work in the replacement attention path. Earlier profile totals were about 10.990 ms for full attention and 0.459 ms for head expansion plus mask conversion. These were motivation, not guaranteed savings: improving attention execution itself matters more than eliminating copies alone.

The kernel handles square, uninterrupted causal self-attention without dropout. It does not implement segment isolation, arbitrary padding masks, rectangular cached attention or backward. The model benchmark checks supplied masks before replacing attention and only patches its own uncached inference instance. Production native-GQA defaults have not been enabled.

## Layout and implementation

Implementation: [native_gqa_kernel.py](../model/decider/native_gqa_kernel.py).

Logical tensors are Q `[B, 8, T, 256]`, K/V `[B, 2, T, 256]`. Every tensor receives its own batch/head/token/feature strides. Feature stride one is required by the wrapper; no preliminary contiguous copies are inserted.

The compiled dump had token-major Q and V, with head-major K. At batch one and length 2048:

| Tensor | Compiled-dump strides in logical B/H/T/D order |
| --- | --- |
| Q | `[4194304, 256, 2048, 1]` |
| K | `[1048576, 524288, 256, 1]` |
| V | `[1048576, 256, 512, 1]` |

The later real-input diagnostic captured eager Q with strides `[4194304, 524288, 256, 1]`; K and V matched the rows above. This is why actual strides are passed instead of assuming every execution has one physical layout. There is no universally best layout independent of the producer and consumer; copying to obtain a preferred attention layout must earn back its cost.

Output storage is allocated physically as `[B, T, 8, 256]`, exposed logically as `[B, 8, T, 256]`. Transposing back for the output gate yields contiguous token-major storage.

Each program owns a query-token tile and one batch/query-head pair. Its KV head is `query_head // 4`. It keeps Q and an FP32 output accumulator resident while iterating over key/value tiles. The online softmax tracks a running maximum and denominator, rescaling the accumulated numerator and denominator whenever the maximum changes.

Pseudocode:

```text
for each batch, query head and query-token tile:
    kv head = query head // 4
    load query tile using Q's strides
    running maximum = negative infinity
    denominator = 0
    output accumulator = 0 in FP32

    for key tiles up to the end of this query tile:
        load K tile using K's strides
        scores = dot(Q, K) / sqrt(256)
        mask future keys and positions outside the sequence
        new maximum = max(previous maximum, row maximum of scores)
        correction = exp(previous maximum - new maximum)
        probabilities = exp(scores - new maximum)
        load V tile using V's strides
        accumulator = correction * accumulator + dot(BF16(probabilities), V)
        denominator = correction * denominator + sum(probabilities)
        running maximum = new maximum

    write BF16(accumulator / denominator) using output strides
```

Wholly future key tiles are already skipped by the loop bound. The partially overlapping causal region still computes a full dot-product tile and masks its future scores. Eliminating all triangular work is not a simple element mask because tensor-core matrix products operate on tiles. A separate fully visible loop and masked diagonal loop remains an unimplemented candidate.

The launcher allocates output and supplies the fixed tile configuration on the current CUDA stream. `num_warps` and `num_stages` are Triton launch/compiler options, not arguments declared in the JIT kernel signature.

## Offline configuration sweep

We chose an offline sweep instead of runtime autotuning. The pointwise heuristic does not generate attention tile choices for the query/key loop. Explicit offline measurements let normal launches use a fixed choice with no configuration search at startup.

[bench_native_gqa_sweep.py](../src/bench_native_gqa_sweep.py) tested:

- Query tile `BLOCK_M`: 16, 32, 64.
- Key tile `BLOCK_N`: 32, 64.
- Warps: 4, 8; stages: 1, 2.

That is 24 configurations per batch, 72 measurements across batches 1, 2 and 4, at sequence length 2048. Twenty-two configurations per batch passed the FP32 attention check. Both 64-by-64, two-stage configurations exceeded the L4 shared-memory limit: required 106496 bytes versus 101376 available.

The winner at every batch was **BLOCK_M=64, BLOCK_N=32, four warps, two stages**.

| Batch | Winner median, microseconds | Runner-up median, microseconds |
| --- | ---: | ---: |
| 1 | 538.163 | 552.602 |
| 2 | 1014.886 | 1036.749 |
| 4 | 2019.635 | 2097.971 |

Fifteen samples, twenty replays per sample, randomized configuration order, CUDA-event timing and compilation excluded. This supported an initial fixed configuration, not universal optimality. [Raw sweep results](../data/native-gqa-sweep.json).

## Standalone BF16 benchmark

[bench_native_gqa.py](../src/bench_native_gqa.py) compares compiled explicit KV expansion, causal-mask construction and SDPA against the fixed native kernel. Inputs are synthetic BF16 with the compiled model's layouts. It is not an exact replay of the generated-source wrapper. Both paths are checked against FP32 math SDPA before timing.

NVIDIA L4, torch 2.14.1+cu130, Triton 3.8.0, sequence length 2048; thirty samples of one hundred CUDA-graph replays each. Configurations are interleaved; compilation/capture are excluded.

| Batch | Expanded SDPA median, us | Native BF16 median, us | Speedup | Saved, us |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1594.916 | 557.322 | 2.862x | 1037.594 |
| 2 | 3350.907 | 1010.314 | 3.317x | 2340.593 |
| 4 | 6744.673 | 2080.584 | 3.242x | 4664.090 |

Maximum FP32-reference errors were identical for the two paths in this run: 0.0083423, 0.0083392 and 0.0082250 respectively. Equal maximum errors do not imply equal tensors. Checks used absolute and relative tolerance 0.02.

Extrapolating the batch-one saving across six attention layers suggested about 6.23 ms, pending integrated measurement.

## Initial model validation failure

[bench_native_gqa_model.py](../src/bench_native_gqa_model.py) preserves projections, Q/K normalization, rotary embeddings, the output gate and projection. It substitutes attention through a `torch.library.custom_op` with fake outputs describing the correct physical layout. It retains baseline and candidate graphs in separate memory pools and interleaves replay timing.

Both initial configurations already included packed convolution. The candidate replaced six full-attention calls. The failed combined-gate and persistent residual RMSNorm patches were never installed.

The initial BF16 model comparison failed before timing or decision-probability checks:

- 5421 of 522240 logits, about 1.0%, exceeded `atol=0.05, rtol=0.02`.
- Maximum absolute difference: 0.2109375.
- Reported maximum relative difference: infinity.

Infinite relative error can result from a zero baseline and a finite nonzero candidate. It does not establish infinite tensor contents. These thresholds are diagnostic thresholds, not an established decision-quality boundary. Passing local attention tolerances does not guarantee passing final-logit tolerances after subsequent nonlinear layers.

## First full-attention layer diagnostics

[diagnose_native_gqa.py](../src/diagnose_native_gqa.py) captures real inputs to the first full-attention layer, layer index 3, using the first real request in a 2048-token bucket. It compares attention on identical Q/K/V and complete eager/compiled layer outputs. A control puts existing SDPA behind the same custom-op boundary.

The first diagnostic attempt failed because `Decider(use_graphs=False)` did not create an Engine. The harness was corrected to create the Engine while calling the eager forward directly. The successful run used the SDPA backend. It is a first-layer diagnostic on inputs from eager preceding layers, not a full compiled-model trace.

| BF16 diagnostic | Maximum absolute difference | RMS difference |
| --- | ---: | ---: |
| Native attention vs existing attention | 0.03125 | 0.000508120 |
| Native attention vs FP32 | 0.0155373 | 0.000920803 |
| Existing attention vs FP32 | 0.0157127 | 0.000922086 |
| Native vs existing attention on compiled-prepared Q/K/V | 0.015625 | 0.000513696 |
| Native complete eager layer vs original | 0.00390625 | 0.0000745621 |
| Native complete compiled layer vs compiled original | 0.0078125 | 0.0000748709 |
| Existing attention behind custom-op boundary vs compiled original | 0 | 0 |

All attention/layer comparisons passed their respective tolerances and all compared outputs were finite. Compiling Q/K preparation itself changed values from eager by up to 0.0625; V and gate preparation were identical. This is evidence of normal compiler-dependent rounding, not proof it caused the end-to-end failure.

Static inspection found no obvious stride, head-mapping or online-softmax recurrence error. The main numerical suspect was converting unnormalized probabilities to BF16 before the probability-V dot product, along with a different softmax reduction order. This is common attention arithmetic, not inherently a bug. The exact SDPA backend's scaling, tile/reduction order and exponential implementation would need matching to pursue bitwise equality. Improving FP32-reference accuracy is a different objective from matching SDPA's rounding.

The boundary-only control was bitwise identical in this first layer. That weakens a boundary-only explanation here, but does not rule out later-layer problems. Progressive replacement of one through six attention layers was proposed but not performed.

## FP16 probability-V experiment and rollback

We temporarily cast both probabilities and the loaded V tile to FP16 for the second dot product, leaving Q/K and final output BF16 and accumulation FP32. Probabilities lie in [0,1], where FP16 offers more precision than BF16. V must have a compatible dot operand dtype; its conversion also introduces FP16's narrower exponent range. FP32 probability-V multiplication was discussed as an accuracy control but not implemented. Larger key tiles and a more accurate exponential were also only proposed.

New real-input diagnostics:

| FP16 diagnostic | Maximum absolute difference | RMS difference |
| --- | ---: | ---: |
| Native attention vs existing attention | 0.015625 | 0.000571999 |
| Native attention vs FP32 | 0.0153499 | 0.000906913 |
| Existing attention vs FP32 | 0.0153499 | 0.000921477 |
| Native compiled layer vs compiled original | 0.00390625 | 0.0000801807 |
| Boundary-only control vs compiled original | 0 | 0 |

Native FP32-reference RMS error was about 1.6% lower than existing SDPA in this run. The existing path's values also changed between runs, so before/after maximum comparisons are not a controlled measurement on identical saved inputs.

Standalone FP16 results:

| Batch | Expanded SDPA median, us | Native FP16 median, us | Speedup | Native max FP32 error |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1576.520 | 576.686 | 2.734x | 0.00781846 |
| 2 | 3280.932 | 1046.031 | 3.137x | 0.00775027 |
| 4 | 6677.878 | 2171.412 | 3.075x | 0.00785780 |

Native latency was about 3.5–4.4% higher than the separate BF16 run, subject to cross-run variability. Maximum FP32-reference error decreased at each batch.

Nevertheless, model-wide FP16 validation still failed: 6045/522240 logits (1.2%) outside tolerance, maximum absolute difference 0.21875. It stopped before probabilities and timing. Better isolated FP32 accuracy did not establish integrated agreement.

At the user's request, the kernel was **returned to BF16**, and the model benchmark's numerical comparison assertions were disabled. It still reports logit/probability differences, changed decisions and replay differences. Mask validation remains enabled. The status explicitly says numerical assertions are disabled and does not claim validation passed. Standalone checks remain enabled.

## Integrated BF16 result: incremental benefit over packed convolution

Four real requests, batch one, 2048-token bucket, thirty samples and ten graph replays per sample. Model revision `ff99089b6810691bd79134ec6febb5d9afa6e619`.

| Configuration | GPU median, ms | GPU minimum, ms | p95 sample mean, ms | Wall median, ms |
| --- | ---: | ---: | ---: | ---: |
| Packed convolution + SDPA | 140.461 | 138.232 | 146.276 | 140.064 |
| Packed convolution + native GQA | 134.518 | 131.614 | 139.753 | 134.086 |

Incremental saving: **5.943 ms, 4.23% latency reduction, 1.044x speedup**. This is additional to packed convolution and closely matches the standalone extrapolation.

No decisions changed across four checked slots. Maximum probability difference was 0.00703913, about 0.70 percentage points; maximum logit difference was 0.234375. There were 7688 logits outside the diagnostic tolerance. Replay maximum difference was zero; 24 attention masks/calls were checked. Four decisions are not a broad quality evaluation.

## Combined result: original model versus both kernels

The benchmark gained `--original-baseline`: start with packed convolution disabled, capture the original baseline, then install packed convolution on eighteen delta layers and native GQA on six full-attention layers. It keeps the original baseline graphs alive for interleaved comparisons. No failed experiment is enabled.

[bench_combined_speedups.sh](../scripts/bench_combined_speedups.sh) invokes this mode and saves `data/combined-speedups-model.json`.

| Configuration | GPU median, ms | GPU minimum, ms | p95 sample mean, ms | Wall median, ms |
| --- | ---: | ---: | ---: | ---: |
| Original convolution + SDPA | 142.885 | 139.284 | 145.706 | 142.503 |
| Packed convolution + native GQA | 126.938 | 124.266 | 130.889 | 127.165 |

Combined saving: **15.946 ms, 11.16% latency reduction, 1.126x speedup**. Same L4/software/model revision, batch one, four requests, thirty samples and ten replays. Absolute timings vary between runs; do not infer an interaction effect by subtracting medians across separate runs.

No decisions changed across four slots. Maximum probability difference was 0.00849801, about 0.85 percentage points; maximum logit difference was 0.2109375. There were 7177 logits outside tolerance. Replay error was zero and 24 masks/calls were checked. Numerical assertions were disabled. The speedup is established for this workload; preserved quality across representative requests is still unestablished.

GPU timings exclude input copies, tokenization and readout. Separate wall timings include input copy, graph launch and synchronization, but exclude tokenization/readout. Reported p95 values are percentiles of repeated-replay sample means, not individual request tail latency.

## Commands and current state

```bash
# Configuration sweep; this was run with the original BF16 kernel.
.venv/bin/modal run src/bench_native_gqa_sweep.py

# Real first-layer diagnostics; JSON is overwritten on subsequent runs.
.venv/bin/modal run src/diagnose_native_gqa.py

# Synthetic attention comparison at batches 1, 2, 4.
.venv/bin/modal run src/bench_native_gqa.py

# Incremental native-GQA benefit, packed convolution on both sides.
.venv/bin/modal run src/bench_native_gqa_model.py --batch-size 1

# Neither new kernel versus both together.
sh scripts/bench_combined_speedups.sh
```

Standalone and model-wide correctness-only scripts were initially written, then removed/replaced with benchmarks at the user's correction. The native-GQA kernel currently uses BF16 probabilities/V, fixed 64-by-32 tiles, four warps and two stages. Packed convolution remains the established default; native GQA is an opt-in benchmark patch. Combined gates and persistent residual RMSNorm remain disabled.

## Candidates discussed after this result

**GQA diagonal handling:** split fully visible key tiles from diagonal tiles to avoid repeated causal comparisons and masking on the former. Future tiles are already skipped. This is a contained kernel experiment; no gain has been measured.

**MLP projection and SwiGLU:** the two projections write large intermediates that the activation/product reads. A suitable GEMM epilogue could remove traffic, but preserving GEMM efficiency matters. A library GEMM followed by Triton still materializes that boundary. The earlier roughly 6 ms SwiGLU total requires a fresh profile before prioritizing a larger implementation.

**Decision-slot projection:** the engine currently computes a 255-option projection at every token, while decision readout only uses selected positions. For one 2048-token request, it projects a `[2048,2048]` hidden matrix to `[2048,255]`. Gathering one decision hidden vector first would instead project `[1,2048]` to `[1,255]`. The transformer still processes the full sequence.

An isolated prototype would pass slot indices into an alternative forward, gather final hidden vectors and use the existing projection. Multiple decisions need stable ordering; CUDA graphs need fixed-size slot-index buffers with padded unused entries or suitable buckets. No custom Triton kernel is initially required. The existing all-position path can stay intact. The reduction in arithmetic does not guarantee proportional latency savings because the resulting GEMM is tiny. This experiment was discussed but not started before this logging update.

## Decision-slot projection prototype and result

Implemented the opt-in [SlotProjectionForward](../model/decider/slot_projection.py) and [model benchmark](../src/bench_slot_projection.py). The forward computes the full transformer hidden sequence, gathers requested row/token vectors, then applies the existing 255-option linear projection only to those vectors. It does not skip any transformer computation or modify the Engine's serving interface.

Pseudocode:

```text
hidden = transformer(input_ids, use_cache=False)
selected_hidden = hidden[slot_rows, slot_positions]
slot_logits = FP32(linear(selected_hidden, existing_option_weights))
```

Fixed-size CUDA int64 slot-row and slot-position buffers allow graph replay with changing positions. Multiple decisions are flattened in request order; unused capacity points to row/token zero and its outputs are discarded. By default, capacity is the maximum slot count in the benchmark's batches. Syntax and a small CPU check covering ordering, repeated indices and padded entries passed before the user ran the GPU benchmark.

Both benchmark variants enable packed convolution on eighteen delta layers and BF16 native GQA on six full-attention layers. The baseline projects all positions; the candidate gathers hidden states before projection. Both graphs remain alive and are timed in randomized interleaved order. Comparisons report numerical differences without blocking timing; nonfinite decision logits and unsupported masks still fail.

Run:

```bash
.venv/bin/modal run src/bench_slot_projection.py \
  --batch-size 1 \
  --output data/slot-projection-b1.json
```

Measured on NVIDIA L4, torch 2.14.1+cu130, model revision `ff99089b6810691bd79134ec6febb5d9afa6e619`. Four real requests, batch one, sequence length 2048, slot capacity one, thirty samples of ten replays each.

| Configuration | GPU median, ms | GPU minimum, ms | p95 sample mean, ms | Wall median, ms |
| --- | ---: | ---: | ---: | ---: |
| All-position projection | 131.382782 | 127.987305 | 134.524829 | 131.292017 |
| Slot-only projection | 131.342743 | 127.947986 | 134.857727 | 131.093230 |

GPU median saving was **40.039 microseconds, 0.0305%, 1.000305x speedup**. Wall median decreased by about 0.199 ms, while p95 GPU sample-mean latency increased slightly. This run establishes no meaningful model-wide latency improvement; the median difference is too small to distinguish confidently from timing variability. Reduced projection arithmetic did not translate into substantial total latency savings. The measurement does not independently isolate the projection's cost.

GPU scope includes the full forward and each variant's projection, with input copies excluded. Wall scope includes copies of input IDs and slot buffers, forward, the baseline's output gather and synchronization. Neither includes softmax, CPU readout or tokenization. Percentiles describe replay sample means, not individual request tail latency.

Numerical report:

- Four slots checked, with zero changed decisions.
- Maximum logit difference: 0.0625; zero logits outside `atol=0.05, rtol=0.02` because the allowed absolute difference also depends on reference magnitude.
- Maximum calibrated probability difference: 0.00849801, about 0.85 percentage points. Passing the logit criterion does not imply negligible probability drift.
- Replay maximum difference: zero; 24 attention masks/calls checked.

**Status: no demonstrated useful batch-one speedup.** Keep the prototype opt-in and serving defaults unchanged. Batch-two/four results have not been supplied. The next discussed targets remain MLP projection/SwiGLU intermediate traffic and removing causal masking from fully visible tiles in native GQA; neither has been implemented as part of this experiment.
