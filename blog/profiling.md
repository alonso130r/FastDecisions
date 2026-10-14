# Inference profiling notes for a later blog post

- **Goal:** determine where `Mapika/decider-2b` spends time during inference on an NVIDIA L4, then identify operations worth considering for custom Triton kernels.
- **Workload:** prepared a 500-row DecisionBench JSONL sample at `data/decisionbench_profile.jsonl` from the Hugging Face `Hanno-Labs/decision-bench` evaluation split. The conversion script uses seed 17 by default and preserves primitive, candidate-count, and state-length diversity. It converts binary classification, candidate selection, and ordinal scoring rows to Decider's question format.
- **Benchmark harness:** `src/benchmark.py` measures warmed batches, synchronizes CUDA around timed calls, reports median and p95, and can collect PyTorch profiler traces with shapes and memory. The Modal function targets an NVIDIA L4. CUDA profiling temporarily disables CUDA graphs and restores the benchmark configuration afterward. The profiler's kernel totals are sums across the capture, not per-request latency.
- **Early benchmark baseline:** the run recorded in Modal logs used batch size 8, 8 rows, and 5 timed repeats. Eager execution reported a 1,479.60 ms median batch latency and 1,482.39 ms p95, about 5.41 decisions/s. The compiled run reported 1,449.50 ms median and 1,455.01 ms p95, about 5.52 decisions/s. That is a 30.10 ms or roughly 2.0% median improvement. With only five repeats, this is a small, noisy difference, not evidence of a substantial speedup.
- **Profile capture caveat:** the saved Chrome traces in `data/decider-l4-profile.json` and `data/decider-l4-profile-graphs.json` are separate profiler captures, not the five-repeat benchmark above. Their capture windows contain multiple score-batch events and profiler overhead. Treat their kernel totals as ranking evidence within each capture, not as a direct eager-versus-compiled latency comparison.
- **Eager trace:** the saved trace has 88,510 events and two `stage.end_to_end_score_batch` events of about 2,215 ms and 2,193 ms. It contains 7,139 CUDA kernel events. Largest aggregated entries include:
  - BF16 CUTLASS GEMM: 516.58 ms over 150 launches.
  - Generic FP32 elementwise multiply: 268.28 ms over 806 launches.
  - Batched triangular solve: 120.21 ms over 36 launches.
  - Depthwise convolution: 82.10 ms over 18 launches.
  - Scalar power: 77.46 ms over 79 launches.
  - Efficient-attention CUTLASS kernel: 71.88 ms over 6 launches.
- **Graph trace:** the separate saved trace has 15,812 events and two `stage.end_to_end_score_batch` events of about 2,703 ms and 2,687 ms. It contains 7,774 CUDA kernel events. Its largest aggregated entries include BF16 CUTLASS GEMM at 625.09 ms over 138 launches, generic FP32 elementwise multiply at 296.14 ms over 860 launches, batched triangular solve at 140.26 ms over 36 launches, and efficient attention at 90.37 ms over 6 launches. Because the capture configuration and timings differ, this trace should not be interpreted as proof that graphs make inference slower or faster.
- **Inductor inspection:** used PyTorch's `torch._inductor.utils.run_and_get_code` to capture generated code for a representative CUDA forward. The saved raw dump, `data/compiled_code.py`, targets CUDA capability 8.9 (L4) with token IDs shaped `[8, 2048]` and hidden states shaped `[8, 2048, 2048]` in BF16. The readable, labeled version is `data/compiled_code_labeled.md`.
- **What the generated code shows:** The readable labeled view presents 14 Triton JIT functions; the raw dump also contains additional query-preparation and delta-path kernels omitted from that view. Inductor already fuses several pointwise/reduction paths, including RMSNorm-related operations, rotary/query-key preparation, gated-delta transforms, attention output gating, SwiGLU activation, and residual operations. Dense matrix multiplies remain external `extern_kernels.mm` calls; efficient attention remains an opaque `aten._scaled_dot_product_efficient_attention` call in this dump. Generated source does not include timings, so a function's size or element count alone does not establish that it is a bottleneck.
- **Fusion boundaries we found:** the dump is not one fused model kernel. It contains narrow Triton kernels around separate matrix-multiply and attention calls:
  - RMSNorm is fused internally: cast/square/reduction, epsilon, reciprocal square root, scale, and output conversion are handled together in a reduction kernel. The following dense projection is still a separate `extern_kernels.mm` call.
  - Query and key preparation combines normalization or rotary arithmetic with reshaping, splitting, and packing in generated Triton kernels. The query/key/value projection GEMMs that feed these paths remain separate external matrix multiplies; key normalization's reduction is also a separate Triton launch from its later rotary/packing work.
  - Gated-delta decay combines exponential, softplus, and multiply work in one pointwise kernel. The beta sigmoid is another kernel, and the projections that produce their inputs are separate GEMMs.
  - SwiGLU's sigmoid-weighted activation (`silu` and multiply) is fused into one Triton kernel, but its up/gate projections and down projection remain separate GEMMs. Residual addition plus the next RMSNorm is fused together in one kernel; the preceding projection is not part of it.
  - Attention setup is only partly fused. The dump has separate Triton launches for K expansion, V expansion, and mask conversion, followed by a framework efficient-attention call. Attention output gating is another Triton kernel after attention. Inductor did not fuse the attention computation into its surrounding pointwise kernels.
  - Across the forward, the large `extern_kernels.mm` calls are the main visible fusion boundary: Inductor dispatches them to optimized library GEMMs, with separate Triton work before or after. Those calls are not evidence that the GEMMs are unoptimized, and they are not automatically candidates for replacement.
- **Interpretation:** fusion reduces intermediate launches and memory traffic inside each generated Triton function, but operations separated by GEMMs, reductions, copies, or the efficient-attention operator still cross kernel boundaries. The profiler must tell us which remaining boundaries cost enough to matter; the code dump alone only shows what Inductor generated for this one shape.
- **Large materialization candidate to measure:** the generated wrapper expands grouped-query K and V tensors from 2 to 8 heads. The labeled dump estimates each copy writes about 64 MiB at the inspected shape. It also converts a mask into a roughly 64 MiB BF16 tensor whose head dimension is broadcast with a zero stride. This suggests measuring whether the K/V expansion copies are costly and whether they can be avoided; it does not establish the expected speedup or justify replacing attention with a custom kernel.
- **Current optimization direction:** first rank operations using a representative, consistently configured profile; map expensive ATen operations and shapes to the generated Triton functions or external library calls; then prototype a narrow candidate such as avoiding K/V expansion only if the profile confirms meaningful time or bandwidth cost. The large CUTLASS GEMMs are already handled by optimized library kernels, so a custom Triton GEMM is not the obvious first target. No custom Triton kernel has been implemented or measured yet.
- **Measurement limitations:** the latency comparison has five repeats; profiler captures perturb execution and disable CUDA graphs for their capture; trace kernel totals aggregate many launches and can include warmup/profile setup; the compiled-code dump is from one representative shape; and no arithmetic-intensity or achieved-bandwidth analysis has yet been done. Conclusions should be presented as profiling leads, not a completed optimization study.
- **Metal profiling attempt:** on an M4 Pro, an `xctrace` Metal System Trace was attempted. A first capture produced mostly unrelated processes such as WindowServer and `iconservicesagent`, rather than useful per-model GPU kernels. A later capture path began generating an enormous `0000-...` trace directory (over 400 GB was reported), and it was deleted to protect disk space. The available Instruments screenshots showed GPU channels such as Compute, but did not provide useful model-attributed kernel measurements. The more actionable profiling work so far is the bounded Modal L4 / PyTorch Kineto trace.
- **Modal setup issue resolved in the workflow:** installing `causal-conv1d` from source initially failed because the image lacked `nvcc`; the profiling image was changed to a CUDA development image and the package build was adjusted. The later Modal logs showed model loading and warmup progressing, with `torch.compile` autotuning overhead on the L4. This setup cost is separate from steady-state inference latency.


## Follow-up: profiling the compiled one-pass decision workload

### Calling pattern determines what to optimize

Decider is a system-one style, one-pass decision model. It processes the input and returns option-letter probabilities at designated answer slots, with no generated output tokens. Optimize time to decisions rather than autoregressive token throughput.

The execution paths differ:

| Calling pattern | Relevant work |
| --- | --- |
| One question per unrelated state | Full forward, padding, batching |
| Multiple independent questions over one state | Shared prefix and question suffix execution |
| Repeated schema over changing states | Existing schema cache and its layout semantics |
| Small requests using CUDA graphs | Device work and memory traffic under replay |

`Engine.score_shared` already shares attention KV and delta-net convolution/recurrent state across question suffixes. The benchmark's `score_batch` sends one question per dataset row through `decide_batch`; it does not represent the multi-question shared-state or schema-cache paths. Those need their own representative fixtures before making production claims.

The inspected dump represents bucketed token IDs `[8, 2048]`, not every serving path. Right padding can introduce substantial work; record actual token lengths and bucketed lengths when evaluating changes.

The graph-backed engine currently projects every final hidden position into option-letter logits, then gathers answer slots. `DecisionModel.slot_logits` instead gathers hidden states before projecting. Moving the gather before the engine projection is a model-specific candidate, but requires accommodating slot metadata in the compiled/CUDA-graph interface. Earlier backbone layers still need context-token representations.

### Reproducible Modal capture

Run from the repository root. The Modal entry point defaults to eager profiling, so explicitly request compiled profiling:

```bash
modal run src/benchmark.py::modal_main \
  --rows 8 \
  --batch-size 8 \
  --warmup-batches 3 \
  --repeats 50 \
  --profile-batches 3 \
  --use-compile \
  --use-cuda-graphs \
  --profile-mode compiled \
  2>&1 | tee data/l4-compiled-b8.log
```

Using exactly eight rows repeats one batch and matches the dump generator's first eight rows. Warmup and initial compilation occur outside capture. The profiler temporarily disables graph replay to expose kernels, then restores graphs for the separately timed benchmark. Check the log for `compile=True`, `cuda_graphs=True`, and `profile_mode=compiled`.

Download the trace before another compiled capture overwrites the fixed remote filename:

```bash
modal volume get decider-hf-cache \
  /profiles/decider-l4-compiled.json \
  data/l4-compiled-b8.json
```

Volume paths omit the container mount prefix `/cache`. If matching source is needed:

```bash
modal run src/dump_compiled_code.py \
  --rows 8 \
  --batch-size 8 \
  2>&1 | tee data/l4-dump-b8.log

modal volume get decider-hf-cache \
  /compiled-code/ \
  data/l4-compiled-source-b8
```

These downloads are different artifacts: the JSON trace records runtime activity; the compiled-code directory contains generated Python/Triton. Source regeneration is optional if the existing source already matches the model revision, shape and compiler environment. Pass the same immutable Hugging Face commit using `--revision` to both commands for reproducibility; both otherwise default to `main`.

The observed dump configuration was:

```text
Compiling decider-2b-v11 on NVIDIA L4: input_ids shape=(8, 2048), dtype=torch.int64, torch=2.14.1+cu130
```

`int64` describes token IDs; hidden activations are BF16. This line confirms input configuration, not compilation completion. Wait for the generated-source paths before downloading.

For single-request latency, repeat the benchmark with `--rows 1 --batch-size 1`, preserving the other settings, and immediately download to a distinct local filename such as `data/l4-compiled-b1.json`. Keep batch-one and batch-eight results separate. No single-request result from this procedure has been established here.

### Reading Perfetto correctly

Load the trace in https://ui.perfetto.dev/. Inspect a warmed `stage.end_to_end_score_batch` interval and its CPU ranges:

- `stage.prompt_build_and_tokenize`
- `stage.engine_prepare_forward_and_readout`
- `stage.model_forward`

CPU annotation duration is not GPU execution time. CUDA launches are asynchronous, so GPU kernels can execute beyond the end of the CPU range that launched them.

`gpu_user_annotation` denotes a named grouping of associated GPU activity. Its span can include gaps and overlap the actual kernels. Do not sum annotation durations with kernel durations. Rank device computation using category `kernel`; examine `gpu_memcpy` separately for transfers.

A `## Call CompiledFxGraph <identifier> ##` event marks execution of an Inductor-compiled FX graph portion. The identifier relates to the compiled graph cache. It is not a compilation event, a single kernel, or necessarily CUDA graph replay. Its wrapper can launch generated Triton and existing library GEMMs. Several calls can appear around external operations or graph breaks.

`cudaLaunchKernel` is a CUDA Runtime API launch. `cuLaunchKernelEx` is a Driver API launch with extended configuration support. Neither identifies eager versus compiled execution or guarantees better performance. Follow the GPU event's correlation/flow link to its CPU launch and enclosing scopes.

Apparent gaps can be CPU launch delays, synchronization, work on another stream, profiler overhead, or compilation of a new shape. Inspect all streams and CUDA API calls at the gap. Graph replay is disabled in this capture, so launch gaps need not reproduce in the production graph-backed benchmark.

For slice extraction, selecting an annotation and querying `descendant_slice(slice_id)` follows actual nesting, not launch ownership. Filtering names with `NOT GLOB '## Call CompiledFxGraph*'` removes the annotation rows only; excluding descendants of those annotations additionally requires checking `ancestor_slice(s.id)`. Neither technique alone identifies all eager kernels.

A crucial correction from the wider capture: relative depth is useful for navigation but cannot reliably classify compiled versus noncompiled work. The same generated Triton names occur at different depths. Aggregate identical names across depths, and use flows/CPU callers for ownership.

### Kernel names encountered

| Name or family | Interpretation |
| --- | --- |
| `direct_copy_kernel_cuda` with BF16 | Device compute kernel copying BF16 tensor elements; inspect caller and strides for the reason |
| `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | Delta-rule recurrent/chunk state computation |
| `chunk_fwd_kernel_o` | Computes attention output from earlier-chunk state and causal within-chunk interactions |
| `l2norm_fwd_kernel` | L2 normalization, typically query/key preparation in this path |
| `recompute_w_u_fwd_kernel` | Intermediate representation construction; the name does not prove redundant work |
| `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | Chunk-local solve/preparation for the delta-rule algorithm |
| `Memcpy HtoD`, `DtoH`, `DtoD` | Transfer events, distinct from compute kernels performing tensor copies |

The FLA kernels outside Inductor calls are already specialized Triton kernels. Calling them noncompiled would be misleading. Their computation is separate from standard softmax attention.

### Evidence from the pasted leaf-slice tables

These are aggregates supplied from Perfetto, not independently reconstructed from a saved full trace in this follow-up. Their exact capture settings and wall-clock duration should be retained with the original trace before publishing comparisons.

The first selection, outside the labeled compiled calls, summed to **183.45 ms**:

| Family | Calls | Total ms | Share of that selection |
| --- | ---: | ---: | ---: |
| BF16 direct copy | 54 | 65.806 | 35.9% |
| Delta state, h | 18 | 32.147 | 17.5% |
| Attention output, o | 18 | 29.964 | 16.3% |
| Recompute w/u | 18 | 23.383 | 12.7% |
| L2 normalization | 36 | 19.755 | 10.8% |
| KKT solve | 18 | 9.823 | 5.4% |
| BF16 fill | 18 | 1.870 | 1.0% |
| Local cumulative sum | 18 | 0.439 | 0.2% |

The repeating sequence is three copies, two normalizations, cumulative sum, fill, solve, w/u preparation, state computation, and output computation. The hypothesis that the three copies prepare Q/K/V contiguity is plausible but unverified. Inspect CPU callers, shapes and strides before claiming it.

The next selection under `stage.model_forward` summed to **1133.01 ms**, with 18 delta-attention calls but only five full-attention calls and 22 SwiGLU calls. The local model configuration has 24 layers: 18 linear-attention and six full-attention. This discrepancy prompted checking the wider enclosing scope.

The selection under `stage.engine_prepare_forward_and_readout` recovered six full-attention calls and 24 SwiGLU calls, consistent with the architecture. It also showed generated Triton kernels at both relative depths 2 and 3. This supports an annotation attribution/boundary explanation for the narrower omissions, rather than skipped model execution.

Across depths, the wider selection totals **1213.85 ms** of listed GPU activity:

| Family | Calls | Total ms | Share of listed activity |
| --- | ---: | ---: | ---: |
| GEMM families with `gemm` in their names | 187 | 752.611 | 62.0% |
| Largest CUTLASS GEMM family, subset of above | 138 | 731.786 | 60.3% |
| Full softmax attention | 6 | 101.944 | 8.4% |
| BF16 direct copies | 54 | 65.806 | 5.4% |
| SwiGLU, both generated names | 24 | 59.574 | 4.9% |

The GEMM rows overlap and must not be added together. The denominator includes listed transfers but excludes idle gaps. Summed GPU activity is not automatically request latency, particularly with overlapping streams.

Transfer durations in the wider table were tiny: three HtoD events totaled 0.014592 ms, one DtoH event 0.003200 ms, and one DtoD event 0.001920 ms. These are distinct from the 65.806 ms of BF16 tensor-copy compute kernels. Synchronization or host preparation costs still require separate inspection.

### How to rank optimization opportunities

Rank by measured total cost, then estimate how much of that work can be avoided. A large duration establishes importance, not inefficiency.

1. **Split the dominant GEMM family by matrix shape and model role.** The same implementation name covers many calls. Attribute MLP and attention projections using CPU callers and operands. For `[M,K] @ [K,N]`, estimate work as $2MKN$ FLOPs and divide by duration for achieved throughput. Do not infer actual activation semantics from substrings such as `relu` in the library kernel name.
2. **Inspect full-attention backend selection.** The six calls total 101.944 ms. Investigate mask semantics, layout and native GQA support. The expansion and mask-conversion kernels are relatively small, so the strongest case for a backend change would also improve attention itself.
3. **Investigate the repeated BF16 copy boundary.** Determine whether the producer can emit the required layout directly, the consumer can accept existing strides, or copying can be fused with normalization. Eliminating half the observed copy cost would save about 32.90 ms, approximately 2.7% of the wider listed activity, before accounting for replacement costs and production effects.
4. **Evaluate MLP fusion as a sequence.** SwiGLU totals 59.574 ms, and surrounding GEMMs dominate overall cost. Fusion that saves activation traffic but slows matrix multiplication can lose overall.
5. **Deprioritize tiny gates and cumulative sums.** The entire local cumulative-sum family totals 0.439 ms. Easy-to-edit code is not necessarily a useful target.

For one copy triplet and its two following normalizations, gather the enclosing CPU operations, input/output shapes, strides, producer, and consumer. If a strided Q/K input is copied and then normalized, a candidate is to write normalized values directly into the consumer's required layout. This is a hypothesis requiring numerical and latency validation.

### Reading the labeled Triton as equations and memory traffic

- **K01 RMSNorm:** `xnumel=16384` counts rows, each reducing over 2048 features. It is not directly comparable to a pointwise kernel's element count. Preserve the learned scale `1 + weight` and epsilon. Two source-level loads do not prove two DRAM reads; caching matters. Retaining a row can reduce loads but increase register pressure.
- **K08/K09 GQA expansion:** at `[8, 2048]`, eight query heads share two KV heads. Each expansion writes 33,554,432 BF16 elements, or 64 MiB. Native GQA could avoid the expanded representation, subject to backend support and actual dispatch.
- **K10 mask:** the wrapper allocates `[8, 1, 2048, 2048]` BF16 and broadcasts the head dimension with zero stride. It writes 64 MiB. If semantics are purely causal, internal causal masking is a candidate; cached rectangular suffix attention needs separate correctness reasoning.
- **K05-K07 and query counterparts:** separate normalization reduction, rotary preparation and packing suggest a possible head-normalization/partial-RoPE fusion. Preserve partial rotary dimensions and consider changed BF16 rounding boundaries.
- **K13 SwiGLU:** 100,663,296 elements imply approximately 576 MiB of logical traffic for two BF16 reads and one BF16 write. That is an estimate, not a measured DRAM counter. Fusion must preserve efficient GEMMs.

For the narrower full-forward table, K/V expansion plus mask conversion totaled only 4.896 ms, about 0.43% of that selection. This corrected the initial emphasis based solely on large tensor sizes. The wider table adds the sixth execution of each.

Useful estimates:

$$
\text{effective bandwidth}=\frac{\text{estimated transferred bytes}}{\text{kernel duration}},
\qquad
\text{potential saving}=\text{measured cost}\times\text{removable fraction}.
$$

Amdahl's law bounds end-to-end gains:

$$
\text{speedup}=\frac{1}{(1-f)+f/s}.
$$

Here $f$ is the fraction of latency affected and $s$ its speedup. Kernel-time shares are only proxies for latency shares when critical-path and overlap assumptions hold.

### Remaining evidence needed

- Preserve the original trace and exact run configuration for these pasted aggregates.
- Aggregate names across relative depths and confirm layer counts, as the wider table now does.
- Split dominant GEMMs by operand shapes and role.
- Identify callers and strides for one repeated copy/normalization sequence.
- Check full-attention mask semantics and backend eligibility.
- Use targeted Nsight Compute hardware counters only after choosing a hot kernel; the PyTorch trace does not establish actual DRAM bandwidth, spills or tensor-core utilization.
- Validate candidate changes against representative slot logits and calibrated probabilities, not only argmax decisions, then measure warmed graph-backed end-to-end latency.

No optimization has yet been implemented or its speedup established.

### Implementation references

- [Modal volume CLI](https://modal.com/docs/cli/latest/volume)
- [PyTorch scaled dot product attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention)
- [Triton normalization tutorial](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html)
- [Kineto trace annotation handling](https://github.com/pytorch/kineto/blob/main/libkineto/src/output_json.cpp)
- [FLA gated delta-rule implementation](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/chunk.py)

These upstream links explain concepts; the installed Modal package version remains the authoritative source for the measured kernels.


### Running the pre-guard layout diagnostic

`src/benchmark.py` now exposes `inspect_layouts`, which wraps the FLA autograd forward outside its existing decorators, logs the first call's tensor metadata, and restores the original method. It does not edit the installed package or bypass its guard. It runs one eager request with compilation and CUDA graphs disabled.

```bash
modal run src/benchmark.py::inspect_layouts --batch-size 1 2>&1 | tee data/l4-layouts-b1.log
```

Look for `FLA_PRE_GUARD` lines containing argument names, shapes, strides and contiguity. The final JSON includes the FLA version/source location and number of forward calls. This is a diagnostic, not a latency benchmark; the logged layouts come from eager execution and must be compared with compiled producer layouts before implementing an optimization. Because the file now has multiple local entry points, select `::modal_main` explicitly for ordinary benchmark runs.


## Follow-up implementation notes

- [Packed convolution motivation, implementation and pseudocode](batch-one-fla-layouts.md#packed-convolution-motivation-and-implementation-in-words), including standalone results and the integrated forward improvement.
- [Fresh compiled structure and fusion targets](compiled-structure-and-fusion-targets.md), including the source-reviewed combined gate-transform prototype, its motivation and pseudocode.

Earlier candidate rankings and statements about unimplemented kernels are historical observations. The follow-up notes distinguish completed GPU checks from proposed work.
