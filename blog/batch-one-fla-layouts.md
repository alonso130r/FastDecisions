# Low-concurrency inference and the FLA input-layout boundary

Follow-up to [the initial profiling notes](profiling.md). This records the workload clarification, batch-one measurements, and investigation of the repeated copies before delta attention. No custom kernel has been implemented or a speedup established.

## Actual optimization objective

The goal is to run the decision model as fast as possible for one to a few independent decision streams. Each stream must receive its previous decision before submitting the next dependent decision. Available batch sizes will therefore usually remain low.

For a stream with dependent decisions, completion time is approximately

$$
T_{\mathrm{stream}}=\sum_i T_{\mathrm{decision},i}.
$$

Optimize warmed per-decision latency and stream completion time, rather than throughput at large batch sizes. Test one stream first, then concurrency two and four, with each stream preserving its dependency chain. Batch ready work when available without waiting unnecessarily to fill a batch.

Custom Triton kernels are an explicit part of the optimization effort, but must produce measured end-to-end improvements. Workload changes, attention dispatch and precision changes remain relevant alongside kernel development.

Replacing established library GEMMs is not currently justified by evidence. GEMMs deserve investigation at the workload and integration levels: reduce padding, share unchanged prefixes where semantics allow, consider precision changes, and avoid projecting unused positions at final readout. Sequential decisions alone do not establish that their model states or prefixes are reusable.

## Batch-one measurement procedure

Run from the repository root:

```bash
modal run src/benchmark.py::modal_main \
  --rows 1 \
  --batch-size 1 \
  --warmup-batches 3 \
  --repeats 100 \
  --profile-batches 3 \
  --use-compile \
  --use-cuda-graphs \
  --profile-mode compiled \
  2>&1 | tee data/l4-compiled-b1.log
```

Download immediately, because another compiled capture overwrites the remote filename:

```bash
modal volume get decider-hf-cache \
  /profiles/decider-l4-compiled.json \
  data/l4-compiled-b1.json
```

The final median and p95 measure warmed request execution with CUDA graphs. The profiler temporarily disables graph replay to expose individual kernels. This repeatedly measures the first dataset row, not the full request distribution. Preserve the model revision and sequence bucket with the result.

In Perfetto, extract leaf GPU slices under `stage.engine_prepare_forward_and_readout` and aggregate identical names across relative depths. Divide by three only if the selection actually contains all three requests. Layer counts provide a useful sanity check.

## Batch-one evidence

The supplied leaf-slice table contains six full-attention calls, eighteen delta-state/output calls, and twenty-four SwiGLU calls. Those match one model forward, so its totals should not be divided by three.

| Work | Total ms | Share of listed activity |
| --- | ---: | ---: |
| GEMMs and split-K reduction | 87.734 | 67.8% |
| Full attention | 10.990 | 8.5% |
| Delta solve, w/u, h and o computation | 8.048 | 6.2% |
| BF16 direct copies | 7.149 | 5.5% |
| SwiGLU | 5.966 | 4.6% |
| Convolution preparation | 2.757 | 2.1% |
| L2 normalization | 1.424 | 1.1% |

Total listed GPU activity is **129.380 ms**. It excludes gaps and is not the graph-backed request latency. The final benchmark median and p95 have not been supplied in this discussion.

There are 54 BF16 copies and 36 normalizations: three copies and two normalizations per delta-attention layer. Copies average 132.4 microseconds, and normalizations average 39.6 microseconds. Together, this preparation costs approximately

$$
3(132.4)+2(39.6)\approx476\ \mu\mathrm{s}
$$

per delta layer, or **8.573 ms**, approximately **6.6%** of listed activity across the forward. This is the existing cost, not a predicted saving: replacement kernels still have to normalize and move necessary data.

The GQA expansion and attention-mask conversion kernels total approximately **0.459 ms**. Full attention itself costs **10.990 ms**, so an attention backend change should be judged primarily by its effect on attention computation.

Low batch size does not necessarily imply tiny GEMMs. Full-forward projections have row dimension

$$
M=BT,
$$

so a long sequence can still create substantial matrix multiplications at batch one. Confirm sequence lengths before comparing batch-one and batch-eight scaling.

## Locating the contiguous copies

A supplied Perfetto screenshot showed three nested CPU towers beneath `ChunkGatedDeltaRuleFunction`:

```text
aten::contiguous
  aten::clone
    aten::empty_like / aten::empty
    aten::copy_
      cudaLaunchKernel
```

Each tower represents one contiguous materialization, not several independent copies. This connects the observed device-copy kernels to input preparation at the FLA boundary.

Earlier guidance guessed `fla/utils.py`; the installed package instead uses a `fla/utils/` directory. The reliable procedure is to follow the import from the actual installed `ChunkGatedDeltaRuleFunction` file. Upstream paths can differ from the Modal installation.

The supplied forward is decorated with `@input_guard` outside `@autocast_custom_fwd`. Its Q/K normalization branch explicitly calls `l2norm_fwd(q)` and `l2norm_fwd(k)` when `use_qk_l2norm_in_kernel` is enabled. Despite that flag's name, these are separate normalization launches in this implementation.

The supplied `input_guard` binds parameter names and processes all positional and keyword arguments through `_contiguous_if_needed`. Plain `@input_guard` exempts no parameters. It also selects the device context before invoking the wrapped function.

The guard supports `no_guard_contiguous=True` or named exemptions. That provides a possible integration seam, but removing the guard without preserving downstream layout assumptions would be incorrect. The helper's implementation was not supplied; the observed copy towers and subsequent tensor logs establish which inputs require materialization in this run.

## Existing L2 normalization assumptions

The supplied Triton implementation calculates input and output addresses as

$$
p_x=x+tD+d,\qquad p_y=y+tD+d.
$$

It receives no input strides and assumes packed rows with contiguous features. The Python wrapper first uses `x.view(-1, x.shape[-1])`; that view may fail for some layouts, and a successful view does not prove the kernel's packed-row assumption is valid.

The normalization is

$$
y_i=\frac{x_i}{\sqrt{\sum_j x_j^2+\epsilon}}.
$$

Preserve the sum rather than a mean, FP32 accumulation, and epsilon inside the square root. The implementation also returns FP32 reciprocal norms, `rstd`; preserve that interface initially. The wrapper dispatches to a different row kernel for feature dimensions above 512; the observed Q/K head dimension is 128.

A replacement should allocate contiguous outputs explicitly. `empty_like` can preserve a noncontiguous input layout, and checking only final-dimension stride does not establish packed rows.

## Diagnostic implementation

Added `inspect_layouts` to `src/benchmark.py`, reusing the current Modal L4 image. It wraps `ChunkGatedDeltaRuleFunction.forward` outside the existing decorators, logs tensor metadata for the first call, invokes the original guarded method, and restores its descriptor afterward. It does not edit the installed FLA package or bypass its guard.

```bash
modal run src/benchmark.py::inspect_layouts \
  --batch-size 1 \
  2>&1 | tee data/l4-layouts-b1.log
```

The diagnostic runs one eager request with compilation and CUDA graphs disabled. Its final JSON reports FLA version/source location and forward-call count, although those fields were not included in the pasted result. Local syntax and diff checks passed; the user subsequently ran the remote diagnostic and supplied the layout lines below.

Because the benchmark file now has multiple local entry points, ordinary benchmark commands must explicitly select `::modal_main`.

## Observed pre-guard layouts

All tensors were on `cuda:0`.

| Argument | Shape | Stride | Contiguous | Dtype |
| --- | --- | --- | --- | --- |
| q | `[1, 1664, 16, 128]` | `[10223616, 6144, 128, 1]` | false | BF16 |
| k | `[1, 1664, 16, 128]` | `[10223616, 6144, 128, 1]` | false | BF16 |
| v | `[1, 1664, 16, 128]` | `[10223616, 6144, 128, 1]` | false | BF16 |
| g | `[1, 1664, 16]` | `[26624, 16, 1]` | true | FP32 |
| beta | `[1, 1664, 16]` | `[26624, 16, 1]` | true | BF16 |

These identify Q, K and V as the noncontiguous inputs. Within each head, features are contiguous. Each tensor contains 2048 elements per token, but consecutive tokens are separated by 6144 elements:

$$
16\times128=2048,\qquad6144=3\times2048.
$$

This is consistent with views into a combined QKV tensor, with three regions per token. Storage pointers and offsets are still needed to prove shared allocation and region ordering. Size-one batch stride alone does not establish the layout for larger batches.

The eager diagnostic processed **1664 tokens**. The graph-backed engine would normally bucket this length to **2048**. Do not assume eager layouts and compiled producer layouts are identical.

## Concrete Triton candidate

Read original strided Q/K, normalize each 128-feature head, and write packed contiguous outputs for the existing delta kernels. For the observed batch-one layout, input and output offsets are

$$
\mathrm{input}(t,h,d)=6144t+128h+d,
$$

$$
\mathrm{output}(t,h,d)=2048t+128h+d.
$$

Offsets are relative to each tensor's own data pointer. A general implementation must handle the relevant batch stride and validated layouts rather than assuming this batch-one case covers every request.

This would replace Q/K preliminary copies plus their normalization calls. Leave V's copy in place initially, since V does not pass through normalization. V needs a separate producer-layout or consumer-stride change to eliminate its copy.

## Next evidence and acceptance conditions

1. Capture equivalent metadata during an untimed compiled run, without introducing logging-induced graph breaks. Confirm the bucketed Q/K layouts and producer boundary.
2. Implement strided-input Q/K normalization and narrowly integrate it with the guard exemptions or a dedicated preparation path.
3. Compare normalized outputs and reciprocal norms, then representative decision probabilities against the existing path.
4. Benchmark the preparation sequence and warmed graph-backed request latency at batch one. Preserve input, revision and environment across comparisons.
5. Test the few-stream workload after establishing a single-stream gain.

The objective is a measured inference improvement. The diagnostic has identified a plausible removable boundary; it has not yet established the compiled layout or demonstrated a speedup.
