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

## Follow-up: compiled layout changes the candidate

The sections above record the initial eager-layout hypothesis. Subsequent compiled-trace evidence supersedes its proposed next step: we now have the compiled strides, have identified the producer, and have implemented a packed-output convolution prototype instead of standalone strided normalization.

### Compiled input metadata from Perfetto

The three `aten::contiguous` calls reported identical BF16 tensor shapes and strides:

| Field | Value |
| --- | --- |
| Shape | `[1, 2048, 16, 128]` |
| Stride | `[12582912, 1, 262144, 2048]` |

This was available under Input Dims and Input Strides in the CPU event details. These are element strides, not byte strides. Tokens are physically contiguous; features are separated by 2048 elements. Ignoring the size-one batch dimension, the address is

$$
\mathrm{offset}(t,h,d)=t+262144h+2048d.
$$

The eager layout instead had contiguous features and a token stride of 6144. Therefore, the initial claim that the proposed normalization kernel could load each head's features consecutively applied only to eager execution. A fused conversion/normalization kernel for the compiled layout would need tiling that loads adjacent tokens efficiently and rearranges values for feature-contiguous output stores.

The original standalone normalization proposal would read one input tensor per invocation, Q or K, and write two separately allocated outputs: normalized values and FP32 reciprocal norms. Input pointer, output pointers and metadata are all kernel parameters; “one input” never meant one total parameter. Dimensions and strides describe addressing, epsilon describes normalization, and tile sizes/warps are separate tuning parameters.

The existing L2 norm uses a sum of squares over the 128 head features, not a mean. A fused implementation must preserve FP32 accumulation and BF16 output. Whether eliminating an intermediate outweighs transpose/reduction costs requires measurement.

### Checking whether the producer needs this layout

The immediately preceding launch in time was beta's `triton_poi_fused__unsafe_view_sigmoid_2`. Reading its body established that it only updates beta in place. It is not the QKV producer.

The producer is `triton_poi_fused__unsafe_view_add_constant_pad_nd_mul_select_silu_slice_squeeze_transpose_unsqueeze_0`. Its wrapper allocates a contiguous `[B, 6144, 2048]` buffer, then returns Q/K/V views with the observed strides and offsets into that buffer.

Comparison with the eager model path shows projection, transpose to channel/token order, convolution, transpose back, split, and reshape. The local engine replaces convolution with shifted multiply-adds plus SiLU. This arithmetic does not require token-contiguous output. Eager and compiled execution already demonstrate different physical layouts for the same computation. This does not establish that the compiled layout is useless: it could benefit producer stores, but its complete downstream cost must be measured.

The complete generated producer definition supplied later confirms:

- Input is contiguous combined projection output with 6144 channels per token.
- Each channel performs a four-tap causal depthwise convolution, using the current and three preceding tokens, with sequence-start masks.
- Weights are stored as four consecutive values per channel.
- The accumulation is converted through FP32 arithmetic and SiLU, then stored as BF16.
- Output addressing changes the physical layout from token-major input to channel-major output.

Input and old output addresses are

$$
\mathrm{input}(b,t,c)=6144(bT+t)+c,
$$

$$
\mathrm{oldOutput}(b,t,c)=b(6144T)+cT+t.
$$

There is no calculation after the final store in this producer that requires the old output order. This motivated testing direct packed outputs before adding normalization fusion.

### Source-generation and download issue

The original `data/compiled_code.py` was truncated around this producer's definition, although its call and final store survived. A downloaded local path named `data/l4-compiled-source-b8` was also a single incomplete file rather than a directory containing all generated modules.

The dump log showed successful generation of eight separate files, `compiled_forward_01.py` through `compiled_forward_08.py`. The exact point where local contents were lost was not established. Do not infer a generator failure from those incomplete local artifacts.

The reliable recovery procedure is to list the Volume directory and download files individually into a real directory:

```bash
mkdir -p data/l4-source-b8
modal volume ls decider-hf-cache /compiled-code/
for i in 01 02 03 04 05 06 07 08; do
  modal volume get decider-hf-cache \
    "/compiled-code/compiled_forward_${i}.py" \
    "data/l4-source-b8/compiled_forward_${i}.py"
done
python3 -m compileall -q data/l4-source-b8
rg -n 'def triton_poi_fused__unsafe_view_add_constant_pad' data/l4-source-b8
```

For a fresh batch-one dump, use `modal run src/dump_compiled_code.py --rows 1 --batch-size 1`, wait for its final output-file list, and download each listed file into a new directory. File counts can differ. Preserve the individual modules rather than reconstructing Python from terminal output.

## Prototype choices and integration plan

Two useful variants were identified:

1. Four-tap convolution plus SiLU writing separate contiguous Q/K/V, retaining existing FLA normalization.
2. The same producer with Q/K L2 normalization fused, leaving V unnormalized and disabling FLA's Q/K normalization.

The second variant needs reductions across full heads and may increase register pressure. It must round convolution/SiLU results to BF16 before converting back to FP32 for normalization, matching the existing intermediate store. This rounding can occur in registers. The second variant has not been implemented.

For model integration, change the source-level convolution-to-FLA preparation boundary rather than editing disposable Inductor-generated code. Replacing only `causal_conv1d_fn` is awkward because it returns one combined tensor. A combined token-major allocation split into Q/K/V would still contain token gaps in each view. Returning three separately packed tensors removes that problem.

Initially scope integration to the uncached full-forward path. Cached convolution/recurrent-state execution needs separate handling. Verify that the compiled trace loses the three copy launches and, only for a future normalization-fused variant, the two L2 launches. A supported PyTorch/Triton integration with visible output shapes/layouts is still needed; the standalone private `.run` wrapper is not yet established as traceable by `torch.compile`.

## Packed-output kernel implementation

The prototype is in `src/kernels/packed_output_conv.py`. It preserves the original convolution and SiLU arithmetic and changes the final store into three masked stores.

For flattened batch/token row $y$ and combined channel $c$:

| Output | Channel interval | Output element offset |
| --- | --- | --- |
| Q | $0\le c<2048$ | $2048y+c$ |
| K | $2048\le c<4096$ | $2048y+c-2048$ |
| V | $4096\le c<6144$ | $2048y+c-4096$ |

Each computed value is written exactly once. These addresses naturally support multiple batches without a separate batch-index term.

The row mask is $y<BT$. The original generated `ynumel` assignment and all-true `ymask` did not affect loads/stores. Merely parameterizing unused variables would not help. The generalized prototype uses the actual `ynumel` in its mask and includes row validity in all four activation loads and all output stores. Weight loads need only the channel mask because their addresses do not depend on rows.

Historical-token conditions remain necessary to prevent reads from the previous batch. Token position is computed modulo `SEQ_LEN`. Tensor masks use explicit elementwise conjunctions, rather than Python chained comparisons.

### Retaining the Inductor decorator

The user chose to retain `triton_heuristics.pointwise` instead of switching to standard `triton.autotune`. This is a private PyTorch interface. In the pinned implementation it selects configurations and returns a cached autotuning wrapper; the docstring's description as constructing heuristics does not describe the full launch machinery.

The updated metadata:

- Declares all five BF16 pointers in function-parameter order.
- Declares `ynumel` and `xnumel` as runtime integers and block sizes/`SEQ_LEN` as constexprs.
- Sets `constants={"SEQ_LEN": 2048}` for the current specialization. Adding a constexpr signature entry alone does not supply its compile-time value to this private compiler path.
- Uses batch-one size hints `y=2048`, `x=8192`. Hints guide configuration selection; actual grid dimensions and masks govern execution.
- Sets `num_store=3` and gives the modified kernel a distinct name.
- Removes copied tiling scores, which described the old output layout.
- Retains L4-specific device properties and alignment attributes. The pointer alignment assumptions must hold for inputs as well as newly allocated outputs.

The current implementation supports different positive batch sizes at fixed sequence length 2048 and channel count 6144. Other sequence lengths require matching constants/specializations. Fixed batch-one hints do not limit correctness but may affect performance at other batches.

### Wrapper

`prep_qkv` accepts contiguous BF16 `[B, 2048, 6144]` combined projection output and contiguous BF16 weights shaped `[6144, 4]` or `[6144, 1, 4]` on the same CUDA device. It checks input alignment, allocates three contiguous `[B, 2048, 16, 128]` outputs, selects the device/current stream, and calls the private kernel `.run` with five tensors, `BT` and 6144.

The decorator's `Grid2D` machinery derives the launch grid from row/channel counts and selected block sizes. The wrapper does not pass its own grid, blocks, or sequence-length argument because those are supplied by the retained machinery and constants. It returns Q/K/V directly without output copies or synchronization.

## Correctness tests and Modal runner

Added `src/check_packed_output_conv.py`. It reuses the profiling image and runs on an L4:

```bash
modal run src/check_packed_output_conv.py \
  2>&1 | tee data/packed-output-conv-correctness.log
```

Four test methods cover:

- Random inputs at batch sizes one, two and four, with both accepted weight shapes.
- Each causal convolution tap separately, distinct Q/K/V regions, and values at the end of one batch that must not leak into the next.
- Zero input producing zero output.
- CUDA graph replay after changing the input, with compilation/autotuning warmed before capture.

The reference uses independently indexed token-major FP64 accumulation followed by SiLU and BF16 conversion. Checks include output shape, dtype, device, contiguity and finite values. Numerical tolerance is `rtol=0.008`, `atol=0.002`, allowing BF16 rounding differences. Passing establishes agreement within that tolerance, not bitwise identity with the original generated kernel or end-to-end model equivalence.

The first Modal attempt failed before executing tests because the runner imported local `benchmark` remotely, where it had not been uploaded. Fixed image construction by importing `benchmark.modal_image` only under `modal.is_local()`. The remote function receives the kernel directory explicitly and no longer imports the image-building module. A simulated remote-initialization check, Python syntax checks and diff checks passed locally.

The user reran the remote suite and reported:

```text
Ran 4 tests in 4.711s

OK
```

This is user-reported GPU correctness evidence for the standalone prototype. No standalone speed or integrated model latency result has been reported.

## Next measurement

Benchmark the complete preparation sequence, with compilation and autotuning outside timing:

- Baseline: original convolution/SiLU plus three contiguous copies.
- Candidate: packed-output convolution.

Keep Q/K normalization unchanged in both. Compare graph-replayed execution at batch sizes one, two and four, then integrate only if the sequence improves and verify warmed end-to-end request latency and answer probabilities.

The existing batch-one table attributes approximately $2.76+7.15=9.91$ ms across eighteen layers to convolution preparation plus copies. This is a target budget from an instrumented trace, not a promised speedup or graph-backed baseline. The prototype still performs convolution, activation, reads and output writes.
