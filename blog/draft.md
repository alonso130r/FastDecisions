# Why is my PyTorch model still slow?

## What I learned optimizing LLM inference with custom Triton kernels

`torch.compile` made my model about 2% faster. That left me with almost the same latency, and very little idea of what to change next.

I was optimizing a system-one decision model: one forward pass produces probabilities over possible actions, without generating an answer token by token. The workload I cared about was a stream of dependent decisions. Each decision had to finish before the next could begin, so batching my way to better throughput would not solve the problem.

I started profiling the compiled model, following expensive operations into the generated kernels, and testing replacements in Triton. Some changes produced large isolated speedups that disappeared inside the model. One useful change came from noticing that a kernel wrote a tensor in a layout the next operation immediately undid.

This is the path from that disappointing benchmark to the kernels that actually reduced forward latency, including the experiments I left disabled.

## I was measuring the wrong workload

- Describe the initial batch-eight comparison: eager 1,479.60 ms, compiled 1,449.50 ms, five timed repeats. The roughly 2% improvement was preliminary.
- Explain why the actual objective was warmed latency for one to a few dependent decision streams.
- Establish the model, GPU, sequence bucket and measurement boundaries.
- Keep the early batch-eight comparison separate from later batch-one results.

### Evidence: the initial compile comparison

| Execution | Median batch latency | p95 batch latency |
| --- | ---: | ---: |
| Eager | 1,479.60 ms | 1,482.39 ms |
| Compiled | 1,449.50 ms | 1,455.01 ms |

These early measurements used batch eight and five timed repeats. The median fell by 30.10 ms, approximately 2%. This was my starting observation, not a robust estimate for the batch-one workload I later optimized.

Source: [initial recorded benchmark results](profiling.md). The original eager comparison is recorded in the notes; do not substitute a later compiled log as its matched baseline.

## Most of the runtime was somewhere I wasn't going to start

- Introduce the PyTorch profiler capture and Perfetto inspection.
- Show a breakdown of the batch-one GPU activity, clearly distinguishing it from request latency.
- Explain why dominant library GEMMs were not the first custom-kernel target. This was a prioritization decision, not proof that their cost was irreducible.
- Introduce the repeated BF16 copies as a lead worth tracing.

### Evidence: the batch-one profile

| GPU work | Listed activity | Share |
| --- | ---: | ---: |
| GEMMs and split-K reduction | 87.734 ms | 67.8% |
| Full attention | 10.990 ms | 8.5% |
| Delta solve and intermediate/state/output computation | 8.048 ms | 6.2% |
| BF16 direct copies | 7.149 ms | 5.5% |
| SwiGLU | 5.966 ms | 4.6% |
| Convolution preparation | 2.757 ms | 2.1% |
| L2 normalization | 1.424 ms | 1.1% |

The denominator was 129.380 ms of listed GPU activity. These rows are a subset, and this total excludes gaps. It is not graph-backed request latency. The supplied selection had six full-attention calls and eighteen delta calls, consistent with one forward.

**Figure to add:** a horizontal bar chart of these activity totals, with the denominator and timing scope in the caption.

Sources: [recorded profile aggregation](batch-one-fla-layouts.md#batch-one-evidence), [batch-one trace](../data/l4-compiled-b1.json), [run log](../data/l4-compiled-b1.log). The table was recorded from a supplied Perfetto selection; its aggregation has not been independently regenerated here.

## Three copies per layer became the first lead

- Follow the repeated Q/K/V contiguous materializations to the FLA boundary.
- Describe the initial eager-layout hypothesis.
- Show how compiled strides changed that hypothesis.
- Identify the actual producer through generated source, rather than assuming the preceding launch produced Q/K/V.

### Evidence: the layout that changed my hypothesis

The compiled Q/K/V tensors had shape \( [1, 2048, 16, 128] \) and element strides \( [12582912, 1, 262144, 2048] \). Tokens were adjacent in memory; features within a head were separated by 2048 elements.

Ignoring the size-one batch dimension, the address was

$$
\operatorname{offset}(t,h,d)=t+262144h+2048d.
$$

Across eighteen delta layers, the profile contained 54 BF16 copies and 36 L2 normalizations. Following one copy to its producer showed that the convolution wrote channel-major storage before FLA requested contiguous inputs.

**Figure to add:** a cropped Perfetto view of one contiguous-copy tower with input dimensions and strides, followed by the producer's output-allocation and addressing excerpt. Label the actual producer rather than the nearest preceding launch.

Sources: [compiled-layout investigation](batch-one-fla-layouts.md#follow-up-compiled-layout-changes-the-candidate), [compiled source](../data/compiled_code.py). The [eager layout diagnostic](../data/l4-layouts-b1.log) is useful as a comparison, not proof of compiled strides.

## The producer was creating work for the consumer

- Show the convolution producer's channel-major output and the consumer's packed-layout requirement.
- Explain the packed-output replacement: retain four-tap causal convolution and SiLU, then store directly into separate contiguous Q/K/V outputs.
- Add a before/after dataflow diagram.
- Explain the relevant addressing, sequence-start masks and BF16 rounding behavior.
- Describe integration with compilation and CUDA graphs, including supported shapes and fallback behavior.
- Report the 7.84x standalone sequence speedup beside the preliminary 5.3% integrated forward latency reduction, with their benchmark conditions.

### Evidence: packed convolution earned an integrated benchmark

| Batch | Original convolution plus copies | Packed-output kernel | Isolated speedup |
| --- | ---: | ---: | ---: |
| 1 | 672.804 μs | 85.775 μs | 7.844× |
| 2 | 1,363.528 μs | 419.768 μs | 3.248× |
| 4 | 2,858.045 μs | 837.028 μs | 3.415× |

These isolated measurements used sequence length 2048. Correctness checks passed in the saved benchmark. The comparison includes the original copies, so the speedup describes the replacement sequence, not convolution arithmetic alone.

A later saved model comparison measured the original forward at 141.206 ms and packed convolution at 133.316 ms, a 7.890 ms reduction, approximately 5.6%. It used one real request, batch one, a 2048-token bucket and thirty repeats. Configurations ran sequentially. Tested slot probabilities were unchanged; this was not a broad quality evaluation.

The earlier ten-measurement integration result was 148.59 ms to 140.72 ms, approximately 5.3%. Use one matched run for a headline, rather than mixing absolute times across runs.

Sources: [isolated benchmark log](../data/packed-output-conv-benchmark.log), [correctness log](../data/packed-output-conv-correctness.log), [saved model comparison](../data/combined-gates-model-benchmark.log).

**Figure to add:** separate panels for isolated sequence time and model-forward time. Use their own scales and label the workloads.

### Putting the kernel back into Inductor's execution path

Changing the stores was only part of the work. I also had to put the replacement back into the compiled model without losing the launch machinery that made the generated kernel practical to run. I retained Inductor's `triton_heuristics.pointwise` wrapper, which selects launch configurations and caches the tuned launcher, rather than introducing a separate Triton autotuning path for this kernel.

That meant updating the compiler metadata along with the arithmetic. The replacement had five tensor pointers and three outputs, and its metadata needed to describe that signature and store count. I removed inherited tiling scores that referred to the old layout. The shared implementation builds and caches launchers by batch size and sequence length, supplies the sequence-length specialization, and updates the row-size hint for the workload. Size hints guide tuning; the actual launch dimensions and masks determine which elements execute. Retaining the decorator alone would not make stale metadata appropriate for the new kernel.

I then exposed the kernel as a CUDA custom operation, with a fake implementation declaring the shape, dtype and contiguous strides of its three outputs. This let Inductor represent the replacement inside the compiled graph while keeping the private launcher outside tracing. The fresh generated wrapper contained the QKV projection, custom convolution call and gate preparation together, and checked the output layout I had promised. These remained separate GPU launches; appearing in one compiled wrapper does not imply kernel fusion. The opaque custom-op boundary also limits Inductor's ability to optimize through the replacement.

This integration let the standalone benchmark and model use the same kernel implementation. Fullgraph capture passed in the integration check, and graph-backed model timing tested the replacement in context. A fresh runtime trace is still needed to directly verify removal of the old copy launches. Using Inductor's tuning machinery keeps configuration selection consistent with this generated pointwise path; it does not guarantee an optimal configuration or a model-level speedup. The retained interface is private and tied to the pinned PyTorch environment.

**Excerpt to add:** the revised launcher metadata beside the generated wrapper's custom-op call and output-stride checks. This shows how the hand-edited kernel rejoined the compiled execution path.

Sources: [shared kernel and launcher factory](../model/decider/packed_conv_kernel.py), [custom-op registration and fake outputs](../model/decider/packed_conv.py), [post-integration compiled structure](compiled-structure-and-fusion-targets.md#what-the-custom-op-boundary-means), [generated wrapper](../data/l4-source-b1-packed/compiled_forward_04.py).

## A fast microbenchmark wasn't enough

- Explain why isolated savings need model-level validation.
- Combined gates: approximately 1.87x faster in isolation, but no demonstrated integrated gain. The extrapolated batch-one saving was only about 0.295 ms across eighteen layers.
- Persistent residual RMSNorm: approximately 1.89x faster in isolation, but no demonstrated integrated gain and failed full-logit validation.
- Explain why these experiments remained disabled. Avoid claiming a stable regression from sequential timing alone.
- Optionally include the decision-slot projection experiment, which showed no useful model-wide improvement.

### Evidence: two local wins I left disabled

| Experiment | Isolated baseline → candidate | Model baseline → candidate | Decision |
| --- | ---: | ---: | --- |
| Combined gates | 35.333 → 18.929 μs | 133.316 → 134.075 ms | No demonstrated integrated gain |
| Persistent residual RMSNorm | 39.373 → 20.838 μs | 132.706 → 132.985 ms | No demonstrated gain; numerical validation failed |

Both model comparisons used packed convolution in the baseline and ran configurations sequentially. The small median increases alone do not establish stable regressions. Combined gates preserved the tested slot probabilities. RMSNorm failed full-logit validation on 6,902 of 522,240 entries, with maximum absolute difference 0.2421875; slot-probability checks were not reached.

Sources: [gate microbenchmark](../data/unpack-gate-benchmark.log), [gate model benchmark](../data/combined-gates-model-benchmark.log), [RMSNorm microbenchmark](../data/persistent-rms-benchmark.log), [RMSNorm model log](../data/persistent-rms-model-b1.log).

## Attention offered a different kind of opportunity

- Introduce native GQA as a separate investigation in the six full-attention layers.
- Explain why the small expansion/mask cost alone was insufficient motivation; attention execution itself offered the larger opportunity.
- Describe grouped-head indexing, actual input strides and tiled online softmax.
- Summarize the offline tile sweep and the selected L4 configuration.
- Report the matched combined result: 142.885 ms to 126.938 ms GPU forward median, an 11.16% latency reduction with packed convolution and native GQA.
- Keep the numerical qualification beside the result: assertions were disabled, logits exceeded diagnostic tolerance, and four checked decisions were unchanged. Broader quality preservation remains unestablished.
- State that GPU forward timings exclude tokenization, input copies and readout. Reported p95 values are percentiles of replay sample means.

### Evidence: attention tuning and the final matched comparison

The offline attention sweep tested 24 configurations per batch at batches one, two and four. The selected configuration used a 64-token query tile, a 32-token key tile, four warps and two stages. The batch-one winner measured 538.163 μs in the sweep. Two configurations per batch exceeded the L4 shared-memory limit.

In the separate standalone attention benchmark, the batch-one baseline measured 1,594.916 μs and native GQA measured 557.322 μs, a 2.862× speedup. This comparison used synthetic inputs with compiled-model layouts and included expansion and mask construction in the baseline.

| Matched model configuration | Median GPU forward | p95 of replay sample means |
| --- | ---: | ---: |
| Original convolution + SDPA | 142.885 ms | 145.706 ms |
| Packed convolution + native GQA | 126.938 ms | 130.889 ms |

This saved comparison used an NVIDIA L4, batch one, four real requests, thirty samples and ten graph replays per sample. The configurations were interleaved. It saved 15.946 ms, an 11.16% latency reduction or 1.126× speedup. Compilation was warmed, and GPU timings excluded input copies and readout; tokenization was also outside the measurement. The p95 column does not measure individual-request tail latency.

**Numerical result:** assertions were disabled. There were 7,177 logits outside diagnostic tolerance, maximum logit difference 0.2109375, and maximum slot-probability difference 0.008498, about 0.85 percentage points. None of four checked decisions changed. This is measured performance with unresolved broader quality validation.

Sources: [configuration sweep](../data/native-gqa-sweep.json), [standalone attention benchmark](../data/native-gqa-benchmark.json), [matched combined model results](../data/combined-speedups-model.json). Model revision: `ff99089b6810691bd79134ec6febb5d9afa6e619`; PyTorch `2.14.1+cu130`.

**Figure to add:** matched baseline/candidate forward timings, captioned with the numerical qualification. Do not combine this with the initial batch-eight compile comparison into a cumulative speedup chart.

## What I would investigate next

- Identify the remaining bottleneck using fresh evidence.
- Explain the absolute savings budget and what would justify another experiment.
- Close with the next concrete decision rather than a generic optimization checklist.

## Source notes

- [Initial profiling](profiling.md)
- [Packed convolution and layout investigation](batch-one-fla-layouts.md)
- [Compiled structure and combined gates](compiled-structure-and-fusion-targets.md)
- [Persistent residual RMSNorm](persistent-residual-rmsnorm.md)
- [Native GQA and decision-slot projection](native-gqa.md)
