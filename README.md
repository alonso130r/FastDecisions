# FastDecisions

Making a one-pass decision model faster, one measured GPU bottleneck at a time.

This repository accompanies a blog post about profiling and optimizing inference for [Mapika/decider-2b](https://huggingface.co/Mapika/decider-2b). It contains the benchmark harness, custom Triton kernels, and experiment notes behind the results. The focus is low-concurrency inference on an NVIDIA L4: one or a few decision streams, where each decision must finish before the next dependent request can begin.

## What is a one-pass decision model?

Decider reads a state and a question with a fixed set of possible answers. A single model forward pass returns probabilities over those answers. For example, software could supply a support ticket and ask which queue should receive it. The model scores the supplied queues without generating a text response.

That changes the optimization problem. There is no token-by-token decoding loop to accelerate. The work here targets the forward pass, including tensor layouts and attention execution.

The benchmark uses a sample from [Hanno-Labs/decision-bench](https://huggingface.co/datasets/Hanno-Labs/decision-bench), converted into Decider's question format. The default preparation produces 500 rows with seed 17, preserving diversity in question type, candidate count, and state length.

## What the experiments found

Two changes improved the measured model-forward workload:

- **Packed convolution:** write the causal convolution output in the layout required by the following linear-attention operations, avoiding a repeated copy boundary.
- **Native grouped-query attention (GQA):** let query heads share the original key/value heads directly, with causal masking inside the attention kernel.

The combined benchmark recorded:

| Configuration | Median GPU forward time |
| --- | ---: |
| Original convolution + PyTorch SDPA | 142.885 ms |
| Packed convolution + native GQA | 126.938 ms |

That is an **11.16% latency reduction**, or **1.126× speedup**, for the recorded L4 workload: batch size one, four requests, thirty timing samples, and ten graph replays per sample. GPU timings exclude input copies, tokenization, and readout. These are model-forward measurements, not serving latency.

The numerical result needs equal attention. All four checked decisions stayed the same, but the maximum probability difference was about 0.85 percentage points and some logits exceeded the diagnostic tolerance. Numerical assertions were disabled for that run. The timing improvement is measured; quality preservation across representative requests remains unestablished. Native GQA is installed by the experiment runner and is not enabled as a production default.

Other experiments explain why isolated kernel speedups are insufficient:

| Experiment | Outcome |
| --- | --- |
| Combined gate projections and transforms | Faster in isolation; no demonstrated integrated model speedup. |
| Persistent residual RMSNorm | Faster in isolation; integrated numerical validation failed and model latency did not improve. |
| Project only decision-slot hidden states | No meaningful model-wide latency improvement in the recorded run. |

See the [combined benchmark details](blog/native-gqa.md#combined-result-original-model-versus-both-kernels) for measurement conditions and numerical diagnostics. Results here summarize recorded experiments, rather than promising the same gains on other GPUs or input shapes.

## Read the experiment notes

The notes follow the investigation from profiling to kernel experiments. Earlier sections sometimes describe work that was subsequently completed; read their follow-up and outcome sections for the final results.

1. [Profiling the decision workload](blog/profiling.md): how the workload was measured and how to interpret traces and generated code.
2. [Batch-one layouts and packed convolution](blog/batch-one-fla-layouts.md): locating the copy boundary and changing the producer's output layout.
3. [Compiled structure and combined gates](blog/compiled-structure-and-fusion-targets.md): fusion boundaries and the limits of reducing launches.
4. [Persistent residual RMSNorm](blog/persistent-residual-rmsnorm.md): an isolated improvement that failed integrated validation.
5. [Native GQA and combined results](blog/native-gqa.md): attention implementation, numerical differences, and the combined benchmark. Later sections also cover decision-slot projection.

## Reproduce a GPU run

Run these commands from the repository root. The main GPU workflow uses Modal, so a local NVIDIA GPU is not required. You need Python, a Modal account with GPU access, and network access to download the dataset and model. Modal runs incur cloud compute charges.

### Prepare the workload

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install modal huggingface_hub pyarrow tqdm
python -m modal setup
python prepare_decisionbench.py --rows 500 --seed 17
```

This writes `data/decisionbench_profile.jsonl`. The Modal image in `src/benchmark.py` installs the GPU dependencies, including PyTorch, Transformers, Flash Linear Attention, and causal-conv1d, and uploads the local inference code. The remote runner downloads model weights into a persistent cache. A local weight download is unnecessary for this path.

### Run the combined comparison

```bash
sh scripts/bench_combined_speedups.sh
```

The script compares original convolution plus SDPA against packed convolution plus native GQA. It saves timings and numerical diagnostics to `data/combined-speedups-model.json`. Read those diagnostics alongside latency before drawing a conclusion about the replacement kernels.

The runner accepts additional options through the script. For a repeatable model selection, supply a specific Hugging Face commit:

```bash
sh scripts/bench_combined_speedups.sh --revision <model-commit>
```

The combined runner defaults to model commit `ff99089b6810691bd79134ec6febb5d9afa6e619`. Record the model commit and software environment with any published result. Initial image builds, downloads, compilation, and autotuning can take substantially longer than warmed inference.

### Capture a compiled profile

```bash
modal run src/benchmark.py::modal_main \
  --batch-size 1 --rows 1 \
  --warmup-batches 3 --repeats 20 \
  --use-compile --use-cuda-graphs --profile-mode compiled
```

Packed convolution is enabled by default for supported engine configurations; add `--no-packed-conv` for the original convolution path. This profiling entry point does not install the experimental native-GQA replacement.

The command prints benchmark results and the remote trace location. This entry point defaults to model revision `main`; use `--revision` to pin a commit. Traces are stored in the Modal volume `decider-hf-cache`. Profiling temporarily disables CUDA graph replay to expose individual kernels, so summed trace activity should not be treated as graph-backed request latency. Percentiles from the combined replay benchmark describe sample means, not individual request tail latency.

## Repository map

| Path | Purpose |
| --- | --- |
| `src/benchmark.py` | Workload timing and profiling, with local CUDA/MPS and remote Modal entry points. |
| `src/bench_*.py`, `src/check_*.py` | Isolated kernel checks and integrated model experiments. |
| `src/dump_compiled_code.py` | Capture generated compiled code for inspection. |
| `model/decider/` | Local Decider inference code and experimental kernel integrations. |
| `src/kernels/` | Standalone convolution prototypes and reference implementations. |
| `prepare_decisionbench.py` | Download, sample, and convert the evaluation workload. |
| `download_model.py` | Download model files for local inference. |
| `scripts/bench_combined_speedups.sh` | Run the original-versus-combined model comparison. |
| `blog/` | Detailed experiment narrative and recorded measurements. |

Model weights and generated `data/` artifacts are ignored by Git. Links in the notes to traces, logs, and compiled dumps refer to artifacts from the original investigation; they are not included in a fresh clone. Downloading the model alone also does not install the dependencies required for local inference. The Modal image is the environment definition used by the GPU workflow.

## Attribution

The underlying model and inference package come from [Mapika/decider](https://github.com/Mapika/decider), based on [Qwen3.5-2B-Base](https://huggingface.co/Qwen/Qwen3.5-2B-Base). The evaluation workload comes from Hanno Labs' DecisionBench. This repository focuses on inference profiling and optimization; consult the upstream model documentation for training, calibration, and broader evaluation results.
