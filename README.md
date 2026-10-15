# FastDecisions

Making a one-pass decision model faster, one measured GPU bottleneck at a time.

Read the companion blog post: [Why is my PyTorch model still slow?](https://medium.com/@vijaygoyal0770/why-is-my-pytorch-model-still-slow-d7c97bf5761b).

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

Results summarize recorded experiments rather than promising the same gains on other GPUs or input shapes. See [the experiment guide](experiments/README.md) for entry points and outcomes.

## Reproduce a GPU run

Run these commands from the repository root. The main GPU workflow uses Modal, so a local NVIDIA GPU is not required. You need Python, a Modal account with GPU access, and network access to download the dataset and model. Modal runs incur cloud compute charges.

### Prepare the workload

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install modal huggingface_hub pyarrow tqdm
python -m modal setup
python scripts/prepare_decisionbench.py --rows 500 --seed 17
```

This writes `artifacts/decisionbench_profile.jsonl`. The Modal image in `benchmarks/benchmark.py` installs the GPU dependencies, including PyTorch, Transformers, Flash Linear Attention, and causal-conv1d, and uploads the local inference code. The remote runner downloads model weights into a persistent cache. A local weight download is unnecessary for this path.

### Run the combined comparison

```bash
sh scripts/bench_combined_speedups.sh
```

The script compares original convolution plus SDPA against packed convolution plus native GQA. It saves timings and numerical diagnostics to `artifacts/combined-speedups-model.json`. Read those diagnostics alongside latency before drawing a conclusion about the replacement kernels.

The runner accepts additional options through the script. For a repeatable model selection, supply a specific Hugging Face commit:

```bash
sh scripts/bench_combined_speedups.sh --revision <model-commit>
```

The combined runner defaults to model commit `ff99089b6810691bd79134ec6febb5d9afa6e619`. Record the model commit and software environment with any published result. Initial image builds, downloads, compilation, and autotuning can take substantially longer than warmed inference.

### Capture a compiled profile

```bash
modal run benchmarks/benchmark.py::modal_main \
  --batch-size 1 --rows 1 \
  --warmup-batches 3 --repeats 20 \
  --use-compile --use-cuda-graphs --profile-mode compiled
```

Packed convolution is enabled by default for supported engine configurations; add `--no-packed-conv` for the original convolution path. This profiling entry point does not install the experimental native-GQA replacement.

The command prints benchmark results and the remote trace location. This entry point defaults to model revision `main`; use `--revision` to pin a commit. Traces are stored in the Modal volume `decider-hf-cache`. Profiling temporarily disables CUDA graph replay to expose individual kernels, so summed trace activity should not be treated as graph-backed request latency. Percentiles from the combined replay benchmark describe sample means, not individual request tail latency.

## Repository map

| Path | Purpose |
| --- | --- |
| `benchmarks/benchmark.py` | Workload timing and profiling, with local CUDA/MPS and remote Modal entry points. |
| `experiments/` | Benchmarks and checks grouped by optimization, with an [experiment guide](experiments/README.md). |
| `profiling/dump_compiled_code.py` | Capture generated compiled code for inspection. |
| `src/decider/` | Local Decider inference code and integrated kernels. |
| `scripts/prepare_decisionbench.py` | Download, sample, and convert the evaluation workload. |
| `scripts/download_model.py` | Download model files into `models/` for local inference. |
| `scripts/bench_combined_speedups.sh` | Run the original-versus-combined model comparison. |
| `artifacts/` | Generated datasets, traces, logs, and compiled dumps, ignored by Git. |
| `models/` | Downloaded weights and model configuration, ignored by Git. |

Install local preparation tools with `python -m pip install -r requirements.txt`. The Modal image in `benchmarks/benchmark.py` defines the GPU dependencies separately. Downloading the model alone does not install dependencies required for local inference.

Generated artifacts are not included in a fresh clone. Blog drafts remain local and are ignored by Git; the published companion article is linked above.

## Attribution

The underlying model and inference package come from [Mapika/decider](https://github.com/Mapika/decider), based on [Qwen3.5-2B-Base](https://huggingface.co/Qwen/Qwen3.5-2B-Base). The evaluation workload comes from Hanno Labs' DecisionBench. This repository focuses on inference profiling and optimization; consult the upstream model documentation for training, calibration, and broader evaluation results.
