# Experiment guide

Run commands from the repository root after preparing the workload as described in the [main README](../README.md). Each runner builds on the shared Modal image in `benchmarks/benchmark.py`. Integrated kernels live in `src/decider/` so the uploaded inference package stays self-contained.

| Directory | Experiment | Recorded outcome |
| --- | --- | --- |
| `packed_conv/` | Emit convolution outputs in the consumer layout | Improved the measured integrated forward. |
| `native_gqa/` | Read shared key/value heads directly | Improved measured forward time, with numerical differences requiring broader quality evaluation. |
| `combined_gates/` | Combine gate projections and transforms | Isolated speedup without demonstrated model speedup. |
| `residual_rms/` | Retain residual sums through RMS normalization | Isolated speedup; integrated numerical validation failed. |
| `slot_projection/` | Project only decision slots | No meaningful measured model-wide improvement. |

## Combined model comparison

Run `sh scripts/bench_combined_speedups.sh`. It compares original convolution plus SDPA against packed convolution plus native GQA and writes `artifacts/combined-speedups-model.json`.

## Individual entry points

Prefix each path below with `modal run`:

| Purpose | Entry point |
| --- | --- |
| Convolution correctness | `experiments/packed_conv/check_packed_output_conv.py` |
| Convolution timing | `experiments/packed_conv/bench_packed_output_conv.py` |
| GQA isolated timing | `experiments/native_gqa/bench_native_gqa.py` |
| GQA configuration sweep | `experiments/native_gqa/bench_native_gqa_sweep.py` |
| GQA numerical diagnostics | `experiments/native_gqa/diagnose_native_gqa.py` |
| GQA model comparison | `experiments/native_gqa/bench_native_gqa_model.py` |
| Gate isolated timing | `experiments/combined_gates/bench_unpack_gate.py` |
| Gate model comparison | `experiments/combined_gates/bench_combined_gates_integration.py` |
| RMSNorm isolated timing | `experiments/residual_rms/bench_persistent_residual_rms.py` |
| RMSNorm model comparison | `experiments/residual_rms/bench_persistent_residual_rms_model.py` |
| Decision-slot projection | `experiments/slot_projection/bench_slot_projection.py` |

These commands launch paid remote GPU work. Inspect runner defaults for the model revision, sample count, and output location. Timings and numerical diagnostics must be interpreted together; unchanged decisions on a small sample do not establish quality preservation.
