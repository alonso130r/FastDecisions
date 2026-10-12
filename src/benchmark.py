"""Benchmark inference runtimes against the prepared DecisionBench JSONL workload."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import statistics
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Sequence

from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))

DEFAULT_DATA = ROOT / "data" / "decisionbench_profile.jsonl"
DEFAULT_MODEL = ROOT / "model"


class InferenceRuntime(ABC):
    """Adapter contract separating the benchmark harness from model runtimes.

    Implement `score_batch` only when the runtime performs a real batched
    inference call. The base implementation preserves correctness by scoring
    rows sequentially, but advertises no batch support.
    """

    name: str
    device: str

    @classmethod
    @abstractmethod
    def load(cls, model_path: Path, device: str | None, options: argparse.Namespace) -> "InferenceRuntime":
        """Load the model and return its runtime adapter."""

    @abstractmethod
    def score_one(self, row: dict[str, Any]) -> Any:
        """Score one dataset row, returning the runtime result."""

    def score_batch(self, rows: Sequence[dict[str, Any]]) -> list[Any]:
        """Score a batch. Override for true batching; base implementation is sequential."""
        return [self.score_one(row) for row in rows]

    @property
    def supports_batching(self) -> bool:
        return type(self).score_batch is not InferenceRuntime.score_batch

    def warmup(self, rows: Sequence[dict[str, Any]]) -> None:
        """Run untimed warmup requests. Override for runtime-specific setup."""
        self.score_batch(rows)

    def synchronize(self) -> None:
        """Synchronize asynchronous device work before/after timed sections."""


class DeciderRuntime(InferenceRuntime):
    """Adapter for the downloaded model's Decider.system_one interface."""

    def __init__(self, decider: Any):
        self.decider = decider
        self.name = decider.name
        self.device = str(decider.dev)

    @classmethod
    def load(cls, model_path: Path, device: str | None, options: argparse.Namespace) -> "DeciderRuntime":
        import torch

        from decider.infer import Decider

        if device in (None, "auto"):
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                raise RuntimeError(
                    "No supported accelerator is available. This benchmark requires CUDA or Apple Metal (MPS); "
                    "PyTorch reports neither as available."
                )
        elif device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"Requested device {device!r}, but CUDA is unavailable in this PyTorch environment")
        elif device.startswith("mps") and not (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ):
            raise RuntimeError(f"Requested device {device!r}, but Apple Metal (MPS) is unavailable")
        elif not device.startswith(("cuda", "mps")):
            raise ValueError("Device must be 'auto', 'cuda', or 'mps'; CPU is not supported by this benchmark")

        use_graphs = None if not options.no_graphs else False
        compile_model = not getattr(options, "no_compile", False)
        return cls(Decider(model_path, device=device, use_graphs=use_graphs, compile=compile_model))

    def score_one(self, row: dict[str, Any]) -> Any:
        return self.decider.system_one(row["state"], row["questions"])

    def score_batch(self, rows: Sequence[dict[str, Any]]) -> list[Any]:
        requests = []
        for row in rows:
            # DecisionBench rows have one question. Convert its rendered typed
            # question to Decider's lower-level batch question representation.
            _, spec = next(iter(row["questions"].items()))
            from decider.systemone import render_question

            # DecisionBench represents binary classes as ordered candidates;
            # the decider's NOUL form represents the first/second candidate as
            # false/true and requires a criteria map instead of an option list.
            if spec.get("type") in ("noul", "bool") and isinstance(spec.get("criteria"), list):
                candidates = spec["criteria"]
                spec = {
                    **spec,
                    "criteria": {"false": candidates[0], "true": candidates[1]},
                }
            question = render_question(spec)
            requests.append((row["state"], [{
                "question": question["question"],
                "options": question["options"],
                "_type": question["type"],
            }]))
        return self.decider.decide_batch(requests)

    @property
    def supports_batching(self) -> bool:
        # Decider.decide_batch collates eager rows into one model batch and
        # Engine.score_items accepts a list of rows for the graph-backed path.
        return True

    def synchronize(self) -> None:
        import torch

        if self.device.startswith("cuda"):
            torch.cuda.synchronize(self.device)
        elif self.device.startswith("mps"):
            torch.mps.synchronize()


def load_rows(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    if not rows:
        raise ValueError(f"No benchmark rows found in {path}")
    return rows


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    pos = (len(ordered) - 1) * pct / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def profile_mps(runtime: InferenceRuntime, batches: Sequence[Sequence[dict[str, Any]]], trace_path: Path) -> None:
    """Collect CPU-visible PyTorch ops and estimate dense-linear arithmetic intensity.

    MPS does not currently expose the per-kernel hardware counters required for
    measured FLOPs/DRAM traffic. The report therefore labels these as estimates
    and treats model-weight traffic as a range from ideal reuse to reload per op.
    Fused attention/linear-attention kernels are reported as unmodeled.
    """
    import torch

    if not runtime.device.startswith("mps"):
        raise ValueError("The MPS estimate report requires an MPS runtime")

    profiler = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    )
    with profiler:
        for batch in tqdm(batches, desc="MPS profiling", unit="batch", leave=False):
            runtime.score_batch(batch)
            profiler.step()
            runtime.synchronize()

    trace_path.parent.mkdir(parents=True, exist_ok=True)
    profiler.export_chrome_trace(str(trace_path))
    print(f"MPS profiler trace: {trace_path}")
    print("Top PyTorch operators by self CPU time (MPS kernels are not individually timed):")
    print(profiler.key_averages(group_by_input_shape=True).table(sort_by="self_cpu_time_total", row_limit=25))

    # Aggregate only operations with a well-defined dense matrix multiply shape.
    # Shapes are input shape(s), e.g. [B,T,K] and [K,H] for torch.nn.functional.linear.
    totals: dict[str, float] = {"flops": 0.0, "weight_bytes_once": 0.0, "weight_bytes_per_call": 0.0,
                                "activation_bytes": 0.0, "calls": 0.0}
    modeled_names = {"aten::linear", "aten::mm", "aten::addmm", "aten::matmul", "aten::bmm"}
    unmodeled = []
    for event in profiler.key_averages(group_by_input_shape=True):
        if event.key not in modeled_names:
            continue
        try:
            shapes = event.input_shapes
            # Count only shapes that look like a matrix op: first operand and weights.
            tensor_shapes = [s for s in shapes if isinstance(s, (list, tuple)) and s and all(isinstance(x, int) for x in s)]
            if len(tensor_shapes) < 2:
                continue
            a, w = tensor_shapes[0], tensor_shapes[1]
            if len(a) < 2 or len(w) != 2 or a[-1] != w[-1]:
                continue
            m = math.prod(a[:-1])
            k, n = w[1], w[0] if event.key == "aten::linear" else w[-1]
            # For linear, PyTorch weight is [out_features, in_features]. For
            # mm/addmm, the second operand is [K,N].
            if event.key == "aten::linear":
                k, n = w[1], w[0]
            else:
                k, n = w[0], w[1]
            calls = max(1, int(event.count))
            dtype_bytes = 2  # model's MPS path loads float16 weights/activations
            totals["flops"] += 2 * m * k * n * calls
            totals["weight_bytes_once"] += k * n * dtype_bytes
            totals["weight_bytes_per_call"] += k * n * dtype_bytes * calls
            totals["activation_bytes"] += (m * k + m * n) * dtype_bytes * calls
            totals["calls"] += calls
        except (TypeError, ValueError, IndexError):
            unmodeled.append(event.key)

    flops = totals["flops"]
    bytes_ideal = totals["weight_bytes_once"] + totals["activation_bytes"]
    bytes_no_reuse = totals["weight_bytes_per_call"] + totals["activation_bytes"]
    if flops and bytes_ideal and bytes_no_reuse:
        print("Dense linear arithmetic estimate (not a hardware-counter measurement):")
        print(f"  modeled calls: {int(totals['calls'])}")
        print(f"  estimated operations: {flops / 1e12:.4f} TFLOP (2 FLOPs per multiply-accumulate)")
        print(f"  estimated traffic: {bytes_ideal / 1e9:.3f} GB with ideal weight reuse to {bytes_no_reuse / 1e9:.3f} GB without reuse")
        print(f"  arithmetic intensity range: {flops / bytes_no_reuse:.2f} to {flops / bytes_ideal:.2f} FLOP/byte")
        print("  excludes embedding, normalization, activations, attention/linear-attention internals, and kernel workspace traffic")
    else:
        print("No dense matrix operators with usable shape metadata were found; arithmetic intensity unavailable.")
    print("  Qwen3.5 fused attention and gated-delta operations are not included in this estimate.")


@contextlib.contextmanager
def _cuda_profile_ranges(runtime: InferenceRuntime, torch: Any, profile_mode: str):
    """Label Decider stages and select eager or compiled forward for a CUDA trace."""
    originals: list[tuple[Any, str, Any]] = []
    engine_profile_state: tuple[Any, bool, Any] | None = None

    def wrap_method(owner: Any, method_name: str, label: str) -> None:
        original = getattr(owner, method_name)

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            with torch.profiler.record_function(label):
                return original(*args, **kwargs)

        originals.append((owner, method_name, original))
        setattr(owner, method_name, wrapped)

    if isinstance(runtime, DeciderRuntime):
        decider = runtime.decider
        wrap_method(decider, "_decide_items", "stage.prompt_build_and_tokenize")
        if decider.eng is not None:
            engine = decider.eng
            engine_profile_state = (engine, engine.use_graphs, engine._fwd_impl)
            if profile_mode == "compiled" and not engine.cfg.get("compile", False):
                raise ValueError("--profile-mode compiled requires --use-compile")
            # Graph replay hides the underlying operations from Kineto. Disable
            # graphs for the capture, while preserving the compiled callable
            # when compiled profiling was requested.
            engine.use_graphs = False
            if profile_mode == "eager":
                engine._fwd_impl = engine._fwd_eager
            wrap_method(engine, "_fwd", "stage.model_forward")
            wrap_method(decider.eng, "score_items", "stage.engine_prepare_forward_and_readout")
        else:
            wrap_method(decider.m, "slot_logits", "stage.model_forward_eager")
    try:
        yield
    finally:
        for owner, method_name, original in reversed(originals):
            setattr(owner, method_name, original)
        if engine_profile_state is not None:
            engine, use_graphs, fwd_impl = engine_profile_state
            engine.use_graphs = use_graphs
            engine._fwd_impl = fwd_impl


def _cuda_kernel_summary(profiler: Any, limit: int = 30) -> list[tuple[str, int, float]]:
    """Aggregate CUDA kernel names from Kineto events where this PyTorch build exposes them."""
    kernels: dict[str, list[float]] = {}
    for event in profiler.events():
        for kernel in getattr(event, "kernels", ()) or ():
            name = str(getattr(kernel, "name", "unknown CUDA kernel"))
            duration = getattr(kernel, "device_duration", None)
            if duration is None:
                duration = getattr(kernel, "duration", 0.0)
            kernels.setdefault(name, []).append(float(duration or 0.0))
    ranked = sorted(
        ((name, len(durations), sum(durations)) for name, durations in kernels.items()),
        key=lambda item: item[2],
        reverse=True,
    )
    return ranked[:limit]


def _print_cuda_profile(profiler: Any, profile_mode: str) -> None:
    """Print per-kernel and shape-grouped operator evidence for Triton candidate selection."""
    kernels = _cuda_kernel_summary(profiler)
    if kernels:
        print("Top CUDA kernels by total device time (kernel timings are aggregated over the capture):")
        print(f"{'CUDA kernel':76} {'calls':>8} {'total ms':>12} {'avg us':>12}")
        for name, count, total_us in kernels:
            print(f"{name[:76]:76} {count:>8} {total_us / 1000:>12.3f} {total_us / count:>12.2f}")
    else:
        print("This PyTorch profiler build did not expose individual CUDA kernel records; inspect the exported Chrome trace.")

    averages = profiler.key_averages(group_by_input_shape=True)
    # Kineto includes CUDA kernel records in key_averages(), but those rows do
    # not have operator input shapes. Keep the ATen operator rows separate so
    # this section gives the tensor shapes needed to map work to Triton.
    operators = [event for event in averages if str(event.key).startswith("aten::")]
    device_time_fields = (
        "self_device_time_total",
        "self_cuda_time_total",
        "device_time_total",
        "cuda_time_total",
    )
    device_time_field = next(
        (field for field in device_time_fields if operators and hasattr(operators[0], field)),
        None,
    )
    if operators and device_time_field:
        operators.sort(key=lambda event: getattr(event, device_time_field, 0.0), reverse=True)
        print("Top ATen operators by device time, grouped by input shape:")
        print(f"{'operator':36} {'calls':>7} {'device ms':>12} {'avg us':>12}  input shapes")
        for event in operators[:30]:
            total_us = float(getattr(event, device_time_field, 0.0) or 0.0)
            count = int(getattr(event, "count", 0) or 0)
            shapes = getattr(event, "input_shapes", None) or []
            print(
                f"{str(event.key)[:36]:36} {count:>7} {total_us / 1000:>12.3f} "
                f"{total_us / count if count else 0.0:>12.2f}  {shapes}"
            )
    else:
        print("No shape-grouped ATen device-time rows were available in this PyTorch profiler capture.")

    print(
        f"CUDA profile forward={profile_mode}; CUDA graphs are disabled during capture; "
        "timed benchmark settings are restored afterward."
    )
    print("Stage totals in the trace: stage.prompt_build_and_tokenize, stage.model_forward, and")
    print("stage.engine_prepare_forward_and_readout. Prioritize repeated, expensive kernels with regular tensor shapes;")
    print("small kernels dominated by launch overhead are better fused into a surrounding Triton kernel.")


def timed_batches(
    runtime: InferenceRuntime,
    rows: list[dict[str, Any]],
    batch_size: int,
    warmup_batches: int,
    repeats: int,
    profile_batches: int = 0,
    trace_path: Path | None = None,
    profile_mode: str = "eager",
) -> dict[str, float | int]:
    if batch_size > 1 and not runtime.supports_batching:
        raise ValueError(f"Runtime {runtime.name!r} does not support true batching")

    batches = [rows[i : i + batch_size] for i in range(0, len(rows), batch_size)]
    batches = [batch for batch in batches if len(batch) == batch_size]
    if not batches:
        raise ValueError(f"Need at least {batch_size} rows for batch size {batch_size}")

    warmups = tqdm(range(warmup_batches), desc=f"batch {batch_size} warmup", unit="batch", leave=False)
    for i in warmups:
        runtime.score_batch(batches[i % len(batches)])
    runtime.synchronize()

    if profile_batches and runtime.device.startswith("mps"):
        profile_mps(runtime, batches[:profile_batches], trace_path or ROOT / "data" / "mps_profile_trace.json")
    elif profile_batches:
        profiler = None
        import torch

        activities = [torch.profiler.ProfilerActivity.CPU]
        if runtime.device.startswith("cuda"):
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        profiler = torch.profiler.profile(activities=activities, record_shapes=True, profile_memory=True)
        with _cuda_profile_ranges(runtime, torch, profile_mode) if runtime.device.startswith("cuda") else contextlib.nullcontext():
            if runtime.device.startswith("cuda"):
                # Compile/JIT kernels before collecting the profile so
                # one-time setup does not pollute the trace.
                runtime.score_batch(batches[0])
                runtime.synchronize()
            with profiler:
                for i in tqdm(range(profile_batches), desc=f"batch {batch_size} profiling", unit="batch", leave=False):
                    with torch.profiler.record_function("stage.end_to_end_score_batch"):
                        runtime.score_batch(batches[i % len(batches)])
                    profiler.step()
                    runtime.synchronize()
        if trace_path is not None:
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            profiler.export_chrome_trace(str(trace_path))
            print(f"profiler trace: {trace_path}")
        if runtime.device.startswith("cuda"):
            _print_cuda_profile(profiler, profile_mode)
        else:
            print("Profiler operators (self CPU time):")
            print(profiler.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))

    timings: list[float] = []
    total_rows = 0
    timed = tqdm(range(repeats), desc=f"batch {batch_size} timed", unit="batch", leave=False)
    for i in timed:
        batch = batches[i % len(batches)]
        runtime.synchronize()
        start = time.perf_counter()
        runtime.score_batch(batch)
        runtime.synchronize()
        timings.append(time.perf_counter() - start)
        total_rows += len(batch)

    median = statistics.median(timings)
    return {
        "batch_size": batch_size,
        "batches_timed": len(timings),
        "median_batch_ms": median * 1000,
        "p95_batch_ms": percentile(timings, 95) * 1000,
        "median_ms_per_decision": median * 1000 / batch_size,
        "decisions_per_second": total_rows / sum(timings),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--runtime", choices=["decider"], default="decider")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1])
    parser.add_argument("--warmup-batches", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--rows", type=int, default=None, help="limit rows loaded from the JSONL")
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps"],
        default="auto",
        help="accelerator to use; auto prefers CUDA, then Apple Metal (MPS), and errors if neither is available",
    )
    parser.add_argument("--no-graphs", action="store_true", help="disable CUDA graph execution")
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="disable torch.compile while retaining CUDA graphs; avoids first-run Inductor compile/autotune cost",
    )
    parser.add_argument("--profile", action="store_true", help="profile one batch size and export a PyTorch trace")
    parser.add_argument("--profile-batches", type=int, default=5, help="number of batches included in the profiler trace")
    parser.add_argument("--trace", type=Path, default=ROOT / "data" / "benchmark_trace.json", help="Chrome trace output path")
    parser.add_argument(
        "--wait-before-run",
        action="store_true",
        help="pause after model load and warmup until Enter is pressed, for attaching a profiler",
    )
    parser.add_argument(
        "--wait-after-run",
        action="store_true",
        help="pause after benchmark inference so an externally attached profiler can be stopped cleanly",
    )
    args = parser.parse_args()

    if args.warmup_batches < 0 or args.repeats < 1:
        parser.error("--warmup-batches must be nonnegative and --repeats must be positive")
    if args.profile_batches < 1:
        parser.error("--profile-batches must be positive")
    if any(size < 1 for size in args.batch_sizes):
        parser.error("batch sizes must be positive")

    rows = load_rows(args.data, args.rows)
    started = time.perf_counter()
    load_progress = tqdm(total=1, desc="loading model", unit="model")
    runtime = DeciderRuntime.load(args.model, args.device, args)
    load_progress.update(1)
    load_progress.close()
    load_seconds = time.perf_counter() - started
    print(f"runtime={runtime.name} device={runtime.device} load_seconds={load_seconds:.3f} rows={len(rows)}")
    if not runtime.supports_batching and any(size > 1 for size in args.batch_sizes):
        parser.error("This runtime has no true batch API; use --batch-sizes 1")

    for index, size in enumerate(args.batch_sizes):
        if args.wait_before_run and index == 0:
            # Initialize the model and warm it before an external profiler attaches.
            warm_rows = rows[: size * args.warmup_batches]
            if len(warm_rows) < size:
                parser.error(f"Need at least {size} rows for batch size {size}")
            warm_batches = [warm_rows[i : i + size] for i in range(0, len(warm_rows), size)]
            warm_batches = [batch for batch in warm_batches if len(batch) == size]
            for warm_batch in tqdm(warm_batches, desc=f"batch {size} warmup", unit="batch", leave=False):
                runtime.score_batch(warm_batch)
            runtime.synchronize()
            input("Model loaded and warmed. Attach/start Instruments, then press Enter to run the benchmark: ")
            # Avoid duplicating warmup after the profiler is attached.
            warmup_count = 0
        else:
            warmup_count = args.warmup_batches
        should_profile = args.profile and index == 0
        result = timed_batches(
            runtime,
            rows,
            size,
            warmup_count,
            args.repeats,
            profile_batches=args.profile_batches if should_profile else 0,
            trace_path=args.trace if should_profile else None,
        )
        mode = "batched" if size > 1 else "single-request"
        print(
            f"mode={mode} batch={size:>3} "
            f"median={result['median_batch_ms']:.2f} ms "
            f"p95={result['p95_batch_ms']:.2f} ms "
            f"median/decision={result['median_ms_per_decision']:.2f} ms "
            f"throughput={result['decisions_per_second']:.2f} decisions/s"
        )
        if args.wait_after_run and index == 0:
            input("Benchmark inference finished. Stop Instruments recording, then press Enter to exit: ")


# Optional Modal entrypoint. The ordinary local benchmark above does not require
# Modal to be installed. `modal run src/benchmark.py` uses this L4 function.
try:
    import modal
except ImportError:  # pragma: no cover - intentionally optional local dependency
    modal = None


if modal is not None:
    MODAL_MODEL_ID = "Mapika/decider-2b"
    MODAL_CACHE_PATH = "/cache"
    MODAL_DATA_PATH = "/root/data/decisionbench_profile.jsonl"
    MODAL_TRACE_PATH = "/cache/profiles/decider-l4-profile.json"
    modal_cache = modal.Volume.from_name("decider-hf-cache", create_if_missing=True)
    modal_image = (
        modal.Image.from_registry("nvidia/cuda:13.0.3-devel-ubuntu22.04", add_python="3.11")
        .entrypoint([])
        .apt_install("build-essential")
        .pip_install(
            "torch==2.14.1",
            "transformers==5.17.0",
            "huggingface_hub",
            "safetensors",
            "numpy",
            "tqdm",
        )
        .pip_install("flash-linear-attention[cuda]")
        .pip_install("ninja", "packaging", "wheel")
        .env({
            "CC": "gcc",
            "CXX": "g++",
            "HF_HOME": f"{MODAL_CACHE_PATH}/huggingface",
            "TOKENIZERS_PARALLELISM": "false",
        })
        .pip_install("causal-conv1d", extra_options="--no-build-isolation")
        .add_local_dir(str(ROOT / "model" / "decider"), remote_path="/root/model/decider")
        .add_local_file(str(DEFAULT_DATA), remote_path=MODAL_DATA_PATH)
    )
    modal_app = modal.App("decider-kernel-profile")

    @modal_app.function(
        image=modal_image,
        gpu="L4",
        volumes={MODAL_CACHE_PATH: modal_cache},
        timeout=60 * 60,
    )
    def _modal_profile(
        model_id: str,
        revision: str,
        row_limit: int,
        batch_size: int,
        warmup_batches: int,
        repeats: int,
        profile_batches: int,
        use_cuda_graphs: bool,
        use_compile: bool,
        profile_mode: str,
        torch_logs: str,
    ) -> dict[str, Any]:
        import os

        if torch_logs:
            # Must be set before importing torch for TORCH_LOGS to take effect.
            os.environ["TORCH_LOGS"] = torch_logs

        import torch
        from huggingface_hub import snapshot_download

        if not torch.cuda.is_available():
            raise RuntimeError("Modal function did not receive a CUDA GPU")
        sys.path.insert(0, "/root/model")

        model_path = Path(snapshot_download(repo_id=model_id, revision=revision))
        rows = load_rows(Path(MODAL_DATA_PATH), row_limit)
        options = argparse.Namespace(no_graphs=not use_cuda_graphs, no_compile=not use_compile)
        started = time.perf_counter()
        runtime = DeciderRuntime.load(model_path, "cuda", options)
        load_seconds = time.perf_counter() - started
        print(
            f"runtime={runtime.name} device={runtime.device} gpu={torch.cuda.get_device_name()} "
            f"load_seconds={load_seconds:.2f} rows={len(rows)} "
            f"cuda_graphs={use_cuda_graphs} compile={use_compile} "
            f"profile_mode={profile_mode} TORCH_LOGS={torch_logs or '(unset)'}"
        )

        trace_path = Path(MODAL_TRACE_PATH).with_name(f"decider-l4-{profile_mode}.json")
        result = timed_batches(
            runtime,
            rows,
            batch_size,
            warmup_batches,
            repeats,
            profile_batches=profile_batches,
            trace_path=trace_path,
            profile_mode=profile_mode,
        )
        print(
            f"CUDA profile summary: batch={batch_size} median={result['median_batch_ms']:.2f} ms "
            f"p95={result['p95_batch_ms']:.2f} ms "
            f"throughput={result['decisions_per_second']:.2f} decisions/s"
        )
        print(f"Chrome trace saved to Modal Volume: {trace_path}")
        modal_cache.commit()
        return {
            "runtime": runtime.name,
            "gpu": torch.cuda.get_device_name(),
            "device": runtime.device,
            "cuda_graphs": use_cuda_graphs,
            "compile": use_compile,
            "profile_mode": profile_mode,
            "torch_logs": torch_logs,
            "load_seconds": load_seconds,
            **result,
            "trace_path": trace_path.as_posix(),
        }

    @modal_app.function(
        image=modal_image,
        gpu="L4",
        volumes={MODAL_CACHE_PATH: modal_cache},
        timeout=60 * 60,
    )
    def _modal_inspect_layouts(model_id: str, revision: str, batch_size: int) -> dict[str, Any]:
        """Inspect inputs before FLA's existing contiguity guard, without editing FLA."""
        import functools
        import inspect
        import importlib.metadata

        import torch
        from huggingface_hub import snapshot_download

        sys.path.insert(0, "/root/model")
        from fla.ops.gated_delta_rule.chunk import ChunkGatedDeltaRuleFunction

        original_descriptor = inspect.getattr_static(ChunkGatedDeltaRuleFunction, "forward")
        original = ChunkGatedDeltaRuleFunction.forward
        signature = inspect.signature(original)
        layouts: list[dict[str, Any]] = []
        calls = 0

        @functools.wraps(original)
        def logged_forward(*args: Any, **kwargs: Any):
            nonlocal calls
            calls += 1
            if calls == 1:
                bound = signature.bind_partial(*args, **kwargs)
                for name, value in bound.arguments.items():
                    if isinstance(value, torch.Tensor):
                        entry = {
                            "argument": name,
                            "shape": list(value.shape),
                            "stride": list(value.stride()),
                            "contiguous": value.is_contiguous(),
                            "dtype": str(value.dtype),
                            "device": str(value.device),
                        }
                        layouts.append(entry)
                        print("FLA_PRE_GUARD " + json.dumps(entry), flush=True)
            return original(*args, **kwargs)

        model_path = Path(snapshot_download(repo_id=model_id, revision=revision))
        rows = load_rows(Path(MODAL_DATA_PATH), batch_size)
        runtime = DeciderRuntime.load(
            model_path, "cuda", argparse.Namespace(no_graphs=True, no_compile=True)
        )
        ChunkGatedDeltaRuleFunction.forward = staticmethod(logged_forward)
        try:
            with torch.no_grad():
                runtime.score_batch(rows)
            runtime.synchronize()
        finally:
            ChunkGatedDeltaRuleFunction.forward = original_descriptor
        if not layouts:
            raise RuntimeError("No FLA inputs were captured; this runtime did not call the patched forward")
        return {
            "diagnostic_only": True,
            "gpu": torch.cuda.get_device_name(),
            "torch_version": torch.__version__,
            "fla_version": importlib.metadata.version("flash-linear-attention"),
            "fla_source": inspect.getsourcefile(inspect.unwrap(original)),
            "model_revision": revision,
            "batch_size": batch_size,
            "forward_calls": calls,
            "first_call_layouts": layouts,
        }

    @modal_app.local_entrypoint()
    def inspect_layouts(
        model_id: str = MODAL_MODEL_ID,
        revision: str = "main",
        batch_size: int = 1,
    ) -> None:
        """Log one eager request's FLA inputs before the contiguity guard."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        result = _modal_inspect_layouts.remote(model_id, revision, batch_size)
        print(json.dumps(result, indent=2))

    @modal_app.local_entrypoint()
    def modal_main(
        model_id: str = MODAL_MODEL_ID,
        revision: str = "main",
        rows: int = 64,
        batch_size: int = 8,
        warmup_batches: int = 3,
        repeats: int = 10,
        profile_batches: int = 1,
        use_cuda_graphs: bool = False,
        use_compile: bool = False,
        profile_mode: str = "eager",
        torch_logs: str = "",
    ) -> None:
        """Profile warmed DecisionBench inference on a Modal NVIDIA L4."""
        if rows < batch_size:
            raise ValueError("--rows must be at least --batch-size")
        if min(warmup_batches, profile_batches) < 1 or repeats < 1:
            raise ValueError("warmup_batches and profile_batches must be positive; repeats must be positive")
        if profile_mode not in ("eager", "compiled"):
            raise ValueError("--profile-mode must be 'eager' or 'compiled'")
        if profile_mode == "compiled" and not use_compile:
            raise ValueError("--profile-mode compiled requires --use-compile")
        result = _modal_profile.remote(
            model_id,
            revision,
            rows,
            batch_size,
            warmup_batches,
            repeats,
            profile_batches,
            use_cuda_graphs,
            use_compile,
            profile_mode,
            torch_logs,
        )
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
