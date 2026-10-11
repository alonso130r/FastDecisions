"""Compile one representative Decider forward and save PyTorch Inductor's generated source."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))

DEFAULT_DATA = ROOT / "data" / "decisionbench_profile.jsonl"
DEFAULT_MODEL = ROOT / "model"
DEFAULT_OUTPUT = ROOT / "data" / "compiled_code"
MODAL_CACHE_PATH = "/cache"
MODAL_DATA_PATH = "/root/data/decisionbench_profile.jsonl"
MODAL_OUTPUT_PATH = "/cache/compiled-code"


def load_rows(path: Path, limit: int) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
                if len(rows) == limit:
                    break
    if len(rows) < limit:
        raise ValueError(f"Requested {limit} rows, but {path} contains only {len(rows)}")
    return rows


def make_requests(rows: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    from decider.systemone import render_question

    requests = []
    for row in rows:
        _, spec = next(iter(row["questions"].items()))
        if spec.get("type") in ("noul", "bool") and isinstance(spec.get("criteria"), list):
            candidates = spec["criteria"]
            spec = {**spec, "criteria": {"false": candidates[0], "true": candidates[1]}}
        question = render_question(spec)
        requests.append((row["state"], [{
            "question": question["question"],
            "options": question["options"],
            "_type": question["type"],
        }]))
    return requests


def make_input_ids(engine: Any, items: list[dict[str, Any]], device: str):
    from decider.engine import B_BUCKETS, GRAPH_MAX_T, LONG_STEP, T_BUCKETS, _bucket, fill_ids

    max_length = max(len(item["ids"]) for item in items)
    seq_len = _bucket(max_length, T_BUCKETS) or math.ceil(max_length / LONG_STEP) * LONG_STEP
    batch = _bucket(len(items), B_BUCKETS) if seq_len <= GRAPH_MAX_T else None
    batch = batch or len(items)
    ids = fill_ids([item["ids"] for item in items], batch, seq_len, engine.tok.pad_token_id)
    return ids.to(device, non_blocking=device.startswith("cuda"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    import torch
    from decider.infer import Decider
    from torch._inductor.utils import run_and_get_code

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but this local PyTorch installation has no available CUDA device")
    if device == "cpu":
        print("Using CPU Inductor. This output is not CUDA/Triton code for the L4.")

    rows = load_rows(args.data, args.batch_size)
    decider = Decider(args.model, device=device, use_graphs=True, compile=True)
    if decider.eng is None:
        raise RuntimeError("Decider did not create its compiled Engine")
    engine = decider.eng
    engine.use_graphs = False

    _, items = decider._decide_items(make_requests(rows))
    ids = make_input_ids(engine, items, device)
    print(
        f"Compiling {decider.name} forward on {device}: input_ids shape={tuple(ids.shape)}, "
        f"dtype={ids.dtype}, torch={torch.__version__}"
    )

    with torch.no_grad():
        _, generated_sources = run_and_get_code(engine._fwd_impl, ids)
    if device == "cuda":
        torch.cuda.synchronize()
    if not generated_sources:
        raise RuntimeError("Inductor returned no generated source for this forward")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for index, source in enumerate(generated_sources, start=1):
        output_path = args.output_dir / f"compiled_forward_{index:02d}.py"
        output_path.write_text(source, encoding="utf-8")
        written.append(output_path)
        print(f"Generated source: {output_path} ({len(source):,} characters)")
    print(f"Saved {len(written)} generated source file(s) to {args.output_dir}")


try:
    import modal
except ImportError:  # pragma: no cover - Modal remains optional for local use
    modal = None


if modal is not None:
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
    modal_app = modal.App("decider-compiled-code")

    @modal_app.function(
        image=modal_image,
        gpu="L4",
        volumes={MODAL_CACHE_PATH: modal_cache},
        timeout=60 * 60,
    )
    def _modal_dump_compiled_code(
        model_id: str,
        revision: str,
        row_limit: int,
        batch_size: int,
    ) -> dict[str, Any]:
        import torch
        from huggingface_hub import snapshot_download

        sys.path.insert(0, "/root/model")
        from decider.infer import Decider
        from torch._inductor.utils import run_and_get_code

        if not torch.cuda.is_available():
            raise RuntimeError("Modal function did not receive a CUDA GPU")
        model_path = Path(snapshot_download(repo_id=model_id, revision=revision))
        rows = load_rows(Path(MODAL_DATA_PATH), row_limit)
        decider = Decider(model_path, device="cuda", use_graphs=True, compile=True)
        if decider.eng is None:
            raise RuntimeError("Decider did not create its compiled Engine")
        engine = decider.eng
        engine.use_graphs = False

        _, items = decider._decide_items(make_requests(rows[:batch_size]))
        ids = make_input_ids(engine, items, "cuda")
        print(
            f"Compiling {decider.name} on {torch.cuda.get_device_name()}: "
            f"input_ids shape={tuple(ids.shape)}, dtype={ids.dtype}, torch={torch.__version__}"
        )

        with torch.no_grad():
            _, generated_sources = run_and_get_code(engine._fwd_impl, ids)
        torch.cuda.synchronize()
        if not generated_sources:
            raise RuntimeError("Inductor returned no generated source for this forward")

        output_dir = Path(MODAL_OUTPUT_PATH)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_files = []
        for index, source in enumerate(generated_sources, start=1):
            output_path = output_dir / f"compiled_forward_{index:02d}.py"
            output_path.write_text(source, encoding="utf-8")
            output_files.append(output_path.as_posix())
            print(f"Generated source: {output_path} ({len(source):,} characters)")
        modal_cache.commit()
        return {
            "model": decider.name,
            "gpu": torch.cuda.get_device_name(),
            "input_shape": list(ids.shape),
            "torch_version": torch.__version__,
            "output_files": output_files,
        }

    @modal_app.local_entrypoint()
    def modal_main(
        model_id: str = "Mapika/decider-2b",
        revision: str = "main",
        rows: int = 8,
        batch_size: int = 8,
    ) -> None:
        if batch_size < 1 or rows < batch_size:
            raise ValueError("rows must be at least batch_size, and batch_size must be positive")
        result = _modal_dump_compiled_code.remote(model_id, revision, rows, batch_size)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
