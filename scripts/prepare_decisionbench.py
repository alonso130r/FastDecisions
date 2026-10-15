#!/usr/bin/env python3
"""Download and convert a fixed DecisionBench sample for decider inference profiling.

Usage:
    .venv/bin/python scripts/prepare_decisionbench.py
    .venv/bin/python scripts/prepare_decisionbench.py --rows 1000 --seed 17

Writes artifacts/decisionbench_profile.jsonl. Each line has `state`, `questions`, and
metadata fields. `questions` can be passed directly to Decider.system_one().
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


REPO_ID = "Hanno-Labs/decision-bench"
SPLIT = "eval"
OUT = Path(__file__).resolve().parents[1] / "artifacts" / "decisionbench_profile.jsonl"


def get(row: dict[str, Any], *keys: str) -> Any:
    """Read one of the dataset's known field aliases."""
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    raise ValueError(f"DecisionBench row missing required field; expected one of {keys}")


def as_text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def convert(row: dict[str, Any]) -> dict[str, Any]:
    primitive = get(row, "primitive")
    state = get(row, "state", "state_json", "input", "context")
    instruction = get(row, "instruction", "question", "prompt")
    candidates = get(row, "candidates", "candidates_json", "options", "choices")
    if isinstance(state, str) and "state_json" in row and row.get("state") is None:
        state = json.loads(state)
    if isinstance(candidates, str):
        candidates = json.loads(candidates)

    if isinstance(candidates, dict):
        entries = list(candidates.items())
    elif isinstance(candidates, list):
        entries = [(str(i), candidate) for i, candidate in enumerate(candidates)]
    else:
        raise ValueError(f"Unsupported candidates value: {type(candidates).__name__}")

    options: list[str] = []
    for key, candidate in entries:
        if isinstance(candidate, dict):
            label = candidate.get("label", candidate.get("name", key))
            description = candidate.get("description")
            text = str(label) if description in (None, "") else f"{label}: {as_text(description)}"
        else:
            text = as_text(candidate)
        options.append(text)

    if primitive == "binary_classification":
        if len(options) != 2:
            raise ValueError(f"Binary row has {len(options)} candidates, expected 2")
        qtype = "noul"
    elif primitive == "candidate_selection":
        qtype = "choice"
    elif primitive == "ordinal_scoring":
        qtype = "score"
    else:
        raise ValueError(f"Unknown DecisionBench primitive: {primitive!r}")

    question = {
        "type": qtype,
        "instructions": as_text(instruction),
        "criteria": options,
    }

    candidate_ids = [str(entry.get("id", i)) if isinstance(entry, dict) else str(i)
                     for i, (_, entry) in enumerate(entries)]
    target_id = row.get("gold_candidate_id", row.get("target", row.get("gold", row.get("label"))))
    target = candidate_ids.index(str(target_id)) if target_id is not None and str(target_id) in candidate_ids else target_id
    # Keep the complete source row outside the inference arguments for scoring and filtering.
    return {
        "id": str(row.get("id", row.get("row_id", ""))),
        "task_id": row.get("task_id"),
        "primitive": primitive,
        "domain": row.get("domain"),
        "family": row.get("family"),
        "reasoning_required": row.get("reasoning_required"),
        "candidate_count": len(options),
        "state": as_text(state),
        "questions": {"decision": question},
        "target": target,
        "target_candidate_id": target_id,
        "target_distribution": row.get("target_distribution", row.get("gold_probabilities", row.get("probabilities"))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=500, help="maximum number of rows in the output (default: 500)")
    parser.add_argument("--seed", type=int, default=17, help="deterministic sampling seed")
    args = parser.parse_args()
    if args.rows < 1:
        parser.error("--rows must be positive")

    try:
        from huggingface_hub import HfApi, hf_hub_download, try_to_load_from_cache
    except ImportError:
        raise SystemExit("Missing dependency. Install with: .venv/bin/pip install huggingface_hub") from None

    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise SystemExit("Missing dependency. Install with: .venv/bin/pip install pyarrow") from None

    cached = try_to_load_from_cache(REPO_ID, "data/eval-00000-of-00001.parquet", repo_type="dataset")
    if cached:
        paths = [str(cached)]
    else:
        files = HfApi().list_repo_files(REPO_ID, repo_type="dataset")
        shards = sorted(path for path in files if path.startswith("data/") and path.endswith(".parquet"))
        if not shards:
            raise SystemExit(f"No Parquet data shards found in {REPO_ID}")
        # The current Hub release stores one eval split under data/. Read shards in order.
        paths = [hf_hub_download(REPO_ID, path, repo_type="dataset") for path in shards]
    # Stratify across primitive and candidate-count ranges, retaining representative
    # short/long contexts through deterministic shuffle within each stratum.
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    total_rows = 0
    for path in paths:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=1024):
            for raw in batch.to_pylist():
                total_rows += 1
                row = dict(raw)
                raw_candidates = row.get("candidates", row.get("candidates_json", row.get("options", [])))
                candidates = json.loads(raw_candidates) if isinstance(raw_candidates, str) else raw_candidates
                count = len(candidates) if candidates is not None else 0
                count_band = "2-4" if count <= 4 else "5-10" if count <= 10 else "11+"
                state = row.get("state", row.get("state_json", ""))
                if isinstance(state, str) and "state_json" in row and row.get("state") is None:
                    state_length = len(state)
                else:
                    state_length = len(as_text(state))
                length_band = "short" if state_length < 500 else "medium" if state_length < 2000 else "long"
                groups.setdefault((str(row.get("primitive", "unknown")), count_band, length_band), []).append(row)

    rng = random.Random(args.seed)
    # Stable deterministic selection; randomized order within each stratum, but
    # preserve workload diversity in both candidate count and state length.
    for key, rows in groups.items():
        rng.shuffle(rows)
        rows.sort(key=lambda row: len(as_text(row.get("state", row.get("state_json", "")))))
        # Interleave short and long examples before the seeded shuffle order.
        ordered = []
        lo, hi = 0, len(rows) - 1
        while lo <= hi:
            ordered.append(rows[lo])
            lo += 1
            if lo <= hi:
                ordered.append(rows[hi])
                hi -= 1
        groups[key] = ordered
    keys = sorted(groups)
    selected: list[dict[str, Any]] = []
    while len(selected) < min(args.rows, total_rows):
        progressed = False
        for key in keys:
            if groups[key] and len(selected) < args.rows:
                selected.append(groups[key].pop())
                progressed = True
        if not progressed:
            break

    # Persist a stable order independent of the source parquet shard ordering.
    converted = [convert(row) for row in selected]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for record in converted:
            f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

    print(f"Wrote {len(converted)} rows from {REPO_ID}:{SPLIT} to {OUT}")
    print("Primitive counts:", json.dumps(counts(converted), sort_keys=True))


def counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    result: dict[str, int] = {}
    for row in rows:
        k = str(row["primitive"])
        result[k] = result.get(k, 0) + 1
    return result


if __name__ == "__main__":
    main()
