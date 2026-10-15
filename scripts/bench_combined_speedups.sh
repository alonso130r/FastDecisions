#!/bin/sh
# Original convolution + SDPA versus packed convolution + native GQA.
set -eu
cd "$(dirname "$0")/.."
exec .venv/bin/modal run experiments/native_gqa/bench_native_gqa_model.py \
    --original-baseline \
    --batch-size 1 \
    --rows 4 \
    --output artifacts/combined-speedups-model.json \
    "$@"
