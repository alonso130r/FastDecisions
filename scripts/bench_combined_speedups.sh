#!/bin/sh
# Original convolution + SDPA versus packed convolution + native GQA.
set -eu
cd "$(dirname "$0")/.."
exec .venv/bin/modal run src/bench_native_gqa_model.py \
    --original-baseline \
    --batch-size 1 \
    --rows 4 \
    --output data/combined-speedups-model.json \
    "$@"
