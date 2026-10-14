"""Isolated compiled expanded-head SDPA versus fixed native GQA on Modal L4."""
from pathlib import Path
import json
import sys
import modal

app = modal.App('native-gqa-benchmark')
image = None
if modal.is_local():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from benchmark import modal_image
    image = modal_image


@app.function(image=image, gpu='L4', timeout=3600)
def bench_remote(samples: int, replays: int):
    import random
    import statistics
    import torch
    import triton
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel
    sys.path.insert(0, '/root/model')
    from decider.native_gqa_kernel import native_gqa_attention
    from decider.engine import set_attention_backend_policy
    set_attention_backend_policy()
    torch.manual_seed(17)
    rng = random.Random(17)
    reports = []

    def expanded_sdpa(q, k, v):
        batch, _, t, d = k.shape
        k = k[:, :, None].expand(batch, 2, 4, t, d).reshape(batch, 8, t, d)
        v = v[:, :, None].expand(batch, 2, 4, t, d).reshape(batch, 8, t, d)
        positions = torch.arange(t, device=q.device)
        mask = positions[:, None] >= positions[None, :]
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)

    compiled_baseline = torch.compile(expanded_sdpa, dynamic=False)
    with torch.no_grad():
        for batch in (1, 2, 4):
            t = 2048
            q = torch.randn((batch, t, 8, 256), device='cuda', dtype=torch.bfloat16).permute(0, 2, 1, 3)
            k = torch.randn((batch, 2, t, 256), device='cuda', dtype=torch.bfloat16)
            v = torch.randn((batch, t, 2, 256), device='cuda', dtype=torch.bfloat16).permute(0, 2, 1, 3)
            with sdpa_kernel(SDPBackend.MATH):
                reference = F.scaled_dot_product_attention(q.float(), k.float(), v.float(),
                                                          is_causal=True, enable_gqa=True)
            functions = {'expanded_sdpa': compiled_baseline, 'native_gqa': native_gqa_attention}
            graphs, outputs = {}, {}
            errors = {}
            for name, fn in functions.items():
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    for _ in range(3):
                        got = fn(q, k, v)
                torch.cuda.current_stream().wait_stream(side)
                torch.cuda.synchronize()
                torch.testing.assert_close(got.float(), reference, atol=.02, rtol=.02)
                errors[name] = (got.float()-reference).abs().max().item()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    outputs[name] = fn(q, k, v)
                graphs[name] = graph
                for _ in range(10):
                    graph.replay()
            torch.cuda.synchronize()
            timings = {name: [] for name in graphs}
            for _ in range(samples):
                order = list(graphs)
                rng.shuffle(order)
                for name in order:
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(replays):
                        graphs[name].replay()
                    end.record()
                    end.synchronize()
                    timings[name].append(start.elapsed_time(end)*1000/replays)
            result = dict(batch=batch, sequence_length=t, correctness='passed within tolerance',
                          max_abs_error_vs_fp32=errors, results={})
            for name, values in timings.items():
                result['results'][name] = dict(median_us=statistics.median(values), min_us=min(values),
                    p95_sample_mean_us=sorted(values)[min(len(values)-1, int(.95*len(values)))])
            old = result['results']['expanded_sdpa']['median_us']
            new = result['results']['native_gqa']['median_us']
            result.update(speedup=old/new, saved_us=old-new, reduction_percent=100*(old-new)/old)
            reports.append(result)
            print(json.dumps(result), flush=True)
    return dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, triton=triton.__version__,
                samples=samples, replays_per_sample=replays, results=reports,
                baseline='torch.compile explicit 4x KV expansion, causal mask construction, SDPA',
                scope='synthetic attention only, current Q/K/V layouts; not an exact generated-source replay',
                timing='interleaved CUDA-event graph replay means; compilation and capture excluded',
                tolerance=dict(atol=.02, rtol=.02))


@app.local_entrypoint()
def main(samples: int = 30, replays: int = 100, output: str = 'data/native-gqa-benchmark.json'):
    if samples < 2 or replays < 1:
        raise ValueError('Require samples >= 2 and replays >= 1')
    result = bench_remote.remote(samples, replays)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
