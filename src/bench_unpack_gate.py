"""L4 benchmarks for gate transforms alone and complete projection/transform paths."""
import json
from pathlib import Path
import sys
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App('combined-gates-isolated-benchmark')
image = None
if modal.is_local():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from benchmark import modal_image
    image = modal_image


@app.function(image=image, gpu='L4', timeout=60*60)
def bench_remote(samples: int, replays: int):
    import statistics
    import torch
    import torch.nn.functional as F
    import triton
    sys.path.insert(0, '/root/model')
    from decider.unpack_gate_transform_kernel import unpack_gate

    torch.manual_seed(17)

    # Separate compiled functions preserve the original two-transform boundary.
    @torch.compile(fullgraph=True)
    def decay_transform(a, a_log, bias):
        return -a_log.float().exp() * F.softplus(a.float() + bias)

    @torch.compile(fullgraph=True)
    def beta_transform(b):
        return b.sigmoid()

    def verify(reference, actual):
        torch.testing.assert_close(actual[0], reference[0], rtol=.02, atol=.002)
        torch.testing.assert_close(actual[1], reference[1], rtol=.008, atol=.001)

    def measure(paths):
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side), torch.no_grad():
            for fn in paths.values():
                for _ in range(3):
                    fn()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        with torch.no_grad():
            verify(paths['baseline'](), paths['candidate']())
            graphs, outputs = {}, {}
            for name, fn in paths.items():
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    outputs[name] = fn()
                graphs[name] = graph
            for g in graphs.values():
                for _ in range(10):
                    g.replay()
            torch.cuda.synchronize()
            verify(outputs['baseline'], outputs['candidate'])
            timings = {name: [] for name in paths}
            for sample in range(samples):
                order = list(paths)
                if sample % 2:
                    order.reverse()
                for name in order:
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(replays):
                        graphs[name].replay()
                    end.record()
                    end.synchronize()
                    timings[name].append(start.elapsed_time(end)*1000/replays)
        result = {}
        for name, values in timings.items():
            ordered = sorted(values)
            result[name] = {'median_us': statistics.median(values), 'min_us': min(values),
                            'p95_sample_mean_us': ordered[min(len(ordered)-1, int(.95*len(ordered)))]}
        old, new = result['baseline']['median_us'], result['candidate']['median_us']
        result.update(speedup=old/new, saved_us=old-new, reduction_percent=100*(old-new)/old,
                      correctness='passed within tolerance')
        return result

    results = []
    for batch in (1, 2, 4):
        seq = 2048
        x = torch.randn((batch, seq, 2048), device='cuda', dtype=torch.bfloat16)*.5
        wa = torch.randn((16, 2048), device='cuda', dtype=torch.bfloat16)*.02
        wb = torch.randn_like(wa)*.02
        a_log = torch.randn(16, device='cuda', dtype=torch.bfloat16)*.5
        bias = torch.randn_like(a_log)
        # One-time preparation: explicitly excluded from timing.
        combined_weight = torch.cat((wa, wb), dim=0).contiguous()
        combined = F.linear(x, combined_weight)
        a = combined[..., :16].contiguous()
        b = combined[..., 16:].contiguous()

        transforms = measure({
            'baseline': lambda: (decay_transform(a, a_log, bias), beta_transform(b)),
            'candidate': lambda: unpack_gate(combined, a_log, bias),
        })
        sequence = measure({
            'baseline': lambda: (decay_transform(F.linear(x, wa), a_log, bias),
                                 beta_transform(F.linear(x, wb))),
            'candidate': lambda: unpack_gate(F.linear(x, combined_weight), a_log, bias),
        })
        row = {'batch': batch, 'sequence_length': seq,
               'transforms_only': transforms, 'projections_and_transforms': sequence}
        results.append(row)
        print(json.dumps(row), flush=True)
    return {'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
            'triton': triton.__version__, 'samples': samples, 'replays_per_sample': replays,
            'inputs': 'seeded synthetic BF16 states, weights and learned-parameter vectors',
            'timing': 'CUDA event sample means over graph replays; compilation and weight concat excluded',
            'results': results}


@app.local_entrypoint()
def main(samples: int = 30, replays: int = 100):
    if samples < 2 or replays < 1:
        raise ValueError('samples must be >= 2 and replays >= 1')
    print(json.dumps(bench_remote.remote(samples, replays), indent=2))
