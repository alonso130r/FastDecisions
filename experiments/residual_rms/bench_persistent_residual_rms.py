"""Isolated L4 benchmark: generated two-pass versus persistent residual RMSNorm."""
from pathlib import Path
import json
import sys
import modal

app = modal.App('persistent-residual-rms-benchmark')
image = None
if modal.is_local():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))
    from benchmark import modal_image
    image = modal_image


@app.function(image=image, gpu='L4', timeout=30*60)
def bench_remote(samples: int, replays: int):
    import statistics
    import torch
    import triton
    sys.path.insert(0, '/root/model')
    from decider.residual_rms_baseline import get_two_pass_kernel
    from decider.persistent_residual_rms_kernel import get_persistent_residual_rms_kernel

    torch.manual_seed(17)
    reports = []
    for batch in (1, 2, 4):
        n_rows = batch*2048
        a = torch.randn((batch, 2048, 2048), device='cuda', dtype=torch.bfloat16)
        b = torch.randn_like(a)
        weight = torch.randn(2048, device='cuda', dtype=torch.bfloat16)*.25
        buffers = {name: torch.empty_like(a) for name in ('two_pass', 'persistent')}
        kernels = {'two_pass': get_two_pass_kernel(n_rows),
                   'persistent': get_persistent_residual_rms_kernel(n_rows, weight.dtype)}

        def launch(name):
            kernels[name].run(a, b, weight, buffers[name], n_rows, 2048,
                              stream=torch.cuda.current_stream().cuda_stream)

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for name in kernels:
                for _ in range(3):
                    launch(name)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        torch.testing.assert_close(buffers['persistent'], buffers['two_pass'], rtol=.008, atol=.002)
        graphs = {}
        for name in kernels:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                launch(name)
            graphs[name] = graph
        for graph in graphs.values():
            for _ in range(10):
                graph.replay()
        torch.cuda.synchronize()
        timings = {name: [] for name in kernels}
        for sample in range(samples):
            order = list(kernels)
            if sample % 2:
                order.reverse()
            for name in order:
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(replays):
                    graphs[name].replay()
                end.record()
                end.synchronize()
                timings[name].append(start.elapsed_time(end)*1000/replays)
        report = {'batch': batch, 'shape': list(a.shape), 'correctness': 'passed within tolerance'}
        for name, values in timings.items():
            report[name] = {
                'median_us': statistics.median(values), 'min_us': min(values),
                'p95_sample_mean_us': sorted(values)[min(len(values)-1, int(.95*len(values)))],
                'selected_config': str(kernels[name].launchers[0].config),
            }
        old, new = report['two_pass']['median_us'], report['persistent']['median_us']
        report.update(speedup=old/new, saved_us=old-new, reduction_percent=100*(old-new)/old)
        reports.append(report)
        print(json.dumps(report), flush=True)
    return {'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
            'triton': triton.__version__, 'samples': samples, 'replays_per_sample': replays,
            'timing': 'CUDA events around graph replays; buffers preallocated; warm compilation/autotuning',
            'baseline': 'generated two-pass arithmetic with shape-specific reduction tuning',
            'results': reports}


@app.local_entrypoint()
def main(samples: int = 30, replays: int = 100):
    if samples < 2 or replays < 1:
        raise ValueError('samples must be >= 2, replays >= 1')
    print(json.dumps(bench_remote.remote(samples, replays), indent=2))
