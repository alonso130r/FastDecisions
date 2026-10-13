"""Compare packed outputs against generated-style convolution plus copies on L4."""
from pathlib import Path
import json
import sys

import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App('packed-output-conv-benchmark')
image = None
if modal.is_local():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from benchmark import modal_image
    image = modal_image.add_local_dir(str(ROOT / 'src' / 'kernels'),
                                      remote_path='/root/src/kernels')


@app.function(image=image, gpu='L4', timeout=30 * 60)
def bench_remote(samples: int, replays: int):
    import statistics
    import torch
    import triton
    sys.path.insert(0, '/root/src')
    from kernels.packed_output_conv import get_packed_conv_kernel
    from kernels.original_layout_conv import get_original_conv_kernel

    torch.manual_seed(17)
    results = []
    for batch in (1, 2, 4):
        seq = 2048
        x = torch.randn((batch, seq, 6144), device='cuda', dtype=torch.bfloat16) * 0.5
        w = torch.randn((6144, 4), device='cuda', dtype=torch.bfloat16) * 0.25
        original_buffer = torch.empty((batch, 6144, seq), device='cuda', dtype=x.dtype)
        views = tuple(t.transpose(1, 2).reshape(batch, seq, 16, 128)
                      for t in original_buffer.split(2048, dim=1))
        baseline_out = tuple(torch.empty((batch, seq, 16, 128), device='cuda', dtype=x.dtype)
                             for _ in range(3))
        packed_out = tuple(torch.empty_like(t) for t in baseline_out)
        original = get_original_conv_kernel(batch, seq)
        packed = get_packed_conv_kernel(batch, seq)

        def baseline():
            original.run(x, w, original_buffer, batch * seq, 6144,
                         stream=torch.cuda.current_stream().cuda_stream)
            for dst, src in zip(baseline_out, views):
                dst.copy_(src)

        def candidate():
            packed.run(x, w, *packed_out, batch * seq, 6144,
                       stream=torch.cuda.current_stream().cuda_stream)

        # Warm compilation/autotuning before correctness checks and capture.
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                baseline()
                candidate()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        for got, expected in zip(packed_out, baseline_out):
            torch.testing.assert_close(got, expected, rtol=0.008, atol=0.002)
        graphs = {}
        for name, fn in (('original_plus_copies', baseline), ('packed', candidate)):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                fn()
            graphs[name] = graph
        for graph in graphs.values():
            for _ in range(10):
                graph.replay()
        torch.cuda.synchronize()

        timings = {name: [] for name in graphs}
        # Alternate ordering to reduce systematic clock/temperature bias.
        for sample in range(samples):
            order = list(graphs)
            if sample % 2:
                order.reverse()
            for name in order:
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(replays):
                    graphs[name].replay()
                end.record()
                end.synchronize()
                timings[name].append(start.elapsed_time(end) * 1000 / replays)
        row = {'batch': batch, 'sequence_length': seq, 'correctness': 'passed'}
        for name, values in timings.items():
            values_sorted = sorted(values)
            row[name] = {
                'median_us': statistics.median(values),
                'min_us': min(values),
                'p95_sample_mean_us': values_sorted[min(len(values)-1, int(0.95*len(values)))],
            }
        old = row['original_plus_copies']['median_us']
        new = row['packed']['median_us']
        row.update(speedup=old/new, saved_us=old-new, reduction_percent=100*(old-new)/old)
        results.append(row)
        print(json.dumps(row), flush=True)
    return {
        'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
        'triton': triton.__version__, 'samples': samples, 'replays_per_sample': replays,
        'timing': 'CUDA events around repeated single-sequence graph replays; preallocated buffers',
        'baseline': 'generated arithmetic/channel-major stores with matching shape hints, then three copies',
        'results': results,
    }


@app.local_entrypoint()
def main(samples: int = 30, replays: int = 100):
    if samples < 2 or replays < 1:
        raise ValueError('samples must be >= 2 and replays >= 1')
    print(json.dumps(bench_remote.remote(samples, replays), indent=2))
