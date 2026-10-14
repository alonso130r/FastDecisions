"""Offline native-GQA tile sweep on L4. Does not modify production defaults."""
from pathlib import Path
import json
import sys
import modal

app = modal.App('native-gqa-offline-sweep')
image = None
if modal.is_local():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from benchmark import modal_image
    image = modal_image


@app.function(image=image, gpu='L4', timeout=3600)
def sweep_remote(samples: int, replays: int, batches: list[int], sequence_length: int):
    import itertools
    import random
    import statistics
    import torch
    import triton
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel
    sys.path.insert(0, '/root/model')
    from decider.native_gqa_kernel import native_gqa

    torch.manual_seed(17)
    rng = random.Random(17)
    reports = []
    for batch in batches:
        t = sequence_length
        # Match the compiled model: Q and V token-major, K head-major.
        q = torch.randn((batch, t, 8, 256), device='cuda', dtype=torch.bfloat16).permute(0, 2, 1, 3)
        k = torch.randn((batch, 2, t, 256), device='cuda', dtype=torch.bfloat16)
        v = torch.randn((batch, t, 2, 256), device='cuda', dtype=torch.bfloat16).permute(0, 2, 1, 3)
        with sdpa_kernel(SDPBackend.MATH):
            reference = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), is_causal=True, enable_gqa=True)
        output = torch.empty((batch, t, 8, 256), device='cuda', dtype=torch.bfloat16).permute(0, 2, 1, 3)
        records, graphs = [], []
        for bm, bn, warps, stages in itertools.product((16, 32, 64), (32, 64), (4, 8), (1, 2)):
            record = dict(block_m=bm, block_n=bn, num_warps=warps, num_stages=stages)
            try:
                def launch():
                    native_gqa[(triton.cdiv(t, bm), batch*8)](
                        q, k, v, output, *q.stride(), *k.stride(), *v.stride(), *output.stride(),
                        SEQ_LEN=t, N_Q_HEADS=8, N_KV_HEADS=2, HEAD_DIM=256, SCALE=256**-.5,
                        BLOCK_M=bm, BLOCK_N=bn, num_warps=warps, num_stages=stages)
                launch()
                torch.cuda.synchronize()
                difference = (output.float()-reference).abs()
                record['max_abs_error'] = difference.max().item()
                record['rms_error'] = difference.square().mean().sqrt().item()
                torch.testing.assert_close(output.float(), reference, atol=.02, rtol=.02)
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    for _ in range(3):
                        launch()
                torch.cuda.current_stream().wait_stream(side)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    launch()
                for _ in range(5):
                    graph.replay()
                torch.cuda.synchronize()
                record['status'] = 'passed'
                record['sample_means_us'] = []
                graphs.append((record, graph))
            except (triton.OutOfResources, triton.CompilationError, AssertionError) as exc:
                record.update(status='rejected', error=str(exc)[:1500])
            records.append(record)
            print(json.dumps({'batch': batch, 'sequence_length': t, 'configuration': record}), flush=True)
        for _ in range(samples):
            order = list(graphs)
            rng.shuffle(order)
            for record, graph in order:
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(replays):
                    graph.replay()
                end.record()
                end.synchronize()
                record['sample_means_us'].append(start.elapsed_time(end)*1000/replays)
        for record, _ in graphs:
            values = record['sample_means_us']
            record.update(median_us=statistics.median(values), min_us=min(values),
                          p95_sample_mean_us=sorted(values)[min(len(values)-1, int(.95*len(values)))])
        ranked = sorted((r for r in records if r['status']=='passed'), key=lambda r: r['median_us'])
        report = dict(batch=batch, sequence_length=t, strides=dict(q=list(q.stride()), k=list(k.stride()), v=list(v.stride())),
                      winner=ranked[0] if ranked else None, ranked=ranked,
                      rejected=[r for r in records if r['status']!='passed'])
        reports.append(report)
        print(json.dumps({'batch': batch, 'winner': report['winner']}), flush=True)
    return dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, triton=triton.__version__,
                samples=samples, replays_per_sample=replays, seed=17,
                scope='isolated causal attention; synthetic inputs; existing compiled layouts; preallocated output',
                reference='FP32 PyTorch math SDPA with native GQA; atol=rtol=0.02',
                timing='CUDA events around graph replays, configurations randomly interleaved; compilation excluded', results=reports)


@app.local_entrypoint()
def main(samples: int = 15, replays: int = 20, batches: str = '1,2,4', sequence_length: int = 2048,
         output: str = 'data/native-gqa-sweep.json'):
    batch_list = [int(x) for x in batches.split(',')]
    if samples < 2 or replays < 1 or sequence_length < 1 or not batch_list or min(batch_list) < 1:
        raise ValueError('Require samples >= 2 and positive replays, batches and sequence length')
    result = sweep_remote.remote(samples, replays, batch_list, sequence_length)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    print(f'Saved {path}')
