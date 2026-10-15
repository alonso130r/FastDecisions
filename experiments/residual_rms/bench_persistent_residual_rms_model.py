"""Benchmark graph-backed model forward with/without persistent residual RMSNorm.

Real decision rows, fixed 2048-token bucket, numerical validity checks.
Excludes tokenization/readout; does not enable a default serving patch.
"""
from pathlib import Path
import json
import sys
import modal

app = modal.App('persistent-residual-rms-model-benchmark')
image = None
if modal.is_local():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))
    from benchmark import modal_image
    image = modal_image


@app.function(image=image, gpu='L4', timeout=60*60,
              volumes={'/cache': modal.Volume.from_name('decider-hf-cache', create_if_missing=True)})
def bench_remote(rows: int, batch_size: int, revision: str, repeats: int):
    import os
    import statistics
    import time
    import torch
    from huggingface_hub import snapshot_download
    sys.path.insert(0, '/root/model')
    from decider.infer import Decider
    from decider.systemone import render_question
    from decider.engine import fill_ids
    from decider.temperature import for_items, slot_temperatures
    from decider.persistent_residual_rms import install_persistent_residual_rms

    # Isolate this change against the existing packed-convolution configuration.
    # Combined gates are not installed by the Engine default.
    os.environ['DECIDER_PACKED_CONV'] = '1'
    path = snapshot_download('Mapika/decider-2b', revision=revision)
    d = Decider(path, device='cuda', use_graphs=True, compile=True)
    engine = d.eng
    assert engine.cfg['packed_conv_layers'] == 18
    raw_rows = [json.loads(line) for line in
                Path('/root/data/decisionbench_profile.jsonl').read_text().splitlines() if line.strip()][:rows]
    if len(raw_rows) != rows:
        raise ValueError('Not enough dataset rows')
    requests = []
    for row in raw_rows:
        spec = next(iter(row['questions'].values()))
        if spec.get('type') in ('noul', 'bool') and isinstance(spec.get('criteria'), list):
            spec = {**spec, 'criteria': dict(zip(('false', 'true'), spec['criteria']))}
        q = render_question(spec)
        requests.append((row['state'], [{'question': q['question'],
                                        'options': q['options'], '_type': q['type']}]))
    _, items = d._decide_items(requests)
    if any(len(it['ids']) > 2048 for it in items):
        raise ValueError('A request exceeds the fixed 2048-token validation bucket')
    batches = [items[i:i+batch_size] for i in range(0, len(items), batch_size)]
    ids = [fill_ids([it['ids'] for it in batch], batch_size, 2048,
                    d.m.tok.pad_token_id).cuda() for batch in batches]
    results = {}
    timings = {}
    with torch.no_grad():
        for mode in ('packed_baseline', 'persistent_candidate'):
            if mode == 'persistent_candidate':
                count = install_persistent_residual_rms(engine.core)
                assert count == 18, count
                engine.graphs.clear()
                engine._fwd_impl = torch.compile(engine._fwd_eager, dynamic=False)
            # Graph capture and compilation precede timing.
            for _ in range(3):
                engine.logits_all(ids[0])
            results[mode] = [engine.logits_all(batch_ids).clone() for batch_ids in ids]
            torch.cuda.synchronize()
            elapsed = []
            for repeat in range(repeats):
                batch_ids = ids[repeat % len(ids)]
                torch.cuda.synchronize()
                start = time.perf_counter()
                engine.logits_all(batch_ids)
                torch.cuda.synchronize()
                elapsed.append((time.perf_counter()-start)*1000)
            ordered = sorted(elapsed)
            timings[mode] = {'median_forward_ms': statistics.median(elapsed),
                             'p95_forward_ms': ordered[min(len(ordered)-1, int(.95*len(ordered)))]}
            print(mode, timings[mode], flush=True)

    max_logit_difference = 0.0
    max_probability_difference = 0.0
    slots_checked = 0
    for batch, baseline, candidate in zip(batches, results['packed_baseline'],
                                         results['persistent_candidate']):
        torch.testing.assert_close(candidate, baseline, rtol=.02, atol=.05)
        max_logit_difference = max(max_logit_difference, (candidate-baseline).abs().max().item())
        for row_index, item in enumerate(batch):
            # Apply the same served per-type temperature to both model outputs.
            temperatures = slot_temperatures(for_items(d.T, d.T_by_type, [item]), [item])
            for local_slot, (slot, n) in enumerate(zip(item['slots'], item['nopts'])):
                temperature = (temperatures[local_slot] if isinstance(temperatures, (list, tuple))
                               else temperatures)
                a = (baseline[row_index, slot, :n].float()/temperature).softmax(-1)
                b = (candidate[row_index, slot, :n].float()/temperature).softmax(-1)
                torch.testing.assert_close(b, a, rtol=.02, atol=.005)
                assert a.argmax() == b.argmax(), 'Selected decision changed'
                max_probability_difference = max(max_probability_difference, (a-b).abs().max().item())
                slots_checked += 1
    old = timings['packed_baseline']['median_forward_ms']
    new = timings['persistent_candidate']['median_forward_ms']
    return {'correctness': 'passed within tolerance', 'rows': rows, 'batch_size': batch_size,
            'repeats': repeats, 'results': timings, 'saved_ms': old-new,
            'reduction_percent': 100*(old-new)/old,
            'timing_scope': 'graph-backed model forward; excludes tokenization/readout',
            'ordering': 'baseline then candidate; sequential comparison',
            'shape_per_batch': [batch_size, 2048], 'patched_delta_layers': 18,
            'slots_checked': slots_checked, 'max_logit_difference': max_logit_difference,
            'max_probability_difference': max_probability_difference,
            'revision': revision, 'resolved_model_path': path,
            'logit_tolerance': {'rtol': .02, 'atol': .05},
            'probability_tolerance': {'rtol': .02, 'atol': .005}}


@app.local_entrypoint()
def main(rows: int = 4, batch_size: int = 1,
         revision: str = 'ff99089b6810691bd79134ec6febb5d9afa6e619', repeats: int = 30):
    if repeats < 2:
        raise ValueError('repeats must be >= 2')
    if rows < 1 or batch_size < 1 or rows % batch_size:
        raise ValueError('rows must be positive and divisible by positive batch_size')
    print(json.dumps(bench_remote.remote(rows, batch_size, revision, repeats), indent=2))
