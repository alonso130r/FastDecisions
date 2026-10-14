"""Compare original, packed convolution, and packed convolution with combined gates."""
import json
from pathlib import Path
import sys
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App('combined-gates-model-benchmark')
image = None
if modal.is_local():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from benchmark import modal_image
    image = modal_image


@app.function(image=image, gpu='L4', timeout=60 * 60,
              volumes={'/cache': modal.Volume.from_name('decider-hf-cache', create_if_missing=True)})
def bench_remote(repeats: int, revision: str):
    import os
    import statistics
    import time
    import torch
    from huggingface_hub import snapshot_download
    sys.path.insert(0, '/root/model')
    from decider.infer import Decider
    from decider.systemone import render_question
    from decider.engine import fill_ids
    from decider.packed_conv import install_packed_conv
    from decider.combined_gates import install_combined_gates

    os.environ['DECIDER_PACKED_CONV'] = '0'
    path = snapshot_download('Mapika/decider-2b', revision=revision)
    d = Decider(path, device='cuda', use_graphs=True, compile=True)
    row = json.loads(Path('/root/data/decisionbench_profile.jsonl').read_text().splitlines()[0])
    spec = next(iter(row['questions'].values()))
    if spec.get('type') in ('noul', 'bool') and isinstance(spec.get('criteria'), list):
        spec = {**spec, 'criteria': dict(zip(('false', 'true'), spec['criteria']))}
    q = render_question(spec)
    _, items = d._decide_items([(row['state'], [{
        'question': q['question'], 'options': q['options'], '_type': q['type'],
    }])])
    assert len(items[0]['ids']) <= 2048
    ids = fill_ids([items[0]['ids']], 1, 2048, d.m.tok.pad_token_id).cuda()
    engine = d.eng
    reports = {}
    outputs = {}
    for name in ('baseline', 'packed', 'packed_combined_gates'):
        if name != 'baseline':
            count = (install_packed_conv(engine.core) if name == 'packed'
                     else install_combined_gates(engine.core))
            assert count == 18, count
            engine._fwd_impl = torch.compile(engine._fwd_eager, dynamic=False)
            # Old graph embeds old launches, so it cannot be reused after patching.
            engine.graphs.clear()
            print(f'Installed {name} on {count} delta layers', flush=True)
        with torch.no_grad():
            for _ in range(3):
                engine.logits_all(ids)
            torch.cuda.synchronize()
            outputs[name] = engine.logits_all(ids).clone()
            elapsed = []
            for _ in range(repeats):
                torch.cuda.synchronize()
                start = time.perf_counter()
                engine.logits_all(ids)
                torch.cuda.synchronize()
                elapsed.append((time.perf_counter()-start)*1000)
        ordered = sorted(elapsed)
        reports[name] = {'median_forward_ms': statistics.median(elapsed),
                         'p95_forward_ms': ordered[min(len(ordered)-1, int(.95*len(ordered)))]}
        print(name, reports[name], flush=True)
    # Keep tolerance identical across candidates; report rather than hide differences.
    # The new GEMM shape can change projection rounding before the transform.
    probability_differences = {}
    for name in ('packed', 'packed_combined_gates'):
        torch.testing.assert_close(outputs[name], outputs['baseline'], rtol=.02, atol=.05)
        max_prob_diff = 0.0
        for slot, n in zip(items[0]['slots'], items[0]['nopts']):
            a = outputs['baseline'][0, slot, :n].float().softmax(-1)
            b = outputs[name][0, slot, :n].float().softmax(-1)
            torch.testing.assert_close(a, b, rtol=.02, atol=.005)
            assert a.argmax() == b.argmax(), f'{name}: decision changed'
            max_prob_diff = max(max_prob_diff, (a-b).abs().max().item())
        probability_differences[name] = max_prob_diff
    old = reports['packed']['median_forward_ms']
    new = reports['packed_combined_gates']['median_forward_ms']
    return {'model_check': 'passed',
            'max_slot_probability_difference_vs_original': probability_differences,
            'gate_incremental_saved_ms': old-new,
            'gate_incremental_reduction_percent': 100*(old-new)/old,
            'revision': revision, 'resolved_model_path': path,
            'shape': list(ids.shape), 'repeats': repeats,
            'timing_scope': 'graph-backed model forward only; excludes tokenization/readout',
            'results': reports}


@app.local_entrypoint()
def main(repeats: int = 30, revision: str = 'main'):
    if repeats < 2:
        raise ValueError('repeats must be >= 2')
    print(json.dumps(bench_remote.remote(repeats, revision), indent=2))
