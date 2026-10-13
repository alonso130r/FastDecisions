"""Modal proof of custom-op capture, full-model equivalence, and graph latency."""
import json
from pathlib import Path
import sys
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App('packed-conv-model-check')
image = None
if modal.is_local():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from benchmark import modal_image
    image = modal_image


@app.function(image=image, gpu='L4', timeout=60 * 60,
              volumes={'/cache': modal.Volume.from_name('decider-hf-cache', create_if_missing=True)})
def check_remote(repeats: int, revision: str):
    import os
    import statistics
    import time
    import torch
    from huggingface_hub import snapshot_download
    sys.path.insert(0, '/root/model')
    from decider.infer import Decider
    from decider.systemone import render_question
    from decider.engine import fill_ids
    from decider.packed_conv import packed_conv, install_packed_conv

    # fullgraph=True fails if our custom-op boundary introduces a graph break.
    x = torch.randn((1, 2048, 6144), device='cuda', dtype=torch.bfloat16) * 0.5
    w = torch.randn((6144, 1, 4), device='cuda', dtype=torch.bfloat16) * 0.25
    with torch.no_grad():
        torch.library.opcheck(packed_conv, (x, w))
        eager = packed_conv(x, w)
        compiled_op = torch.compile(packed_conv, fullgraph=True)
        compiled = compiled_op(x, w)
        for a, b in zip(eager, compiled):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    del x, w, eager, compiled
    print('Custom op registration and fullgraph capture passed', flush=True)

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
    for name in ('baseline', 'packed'):
        if name == 'packed':
            count = install_packed_conv(engine.core)
            assert count == 18, count
            engine._fwd_impl = torch.compile(engine._fwd_eager, dynamic=False)
            # Old graph embeds old launches, so it cannot be reused after patching.
            engine.graphs.clear()
            print(f'Installed packed preparation on {count} delta layers', flush=True)
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
    # Check all logits, plus probabilities at the real decision slots.
    torch.testing.assert_close(outputs['packed'], outputs['baseline'], rtol=.02, atol=.05)
    slots = items[0]['slots']
    nopts = items[0]['nopts']
    max_prob_diff = 0.0
    for slot, n in zip(slots, nopts):
        a = outputs['baseline'][0, slot, :n].float().softmax(-1)
        b = outputs['packed'][0, slot, :n].float().softmax(-1)
        torch.testing.assert_close(a, b, rtol=.02, atol=.005)
        assert a.argmax() == b.argmax()
        max_prob_diff = max(max_prob_diff, (a-b).abs().max().item())
    return {'custom_op_fullgraph': 'passed', 'model_check': 'passed',
            'max_slot_probability_difference': max_prob_diff, 'revision': revision,
            'resolved_model_path': path, 'shape': list(ids.shape), 'repeats': repeats,
            'timing_scope': 'graph-backed model forward only; excludes tokenization/readout',
            'results': reports}


@app.local_entrypoint()
def main(repeats: int = 30, revision: str = 'main'):
    if repeats < 2:
        raise ValueError('repeats must be >= 2')
    print(json.dumps(check_remote.remote(repeats, revision), indent=2))
