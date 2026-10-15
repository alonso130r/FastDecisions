"""Model-wide native GQA benchmark. Numerical differences reported without assertions."""
from pathlib import Path
import json
import sys
import modal

app = modal.App('native-gqa-model-benchmark')
image = None
if modal.is_local():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))
    from benchmark import modal_image
    image = modal_image


@app.function(image=image, gpu='L4', timeout=3600,
              volumes={'/cache': modal.Volume.from_name('decider-hf-cache', create_if_missing=True)})
def bench_remote(rows: int, batch_size: int, revision: str, samples: int, replays: int,
                 original_baseline: bool = False):
    import os
    import random
    import statistics
    import time
    from types import MethodType
    import torch
    from huggingface_hub import snapshot_download
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention, apply_rotary_pos_emb
    sys.path.insert(0, '/root/model')
    from decider.infer import Decider
    from decider.systemone import render_question
    from decider.engine import fill_ids
    from decider.temperature import for_items, slot_temperatures
    from decider.native_gqa_kernel import native_gqa_attention

    @torch.library.custom_op('fastdecisions_test::native_gqa', mutates_args=(), device_types='cuda')
    def attention_op(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return native_gqa_attention(q, k, v)

    @attention_op.register_fake
    def fake(q, k, v):
        b, h, t, d = q.shape
        return q.new_empty((b, t, h, d)).permute(0, 2, 1, 3)

    def candidate_forward(self, hidden_states, position_embeddings, attention_mask,
                          past_key_values=None, **kwargs):
        if past_key_values is not None or self.training or torch.is_grad_enabled():
            raise ValueError('This test patch only supports uncached inference')
        shape = hidden_states.shape[:-1]
        heads_shape = (*shape, -1, self.head_dim)
        q, gate = torch.chunk(self.q_proj(hidden_states).view(*shape, -1, self.head_dim*2), 2, dim=-1)
        gate = gate.reshape(*shape, -1)
        q = self.q_norm(q.reshape(heads_shape)).transpose(1, 2)
        k = self.k_norm(self.k_proj(hidden_states).view(heads_shape)).transpose(1, 2)
        v = self.v_proj(hidden_states).view(heads_shape).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, *position_embeddings)
        output = attention_op(q, k, v).transpose(1, 2).reshape(*shape, -1)
        return self.o_proj(output*torch.sigmoid(gate)), None

    os.environ['DECIDER_PACKED_CONV'] = '0' if original_baseline else '1'
    path = snapshot_download('Mapika/decider-2b', revision=revision)
    d = Decider(path, device='cuda', use_graphs=True, compile=True)
    engine = d.eng
    modules = [m for m in engine.core.modules() if isinstance(m, Qwen3_5Attention)]
    assert len(modules) == 6, len(modules)
    for m in modules:
        assert m.head_dim == 256 and m.num_key_value_groups == 4 and m.scaling == 256**-.5
    raw = [json.loads(line) for line in Path('/root/data/decisionbench_profile.jsonl').read_text().splitlines()
           if line.strip()][:rows]
    if len(raw) != rows:
        raise ValueError('Not enough dataset rows')
    requests = []
    for row in raw:
        spec = next(iter(row['questions'].values()))
        if spec.get('type') in ('noul', 'bool') and isinstance(spec.get('criteria'), list):
            spec = {**spec, 'criteria': dict(zip(('false', 'true'), spec['criteria']))}
        q = render_question(spec)
        requests.append((row['state'], [{'question': q['question'], 'options': q['options'], '_type': q['type']}]))
    _, items = d._decide_items(requests)
    if any(len(item['ids']) > 2048 for item in items):
        raise ValueError('Request exceeds the 2048-token bucket')
    batches = [items[i:i+batch_size] for i in range(0, len(items), batch_size)]
    ids = [fill_ids([it['ids'] for it in batch], batch_size, 2048, d.m.tok.pad_token_id).cuda()
           for batch in batches]

    # Validate actual masks in eager execution before ignoring them in the test patch.
    masks_checked = 0
    def check_mask(module, args, kwargs):
        nonlocal masks_checked
        mask = kwargs.get('attention_mask', args[2] if len(args) > 2 else None)
        if mask is not None:
            t = kwargs.get('hidden_states', args[0] if args else None).shape[1]
            assert mask.shape[-2:] == (t, t), mask.shape
            allowed = torch.ones((t, t), device=mask.device, dtype=torch.bool).tril()
            if mask.dtype == torch.bool:
                assert torch.equal(mask, allowed.expand_as(mask)), 'Noncausal or segment mask unsupported'
            else:
                assert torch.all(mask[..., allowed] == 0).item(), 'Additional attention bias unsupported'
                assert torch.all(mask[..., ~allowed] < -1e4).item(), 'Future positions not fully masked'
        masks_checked += 1

    with torch.no_grad():
        handles = [m.register_forward_pre_hook(check_mask, with_kwargs=True) for m in modules]
        try:
            for batch_ids in ids:
                engine._fwd_eager(batch_ids)
        finally:
            for handle in handles:
                handle.remove()
        baseline = [engine.logits_all(x).clone() for x in ids]
        baseline_graphs = dict(engine.graphs)
        if original_baseline:
            from decider.packed_conv import install_packed_conv
            count = install_packed_conv(engine.core)
            assert count == 18, count
        for module in modules:
            module.forward = MethodType(candidate_forward, module)
        engine.graphs.clear()
        engine.pool = torch.cuda.graph_pool_handle()
        engine._fwd_impl = torch.compile(engine._fwd_eager, dynamic=False)
        candidate = [engine.logits_all(x).clone() for x in ids]
        # A second pass changes the captured inputs when multiple batches exist.
        replay = [engine.logits_all(x).clone() for x in reversed(ids)]
        replay_max_error = max((got-expected).abs().max().item()
                               for got, expected in zip(replay, reversed(candidate)))

    max_logit_error = max_probability_error = 0.0
    slots_checked = 0
    changed_decisions = 0
    logits_outside_tolerance = 0
    for batch, old, new in zip(batches, baseline, candidate):
        logits_outside_tolerance += int(((new-old).abs() > .05+.02*old.abs()).sum().item())
        max_logit_error = max(max_logit_error, (new-old).abs().max().item())
        for row_index, item in enumerate(batch):
            temperatures = slot_temperatures(for_items(d.T, d.T_by_type, [item]), [item])
            for i, (slot, n) in enumerate(zip(item['slots'], item['nopts'])):
                temperature = temperatures[i] if isinstance(temperatures, (list, tuple)) else temperatures
                a = (old[row_index, slot, :n]/temperature).softmax(-1)
                b = (new[row_index, slot, :n]/temperature).softmax(-1)
                changed_decisions += int(a.argmax().item() != b.argmax().item())
                max_probability_error = max(max_probability_error, (a-b).abs().max().item())
                slots_checked += 1
    assert slots_checked > 0
    candidate_graphs = dict(engine.graphs)
    modes = {'baseline': baseline_graphs[(batch_size, 2048)],
             'native_gqa': candidate_graphs[(batch_size, 2048)]}
    timings = {name: [] for name in modes}
    wall = {name: [] for name in modes}
    rng = random.Random(17)
    # Both captured graphs stay alive. Interleave to reduce clock/order bias.
    for _, _, graph in modes.values():
        for _ in range(10):
            graph.replay()
    torch.cuda.synchronize()
    for sample in range(samples):
        batch_ids = ids[sample % len(ids)]
        order = list(modes)
        rng.shuffle(order)
        for name in order:
            static_ids, _, graph = modes[name]
            static_ids.copy_(batch_ids)
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(replays):
                graph.replay()
            end.record()
            end.synchronize()
            timings[name].append(start.elapsed_time(end)*1000/replays)
            # Separate measurement includes the serving input copy and host launch.
            torch.cuda.synchronize()
            before = time.perf_counter()
            static_ids.copy_(batch_ids)
            graph.replay()
            torch.cuda.synchronize()
            wall[name].append((time.perf_counter()-before)*1000)
    summaries = {}
    for name, values in timings.items():
        summaries[name] = dict(median_us=statistics.median(values), min_us=min(values),
            p95_sample_mean_us=sorted(values)[min(len(values)-1, int(.95*len(values)))],
            median_wall_forward_ms=statistics.median(wall[name]))
    old, new = summaries['baseline']['median_us'], summaries['native_gqa']['median_us']
    return dict(status='benchmarked; numerical assertions disabled', rows=rows, batch_size=batch_size,
                baseline_configuration='original convolution + SDPA' if original_baseline else 'packed convolution + SDPA',
                candidate_configuration='packed convolution + native GQA',
                changed_decisions=changed_decisions, logits_outside_tolerance=logits_outside_tolerance,
                replay_max_error=replay_max_error,
                results=summaries, saved_us=old-new, speedup=old/new,
                samples=samples, replays_per_sample=replays, gpu=torch.cuda.get_device_name(),
                torch=torch.__version__,
                timing_scope='CUDA events around graph replays; excludes input copy and readout; warm compilation',
                wall_scope='input copy plus graph launch and synchronization; excludes tokenization/readout',
                revision=revision, patched_attention_layers=len(modules), masks_checked=masks_checked,
                slots_checked=slots_checked, max_logit_error=max_logit_error,
                max_probability_error=max_probability_error,
                logit_tolerance=dict(atol=.05, rtol=.02), probability_tolerance=dict(atol=.005, rtol=.02))


@app.local_entrypoint()
def main(rows: int = 4, batch_size: int = 1,
         revision: str = 'ff99089b6810691bd79134ec6febb5d9afa6e619',
         samples: int = 30, replays: int = 10, output: str = 'artifacts/native-gqa-model.json',
         original_baseline: bool = False):
    if rows < 1 or batch_size < 1 or rows % batch_size:
        raise ValueError('Rows must be positive and divisible by batch size')
    if samples < 2 or replays < 1:
        raise ValueError('Require samples >= 2 and replays >= 1')
    result = bench_remote.remote(rows, batch_size, revision, samples, replays, original_baseline)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
