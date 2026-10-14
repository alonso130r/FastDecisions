"""Slot-only projection versus all-position projection; packed convolution and native GQA on both sides."""
from pathlib import Path
import json
import sys
import modal

app = modal.App('slot-projection-benchmark')
image = None
if modal.is_local():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from benchmark import modal_image
    image = modal_image


@app.function(image=image, gpu='L4', timeout=3600,
              volumes={'/cache': modal.Volume.from_name('decider-hf-cache', create_if_missing=True)})
def bench_remote(rows: int, batch_size: int, revision: str, samples: int, replays: int,
                 slot_capacity: int = 0):
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

    os.environ['DECIDER_PACKED_CONV'] = '1'
    path = snapshot_download('Mapika/decider-2b', revision=revision)
    d = Decider(path, device='cuda', use_graphs=True, compile=True)
    engine = d.eng
    assert engine.cfg['packed_conv_layers'] == 18, engine.cfg
    from decider.slot_projection import SlotProjectionForward
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

    counts = [sum(len(item['slots']) for item in batch) for batch in batches]
    capacity = slot_capacity or max(counts)
    if capacity < max(counts) or capacity < 1:
        raise ValueError('slot_capacity must cover every batch and be positive')
    slot_inputs = []
    for batch in batches:
        rows_flat, positions = [], []
        for row_index, item in enumerate(batch):
            if len(item['slots']) != len(item['nopts']):
                raise ValueError('Slot and option counts differ')
            if any(slot < 0 or slot >= len(item['ids']) for slot in item['slots']):
                raise ValueError('Slot index outside actual request')
            rows_flat.extend([row_index]*len(item['slots']))
            positions.extend(item['slots'])
        padding = capacity-len(positions)
        slot_inputs.append((torch.tensor(rows_flat+[0]*padding, device='cuda', dtype=torch.long),
                            torch.tensor(positions+[0]*padding, device='cuda', dtype=torch.long)))

    with torch.no_grad():
        # Check mask semantics before applying the same GQA patch to both modes.
        handles = [m.register_forward_pre_hook(check_mask, with_kwargs=True) for m in modules]
        try:
            for batch_ids in ids:
                engine._fwd_eager(batch_ids)
        finally:
            for handle in handles:
                handle.remove()
        for module in modules:
            module.forward = MethodType(candidate_forward, module)
        slot_forward = SlotProjectionForward(engine.core, engine.W).eval()
        def all_positions(batch_ids, slot_rows, slot_positions):
            return engine._fwd_eager(batch_ids)
        functions = {'all_positions': torch.compile(all_positions, dynamic=False),
                     'slot_only': torch.compile(slot_forward, dynamic=False)}
        graphs, static_inputs, outputs = {}, {}, {}
        for name, fn in functions.items():
            static = (ids[0].clone(), slot_inputs[0][0].clone(), slot_inputs[0][1].clone())
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    fn(*static)
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            # Separate private pools keep both graph outputs/buffers live.
            with torch.cuda.graph(graph):
                outputs[name] = fn(*static)
            graphs[name], static_inputs[name] = graph, static

        def feed(name, i):
            for destination, source in zip(static_inputs[name], (ids[i], *slot_inputs[i])):
                destination.copy_(source)

        max_logit_error = max_probability_error = 0.0
        logits_outside_tolerance = changed_decisions = slots_checked = 0
        stored = []
        for i, batch in enumerate(batches):
            for name in graphs:
                feed(name, i)
                graphs[name].replay()
            old = outputs['all_positions'][slot_inputs[i][0][:counts[i]], slot_inputs[i][1][:counts[i]]].clone()
            new = outputs['slot_only'][:counts[i]].clone()
            if not torch.isfinite(old).all().item() or not torch.isfinite(new).all().item():
                raise ValueError('Nonfinite decision logits')
            diff = (new-old).abs()
            max_logit_error = max(max_logit_error, diff.max().item())
            logits_outside_tolerance += int((diff > .05+.02*old.abs()).sum().item())
            stored.append(new)
            flat = 0
            for item in batch:
                temperatures = slot_temperatures(for_items(d.T, d.T_by_type, [item]), [item])
                for j, n in enumerate(item['nopts']):
                    temperature = temperatures[j] if isinstance(temperatures, (list, tuple)) else temperatures
                    a = (old[flat, :n]/temperature).softmax(-1)
                    b = (new[flat, :n]/temperature).softmax(-1)
                    max_probability_error = max(max_probability_error, (a-b).abs().max().item())
                    changed_decisions += int(a.argmax().item() != b.argmax().item())
                    flat += 1
                    slots_checked += 1
        replay_max_error = 0.0
        for i in reversed(range(len(batches))):
            feed('slot_only', i)
            graphs['slot_only'].replay()
            replay_max_error = max(replay_max_error,
                (outputs['slot_only'][:counts[i]]-stored[i]).abs().max().item())

        timings = {name: [] for name in graphs}
        wall = {name: [] for name in graphs}
        rng = random.Random(17)
        for name, graph in graphs.items():
            for _ in range(10):
                graph.replay()
        torch.cuda.synchronize()
        for sample in range(samples):
            i = sample % len(batches)
            order = list(graphs)
            rng.shuffle(order)
            for name in order:
                feed(name, i)
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(replays):
                    graphs[name].replay()
                end.record()
                end.synchronize()
                timings[name].append(start.elapsed_time(end)*1000/replays)
                torch.cuda.synchronize()
                before = time.perf_counter()
                feed(name, i)
                graphs[name].replay()
                if name == 'all_positions':
                    selected = outputs[name][static_inputs[name][1], static_inputs[name][2]]
                else:
                    selected = outputs[name]
                torch.cuda.synchronize()
                wall[name].append((time.perf_counter()-before)*1000)

    summaries = {}
    for name, values in timings.items():
        summaries[name] = dict(median_us=statistics.median(values), min_us=min(values),
            p95_sample_mean_us=sorted(values)[min(len(values)-1, int(.95*len(values)))],
            median_wall_forward_and_gather_ms=statistics.median(wall[name]))
    old, new = summaries['all_positions']['median_us'], summaries['slot_only']['median_us']
    return dict(status='benchmarked; numerical differences reported', results=summaries,
                saved_us=old-new, speedup=old/new, reduction_percent=100*(old-new)/old,
                common_configuration='packed convolution on 18 delta layers + native BF16 GQA on 6 full-attention layers',
                rows=rows, batch_size=batch_size, sequence_length=2048,
                slot_capacity=capacity, valid_slots_per_batch=counts, slots_checked=slots_checked,
                changed_decisions=changed_decisions, max_logit_error=max_logit_error,
                max_probability_error=max_probability_error, logits_outside_tolerance=logits_outside_tolerance,
                replay_max_error=replay_max_error, masks_checked=masks_checked,
                samples=samples, replays_per_sample=replays, gpu=torch.cuda.get_device_name(),
                torch=torch.__version__, revision=revision,
                gpu_scope='model forward: all-position projection versus hidden gather + slot projection; input copies excluded',
                wall_scope='copies of ids/slot buffers + forward + baseline output gather + synchronization; excludes softmax, CPU readout and tokenization',
                logit_tolerance=dict(atol=.05, rtol=.02))


@app.local_entrypoint()
def main(rows: int = 4, batch_size: int = 1,
         revision: str = 'ff99089b6810691bd79134ec6febb5d9afa6e619',
         samples: int = 30, replays: int = 10, slot_capacity: int = 0,
         output: str = 'data/slot-projection-model.json'):
    if rows < 1 or batch_size < 1 or rows % batch_size:
        raise ValueError('Rows must be positive and divisible by batch size')
    if samples < 2 or replays < 1 or slot_capacity < 0:
        raise ValueError('Require samples >= 2, replays >= 1, slot_capacity >= 0')
    result = bench_remote.remote(rows, batch_size, revision, samples, replays, slot_capacity)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
