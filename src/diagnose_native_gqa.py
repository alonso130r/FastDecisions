"""First-layer native GQA diagnostics on real model inputs."""
from pathlib import Path
import json
import sys
import modal

app = modal.App('native-gqa-diagnostics')
image = None
if modal.is_local():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from benchmark import modal_image
    image = modal_image


@app.function(image=image, gpu='L4', timeout=3600,
              volumes={'/cache': modal.Volume.from_name('decider-hf-cache', create_if_missing=True)})
def diagnose_remote():
    import torch
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from huggingface_hub import snapshot_download
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention, apply_rotary_pos_emb, eager_attention_forward
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    sys.path.insert(0, '/root/model')
    from decider.infer import Decider
    from decider.systemone import render_question
    from decider.engine import fill_ids
    from decider.native_gqa_kernel import native_gqa_attention

    path = snapshot_download('Mapika/decider-2b', revision='ff99089b6810691bd79134ec6febb5d9afa6e619')
    d = Decider(path, device='cuda', use_graphs=True, compile=False)
    row = json.loads(Path('/root/data/decisionbench_profile.jsonl').read_text().splitlines()[0])
    spec = next(iter(row['questions'].values()))
    if spec.get('type') in ('noul', 'bool') and isinstance(spec.get('criteria'), list):
        spec = {**spec, 'criteria': dict(zip(('false', 'true'), spec['criteria']))}
    question = render_question(spec)
    _, items = d._decide_items([(row['state'], [{'question': question['question'],
        'options': question['options'], '_type': question['type']}])])
    ids = fill_ids([items[0]['ids']], 1, 2048, d.m.tok.pad_token_id).cuda()
    module = next(m for m in d.eng.core.modules() if isinstance(m, Qwen3_5Attention))
    captured = {}
    class Captured(Exception):
        pass
    def hook(m, args, kwargs):
        captured.update(kwargs)
        if args:
            captured['hidden_states'] = args[0]
        raise Captured()
    handle = module.register_forward_pre_hook(hook, with_kwargs=True)
    with torch.no_grad():
        try:
            d.eng._fwd_eager(ids)
        except Captured:
            pass
        finally:
            handle.remove()
    hidden = captured['hidden_states']
    embeddings = captured['position_embeddings']
    mask = captured['attention_mask']
    interface = ALL_ATTENTION_FUNCTIONS.get_interface(module.config._attn_implementation, eager_attention_forward)
    report = dict(layer=module.layer_idx, backend=module.config._attn_implementation,
                  hidden_shape=list(hidden.shape), comparisons={})
    def compare(name, a, b, atol=.02, rtol=.02):
        a, b = a.float(), b.float()
        diff = (a-b).abs()
        report['comparisons'][name] = dict(max_abs=diff.max().item(), rms=diff.square().mean().sqrt().item(),
            mismatched=int((diff > atol+rtol*b.abs()).sum().item()), elements=a.numel(),
            finite=bool(torch.isfinite(a).all().item() and torch.isfinite(b).all().item()))
        print(name, report['comparisons'][name], flush=True)
    def prepare(h, cos, sin):
        shape = h.shape[:-1]
        hshape = (*shape, -1, module.head_dim)
        q, gate = torch.chunk(module.q_proj(h).view(*shape, -1, module.head_dim*2), 2, -1)
        q = module.q_norm(q.reshape(hshape)).transpose(1, 2)
        k = module.k_norm(module.k_proj(h).view(hshape)).transpose(1, 2)
        v = module.v_proj(h).view(hshape).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        return q, k, v, gate.reshape(*shape, -1)
    def existing(q, k, v):
        return interface(module, q, k, v, mask, dropout=0., scaling=module.scaling)[0].transpose(1, 2)

    @torch.library.custom_op('gqa_diagnostic::native', mutates_args=(), device_types='cuda')
    def native_op(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return native_gqa_attention(q, k, v)
    @native_op.register_fake
    def fake(q, k, v):
        b, h, t, dim = q.shape
        return q.new_empty((b, t, h, dim)).permute(0, 2, 1, 3)
    @torch.library.custom_op('gqa_diagnostic::existing', mutates_args=(), device_types='cuda')
    def existing_op(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        result = interface(module, q, k, v, m, dropout=0., scaling=module.scaling)[0]
        return result.contiguous().transpose(1, 2)
    existing_op.register_fake(fake_existing := lambda q, k, v, m: fake(q, k, v))

    def layer(h, cos, sin, mode):
        q, k, v, gate = prepare(h, cos, sin)
        if mode == 'original':
            out = existing(q, k, v)
        elif mode == 'boundary_only':
            out = existing_op(q, k, v, mask)
        else:
            out = native_op(q, k, v)
        out = out.transpose(1, 2).reshape(*h.shape[:-1], -1)
        return module.o_proj(out*torch.sigmoid(gate))

    with torch.no_grad():
        q, k, v, gate = prepare(hidden, *embeddings)
        report['strides'] = dict(q=list(q.stride()), k=list(k.stride()), v=list(v.stride()))
        if mask is not None:
            allowed = torch.ones((2048, 2048), device='cuda', dtype=torch.bool).tril()
            semantic_mask = mask if mask.dtype == torch.bool else mask == 0
            report['ordinary_causal_mask'] = torch.equal(semantic_mask, allowed.expand_as(semantic_mask))
            assert report['ordinary_causal_mask']
        baseline = existing(q, k, v)
        native = native_gqa_attention(q, k, v)
        with sdpa_kernel(SDPBackend.MATH):
            reference = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), is_causal=True, enable_gqa=True)
        compare('attention_native_vs_existing', native, baseline)
        compare('attention_native_vs_fp32', native, reference)
        compare('attention_existing_vs_fp32', baseline, reference)
        compiled_prepare = torch.compile(prepare, dynamic=False)
        cq, ck, cv, cg = compiled_prepare(hidden, *embeddings)
        for name, a, b in zip(('q', 'k', 'v', 'gate'), (cq, ck, cv, cg), (q, k, v, gate)):
            compare('compiled_preparation_'+name, a, b)
        compare('compiled_inputs_native_vs_existing', native_gqa_attention(cq, ck, cv), existing(cq, ck, cv))
        original_eager = module(hidden, embeddings, mask)[0]
        compare('wrapper_existing_eager_vs_original', layer(hidden, *embeddings, 'original'), original_eager, .05, .02)
        compare('layer_native_eager_vs_original', layer(hidden, *embeddings, 'native'), original_eager, .05, .02)
        compiled_results = {}
        for mode in ('original', 'boundary_only', 'native'):
            fn = torch.compile(lambda h, cos, sin: layer(h, cos, sin, mode), dynamic=False)
            compiled_results[mode] = fn(hidden, *embeddings).clone()
            torch.cuda.synchronize()
            compare('layer_compiled_'+mode+'_vs_eager_original', compiled_results[mode], original_eager, .05, .02)
        compare('layer_compiled_boundary_vs_compiled_original', compiled_results['boundary_only'], compiled_results['original'], .05, .02)
        compare('layer_compiled_native_vs_compiled_original', compiled_results['native'], compiled_results['original'], .05, .02)
    return report


@app.local_entrypoint()
def main():
    result = diagnose_remote.remote()
    Path('data/native-gqa-diagnostics.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
