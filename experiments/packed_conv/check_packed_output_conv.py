"""Modal GPU correctness checks for the packed-output convolution prototype.

Run: modal run experiments/packed_conv/check_packed_output_conv.py
This checks correctness only; it does not report performance.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]


def run_checks():
    import unittest
    import torch
    from kernels.packed_output_conv import prep_qkv

    class PackedConvChecks(unittest.TestCase):
        def setUp(self):
            torch.manual_seed(17)

        def reference(self, x, weight):
            # Independent indexing in token-major layout. Accumulate in FP64
            # to avoid sharing the candidate's compiler/FMA implementation.
            w = weight.reshape(6144, 4).double()
            xf = x.double()
            acc = torch.zeros_like(xf)
            for tap in range(4):
                lag = 3 - tap
                acc[:, lag:, :] += xf[:, :2048-lag, :] * w[:, tap]
            expected = (acc * torch.sigmoid(acc)).to(torch.bfloat16)
            return tuple(t.contiguous().reshape(x.shape[0], 2048, 16, 128)
                         for t in expected.split(2048, dim=-1))

        def compare(self, actual, expected):
            self.assertEqual(len(actual), 3)
            for name, got, want in zip(('q', 'k', 'v'), actual, expected):
                self.assertEqual(got.shape, want.shape)
                self.assertEqual(got.dtype, torch.bfloat16)
                self.assertEqual(got.device, want.device)
                self.assertTrue(got.is_contiguous(), name)
                self.assertTrue(torch.isfinite(got).all().item(), name)
                # FP32/FMA/exp differences can cross a BF16 rounding boundary.
                # Allow roughly one BF16 rounding step, plus a small absolute
                # tolerance near zero. This is not a bitwise-equivalence claim.
                torch.testing.assert_close(got, want, rtol=0.008, atol=0.002,
                                           msg=lambda msg: f'{name}: {msg}')

        def test_random_batches_and_weight_shapes(self):
            for batch in (1, 2, 4):
                for weight_shape in ((6144, 4), (6144, 1, 4)):
                    with self.subTest(batch=batch, weight_shape=weight_shape):
                        x = torch.randn((batch, 2048, 6144), device='cuda',
                                        dtype=torch.bfloat16) * 0.5
                        w = torch.randn(weight_shape, device='cuda',
                                        dtype=torch.bfloat16) * 0.25
                        got = prep_qkv(x, w)
                        self.compare(got, self.reference(x, w))

        def test_each_tap_and_batch_boundaries(self):
            # Distinct batches and Q/K/V regions reveal cross-batch reads and
            # incorrect output-region offsets. Exercise all four causal taps.
            x = torch.empty((2, 2048, 6144), device='cuda', dtype=torch.bfloat16)
            for batch in range(2):
                for region in range(3):
                    x[batch, :, region*2048:(region+1)*2048] = (batch*3+region+1)/8
            x[0, -3:, :] = 3  # Must never leak into the next batch's first tokens.
            for tap in range(4):
                with self.subTest(tap=tap):
                    w = torch.zeros((6144, 4), device='cuda', dtype=torch.bfloat16)
                    w[:, tap] = 1
                    self.compare(prep_qkv(x, w), self.reference(x, w))

        def test_zero_input(self):
            x = torch.zeros((1, 2048, 6144), device='cuda', dtype=torch.bfloat16)
            w = torch.randn((6144, 4), device='cuda', dtype=torch.bfloat16)
            for got in prep_qkv(x, w):
                self.assertEqual(torch.count_nonzero(got).item(), 0)

        def test_cuda_graph_replay_with_changed_input(self):
            x = torch.randn((1, 2048, 6144), device='cuda', dtype=torch.bfloat16) * 0.5
            w = torch.randn((6144, 4), device='cuda', dtype=torch.bfloat16) * 0.25
            # Complete compilation/autotuning on a side stream before capture.
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    prep_qkv(x, w)
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                outputs = prep_qkv(x, w)
            x.copy_(torch.randn_like(x) * 0.5)
            graph.replay()
            self.compare(outputs, self.reference(x, w))

    print(f'GPU: {torch.cuda.get_device_name()}; torch: {torch.__version__}', flush=True)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(PackedConvChecks)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise RuntimeError('Packed-output convolution correctness checks failed')
    return {'tests_run': result.testsRun, 'passed': True}


import modal
app = modal.App('packed-output-conv-correctness')
# Image construction is local. The remote module must not import benchmark,
# whose image recipe also references files that only exist on the host.
image = None
if modal.is_local():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))
    from benchmark import modal_image

    image = modal_image.add_local_dir(
        str(ROOT / 'experiments' / 'packed_conv' / 'kernels'), remote_path='/root/src/kernels'
    )


@app.function(image=image, gpu='L4', timeout=30 * 60)
def check_remote():
    sys.path.insert(0, '/root/src')
    return run_checks()


@app.local_entrypoint()
def main():
    print(check_remote.remote())
