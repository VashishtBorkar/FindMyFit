"""Check projector variants and compatibility with saved checkpoints."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from findmyfit.embeddings.metric_projector import MetricProjector
from findmyfit.models.metric import FashionCompatibilityModel


class MetricArchitectureTests(unittest.TestCase):
    def test_default_layout_preserves_original_weights(self):
        original = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 128))
        model = FashionCompatibilityModel(512)
        model.projector.load_state_dict(original.state_dict())
        inputs = torch.randn(3, 512)
        expected = torch.nn.functional.normalize(original(inputs), dim=1)
        torch.testing.assert_close(model(inputs), expected)

    def test_variants_have_normalized_outputs_and_finite_gradients(self):
        for kwargs in (
            {"hidden_dims": [512]},
            {"hidden_dims": [1024]},
            {"hidden_dims": [256] * 5},
            {"hidden_dims": [256], "residual_blocks": 2},
            {"hidden_dims": [512, 256]},
            {"hidden_dims": [512, 256], "dropout": 0.15, "layer_norm": True},
        ):
            with self.subTest(kwargs=kwargs):
                model = FashionCompatibilityModel(512, **kwargs)
                output = model(torch.randn(3, 512))
                self.assertEqual(output.shape, (3, 128))
                torch.testing.assert_close(output.norm(dim=1), torch.ones(3))
                output[:, 0].sum().backward()
                self.assertTrue(all(
                    parameter.grad is not None and torch.isfinite(parameter.grad).all()
                    for parameter in model.parameters()
                ))

    def test_plain_and_residual_have_matched_capacity_and_initial_weights(self):
        torch.manual_seed(42)
        plain = FashionCompatibilityModel(512, hidden_dims=[256] * 5)
        torch.manual_seed(42)
        residual = FashionCompatibilityModel(512, hidden_dims=[256], residual_blocks=2)
        self.assertEqual(sum(p.numel() for p in plain.parameters()), 427392)
        for left, right in zip(plain.parameters(), residual.parameters(), strict=True):
            torch.testing.assert_close(left, right)

    def test_runtime_loads_legacy_and_deep_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            for name, kwargs in (
                ("legacy", {}),
                ("residual", {"hidden_dims": [256], "residual_blocks": 2}),
                ("deep", {"hidden_dims": [512, 256], "dropout": 0.15, "layer_norm": True}),
            ):
                with self.subTest(name=name):
                    model = FashionCompatibilityModel(512, **kwargs).eval()
                    checkpoint = {
                        "model_state_dict": model.state_dict(),
                        "hidden_dim": 256, "output_dim": 128,
                    }
                    if kwargs:
                        checkpoint["model_kwargs"] = {"embedding_dim": 512, **kwargs}
                    path = Path(directory) / f"{name}.pt"
                    torch.save(checkpoint, path)
                    projector = MetricProjector(path, device="cpu")
                    inputs = np.random.default_rng(42).normal(size=(3, 512)).astype(np.float32)
                    with torch.no_grad():
                        expected = model(torch.from_numpy(inputs)).numpy()
                    np.testing.assert_allclose(projector.project_batch(inputs), expected, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
