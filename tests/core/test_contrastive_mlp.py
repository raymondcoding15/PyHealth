import unittest

import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import ContrastiveMLP


class TestContrastiveMLP(unittest.TestCase):
    def setUp(self):
        samples = [
            {
                "patient_id": "p0",
                "visit_id": "v0",
                "sentence_signal": [0.1, 0.2, 0.3, 0.4],
                "meta_signal": [1.0, 0.0],
                "label": 0,
            },
            {
                "patient_id": "p1",
                "visit_id": "v1",
                "sentence_signal": [0.9, 0.7, 0.8, 0.6],
                "meta_signal": [0.0, 1.0],
                "label": 1,
            },
            {
                "patient_id": "p2",
                "visit_id": "v2",
                "sentence_signal": [0.8, 0.8, 0.7, 0.9],
                "meta_signal": [0.1, 0.9],
                "label": 1,
            },
        ]
        self.dataset = create_sample_dataset(
            samples=samples,
            input_schema={"sentence_signal": "tensor", "meta_signal": "tensor"},
            output_schema={"label": "binary"},
            dataset_name="contrastive_mlp_test",
        )
        self.model = ContrastiveMLP(
            dataset=self.dataset,
            hidden_dim=16,
            contrastive_weight=0.2,
            contrastive_margin=0.5,
        )

    def test_initialization(self):
        self.assertEqual(self.model.label_key, "label")
        self.assertEqual(self.model.hidden_dim, 16)
        self.assertIn("sentence_signal", self.model.feature_keys)
        self.assertIn("meta_signal", self.model.feature_keys)

    def test_forward_outputs(self):
        batch = next(iter(get_dataloader(self.dataset, batch_size=3, shuffle=False)))
        with torch.no_grad():
            out = self.model(**batch)
        for key in [
            "loss",
            "task_loss",
            "contrastive_loss",
            "y_prob",
            "y_true",
            "logit",
            "embed",
        ]:
            self.assertIn(key, out)
        self.assertEqual(out["y_prob"].shape[0], 3)
        self.assertEqual(out["embed"].shape[0], 3)
        self.assertEqual(out["loss"].dim(), 0)

    def test_backward_pass(self):
        batch = next(iter(get_dataloader(self.dataset, batch_size=3, shuffle=False)))
        out = self.model(**batch)
        out["loss"].backward()
        self.assertTrue(any(p.grad is not None for p in self.model.parameters() if p.requires_grad))

    def test_ablation_without_contrastive(self):
        model = ContrastiveMLP(dataset=self.dataset, hidden_dim=8, contrastive_weight=0.0)
        batch = next(iter(get_dataloader(self.dataset, batch_size=3, shuffle=False)))
        with torch.no_grad():
            out = model(**batch)
        self.assertGreaterEqual(float(out["loss"]), 0.0)


if __name__ == "__main__":
    unittest.main()
