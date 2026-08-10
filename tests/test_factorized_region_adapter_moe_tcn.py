import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from models.sequence_models import FactorizedRegionAdapterMoETCN, RegionAdapterMoETCN
from scripts.predict_sequence_disorder_model import build_model as build_prediction_model
from scripts.train_sequence_disorder_model import (
    build_model as build_training_model,
    canonical_model_type,
    freeze_shared_backbone_parameters,
    initialize_model_from_checkpoint,
    train_epoch,
)


class FactorizedRegionAdapterMoETCNTest(unittest.TestCase):
    def make_model(self) -> FactorizedRegionAdapterMoETCN:
        return FactorizedRegionAdapterMoETCN(
            input_dim=5,
            hidden_dim=8,
            layers=1,
            dropout=0.0,
            adapter_dim=3,
            gate_temperature=0.7,
        )

    def test_factorized_gate_outputs_are_normalized_and_match_delta(self) -> None:
        torch.manual_seed(3)
        model = self.make_model().eval()
        features = torch.randn(2, 9, 5)
        mask = torch.ones(2, 9)
        output = model(features, mask)

        length_weights = output["length_gate_weights"]
        location_weights = output["location_gate_weights"]
        gate_weights = output["gate_weights"]
        self.assertEqual(tuple(gate_weights.shape), (2, 9, 4))
        torch.testing.assert_close(length_weights.sum(dim=-1), torch.ones(2, 9))
        torch.testing.assert_close(location_weights.sum(dim=-1), torch.ones(2, 9))
        torch.testing.assert_close(gate_weights.sum(dim=-1), torch.ones(2, 9))
        torch.testing.assert_close(gate_weights[..., :2].sum(dim=-1), torch.full((2, 9), 0.5))
        torch.testing.assert_close(gate_weights[..., 2:].sum(dim=-1), torch.full((2, 9), 0.5))
        torch.testing.assert_close(
            gate_weights,
            torch.cat((0.5 * length_weights, 0.5 * location_weights), dim=-1),
        )
        expected_delta = torch.sum(output["expert_logits"] * gate_weights, dim=-1)
        torch.testing.assert_close(
            output["disorder_logits"] - output["generic_logits"],
            expected_delta,
        )

    def test_legacy_region_adapter_gate_rows_initialize_factorized_gates(self) -> None:
        torch.manual_seed(5)
        legacy = RegionAdapterMoETCN(
            input_dim=5,
            hidden_dim=8,
            layers=1,
            dropout=0.0,
            adapter_dim=3,
            gate_temperature=0.7,
        )
        with torch.no_grad():
            legacy.gate_head.weight.copy_(torch.arange(32, dtype=torch.float32).reshape(4, 8))
            legacy.gate_head.bias.copy_(torch.tensor([0.1, 0.2, 0.3, 0.4]))
        target = self.make_model()

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "legacy.pt"
            torch.save({"model_state_dict": legacy.state_dict()}, checkpoint_path)
            loaded_keys = initialize_model_from_checkpoint(target, checkpoint_path, torch.device("cpu"))

        torch.testing.assert_close(target.length_gate_head.weight, legacy.gate_head.weight[:2])
        torch.testing.assert_close(target.location_gate_head.weight, legacy.gate_head.weight[2:])
        torch.testing.assert_close(target.length_gate_head.bias, legacy.gate_head.bias[:2])
        torch.testing.assert_close(target.location_gate_head.bias, legacy.gate_head.bias[2:])
        self.assertIn("length_gate_head.weight", loaded_keys)
        self.assertIn("location_gate_head.weight", loaded_keys)
        self.assertIn("length_gate_head.bias", loaded_keys)
        self.assertIn("location_gate_head.bias", loaded_keys)

    def test_freezing_keeps_only_adapters_experts_and_factorized_gates_trainable(self) -> None:
        model = self.make_model()
        freeze_shared_backbone_parameters(model)
        for name, parameter in model.named_parameters():
            expected = name.startswith(
                ("region_adapters.", "expert_heads.", "length_gate_head.", "location_gate_head.")
            )
            self.assertEqual(parameter.requires_grad, expected, name)
        self.assertTrue(any(parameter.requires_grad for parameter in model.length_gate_head.parameters()))
        self.assertTrue(any(parameter.requires_grad for parameter in model.location_gate_head.parameters()))
        self.assertFalse(any(parameter.requires_grad for parameter in model.input_projection.parameters()))

    def test_cli_reconstruction_and_real_loss_backward(self) -> None:
        self.assertEqual(canonical_model_type("factorized_region_adapter_moe_tcn"), "FactorizedRegionAdapterMoETCN")
        training_model = build_training_model(
            "FactorizedRegionAdapterMoETCN",
            input_dim=5,
            hidden_dim=8,
            layers=1,
            dropout=0.0,
            adapter_dim=3,
            gate_temperature=0.7,
        )
        prediction_model = build_prediction_model(
            {
                "model_type": "FactorizedRegionAdapterMoETCN",
                "input_dim": 5,
                "hidden_dim": 8,
                "layers": 1,
                "dropout": 0.0,
                "adapter_dim": 3,
                "gate_temperature": 0.7,
            }
        )
        self.assertIsInstance(training_model, FactorizedRegionAdapterMoETCN)
        self.assertIsInstance(prediction_model, FactorizedRegionAdapterMoETCN)
        prediction_model.load_state_dict(training_model.state_dict())

        features = np.random.default_rng(7).normal(size=(8, 5)).astype(np.float32)
        labels = np.asarray([0, 1, 1, 0, 0, 1, 0, 0], dtype=np.float32)
        known_mask = np.ones(8, dtype=np.float32)
        auxiliary = np.zeros((8, 4), dtype=np.float32)
        auxiliary[1:3, 0] = 1.0
        auxiliary[1:3, 2] = 1.0
        auxiliary[5, 0] = 1.0
        auxiliary[5, 3] = 1.0
        examples = [
            {
                "id": "toy",
                "sequence": "A" * 8,
                "labels": labels,
                "known_mask": known_mask,
                "auxiliary": auxiliary,
                "length": 8,
                "features": features,
            }
        ]
        optimizer = torch.optim.AdamW(training_model.parameters(), lr=1.0e-3)
        metrics = train_epoch(
            model=training_model,
            examples=examples,
            feature_names=[],
            embedding_dir=None,
            optimizer=optimizer,
            device=torch.device("cpu"),
            max_tokens=1000,
            max_proteins=2,
            disorder_pos_weight=1.0,
            aux_pos_weight=np.ones(4, dtype=np.float32),
            aux_loss_weight=0.2,
            loss_type="bce",
            focal_gamma=2.0,
            asym_gamma_pos=0.0,
            asym_gamma_neg=2.0,
            gate_loss_weight=0.5,
            rng=random.Random(11),
        )
        self.assertTrue(np.isfinite(metrics["train_loss"]))
        self.assertIsNotNone(training_model.length_gate_head.weight.grad)
        self.assertIsNotNone(training_model.location_gate_head.weight.grad)
        self.assertTrue(torch.isfinite(training_model.length_gate_head.weight.grad).all())
        self.assertTrue(torch.isfinite(training_model.location_gate_head.weight.grad).all())


if __name__ == "__main__":
    unittest.main()
