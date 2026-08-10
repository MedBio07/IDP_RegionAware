from __future__ import annotations

import copy
import math
import random
import unittest

import numpy as np
import torch

from scripts.train_sequence_disorder_model import (
    build_optimizer_parameter_groups,
    masked_internal_ordered_pairwise_ranking_loss,
    masked_teacher_logit_huber_loss,
    validate_last_block_options,
    sample_internal_ordered_pairs,
    freeze_shared_backbone_parameters,
    train_epoch,
)
from models.sequence_models import RegionAdapterMoETCN


class P49BTrainingLossTest(unittest.TestCase):
    @staticmethod
    def region_targets(length: int, internal: list[int], terminal: list[int]) -> torch.Tensor:
        targets = torch.zeros((1, length, 4), dtype=torch.float32)
        targets[0, internal, 0] = 1.0
        targets[0, internal, 3] = 1.0
        targets[0, terminal, 0] = 1.0
        targets[0, terminal, 2] = 1.0
        return targets

    def test_pairwise_loss_is_differentiable_segment_balanced_and_excludes_terminal_positive(self) -> None:
        labels = torch.tensor([[0, 1, 1, 0, 1, 0, 0, 1, 0]], dtype=torch.float32)
        known_mask = torch.ones_like(labels)
        auxiliary = self.region_targets(
            length=labels.shape[1],
            internal=[1, 2, 4],
            terminal=[7],
        )
        logits = torch.tensor([[0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, -100.0, 0.0]], requires_grad=True)

        sampled = sample_internal_ordered_pairs(
            labels=labels,
            auxiliary=auxiliary,
            known_mask=known_mask,
            max_per_segment=1,
            rng=random.Random(19),
        )
        self.assertEqual(len(sampled), 2)
        self.assertTrue(all(7 not in negative_indices for _, _, negative_indices in sampled))
        self.assertTrue(all(len(positive_indices) == 1 for _, positive_indices, _ in sampled))
        self.assertTrue(all(len(negative_indices) == 1 for _, _, negative_indices in sampled))

        loss, pair_count, segment_count = masked_internal_ordered_pairwise_ranking_loss(
            logits=logits,
            labels=labels,
            auxiliary=auxiliary,
            known_mask=known_mask,
            margin=1.0,
            max_per_segment=1,
            rng=random.Random(19),
        )
        # The first segment has a zero logit gap and the second a two-logit
        # gap; softplus ranking plus segment averaging gives this value.
        expected = (math.log1p(math.exp(1.0)) + math.log1p(math.exp(-1.0))) / 2.0
        self.assertAlmostEqual(float(loss.detach()), expected, places=6)
        self.assertEqual(pair_count, 2)
        self.assertEqual(segment_count, 2)
        loss.backward()
        self.assertIsNotNone(logits.grad)
        positive_indices = [index for _, indices, _ in sampled for index in indices]
        negative_indices = [index for _, _, indices in sampled for index in indices]
        self.assertTrue(any(float(logits.grad[0, index]) < 0.0 for index in positive_indices))
        self.assertTrue(any(float(logits.grad[0, index]) > 0.0 for index in negative_indices))
        self.assertAlmostEqual(float(logits.grad[0, 7]), 0.0, places=7)

        terminal_changed = logits.detach().clone().requires_grad_(True)
        terminal_changed.data[0, 7] = 100.0
        changed_loss, _, _ = masked_internal_ordered_pairwise_ranking_loss(
            logits=terminal_changed,
            labels=labels,
            auxiliary=auxiliary,
            known_mask=known_mask,
            margin=1.0,
            max_per_segment=1,
            rng=random.Random(19),
        )
        torch.testing.assert_close(changed_loss, loss.detach())

    def test_teacher_huber_protects_ordered_and_terminal_but_not_internal(self) -> None:
        labels = torch.tensor([[0, 1, 1, 0, 1, 0]], dtype=torch.float32)
        known_mask = torch.ones_like(labels)
        auxiliary = self.region_targets(length=6, internal=[1, 2], terminal=[4])
        teacher = torch.zeros_like(labels)
        student = torch.tensor([[2.0, 7.0, -9.0, -2.0, 3.0, 0.5]], requires_grad=True)

        loss = masked_teacher_logit_huber_loss(
            student_logits=student,
            teacher_logits=teacher,
            labels=labels,
            auxiliary=auxiliary,
            known_mask=known_mask,
        )
        # Protected positions are ordered [0, 3, 5] and terminal-positive [4].
        expected = (1.5 + 1.5 + 2.5 + 0.125) / 4.0
        self.assertAlmostEqual(float(loss.detach()), expected, places=6)
        loss.backward()
        self.assertAlmostEqual(float(student.grad[0, 1]), 0.0, places=6)
        self.assertAlmostEqual(float(student.grad[0, 2]), 0.0, places=6)
        self.assertNotEqual(float(student.grad[0, 4]), 0.0)

    def test_zero_main_loss_leaves_only_rank_and_protect_to_drive_optimization(self) -> None:
        class ScalarDisorderModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(0.0))
                self.bias = torch.nn.Parameter(torch.tensor(0.0))

            def forward(self, features: torch.Tensor, pad_mask: torch.Tensor) -> dict[str, torch.Tensor]:
                del pad_mask
                return {"disorder_logits": self.scale * features[..., 0] + self.bias}

        features = np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0],
             [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        labels = np.asarray([0, 1, 1, 0, 0, 1], dtype=np.float32)
        auxiliary = np.zeros((6, 4), dtype=np.float32)
        auxiliary[1:3, [0, 3]] = 1.0
        auxiliary[5, [0, 2]] = 1.0
        example = {
            "id": "toy",
            "sequence": "A" * 6,
            "labels": labels,
            "known_mask": np.ones(6, dtype=np.float32),
            "auxiliary": auxiliary,
            "length": 6,
            "features": features,
        }
        initial_model = ScalarDisorderModel()
        active_model = copy.deepcopy(initial_model)
        inactive_model = copy.deepcopy(initial_model)
        teacher = ScalarDisorderModel()
        with torch.no_grad():
            teacher.bias.fill_(1.0)

        common = {
            "examples": [example],
            "feature_names": [],
            "embedding_dir": None,
            "device": torch.device("cpu"),
            "max_tokens": 1000,
            "max_proteins": 2,
            "disorder_pos_weight": 1.0,
            "aux_pos_weight": np.ones(4, dtype=np.float32),
            "aux_loss_weight": 0.0,
            "loss_type": "bce",
            "focal_gamma": 2.0,
            "asym_gamma_pos": 0.0,
            "asym_gamma_neg": 2.0,
            "gate_loss_weight": 0.0,
            "main_loss_weight": 0.0,
        }
        active_optimizer = torch.optim.SGD(active_model.parameters(), lr=0.1, weight_decay=0.0)
        inactive_optimizer = torch.optim.SGD(inactive_model.parameters(), lr=0.1, weight_decay=0.0)
        active_metrics = train_epoch(
            model=active_model,
            optimizer=active_optimizer,
            rng=random.Random(31),
            pairwise_rank_weight=0.5,
            teacher_protect_weight=0.25,
            pairwise_margin=0.2,
            pairwise_max_per_segment=16,
            teacher=teacher,
            **common,
        )
        inactive_metrics = train_epoch(
            model=inactive_model,
            optimizer=inactive_optimizer,
            rng=random.Random(31),
            pairwise_rank_weight=0.0,
            teacher_protect_weight=0.0,
            pairwise_margin=0.2,
            pairwise_max_per_segment=16,
            teacher=None,
            **common,
        )

        expected_active_loss = (
            0.5 * active_metrics["train_pairwise_rank_loss"]
            + 0.25 * active_metrics["train_teacher_protect_loss"]
        )
        self.assertGreater(active_metrics["train_main_loss"], 0.0)
        self.assertGreater(active_metrics["train_pairwise_pairs"], 0.0)
        self.assertGreater(active_metrics["train_teacher_protect_loss"], 0.0)
        self.assertAlmostEqual(active_metrics["train_loss"], expected_active_loss, places=6)
        self.assertEqual(inactive_metrics["train_loss"], 0.0)
        for before, after in zip(initial_model.parameters(), inactive_model.parameters()):
            torch.testing.assert_close(after, before, rtol=0.0, atol=0.0)
        self.assertTrue(
            any(
                not torch.equal(before, after)
                for before, after in zip(initial_model.parameters(), active_model.parameters())
            )
        )

    def test_zero_weights_match_legacy_train_epoch_behavior(self) -> None:
        torch.manual_seed(23)
        model = RegionAdapterMoETCN(
            input_dim=3,
            hidden_dim=6,
            layers=1,
            dropout=0.0,
            adapter_dim=2,
        )
        legacy_model = copy.deepcopy(model)
        features = np.arange(18, dtype=np.float32).reshape(6, 3) / 10.0
        labels = np.asarray([0, 1, 1, 0, 0, 1], dtype=np.float32)
        auxiliary = np.zeros((6, 4), dtype=np.float32)
        auxiliary[1:3, [0, 3]] = 1.0
        auxiliary[5, [0, 2]] = 1.0
        example = {
            "id": "toy",
            "sequence": "A" * 6,
            "labels": labels,
            "known_mask": np.ones(6, dtype=np.float32),
            "auxiliary": auxiliary,
            "length": 6,
            "features": features,
        }
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
        legacy_optimizer = torch.optim.AdamW(legacy_model.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
        common = {
            "examples": [example],
            "feature_names": [],
            "embedding_dir": None,
            "device": torch.device("cpu"),
            "max_tokens": 1000,
            "max_proteins": 2,
            "disorder_pos_weight": 1.0,
            "aux_pos_weight": np.ones(4, dtype=np.float32),
            "aux_loss_weight": 0.2,
            "loss_type": "bce",
            "focal_gamma": 2.0,
            "asym_gamma_pos": 0.0,
            "asym_gamma_neg": 2.0,
            "gate_loss_weight": 0.5,
        }
        default_metrics = train_epoch(
            model=model,
            optimizer=optimizer,
            rng=random.Random(31),
            **common,
        )
        zero_metrics = train_epoch(
            model=legacy_model,
            optimizer=legacy_optimizer,
            rng=random.Random(31),
            pairwise_rank_weight=0.0,
            teacher_protect_weight=0.0,
            pairwise_margin=0.2,
            pairwise_max_per_segment=16,
            teacher=None,
            main_loss_weight=1.0,
            **common,
        )
        self.assertEqual(default_metrics, zero_metrics)
        for left, right in zip(model.parameters(), legacy_model.parameters()):
            torch.testing.assert_close(left, right)
        self.assertEqual(default_metrics["train_pairwise_pairs"], 0.0)
        self.assertEqual(default_metrics["train_pairwise_segments"], 0.0)
        self.assertEqual(default_metrics["train_pairwise_rank_loss"], 0.0)
        self.assertEqual(default_metrics["train_teacher_protect_loss"], 0.0)

    def test_last_block_unfreeze_has_real_gradient_and_optimizer_boundary(self) -> None:
        torch.manual_seed(41)
        model = RegionAdapterMoETCN(
            input_dim=5,
            hidden_dim=8,
            layers=2,
            dropout=0.0,
            adapter_dim=3,
        )
        freeze_shared_backbone_parameters(model, unfreeze_last_block=True)
        parameter_names = dict(model.named_parameters())
        self.assertTrue(parameter_names["blocks.1.convs.0.weight"].requires_grad)
        self.assertTrue(parameter_names["final_norm.weight"].requires_grad)
        self.assertTrue(parameter_names["region_adapters.0.down.weight"].requires_grad)
        self.assertFalse(parameter_names["blocks.0.convs.0.weight"].requires_grad)
        self.assertFalse(parameter_names["input_projection.weight"].requires_grad)

        parameter_groups = build_optimizer_parameter_groups(
            model=model,
            learning_rate=0.01,
            weight_decay=0.0,
            last_block_learning_rate=0.20,
        )
        self.assertEqual([group["name"] for group in parameter_groups], ["adapter_expert_gate", "last_block_final_norm"])
        self.assertEqual([group["lr"] for group in parameter_groups], [0.01, 0.20])
        group_parameter_ids = [
            {id(parameter) for parameter in group["params"]}
            for group in parameter_groups
        ]
        self.assertIn(id(parameter_names["region_adapters.0.down.weight"]), group_parameter_ids[0])
        self.assertIn(id(parameter_names["blocks.1.convs.0.weight"]), group_parameter_ids[1])
        self.assertIn(id(parameter_names["final_norm.weight"]), group_parameter_ids[1])

        optimizer = torch.optim.SGD(parameter_groups)
        before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        features = torch.randn(1, 12, 5)
        mask = torch.ones(1, 12)
        loss = model(features, mask)["disorder_logits"].square().mean()
        loss.backward()
        self.assertIsNotNone(parameter_names["blocks.1.convs.0.weight"].grad)
        self.assertIsNotNone(parameter_names["final_norm.weight"].grad)
        self.assertIsNone(parameter_names["blocks.0.convs.0.weight"].grad)
        self.assertIsNone(parameter_names["input_projection.weight"].grad)
        optimizer.step()

        self.assertTrue(
            any(
                not torch.equal(before[name], parameter)
                for name, parameter in model.named_parameters()
                if name.startswith(("blocks.1.", "final_norm."))
            )
        )
        for name, parameter in model.named_parameters():
            if name.startswith(("blocks.0.", "input_norm.", "input_projection.")):
                torch.testing.assert_close(parameter, before[name], rtol=0.0, atol=0.0)

    def test_last_block_cli_combinations_are_explicitly_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires --freeze-shared-backbone"):
            validate_last_block_options(False, True, 1.0e-4)
        with self.assertRaisesRegex(ValueError, "requires --unfreeze-last-block"):
            validate_last_block_options(True, False, 1.0e-4)
        with self.assertRaisesRegex(ValueError, "requires --last-block-learning-rate"):
            validate_last_block_options(True, True, None)
        with self.assertRaisesRegex(ValueError, "must be positive"):
            validate_last_block_options(True, True, 0.0)
        validate_last_block_options(True, True, 1.0e-4)
        validate_last_block_options(False, False, None)


if __name__ == "__main__":
    unittest.main()
