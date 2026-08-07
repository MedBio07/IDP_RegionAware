import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from reproduction.idp_edl.official import (
    ModelLoadError,
    _load_parameter_file,
    freeze_components_for_meta,
    validate_runtime_dtype,
)


class _DummyMeta(nn.Module):
    def __init__(self):
        super().__init__()
        self.idp_model = nn.Linear(2, 2)
        self.sdr_model = nn.Linear(2, 2)
        self.ldr_model = nn.Linear(2, 2)
        self.classifier = nn.Linear(6, 2)


class MetaFreezeInvariantTest(unittest.TestCase):
    def test_fp16_runtime_is_rejected_for_official_forward(self):
        with self.assertRaises(ModelLoadError):
            validate_runtime_dtype(torch.float16)
        self.assertEqual(validate_runtime_dtype(None), torch.float32)

    def test_meta_checkpoint_loads_after_components_are_frozen(self):
        meta = _DummyMeta()
        freeze_components_for_meta(meta)
        self.assertTrue(all(not parameter.requires_grad for parameter in meta.idp_model.parameters()))
        self.assertTrue(all(not parameter.requires_grad for parameter in meta.sdr_model.parameters()))
        self.assertTrue(all(not parameter.requires_grad for parameter in meta.ldr_model.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in meta.classifier.parameters()))
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "meta_predictor.pth"
            torch.save(
                {name: value.detach().clone() for name, value in meta.classifier.named_parameters()},
                checkpoint,
            )
            # The official checkpoint names include the classifier prefix.
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            torch.save({"classifier." + name: value for name, value in state.items()}, checkpoint)
            report = _load_parameter_file(meta, checkpoint)
        self.assertEqual(report["loaded_keys"], 2)
        self.assertEqual(report["unexpected_keys"], 0)
