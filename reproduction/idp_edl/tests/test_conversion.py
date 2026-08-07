import json
import tempfile
import unittest
from pathlib import Path

from safetensors.torch import load_file
from transformers import T5Config, T5EncoderModel, T5ForConditionalGeneration

from reproduction.idp_edl.convert import convert_checkpoint, expected_encoder_keys


class ProtT5ConversionTest(unittest.TestCase):
    def _make_full_checkpoint(self, directory):
        config = T5Config(
            vocab_size=32,
            d_model=8,
            d_kv=2,
            d_ff=16,
            num_layers=2,
            num_decoder_layers=2,
            num_heads=4,
        )
        source = Path(directory) / "full"
        T5ForConditionalGeneration(config).save_pretrained(source, safe_serialization=False)
        (source / "spiece.model").write_bytes(b"synthetic-tokenizer")
        (source / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")
        return source, config

    def test_encoder_key_filter_is_strict_and_fp16(self):
        with tempfile.TemporaryDirectory() as directory:
            source, config = self._make_full_checkpoint(directory)
            output = Path(directory) / "encoder"
            manifest = convert_checkpoint(source, output)
            self.assertEqual(manifest["encoder_key_check"], "pass")
            self.assertEqual(manifest["required_encoder_key_count"], 20)
            self.assertEqual(manifest["written_encoder_key_count"], 20)
            target = T5EncoderModel.from_pretrained(output, local_files_only=True)
            self.assertEqual(set(target.state_dict()), expected_encoder_keys(config.to_dict()))
            saved_tensors = load_file(str(output / "model.safetensors"), device="cpu")
            self.assertTrue(all(str(tensor.dtype) == "torch.float16" for tensor in saved_tensors.values()))
            self.assertFalse(any(key.startswith("decoder.") for key in target.state_dict()))
            saved_manifest = json.loads((output / "conversion_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(saved_manifest["source_signature"], manifest["source_signature"])
            self.assertTrue(all((output / item["path"]).is_file() for item in manifest["target_files"]))
            second = convert_checkpoint(source, output)
            self.assertEqual(second["source_signature"], manifest["source_signature"])
