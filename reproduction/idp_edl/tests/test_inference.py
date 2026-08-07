import unittest
from types import SimpleNamespace

import torch

from reproduction.idp_edl.data import LabeledExample
from reproduction.idp_edl.inference import predict_batch


class _BatchTokenizer:
    def __call__(self, texts, **kwargs):
        lengths = [len(text.split()) + 1 for text in texts]
        width = max(lengths)
        attention_mask = torch.zeros((len(texts), width), dtype=torch.long)
        input_ids = torch.zeros((len(texts), width), dtype=torch.long)
        for index, length in enumerate(lengths):
            attention_mask[index, :length] = 1
            input_ids[index, :length] = torch.arange(1, length + 1)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class _BatchModel:
    def __init__(self):
        self.calls = 0
        self.last_shape = None

    def __call__(self, input_ids, attention_mask):
        self.calls += 1
        self.last_shape = tuple(input_ids.shape)
        positive = input_ids.to(torch.float32) / 10.0
        logits = torch.stack((torch.zeros_like(positive), positive), dim=-1)
        return SimpleNamespace(logits=logits)


class BatchedInferenceTest(unittest.TestCase):
    def test_padded_batch_is_one_forward_and_cropped_per_protein(self):
        examples = [
            LabeledExample("test", "a", "a", "a", "AC", "01", "sdr"),
            LabeledExample("test", "b", "b", "b", "ACDE", "0101", "ldr"),
        ]
        model = _BatchModel()

        results = predict_batch(model, _BatchTokenizer(), torch.device("cpu"), examples)

        self.assertEqual(model.calls, 1)
        self.assertEqual(model.last_shape, (2, 5))
        self.assertEqual([len(rows) for rows, _ in results], [2, 4])
        self.assertEqual([row["position"] for row in results[1][0]], [1, 2, 3, 4])
        self.assertEqual([row["label"] for row in results[0][0]], ["0", "1"])


if __name__ == "__main__":
    unittest.main()
