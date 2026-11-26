"""CPU regression tests for Pad With Silence."""
import unittest

import torch

from processing_nodes import PadAudio


class TestPadAudio(unittest.TestCase):
    def test_padding_preserves_each_batch_and_channel(self):
        waveform = torch.arange(1, 21, dtype=torch.float32).reshape(2, 2, 5)
        for start, end in ((0.8, 0.4), (0.8, 0.0), (0.0, 0.4), (0.0, 0.0)):
            with self.subTest(start=start, end=end):
                (output,) = PadAudio().pad(
                    {"waveform": waveform, "sample_rate": 10}, start, end
                )
                left, right = int(start * 10), int(end * 10)
                actual = output["waveform"]
                self.assertEqual(output["sample_rate"], 10)
                self.assertEqual(tuple(actual.shape), (2, 2, left + 5 + right))
                torch.testing.assert_close(actual[..., left:left + 5], waveform)
                self.assertEqual(torch.count_nonzero(actual[..., :left]).item(), 0)
                self.assertEqual(torch.count_nonzero(actual[..., left + 5:]).item(), 0)
