import unittest

from kvcache import InferenceConfig, load_model
from kvcache.api import EchoAdapter


class APITestCase(unittest.TestCase):
    def test_generate_returns_metrics(self) -> None:
        engine = load_model(EchoAdapter(seed=" x"), InferenceConfig())
        text, metrics = engine.generate("hello", max_new_tokens=4)
        self.assertEqual(text, " x x x x")
        self.assertGreaterEqual(metrics.tokens_per_second, 0.0)
        self.assertGreaterEqual(metrics.ttft_ms, 0.0)

    def test_stream_generate_emits_tokens(self) -> None:
        engine = load_model(EchoAdapter(seed=" y"), InferenceConfig())
        output = list(engine.stream_generate("hello", max_new_tokens=3))
        self.assertEqual(output, [" y", " y", " y"])


if __name__ == "__main__":
    unittest.main()
