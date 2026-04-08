from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kvcache import InferenceConfig, load_model
from kvcache.api import EchoAdapter


def main() -> None:
    config = InferenceConfig(cache_policy="paged", enable_continuous_batching=True)
    engine = load_model(model=EchoAdapter(seed=" hi"), config=config)

    print("Streaming output:")
    for token in engine.stream_generate("Hello", max_new_tokens=8):
        print(token, end="")
    print("\n")
    print("Last metrics:", engine.last_metrics)


if __name__ == "__main__":
    main()
