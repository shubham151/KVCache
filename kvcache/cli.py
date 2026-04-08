from .api import EchoAdapter, load_model
from .config import InferenceConfig


def main() -> None:
    config = InferenceConfig()
    engine = load_model(model=EchoAdapter(seed=" demo"), config=config)
    text, metrics = engine.generate("hello", max_new_tokens=8)
    print("Generated:", text)
    print("Metrics:", metrics)


if __name__ == "__main__":
    main()
