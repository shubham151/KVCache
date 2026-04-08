from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Iterable, List

from .api import EchoAdapter, KVCacheEngine, load_model
from .config import InferenceConfig, InferenceMetrics


@dataclass
class BenchmarkResult:
    name: str
    ttft_ms: float
    decode_latency_per_token_ms: float
    tokens_per_second: float
    peak_vram_mb: float


def _summarize(name: str, metrics: List[InferenceMetrics]) -> BenchmarkResult:
    return BenchmarkResult(
        name=name,
        ttft_ms=mean([m.ttft_ms for m in metrics]) if metrics else 0.0,
        decode_latency_per_token_ms=mean([m.decode_latency_per_token_ms for m in metrics]) if metrics else 0.0,
        tokens_per_second=mean([m.tokens_per_second for m in metrics]) if metrics else 0.0,
        peak_vram_mb=mean([m.peak_vram_mb or 0.0 for m in metrics]) if metrics else 0.0,
    )


def run_benchmark(prompts: Iterable[str], max_new_tokens: int = 32) -> List[BenchmarkResult]:
    prompts = list(prompts)
    baseline_cfg = InferenceConfig(cache_policy="simple", enable_continuous_batching=False, attention_backend="torch")
    optimized_cfg = InferenceConfig(cache_policy="paged", enable_continuous_batching=True, attention_backend="auto")

    baseline_engine: KVCacheEngine = load_model(EchoAdapter(seed=" b"), baseline_cfg)
    optimized_engine: KVCacheEngine = load_model(EchoAdapter(seed=" o"), optimized_cfg)

    baseline_metrics: List[InferenceMetrics] = []
    optimized_metrics: List[InferenceMetrics] = []

    for prompt in prompts:
        baseline_engine.generate(prompt, max_new_tokens=max_new_tokens)
        if baseline_engine.last_metrics:
            baseline_metrics.append(baseline_engine.last_metrics)

        optimized_engine.generate(prompt, max_new_tokens=max_new_tokens)
        if optimized_engine.last_metrics:
            optimized_metrics.append(optimized_engine.last_metrics)

    return [
        _summarize("baseline", baseline_metrics),
        _summarize("optimized", optimized_metrics),
    ]


def _print_results(results: List[BenchmarkResult]) -> None:
    print("name,ttft_ms,decode_latency_per_token_ms,tokens_per_second,peak_vram_mb")
    for result in results:
        print(
            f"{result.name},{result.ttft_ms:.3f},{result.decode_latency_per_token_ms:.3f},"
            f"{result.tokens_per_second:.3f},{result.peak_vram_mb:.3f}"
        )


def main() -> None:
    prompts = [
        "Explain KV cache in one sentence.",
        "List three inference optimization ideas.",
        "What is TTFT?",
    ]
    results = run_benchmark(prompts=prompts, max_new_tokens=32)
    _print_results(results)


if __name__ == "__main__":
    main()
