from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Generator, List, Optional, Protocol, Tuple
from uuid import uuid4

from .attention import maybe_peak_vram_mb, select_attention_backend
from .cache import KVCacheManager, create_cache_manager
from .config import InferenceConfig, InferenceMetrics
from .scheduler import ContinuousBatchScheduler


class GenerationAdapter(Protocol):
    def generate_next(self, prompt: str, generated_tokens: List[str], max_new_tokens: int) -> Optional[str]:
        """Return next token chunk or None when generation is complete."""


@dataclass
class EchoAdapter:
    seed: str = " token"

    def generate_next(self, prompt: str, generated_tokens: List[str], max_new_tokens: int) -> Optional[str]:
        if len(generated_tokens) >= max_new_tokens:
            return None
        return self.seed


class KVCacheEngine:
    def __init__(self, model: GenerationAdapter, config: Optional[InferenceConfig] = None) -> None:
        self.model = model
        self.config = config or InferenceConfig()
        self.cache: KVCacheManager = create_cache_manager(self.config.cache_policy)
        self.scheduler = ContinuousBatchScheduler[str](max_batch_size=self.config.max_batch_size)
        self.attention_backend = select_attention_backend(self.config.attention_backend)
        self._last_metrics: Optional[InferenceMetrics] = None

    @property
    def last_metrics(self) -> Optional[InferenceMetrics]:
        return self._last_metrics

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        stop_tokens: Optional[List[str]] = None,
    ) -> Tuple[str, InferenceMetrics]:
        generated: List[str] = []
        for token in self.stream_generate(prompt, max_new_tokens=max_new_tokens, stop_tokens=stop_tokens):
            generated.append(token)
        return "".join(generated), self._last_metrics or InferenceMetrics(0.0, 0.0, 0.0, None)

    def stream_generate(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        stop_tokens: Optional[List[str]] = None,
    ) -> Generator[str, None, None]:
        stop_tokens = stop_tokens or []
        request_id = str(uuid4())
        ttft: Optional[float] = None
        started = perf_counter()
        emitted = 0

        try:
            if self.config.enable_continuous_batching:
                self.scheduler.submit(request_id)
                _ = self.scheduler.next_batch()

            generated_tokens: List[str] = []
            while emitted < max_new_tokens:
                token = self.model.generate_next(prompt, generated_tokens, max_new_tokens)
                if token is None:
                    break
                generated_tokens.append(token)
                self.cache.append(request_id, token)
                emitted += 1
                if ttft is None:
                    ttft = (perf_counter() - started) * 1000.0
                yield token
                if token in stop_tokens:
                    break
        finally:
            elapsed = perf_counter() - started
            decode_time = max((elapsed * 1000.0) - (ttft or 0.0), 0.0) if emitted > 1 else 0.0
            self._last_metrics = InferenceMetrics(
                ttft_ms=ttft or 0.0,
                decode_latency_per_token_ms=(decode_time / (emitted - 1)) if emitted > 1 else 0.0,
                tokens_per_second=(emitted / elapsed) if elapsed > 0 else 0.0,
                peak_vram_mb=maybe_peak_vram_mb(),
            )
            self.cache.clear(request_id)


def load_model(model: Optional[GenerationAdapter] = None, config: Optional[InferenceConfig] = None) -> KVCacheEngine:
    adapter = model or EchoAdapter()
    return KVCacheEngine(model=adapter, config=config)
