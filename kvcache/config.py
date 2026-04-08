from dataclasses import dataclass
from typing import Optional


@dataclass
class InferenceConfig:
    precision: str = "fp16"
    max_batch_size: int = 8
    cache_policy: str = "paged"
    attention_backend: str = "auto"
    device: str = "cuda"
    enable_continuous_batching: bool = True


@dataclass
class InferenceMetrics:
    ttft_ms: float
    decode_latency_per_token_ms: float
    tokens_per_second: float
    peak_vram_mb: Optional[float] = None
