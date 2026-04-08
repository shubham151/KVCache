# KVCache

Library for optimizing LLM inference with KV cache management, batching, and backend selection.

## MVP Scope

- **Target models:** decoder-only style models (Llama/Mistral-like usage patterns via adapters)
- **Hardware target:** single GPU first, with abstraction points for multi-GPU later
- **Inference goals:** lower time-to-first-token (TTFT), improve decode throughput, reduce memory overhead

## Core Optimizations

- Efficient KV cache abstraction (simple and paged-style cache managers)
- Continuous batching scheduler abstraction
- Attention backend selection with priority:
  1. `flash-attn`
  2. `xformers`
  3. `torch` fallback

## Minimal API

```python
from kvcache import InferenceConfig, load_model

config = InferenceConfig()
engine = load_model(model=my_adapter, config=config)

text, metrics = engine.generate("hello", max_new_tokens=16)
for token in engine.stream_generate("hello", max_new_tokens=16):
    print(token, end="")
```

## Install

```bash
pip install -e .
```

## Quickstart Demo

```bash
python examples/demo.py
```

## Benchmark Harness

Run the built-in benchmark comparison:

```bash
python -m kvcache.benchmark
```

This reports:
- TTFT (ms)
- Decode latency per token (ms)
- Tokens/second
- Peak VRAM (if torch CUDA is available)

## Architecture Summary

- `kvcache.api`: high-level engine and public API (`load_model`, `generate`, `stream_generate`)
- `kvcache.cache`: KV cache managers (simple and paged-style abstractions)
- `kvcache.scheduler`: request scheduler for continuous batching flow
- `kvcache.attention`: attention backend auto-selection
- `kvcache.benchmark`: baseline vs optimized benchmark harness

## Notes

This repository currently provides a lightweight, framework and abstractions so you can plug in real model backends quickly.
