from typing import Optional


def _is_available(module_name: str) -> bool:
    try:
        __import__(module_name)
        return True
    except Exception:
        return False


def select_attention_backend(preferred: str = "auto") -> str:
    candidate = (preferred or "auto").strip().lower()
    if candidate in {"flashattention", "flash-attn", "flash_attn"}:
        return "flash_attn" if _is_available("flash_attn") else "torch"
    if candidate == "xformers":
        return "xformers" if _is_available("xformers") else "torch"
    if candidate == "torch":
        return "torch"

    if _is_available("flash_attn"):
        return "flash_attn"
    if _is_available("xformers"):
        return "xformers"
    return "torch"


def maybe_peak_vram_mb() -> Optional[float]:
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return float(torch.cuda.max_memory_allocated() / (1024 * 1024))
    except Exception:
        return None
    return None
