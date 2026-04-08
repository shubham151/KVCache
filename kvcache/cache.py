from dataclasses import dataclass, field
from typing import Dict, List


class KVCacheManager:
    def append(self, request_id: str, token: str) -> None:
        raise NotImplementedError

    def get(self, request_id: str) -> List[str]:
        raise NotImplementedError

    def clear(self, request_id: str) -> None:
        raise NotImplementedError


@dataclass
class SimpleKVCache(KVCacheManager):
    _store: Dict[str, List[str]] = field(default_factory=dict)

    def append(self, request_id: str, token: str) -> None:
        self._store.setdefault(request_id, []).append(token)

    def get(self, request_id: str) -> List[str]:
        return self._store.get(request_id, [])

    def clear(self, request_id: str) -> None:
        self._store.pop(request_id, None)


@dataclass
class PagedKVCache(KVCacheManager):
    block_size: int = 16
    _store: Dict[str, List[List[str]]] = field(default_factory=dict)

    def append(self, request_id: str, token: str) -> None:
        blocks = self._store.setdefault(request_id, [[]])
        if len(blocks[-1]) >= self.block_size:
            blocks.append([])
        blocks[-1].append(token)

    def get(self, request_id: str) -> List[str]:
        blocks = self._store.get(request_id, [])
        return [token for block in blocks for token in block]

    def clear(self, request_id: str) -> None:
        self._store.pop(request_id, None)


def create_cache_manager(policy: str) -> KVCacheManager:
    normalized = (policy or "").strip().lower()
    if normalized == "simple":
        return SimpleKVCache()
    if normalized in {"paged", "efficient"}:
        return PagedKVCache()
    return SimpleKVCache()
