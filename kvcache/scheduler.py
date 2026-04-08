from collections import deque
from dataclasses import dataclass
from typing import Deque, Generic, List, TypeVar

T = TypeVar("T")


@dataclass
class ContinuousBatchScheduler(Generic[T]):
    max_batch_size: int

    def __post_init__(self) -> None:
        self._queue: Deque[T] = deque()

    def submit(self, item: T) -> None:
        self._queue.append(item)

    def next_batch(self) -> List[T]:
        batch: List[T] = []
        while self._queue and len(batch) < self.max_batch_size:
            batch.append(self._queue.popleft())
        return batch

    def has_pending(self) -> bool:
        return bool(self._queue)
