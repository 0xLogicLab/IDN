"""Thread-safe exact-response cache for the IDN prototype."""

from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from threading import RLock
from typing import Optional, Protocol

from idn.framework.types import InferenceRequest, InferenceResponse, ResourceBudget


class ResponseCache(Protocol):
    def get(self, key: str) -> Optional[InferenceResponse]:
        ...

    def put(self, key: str, response: InferenceResponse) -> None:
        ...


class InMemoryResponseCache:
    """Bounded LRU with TTL; only successful exact matches are stored."""

    def __init__(self, capacity: int = 1024, ttl_seconds: float = 300.0) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        self.capacity = int(capacity)
        self.ttl_seconds = float(ttl_seconds)
        self._items: OrderedDict[str, tuple[float, InferenceResponse]] = OrderedDict()
        self._lock = RLock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[InferenceResponse]:
        now = time.monotonic()
        with self._lock:
            item = self._items.get(key)
            if item is None:
                self.misses += 1
                return None
            expires_at, response = item
            if expires_at <= now:
                del self._items[key]
                self.misses += 1
                return None
            self._items.move_to_end(key)
            self.hits += 1
            return response

    def put(self, key: str, response: InferenceResponse) -> None:
        with self._lock:
            self._items[key] = (time.monotonic() + self.ttl_seconds, response)
            self._items.move_to_end(key)
            while len(self._items) > self.capacity:
                self._items.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._items)


def response_cache_key(
    request: InferenceRequest,
    budget: ResourceBudget,
    *,
    deployment_version: str,
) -> str:
    """Hash all inputs that can change the selected package or its output."""

    payload = {
        "deployment_version": deployment_version,
        "tenant_id": request.tenant_id,
        "messages": [
            {"role": message.role, "content": message.content}
            for message in request.messages
        ],
        "max_new_tokens": request.max_new_tokens,
        "budget": {
            "max_memory_gib": budget.max_memory_gib,
            "max_latency_ms": budget.max_latency_ms,
            "min_quality_lower_bound": budget.min_quality_lower_bound,
        },
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
