"""
llm_cache.py — In-memory cache for LLM analysis results.

Prevents duplicate API calls when the same content is analyzed
multiple times (e.g., retried webhooks, identical PRs, bot spam).

Cache entries are keyed by a SHA-256 hash of the normalised input
so structurally identical content always hits the same bucket.

Usage
-----
from cache.llm_cache import LLMCache

cache = LLMCache(max_size=512, ttl_seconds=3600)

key   = cache.make_key(content, extra="reviewer")
entry = cache.get(key)
if entry is None:
    result = await expensive_llm_call(content)
    cache.set(key, result)
else:
    result = entry
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections import OrderedDict
from typing import Any, Optional

logger = logging.getLogger(__name__)


class LLMCache:
    """Thread-safe LRU in-memory cache for LLM responses.

    Parameters
    ----------
    max_size:
        Maximum number of entries to keep in memory.
        Oldest entries are evicted first (LRU).
    ttl_seconds:
        How long (in seconds) a cached entry remains valid.
        Set to 0 or None to disable TTL expiry.
    """

    def __init__(self, max_size: int = 512, ttl_seconds: Optional[float] = 3600) -> None:
        self._max_size = max_size
        self._ttl = ttl_seconds
        # OrderedDict preserves insertion order → easy LRU eviction
        self._store: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def make_key(*parts: str, **kwargs: str) -> str:
        """Return a deterministic SHA-256 hex digest for the given inputs.

        Pass any combination of positional strings (e.g. the content to
        analyse) and keyword arguments (e.g. ``model="llama3"``).

        Example
        -------
        >>> LLMCache.make_key(diff_text, model="llama-3.3-70b", repo="owner/repo")
        'a3f9...'
        """
        raw = "\x00".join(parts)
        if kwargs:
            sorted_kv = "\x00".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
            raw = f"{raw}\x01{sorted_kv}"
        return hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """Return the cached value for *key*, or ``None`` on a cache miss.

        Also handles TTL expiry: stale entries are removed and treated as
        misses.
        """
        if key not in self._store:
            self.misses += 1
            logger.debug("Cache MISS  key=%s  (total misses=%d)", key[:12], self.misses)
            return None

        value, stored_at = self._store[key]

        # TTL check
        if self._ttl and (time.monotonic() - stored_at) > self._ttl:
            del self._store[key]
            self.misses += 1
            logger.debug(
                "Cache MISS (expired)  key=%s  age=%.1fs",
                key[:12],
                time.monotonic() - stored_at,
            )
            return None

        # Move to end to mark as recently used (LRU semantics)
        self._store.move_to_end(key)
        self.hits += 1
        logger.info(
            "Cache HIT  key=%s  (hits=%d  misses=%d  ratio=%.1f%%)",
            key[:12],
            self.hits,
            self.misses,
            self._hit_ratio(),
        )
        return value

    def set(self, key: str, value: Any) -> None:
        """Store *value* in the cache under *key*.

        Evicts the least-recently-used entry if the cache is full.
        """
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = (value, time.monotonic())

        # Evict oldest when over capacity
        while len(self._store) > self._max_size:
            evicted_key, _ = self._store.popitem(last=False)
            logger.debug("Cache EVICT  key=%s  (size=%d)", evicted_key[:12], len(self._store))

        logger.debug("Cache SET  key=%s  (size=%d)", key[:12], len(self._store))

    def invalidate(self, key: str) -> bool:
        """Remove a single entry.  Returns ``True`` if it existed."""
        existed = key in self._store
        self._store.pop(key, None)
        if existed:
            logger.debug("Cache INVALIDATE  key=%s", key[:12])
        return existed

    def clear(self) -> None:
        """Flush all entries and reset statistics."""
        self._store.clear()
        self.hits = 0
        self.misses = 0
        logger.info("Cache CLEAR")

    # ------------------------------------------------------------------
    # Stats helpers
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Current number of entries."""
        return len(self._store)

    def stats(self) -> dict:
        """Return a snapshot of cache statistics."""
        return {
            "size": self.size,
            "max_size": self._max_size,
            "ttl_seconds": self._ttl,
            "hits": self.hits,
            "misses": self.misses,
            "hit_ratio_pct": round(self._hit_ratio(), 2),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _hit_ratio(self) -> float:
        total = self.hits + self.misses
        return (self.hits / total * 100) if total else 0.0

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<LLMCache size={self.size}/{self._max_size} "
            f"hits={self.hits} misses={self.misses}>"
        )
