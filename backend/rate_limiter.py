import time
import threading
from typing import Dict, Tuple, Optional

class TokenBucketRateLimiter:
    """
    Thread-safe in-memory sliding token-bucket rate limiter.
    Limits endpoint abuse per client IP or authenticated session.
    """
    def __init__(self):
        self._lock = threading.Lock()
        # Storage: { bucket_key: (tokens, last_refill_timestamp) }
        self._buckets: Dict[str, Tuple[float, float]] = {}
        self._last_cleanup = time.time()
        self._cleanup_interval = 300.0  # Clean up stale buckets every 5 minutes

        # Default rules: (capacity, refill_rate_per_second)
        # capacity: max burst tokens
        # refill_rate: tokens added per second
        self.rules = {
            '/api/recommend': (30.0, 30.0 / 60.0),       # 30 requests per minute
            '/api/retrain': (10.0, 10.0 / 600.0),        # 10 requests per 10 minutes
            '/api/onboarding/start': (5.0, 5.0 / 600.0), # 5 requests per 10 minutes
            'default': (120.0, 120.0 / 60.0)             # 120 requests per minute
        }

    def _get_rule_for_endpoint(self, endpoint: str) -> Tuple[float, float]:
        for prefix, rule in self.rules.items():
            if prefix != 'default' and endpoint.startswith(prefix):
                return rule
        return self.rules['default']

    def is_allowed(self, identifier: str, endpoint: str, cost: float = 1.0) -> Tuple[bool, int]:
        """
        Checks if a request from `identifier` to `endpoint` is allowed.
        Returns:
            (allowed: bool, retry_after: int)
        """
        with self._lock:
            now = time.time()
            self._maybe_cleanup(now)

            capacity, refill_rate = self._get_rule_for_endpoint(endpoint)
            bucket_key = f"{identifier}:{endpoint.split('?')[0]}"

            if bucket_key not in self._buckets:
                tokens = capacity - cost
                self._buckets[bucket_key] = (tokens, now)
                return True, 0

            tokens, last_refill = self._buckets[bucket_key]
            # Refill tokens based on elapsed time
            elapsed = max(0.0, now - last_refill)
            tokens = min(capacity, tokens + elapsed * refill_rate)

            if tokens >= cost:
                tokens -= cost
                self._buckets[bucket_key] = (tokens, now)
                return True, 0
            else:
                # Calculate wait time in seconds until enough tokens are refilled
                deficit = cost - tokens
                retry_after = int(deficit / refill_rate) + 1 if refill_rate > 0 else 60
                self._buckets[bucket_key] = (tokens, now)
                return False, max(1, retry_after)

    def _maybe_cleanup(self, now: float):
        """Removes buckets idle for more than 10 minutes to prevent memory growth."""
        if now - self._last_cleanup < self._cleanup_interval:
            return

        stale_threshold = now - 600.0
        keys_to_remove = [
            k for k, (_, last_refill) in self._buckets.items()
            if last_refill < stale_threshold
        ]
        for k in keys_to_remove:
            del self._buckets[k]

        self._last_cleanup = now

    def reset(self):
        """Resets all buckets (primarily for unit testing)."""
        with self._lock:
            self._buckets.clear()

# Global rate limiter singleton
rate_limiter = TokenBucketRateLimiter()
