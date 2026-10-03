"""Transformers initialization changes process-global state; serialize cold loads."""

from functools import wraps
from threading import RLock


_load_lock = RLock()


def serialized_model_load(cached_load):
    @wraps(cached_load)
    def load(*args, **kwargs):
        # Lock before the cache lookup so simultaneous misses load only once.
        with _load_lock:
            return cached_load(*args, **kwargs)

    load.cache_clear = cached_load.cache_clear
    load.cache_info = cached_load.cache_info
    return load
