"""Concurrent cache misses must not overlap Transformers initialization."""

from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from threading import Event

from app.model_loading import serialized_model_load


def test_different_loaders_serialize_and_same_key_loads_once():
    entered, release, second_entered = Event(), Event(), Event()
    calls = []
    @serialized_model_load
    @lru_cache(maxsize=1)
    def first(name):
        calls.append(name)
        entered.set()
        assert release.wait(2)
        return object()
    @serialized_model_load
    @lru_cache(maxsize=1)
    def second(name):
        second_entered.set()
        return object()
    with ThreadPoolExecutor(max_workers=3) as pool:
        a = pool.submit(first, "encoder")
        assert entered.wait(2)
        b = pool.submit(second, "reranker")
        c = pool.submit(first, "encoder")
        try:
            assert not second_entered.wait(0.05)
        finally:
            release.set()
        assert a.result() is c.result()
        assert b.result() is not None
    assert calls == ["encoder"] and second_entered.is_set()
    first.cache_clear()
    assert first.cache_info().currsize == 0


def test_all_torch_pretrained_loaders_use_the_shared_guard():
    from app.retrieval import encoder, reranker
    from app.voice import _tts
    for loader in (encoder, reranker, _tts):
        assert hasattr(loader, "__wrapped__") and hasattr(loader.__wrapped__, "cache_info")
