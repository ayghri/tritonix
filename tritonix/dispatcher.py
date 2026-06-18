"""Backend dispatcher: benchmarks multiple implementations and caches the fastest."""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
from contextlib import contextmanager
from typing import Callable, Dict, List, Optional

import torch
import triton

log = logging.getLogger(__name__)


def _tuning_key(key_names: List[str], args: tuple, kwargs: dict, arg_names: List[str]) -> tuple:
    bound = dict(zip(arg_names, args))
    bound.update(kwargs)
    parts: list = []
    for name in key_names:
        if name in bound:
            v = bound[name]
            parts.extend(list(v.shape) + [str(v.dtype)] if isinstance(v, torch.Tensor) else [v])
    keyed = set(key_names)
    for name, v in bound.items():
        if name not in keyed and isinstance(v, torch.Tensor):
            parts.extend(list(v.shape) + [str(v.dtype)])
    return tuple(parts)


def _disk_key(op_name: str, backends: List[str], tuning_key: tuple) -> str:
    parts = [op_name, str(sorted(backends)), str(tuning_key)]
    return hashlib.sha256("-".join(parts).encode()).hexdigest()


class DynamicDispatcher:
    def __init__(
        self,
        fn: Callable,
        key: List[str],
        arg_names: List[str],
        warmup: int = 25,
        rep: int = 100,
        use_disk_cache: bool = True,
    ):
        self._fn = fn
        self._key_names = key
        self._arg_names = arg_names
        self._warmup = warmup
        self._rep = rep
        self._disk = use_disk_cache
        self._backends: Dict[str, Callable] = {}
        self._forced: Optional[str] = None
        self.cache: Dict[tuple, str] = {}
        self.timings: Dict[tuple, Dict[str, float]] = {}
        self.__name__ = fn.__name__
        self.__doc__ = fn.__doc__

    def register(self, name: str, impl: Callable) -> "DynamicDispatcher":
        self._backends[name] = impl
        return self

    def __call__(self, *args, **kwargs):
        if not self._backends:
            raise RuntimeError(f"[{self.__name__}] No backends registered.")
        if self._forced is not None:
            return self._backends[self._forced](*args, **kwargs)

        key = _tuning_key(self._key_names, args, kwargs, self._arg_names)
        if key not in self.cache:
            self._bench_and_cache(key, args, kwargs)
        return self._backends[self.cache[key]](*args, **kwargs)

    def _bench_and_cache(self, key: tuple, args: tuple, kwargs: dict):
        if self._disk:
            winner = self._load_from_disk(key)
            if winner is not None:
                return

        results = {}
        for name, impl in self._backends.items():
            try:
                results[name] = triton.testing.do_bench(
                    lambda: impl(*args, **kwargs), warmup=self._warmup, rep=self._rep
                )
            except Exception as e:
                log.debug("[%s] backend '%s' skipped: %s", self.__name__, name, e)
        if not results:
            raise RuntimeError(f"[{self.__name__}] All backends failed.")
        winner = min(results, key=results.get)
        log.info("[%s] %s", self.__name__, "  ".join(f"{n}={v:.3f}ms" for n, v in results.items()))

        self.cache[key] = winner
        self.timings[key] = results
        if self._disk:
            self._save_to_disk(key, results, winner)

    _CACHE_FILE = "backend_dispatch.json"

    def _cache_manager(self, key: tuple):
        from triton.runtime.cache import get_cache_manager
        return get_cache_manager(_disk_key(self.__name__, list(self._backends.keys()), key))

    def _load_from_disk(self, key: tuple) -> Optional[str]:
        try:
            path = self._cache_manager(key).get_file(self._CACHE_FILE)
            if path is None:
                return None
            with open(path) as f:
                data = json.load(f)
            self.cache[key] = data["winner"]
            self.timings[key] = data["timings"]
            return data["winner"]
        except Exception as exc:
            log.debug("[%s] disk miss: %s", self.__name__, exc)
            return None

    def _save_to_disk(self, key: tuple, results: Dict[str, float], winner: str):
        try:
            self._cache_manager(key).put(
                json.dumps({"op": self.__name__, "key": list(key), "winner": winner, "timings": results}),
                self._CACHE_FILE,
                binary=False,
            )
        except Exception as exc:
            log.warning("[%s] disk write failed: %s", self.__name__, exc)

    @contextmanager
    def force_backend(self, name: str):
        if name not in self._backends:
            raise KeyError(f"Backend '{name}' not registered in '{self.__name__}'.")
        self._forced = name
        try:
            yield
        finally:
            self._forced = None

    def clear_cache(self):
        self.cache.clear()
        self.timings.clear()

    def best_config(self) -> Dict[tuple, str]:
        return dict(self.cache)


def dynamic_dispatch(
    backends: Optional[Dict[str, Callable]] = None,
    key: Optional[List[str]] = None,
    warmup: int = 25,
    rep: int = 100,
    use_disk_cache: bool = True,
):
    def _make(fn: Callable) -> DynamicDispatcher:
        return DynamicDispatcher(
            fn,
            key=key or [],
            arg_names=list(inspect.signature(fn).parameters.keys()),
            warmup=warmup,
            rep=rep,
            use_disk_cache=use_disk_cache,
        )

    if backends is not None:
        first = next(iter(backends.values()))
        d = _make(first)
        for name, impl in backends.items():
            d.register(name, impl)
        return d

    # decorator mode: @dynamic_dispatch(key=[...])
    def decorator(fn: Callable) -> DynamicDispatcher:
        return _make(fn)
    return decorator
