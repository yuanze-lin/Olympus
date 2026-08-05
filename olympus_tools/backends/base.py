"""Backend base classes and the lazy specialist registry.

Everything here is built around one constraint: Olympus routes a single
instruction to several specialists, but a workstation typically has one free GPU.
So backends are *lazily constructed* and the :class:`ModelHost` evicts the
previously used pipeline before building the next one. That keeps peak VRAM at
roughly one specialist rather than the sum of all of them.
"""

import gc
import os
from typing import Any, Callable, Dict, Optional

import torch

_REGISTRY: Dict[str, Callable[..., "Backend"]] = {}


def register(name: str):
    """Class decorator adding a backend under ``name``."""

    def _wrap(cls):
        _REGISTRY[name] = cls
        cls.backend_id = name
        return cls

    return _wrap


def available_backends():
    return sorted(_REGISTRY)


class Backend:
    """A specialist model wrapper.

    Subclasses implement :meth:`load` (build the underlying pipeline) and
    :meth:`run` (execute one step). ``run`` receives the step prompt, an optional
    input artifact path, and an output path stem; it returns the path(s) written.
    """

    backend_id = "base"

    def __init__(self, device: str = "cuda", dtype: str = "fp16",
                 model_id: Optional[str] = None, **kwargs):
        self.device = device
        self.dtype = torch.float16 if dtype == "fp16" else torch.float32
        self.model_id = model_id or getattr(self, "default_model_id", None)
        self.options = kwargs
        self._loaded = False

    # -- lifecycle ---------------------------------------------------------
    def load(self):
        raise NotImplementedError

    def ensure_loaded(self):
        if not self._loaded:
            self.load()
            self._loaded = True

    def unload(self):
        for attr in list(self.__dict__):
            if attr.startswith("_pipe") or attr in ("pipe", "model", "processor"):
                setattr(self, attr, None)
        self._loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -- execution ---------------------------------------------------------
    def run(self, prompt: str, input_path: Optional[str], out_stem: str,
            step=None, **kw) -> Dict[str, Any]:
        raise NotImplementedError


class ModelHost:
    """Builds backends on demand and keeps at most ``max_resident`` alive."""

    def __init__(self, device: str = "cuda", dtype: str = "fp16",
                 max_resident: int = 1, overrides: Optional[Dict[str, dict]] = None):
        self.device = device
        self.dtype = dtype
        self.max_resident = max_resident
        self.overrides = overrides or {}
        self._live: Dict[str, Backend] = {}
        self._order = []

    def get(self, backend_id: str) -> Backend:
        if backend_id in self._live:
            return self._live[backend_id]
        if backend_id not in _REGISTRY:
            raise KeyError(
                f"no backend registered for '{backend_id}'. "
                f"available: {', '.join(available_backends())}"
            )
        while len(self._order) >= self.max_resident:
            victim = self._order.pop(0)
            self._live.pop(victim).unload()
        cfg = dict(self.overrides.get(backend_id, {}))
        backend = _REGISTRY[backend_id](device=self.device, dtype=self.dtype, **cfg)
        backend.ensure_loaded()
        self._live[backend_id] = backend
        self._order.append(backend_id)
        return backend

    def shutdown(self):
        for b in self._live.values():
            b.unload()
        self._live.clear()
        self._order.clear()


def hf_kwargs(dtype) -> dict:
    """Common from_pretrained kwargs, honouring an offline cache if configured."""
    kw = {"torch_dtype": dtype}
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        kw["local_files_only"] = True
    return kw
