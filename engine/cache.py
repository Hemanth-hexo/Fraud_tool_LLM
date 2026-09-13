"""Thin, explicit wrapper around transformers' KV-cache primitive.

`DynamicCache`'s own internals (mask sizing, sliding-window/offloading
support, beam-search reordering) are deep, fast-moving library internals --
reimplementing them from scratch would just be re-typing transformers, not
demonstrating anything. The actual "own the lifecycle instead of hiding it
behind a high-level API" value is here: *we* create the cache, *we* decide
when it grows (one forward call per token, driven by our own loop in
generate.py, not by `model.generate()`), and *we* can inspect/reset/measure
it between steps -- all of which `.generate()` normally hides.
"""
from dataclasses import dataclass

import torch
from transformers.cache_utils import DynamicCache


@dataclass
class CacheStats:
    seq_length: int
    num_layers: int
    bytes_used: int

    @property
    def megabytes(self) -> float:
        return self.bytes_used / (1024 * 1024)


class CacheManager:
    """Owns one `DynamicCache` instance across a generation's prefill + decode steps."""

    def __init__(self, model):
        self.model = model
        self.cache = DynamicCache(config=model.config)

    def seq_length(self) -> int:
        return self.cache.get_seq_length()

    def reset(self) -> None:
        self.cache.reset()

    def stats(self) -> CacheStats:
        """Sum the actual bytes held by every layer's key/value tensors."""
        total_bytes = 0
        num_layers = len(self.cache.layers)
        for layer in self.cache.layers:
            for tensor in (getattr(layer, "keys", None), getattr(layer, "values", None)):
                if isinstance(tensor, torch.Tensor):
                    total_bytes += tensor.element_size() * tensor.nelement()
        return CacheStats(seq_length=self.seq_length(), num_layers=num_layers, bytes_used=total_bytes)
