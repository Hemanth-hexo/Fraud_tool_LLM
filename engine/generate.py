"""Manual autoregressive decoding loop: no `model.generate()`.

Two phases, both driven by us:
  - prefill:  one forward pass over the whole prompt, `use_cache=True`, an
              explicitly-created `CacheManager` supplied as `past_key_values`.
  - decode:   repeated forward passes, each fed *only* the previous step's
              new token (shape [batch, 1]) plus the same growing cache --
              never re-feeding the whole sequence like a naive loop would.

This is the hook point for what Phase 3 builds next: constrained decoding
(mask `logits` before the argmax below) and batching (the batch dimension is
already threaded through everywhere here).
"""
import time
from dataclasses import dataclass, field

import torch

from cache import CacheManager


@dataclass
class GenerationResult:
    token_ids: list[int]
    text: str
    prefill_seconds: float
    decode_seconds: list[float] = field(default_factory=list)

    @property
    def total_seconds(self) -> float:
        return self.prefill_seconds + sum(self.decode_seconds)

    @property
    def tokens_per_second(self) -> float:
        if not self.decode_seconds:
            return 0.0
        return len(self.decode_seconds) / sum(self.decode_seconds)


class ManualGenerator:
    """Greedy decoding with an explicitly owned KV cache. No sampling, no beams --
    those are orthogonal to the cache/batching/constrained-decoding work Phase 3
    is actually about, so kept out to keep the loop legible."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 100, logits_processor=None) -> GenerationResult:
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        cache_manager = CacheManager(self.model)

        t0 = time.perf_counter()
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=cache_manager.cache,
            use_cache=True,
        )
        prefill_seconds = time.perf_counter() - t0

        logits = out.logits[:, -1, :]
        if logits_processor is not None:
            logits = logits_processor(input_ids, logits)
        next_token = torch.argmax(logits, dim=-1, keepdim=True)

        generated = [next_token.item()]
        decode_seconds = []
        eos_id = self.tokenizer.eos_token_id

        for _ in range(max_new_tokens - 1):
            if generated[-1] == eos_id:
                break

            step_attention_mask = torch.ones(
                (attention_mask.shape[0], cache_manager.seq_length() + 1), device=device, dtype=attention_mask.dtype
            )

            t0 = time.perf_counter()
            out = self.model(
                input_ids=next_token,
                attention_mask=step_attention_mask,
                past_key_values=cache_manager.cache,
                use_cache=True,
            )
            decode_seconds.append(time.perf_counter() - t0)

            logits = out.logits[:, -1, :]
            if logits_processor is not None:
                full_ids = torch.cat([input_ids, torch.tensor([generated], device=device)], dim=-1)
                logits = logits_processor(full_ids, logits)
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            generated.append(next_token.item())

        if generated and generated[-1] == eos_id:
            generated = generated[:-1]

        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return GenerationResult(
            token_ids=generated, text=text, prefill_seconds=prefill_seconds, decode_seconds=decode_seconds
        )
