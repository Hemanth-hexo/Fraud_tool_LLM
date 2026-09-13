"""Batched version of the manual decode loop in generate.py: serve several
transactions' tool-plan requests in one forward pass instead of one at a time.

The tricky part isn't the loop, it's that `DynamicCache` holds one tensor per
layer shared across the whole batch -- every row must have the same sequence
length at every step. Two consequences we have to handle ourselves (this is
exactly what `model.generate()` normally hides):

  - Left-padding: prompts have different lengths, so shorter ones need pad
    tokens on the *left* (padding on the right would shift a short prompt's
    real content away from the position new tokens get appended at).
  - Per-row position ids: RoPE needs each row's true position, not just
    "index into this padded tensor" -- computed once from the attention mask
    at prefill, then simply incremented by 1/step for every row after that.
  - Per-row early stopping: a row that hits EOS can't just drop out of the
    batch (the cache tensor can't shrink per-row), so it keeps getting
    greedy-decoded in the background and we simply stop recording its
    tokens once EOS is seen.
"""
import time

import torch

from cache import CacheManager
from generate import GenerationResult


class BatchGenerator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @torch.no_grad()
    def generate(self, prompts: list[str], max_new_tokens: int = 100) -> list[GenerationResult]:
        device = next(self.model.parameters()).device
        eos_id = self.tokenizer.eos_token_id

        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        try:
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        finally:
            self.tokenizer.padding_side = original_padding_side

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        batch_size = input_ids.shape[0]

        # Real position of each token, ignoring left-padding (pad positions are
        # clamped to 0 -- their value is irrelevant since attention_mask zeroes
        # them out of the attention computation anyway).
        position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)

        cache_manager = CacheManager(self.model)

        t0 = time.perf_counter()
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=cache_manager.cache,
            use_cache=True,
        )
        prefill_seconds = time.perf_counter() - t0

        next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        next_position = position_ids[:, -1:] + 1

        generated: list[list[int]] = [[] for _ in range(batch_size)]
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        running_mask = attention_mask
        decode_seconds = []

        for _ in range(max_new_tokens):
            for i in range(batch_size):
                if not finished[i]:
                    generated[i].append(next_token[i].item())
            finished = finished | (next_token.squeeze(-1) == eos_id)
            if finished.all():
                break

            running_mask = torch.cat(
                [running_mask, torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=device)], dim=-1
            )
            t0 = time.perf_counter()
            out = self.model(
                input_ids=next_token,
                attention_mask=running_mask,
                position_ids=next_position,
                past_key_values=cache_manager.cache,
                use_cache=True,
            )
            decode_seconds.append(time.perf_counter() - t0)

            next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
            next_position = next_position + 1

        results = []
        for ids in generated:
            if ids and ids[-1] == eos_id:
                ids = ids[:-1]
            text = self.tokenizer.decode(ids, skip_special_tokens=True)
            results.append(
                GenerationResult(token_ids=ids, text=text, prefill_seconds=prefill_seconds, decode_seconds=decode_seconds)
            )
        return results
