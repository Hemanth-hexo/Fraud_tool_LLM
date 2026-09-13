"""Grammar/schema-constrained decoding, built from scratch: mask logits so the
model is structurally incapable of emitting invalid JSON or an unknown tool
name -- the differentiator PLAN.md calls out for Phase 3.

The output schema is fixed: `{"selectedTools": [...], "reason": "..."}`. Most
of that is deterministic syntax -- braces, quotes, key names, commas -- and
deterministic syntax doesn't need the model's opinion, so it's force-injected
directly (this is the same "fast-forward" trick real grammar-decoding
libraries use for literal spans of a grammar). The model's own (masked)
logits only decide at the two points that are actually open questions:

  1. Which tool name comes next -- masked to a trie over the 5 known names
     (they all share the "query_" prefix, so this genuinely needs token-by-
     token narrowing, not just a single-token check) minus whichever tools
     were already picked.
  2. Whether to add another tool or close the array and move to `reason` --
     masked to exactly those two continuations.

`reason` itself is left as free text (masked only to forbid a stray `"` that
would break the JSON, plus the one specific token that cleanly closes the
string) -- PLAN.md's ask is valid JSON + valid tool names, not literally
hardcoding the heuristic's two known reason strings.
"""
import torch

TOOL_NAMES = [
    "query_fraud_history",
    "query_beneficiary_trust",
    "query_behavioral_norms",
    "query_graph_risk",
    "query_fraud_policies",
]

NEG_INF = float("-inf")


class ConstrainedToolPlanDecoder:
    def __init__(
        self,
        tokenizer,
        prompt_len: int,
        model_vocab_size: int,
        max_tools: int = len(TOOL_NAMES),
        max_reason_tokens: int = 30,
        max_new_tokens: int | None = None,
    ):
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len
        self.max_tools = max_tools
        # Safety net: an untrained/undertrained model has no learned reason to ever
        # choose the closing quote in free text, and would otherwise ramble past
        # max_new_tokens, leaving truncated (invalid) JSON. Once either cap is hit
        # we force the close ourselves -- output is *always* complete, valid JSON,
        # regardless of what the underlying model does or how tight the overall
        # generation budget is. `max_new_tokens` should match whatever the caller
        # passes to the generation loop, so the reason field always leaves enough
        # room to actually close before that budget runs out.
        self.max_reason_tokens = max_reason_tokens
        self.max_new_tokens = max_new_tokens
        self.close_margin = len(tokenizer('"}', add_special_tokens=False)["input_ids"]) + 1
        # The model's output dimension is often padded beyond the tokenizer's real
        # vocab for hardware alignment (e.g. Qwen2.5: 151936 vs. 151665 real tokens) --
        # masks must match the logits tensor's actual size, not the tokenizer's.
        self.vocab_size = model_vocab_size
        self.real_vocab_size = len(tokenizer)

        def ids(s: str) -> list[int]:
            return tokenizer(s, add_special_tokens=False)["input_ids"]

        self.prefix_ids = ids('{"selectedTools": ["')
        self.tool_ids = {name: ids(name) for name in TOOL_NAMES}
        self.continue_ids = ids('", "')
        self.close_ids = ids('"], "reason": "')
        self.suffix_ids = ids('"}')

        raw_tokens = tokenizer.convert_ids_to_tokens(list(range(self.real_vocab_size)))
        # True where a token's raw text contains no literal '"' -- safe to use inside
        # the unquoted `reason` free-text span without corrupting the JSON. Padding
        # rows beyond the tokenizer's real vocab (see above) are never valid tokens,
        # so they default to blocked.
        self.quote_free = [t is None or '"' not in t for t in raw_tokens]
        self.quote_free += [False] * (self.vocab_size - self.real_vocab_size)

    def __call__(self, input_ids: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        generated = input_ids[0, self.prompt_len :].tolist()
        mask = self._mask_for(generated)
        return logits + mask.to(logits.device, logits.dtype)

    # -- mask construction -------------------------------------------------

    def _remaining(self, generated: list[int]) -> int | None:
        if self.max_new_tokens is None:
            return None
        return self.max_new_tokens - len(generated)

    def _single(self, token_id: int) -> torch.Tensor:
        mask = torch.full((self.vocab_size,), NEG_INF)
        mask[token_id] = 0.0
        return mask

    def _choice(self, first_tokens: list[int]) -> torch.Tensor:
        mask = torch.full((self.vocab_size,), NEG_INF)
        for t in set(first_tokens):
            mask[t] = 0.0
        return mask

    def _mask_for(self, generated: list[int]) -> torch.Tensor:
        state, ctx = self._replay(generated)

        if state == "prefix":
            return self._single(self.prefix_ids[ctx["pos"]])

        if state == "tool_select":
            candidates = [
                self.tool_ids[name][ctx["pos"]]
                for name in TOOL_NAMES
                if name not in ctx["used"] and self.tool_ids[name][: ctx["pos"]] == ctx["consumed"]
            ]
            return self._choice(candidates)

        if state == "decide":
            # If the array stayed open, we'd still need room for `close_ids` + at
            # least one reason token + `suffix_ids` before the budget runs out.
            min_close_path = len(self.close_ids) + 1 + len(self.suffix_ids)
            remaining = self._remaining(generated)
            must_close = remaining is not None and remaining <= min_close_path
            options = [self.close_ids]
            if len(ctx["used"]) < self.max_tools and not must_close:
                options.append(self.continue_ids)
            options = [o for o in options if o[: ctx["pos"]] == ctx["consumed"]]
            return self._choice([o[ctx["pos"]] for o in options])

        if state == "reason":
            remaining = self._remaining(generated)
            must_close = ctx["reason_len"] >= self.max_reason_tokens or (
                remaining is not None and remaining <= self.close_margin
            )
            if must_close:
                return self._single(self.suffix_ids[0])  # force the close now
            mask = torch.tensor([0.0 if allowed else NEG_INF for allowed in self.quote_free])
            mask[self.suffix_ids[0]] = 0.0  # the model may also choose to close the string now
            return mask

        if state == "suffix":
            return self._single(self.suffix_ids[ctx["pos"]])

        # state == "done": force end-of-sequence
        return self._single(self.tokenizer.eos_token_id)

    # -- state replay --------------------------------------------------------
    # Recomputes the parser state from the tokens generated so far, rather than
    # mutating shared state across calls -- simpler to reason about, and cheap
    # since these outputs are only ~40-60 tokens long.

    def _replay(self, generated: list[int]):
        state = "prefix"
        pos = 0
        used: list[str] = []
        consumed: list[int] = []
        reason_len = 0

        for tok in generated:
            if state == "prefix":
                pos += 1
                if pos == len(self.prefix_ids):
                    state, pos = "tool_select", 0
            elif state == "tool_select":
                consumed.append(tok)
                pos = len(consumed)
                matches = [name for name in TOOL_NAMES if name not in used and self.tool_ids[name][:pos] == consumed]
                if len(matches) == 1 and len(self.tool_ids[matches[0]]) == pos:
                    used.append(matches[0])
                    state, pos, consumed = "decide", 0, []
            elif state == "decide":
                consumed.append(tok)
                pos = len(consumed)
                options = [self.close_ids, self.continue_ids]
                matches = [o for o in options if o[:pos] == consumed]
                if len(matches) == 1 and len(matches[0]) == pos:
                    state = "reason" if matches[0] is self.close_ids else "tool_select"
                    pos, consumed = 0, []
            elif state == "reason":
                if tok == self.suffix_ids[0]:
                    state, pos = ("done", 0) if len(self.suffix_ids) == 1 else ("suffix", 1)
                else:
                    reason_len += 1
            elif state == "suffix":
                pos += 1
                if pos == len(self.suffix_ids):
                    state = "done"

        return state, {"pos": pos, "used": used, "consumed": consumed, "reason_len": reason_len}
