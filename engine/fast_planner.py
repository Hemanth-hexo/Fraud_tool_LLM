"""Fast constrained tool planning: same model, same greedy decisions, ~5-7x fewer forward
passes than ManualGenerator + ConstrainedToolPlanDecoder.

The slow path feeds the model ONE token per forward pass -- including every token the
grammar has already decided (the JSON skeleton, the shared `query` prefix, the rest of a
tool name once only one tool can match). On CPU each forward pass costs ~100ms whether it
carries 1 token or 8, so a 5-tool plan spent ~55 passes on tokens that were never in
question, plus ~15 more generating a free-text `reason` nothing downstream uses.

Here the model is consulted only at genuine decision points:
  - which tool comes next  (masked argmax over the distinct next tokens the trie allows),
  - continue vs. close the array (masked argmax over the two first tokens).
Every forced token between decision points is appended to a pending buffer and fed in a
single multi-token forward pass with the KV cache. Causal attention makes the hidden state
after "feed k tokens at once" the same as after "feed them one at a time", so the decisions
are the slow path's decisions (verified on the eval set: engine/verify_fast.py).

Two more savings:
  - prefix KV caching: the ~200-token system prompt never changes, so its KV cache is
    computed once at load time and deep-copied per request; only the transaction-specific
    tail is prefilled.
  - no model-generated `reason`: a short factual reason is composed from the plan.
"""
import copy
import time
from dataclasses import dataclass

import torch
from transformers.cache_utils import DynamicCache

from constrained import TOOL_NAMES

NEG_INF = float("-inf")


@dataclass
class FastPlanResult:
    selected_tools: list[str]
    forward_passes: int
    prefill_tokens: int
    seconds: float
    used_prefix_cache: bool


class FastToolPlanner:
    def __init__(self, model, tokenizer, static_prompt_prefix: str | None = None, max_tools: int = len(TOOL_NAMES)):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.max_tools = max_tools
        self.vocab_size = model.get_output_embeddings().weight.shape[0]

        def ids(s: str) -> list[int]:
            return tokenizer(s, add_special_tokens=False)["input_ids"]

        # Same span tokenization as ConstrainedToolPlanDecoder, so the model is conditioned
        # on exactly the token sequence the slow path produces.
        self.prefix_ids = ids('{"selectedTools": ["')
        self.tool_ids = {name: ids(name) for name in TOOL_NAMES}
        self.continue_ids = ids('", "')
        self.close_ids = ids('"], "reason": "')
        self._assert_prefix_free([self.tool_ids[n] for n in TOOL_NAMES])
        self._assert_prefix_free([self.continue_ids, self.close_ids])

        self.static_ids: list[int] = []
        self.static_cache = None
        if static_prompt_prefix:
            self.static_ids = ids(static_prompt_prefix)
            self.static_cache = DynamicCache(config=model.config)
            with torch.no_grad():
                model(input_ids=torch.tensor([self.static_ids], device=self.device),
                      past_key_values=self.static_cache, use_cache=True)

    @staticmethod
    def _assert_prefix_free(seqs: list[list[int]]):
        for a in seqs:
            for b in seqs:
                if a is not b and len(a) < len(b) and b[: len(a)] == a:
                    raise ValueError("candidate token sequences must be prefix-free")

    # -- core loop -------------------------------------------------------------------
    # All per-request decoding state lives in a _Decode object created per call, never on the
    # planner itself: the HTTP server runs requests on a thread pool, and shared state let two
    # overlapping requests write into each other's KV cache.

    class _Decode:
        def __init__(self, planner, cache, pending):
            self.p = planner
            self.cache = cache
            self.pending = pending
            self.last_logits = None
            self.passes = 0

        def logits_for_next(self) -> torch.Tensor:
            """Feed every pending (already-decided) token in ONE forward pass; return the
            next-token logits after the last of them."""
            if self.pending:
                out = self.p.model(
                    input_ids=torch.tensor([self.pending], device=self.p.device),
                    past_key_values=self.cache,
                    use_cache=True,
                )
                self.last_logits = out.logits[0, -1, :]
                self.passes += 1
                self.pending = []
            return self.last_logits

        def choose(self, candidates: list[list[int]]) -> int:
            """Pick one of several token sequences, consulting the model only where they diverge."""
            alive = list(range(len(candidates)))
            pos = 0
            while len(alive) > 1:
                next_tokens = sorted({candidates[i][pos] for i in alive})
                if len(next_tokens) == 1:
                    token = next_tokens[0]  # grammar-forced: no model call
                else:
                    logits = self.logits_for_next()
                    mask = torch.full((self.p.vocab_size,), NEG_INF, device=logits.device, dtype=logits.dtype)
                    mask[next_tokens] = 0.0
                    token = int(torch.argmax(logits + mask))
                self.pending.append(token)
                alive = [i for i in alive if candidates[i][pos] == token]
                pos += 1
            chosen = alive[0]
            self.pending.extend(candidates[chosen][pos:])  # the rest of it is forced
            return chosen

    @torch.no_grad()
    def plan(self, prompt_text: str) -> FastPlanResult:
        t0 = time.perf_counter()
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        n = len(self.static_ids)
        used_prefix_cache = bool(n) and prompt_ids[:n] == self.static_ids
        if used_prefix_cache:
            cache = copy.deepcopy(self.static_cache)
            tail = prompt_ids[n:]
        else:
            cache = DynamicCache(config=self.model.config)
            tail = prompt_ids
        # prompt tail + JSON prefix go in together: one prefill pass
        run = FastToolPlanner._Decode(self, cache, tail + self.prefix_ids)

        used: list[str] = []
        while True:
            remaining = [name for name in TOOL_NAMES if name not in used]
            used.append(remaining[run.choose([self.tool_ids[name] for name in remaining])])
            if len(used) >= self.max_tools:
                break
            if run.choose([self.continue_ids, self.close_ids]) == 1:
                break

        return FastPlanResult(
            selected_tools=used,
            forward_passes=run.passes,
            prefill_tokens=len(tail) + len(self.prefix_ids),
            seconds=time.perf_counter() - t0,
            used_prefix_cache=used_prefix_cache,
        )


def static_prefix_for(tokenizer, build_messages) -> str:
    """The part of the chat-templated prompt that is identical for every transaction: the
    system prompt and the start of the user turn, up to (not including) the features JSON.
    Ends before the space ahead of the JSON so the tokenizer can't merge across the boundary
    (FastToolPlanner verifies the token-level match per request and falls back if not)."""
    sentinel = "<<<FEATURES_JSON>>>"
    messages = build_messages({})
    messages[-1]["content"] = messages[-1]["content"].replace("{}", sentinel)
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return text.split(sentinel)[0].rstrip()
