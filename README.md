# fraud-tool-planner-llm

See [PLAN.md](PLAN.md) for the full project plan.

## Phase 1 — synthetic training data (done)

`data/heuristic.py` is a line-for-line Python port of the heuristic fallback
branch in `aws-final`'s `agentic_ai.service.js` (verified byte-for-byte against
the JS on 2000 random inputs). `data/generate_dataset.py` generates random,
realistic transaction feature vectors and labels each one with that heuristic,
producing `data/train.jsonl` and `data/eval.jsonl`.

```bash
python3 data/generate_dataset.py --n-train 4000 --n-eval 500
```

Each line is `{"features": {...}, "output": {"selectedTools": [...], "reason": "..."}}`.

## Phase 2 — LoRA fine-tune (done)

`train/train.py` LoRA fine-tunes Qwen2.5-1.5B-Instruct on the Phase 1 dataset
via `peft`. One script for both target machines -- `--precision` picks
fp16/bf16 (Mac) or 4-bit QLoRA (Omen, gated to require CUDA).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-train.txt
python3 train/train.py --precision fp32 --device cpu --max-steps 500 \
  --batch-size 2 --max-eval-samples 30 --out-dir checkpoints/lora-adapter-v1
```

**Hardware finding:** on this machine (M5 MacBook Pro), PyTorch 2.14's MPS
backend benchmarked ~2x *slower* than plain CPU for raw forward/backward
passes on this model -- likely immature Metal kernels for the brand-new M5
GPU. `train.py` defaults training to `--device cpu` as a result; re-benchmark
MPS as PyTorch releases catch up, or use `--device cuda` on the Omen.

`train/eval_model.py` checks real generation (not just teacher-forced loss)
against held-out data: JSON validity, known tool names, and exact
`selectedTools` match against the heuristic ground truth.

**Results:** the first 500-step run (fp32/CPU, 0.25 epoch) reached 96% exact
generation match, with a single systematic gap: it under-learned the rare
`ipAddress == "192.168.1.100"` OR-branch of the heuristic, seeing that
combination in only ~4% of the examples it was trained on. A short booster
run -- oversampling that case 3x and resuming from the v1 adapter via
`--resume-adapter`, see `data/train_boost.jsonl` -- brought it to **99.3%**
exact match, with the one remaining miss a reasonable near-confusion on an
IP (`192.122.90.30`) that superficially resembles the trigger IP.

## Phase 3 — custom inference engine (in progress)

`engine/cache.py` + `engine/generate.py`: a manual autoregressive decode loop
-- no `model.generate()`. We create the `DynamicCache` ourselves, run one
prefill forward pass over the prompt, then repeatedly feed just the last
generated token (plus the same growing cache) back in, one step at a time.
Transformers' own cache tensor mechanics are reused (its masking/sliding-
window internals are deep, fast-moving library code not worth reimplementing)
-- what we own is the loop itself: prefill vs. decode are explicit, the cache
lifecycle is fully visible and inspectable (`CacheManager.stats()`), and each
decode step is individually timed. This is also the hook point for what's
next: constrained decoding masks `logits` right before the argmax in
`generate.py`, and batching reuses the same loop with a batch dimension.

`engine/verify.py` checks correctness the only way that matters: token-for-
token identical output against `model.generate()` on real eval examples.

```bash
python3 engine/verify.py --n 10
```

**Result:** 10/10 outputs byte-identical to `model.generate()`, at ~8.7
tokens/sec on CPU with only ~5% wall-clock overhead vs. the built-in path
(52.2s vs. 49.8s total) -- the manual loop isn't just correct, it's not
leaving obvious performance on the table either.

`engine/batch_generate.py` extends the same loop to serve several
transactions in one forward pass. The real work isn't the loop, it's that
`DynamicCache` holds one tensor per layer shared across the whole batch --
every row must share a sequence length at every step. We handle that
ourselves: left-padding (so short prompts don't shift where new tokens get
appended), per-row position ids (computed once from the attention mask,
since RoPE needs each row's true position, not its padded index), and
per-row early stopping (a finished row can't shrink out of the shared cache
tensor, so it keeps getting silently decoded and we just stop recording its
tokens once EOS appears).

```bash
python3 engine/verify_batch.py --batch-size 8
```

**Result:** batched output is token-for-token identical to generating each
prompt alone (proves the padding/position-id/early-stop handling is
correct), and batching is a real win on this CPU-bound box: **1.58x**
speedup at batch size 4, **2.37x** at batch size 8, vs. running the same
prompts one-by-one.

`engine/constrained.py` is the actual differentiator: grammar/schema-
constrained decoding from scratch, built as a logits mask plugged into the
same loop (no external constrained-decoding library). The output schema is
fixed -- `{"selectedTools": [...], "reason": "..."}` -- and most of that is
deterministic syntax (braces, quotes, key names, commas), so it's
force-injected directly rather than left to the model's opinion (the same
"fast-forward" trick real grammar-decoding libraries use for literal grammar
spans). The model's own masked logits only decide at the two genuinely open
points: which tool name comes next (a trie over the 5 names, since they all
share the `query_` prefix and need real token-by-token narrowing, not a
single-token check), and whether to add another tool or close the array and
move to `reason`. `reason` itself stays free text -- PLAN.md's ask is valid
JSON + valid tool names, not hardcoding the heuristic's two known reason
strings -- guarded only against a stray `"` breaking the JSON, with a
budget-aware forced-close safety net so output is always complete and valid
even if the underlying model never learns to stop on its own.

```bash
python3 engine/verify_constrained.py --n 20 --max-new-tokens 80
```

**Result**, at a deliberately tight 80-token budget:

| | unconstrained | constrained |
|---|---|---|
| Fine-tuned model (knows the task) | 20/20 valid, 20/20 exact match | 20/20 valid, 20/20 exact match |
| Base model (never saw this schema) | **10/20** valid JSON | **20/20** valid JSON, 20/20 known tool names |

No regression on the model that already knows the task, and on a model that
doesn't, unconstrained decoding genuinely fails half the time (it rambles in
the free-text `reason` field and runs out of budget before closing the
string) while constrained decoding is 100% structurally valid regardless --
by construction, not by luck. (Its tool *choices* are still wrong on the
base model, 0/20 exact match, exactly as expected: picking the right tools
is a training problem, not a decoding one -- constrained decoding only
guarantees the shape of the answer, not its correctness.)

**Still to build:** the benchmark suite (latency, throughput, memory;
constrained vs. unconstrained; quantized vs. fp16 -- using CPU int8 dynamic
quantization as the comparison point in place of QLoRA/bitsandbytes, which
doesn't run on Apple Silicon). See PLAN.md.
