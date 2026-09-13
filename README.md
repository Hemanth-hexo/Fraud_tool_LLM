# Fraud Tool-Planner LLM

A small, locally fine-tuned LLM paired with a **custom-built inference
engine**, trained to make a structured tool-selection decision in a
fraud-detection pipeline -- entirely on-device, at $0 marginal cost per
decision. See [PLAN.md](PLAN.md) for the original project plan and hardware
constraints.

## What this is

A fraud-detection pipeline needs to decide which of 5 read-only
investigation tools to run per transaction. That decision is currently made
by an external hosted LLM API call -- effective, but with recurring
per-token cost and no structural guarantee against a malformed or
hallucinated response. This project builds a self-contained, local
alternative: synthetic training data generated from the pipeline's own real
decision logic, a LoRA fine-tune of a 1.5B-parameter model on that data, and
a custom inference engine -- built from scratch, not `model.generate()` --
that makes structurally invalid output impossible by construction.
Everything here runs on a CPU-only laptop; no GPU, no cloud spend.

## Results at a glance

| | |
|---|---|
| Tool-plan accuracy vs. the real production heuristic | **99.3%** exact match |
| JSON validity under constrained decoding | **100%**, even on an untrained model (vs. 50% unconstrained) |
| Batching throughput | **2.37x** at batch size 8 |
| int8 quantization | **59.8%** smaller, **+24.5%** faster |
| Constrained-decoding overhead | effectively **0%** |
| Marginal cost per decision | **$0**, fully local inference |

## How it works

```
transaction features
        |
        v
  synthetic data generator  --labeled by the real production heuristic-->  train/eval JSONL
        |
        v
  LoRA fine-tune (Qwen2.5-1.5B-Instruct)  -->  adapter, 99.3% accurate
        |
        v
  custom inference engine
    - manual KV-cache management (own the generation loop, no model.generate())
    - batching (serve several transactions in one forward pass)
    - grammar-constrained decoding (logits masked so invalid JSON / unknown
      tool names are structurally impossible)
    - int8 quantization (smaller + faster on CPU)
        |
        v
  {"selectedTools": [...], "reason": "..."}   -- guaranteed valid, every time
```

## Repo layout

| Path | What's in it |
|---|---|
| [`PLAN.md`](PLAN.md) | Original project plan, constraints, and hardware notes |
| [`data/`](data/) | Synthetic data generator + the ported production heuristic used to label it |
| [`train/`](train/) | LoRA fine-tuning pipeline and generation-quality eval |
| [`engine/`](engine/) | The custom inference engine: cache, batching, constrained decoding, benchmarks |

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Phase 1: generate training data
python3 data/generate_dataset.py --n-train 4000 --n-eval 500

# Phase 2: fine-tune
python3 train/train.py --precision fp32 --device cpu --max-steps 500 \
  --batch-size 2 --max-eval-samples 30 --out-dir checkpoints/lora-adapter-v1

# Phase 3: exercise the inference engine
python3 engine/verify.py --n 10
python3 engine/verify_batch.py --batch-size 8
python3 engine/verify_constrained.py --n 20 --max-new-tokens 80
python3 engine/benchmark.py --n 10 --max-new-tokens 80
```

---

## Phase 1 — synthetic training data

`data/heuristic.py` is a line-for-line Python port of the heuristic fallback
branch in the production service's tool-planner code (verified byte-for-byte
against the original JS on 2000 random inputs). `data/generate_dataset.py`
generates random, realistic transaction feature vectors and labels each one
with that heuristic, producing `data/train.jsonl` and `data/eval.jsonl`.

```bash
python3 data/generate_dataset.py --n-train 4000 --n-eval 500
```

Each line is `{"features": {...}, "output": {"selectedTools": [...], "reason": "..."}}`.

## Phase 2 — LoRA fine-tune

`train/train.py` LoRA fine-tunes Qwen2.5-1.5B-Instruct on the Phase 1 dataset
via `peft`. One script for both target machines -- `--precision` picks
fp16/bf16 (Mac) or 4-bit QLoRA (Omen, gated to require CUDA).

```bash
python3 train/train.py --precision fp32 --device cpu --max-steps 500 \
  --batch-size 2 --max-eval-samples 30 --out-dir checkpoints/lora-adapter-v1
```

**Hardware finding:** on this machine (M5 MacBook Pro), PyTorch 2.14's MPS
backend benchmarked ~2x *slower* than plain CPU for raw forward/backward
passes on this model -- likely immature Metal kernels for the brand-new M5
GPU. `train.py` defaults training to `--device cpu` as a result; re-benchmark
MPS as PyTorch releases catch up, or use `--device cuda` on a CUDA machine.

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

## Phase 3 — custom inference engine

`engine/cache.py` + `engine/generate.py`: a manual autoregressive decode loop
-- no `model.generate()`. We create the `DynamicCache` ourselves, run one
prefill forward pass over the prompt, then repeatedly feed just the last
generated token (plus the same growing cache) back in, one step at a time.
Transformers' own cache tensor mechanics are reused (its masking/sliding-
window internals are deep, fast-moving library code not worth reimplementing)
-- what we own is the loop itself: prefill vs. decode are explicit, the cache
lifecycle is fully visible and inspectable (`CacheManager.stats()`), and each
decode step is individually timed.

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
move to `reason`. `reason` itself stays free text -- the goal is valid JSON
+ valid tool names, not hardcoding the heuristic's two known reason strings
-- guarded only against a stray `"` breaking the JSON, with a budget-aware
forced-close safety net so output is always complete and valid even if the
underlying model never learns to stop on its own.

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

### Benchmarks

`engine/benchmark.py` covers what `verify_batch.py` didn't already measure:
constrained-decoding overhead, KV-cache memory footprint, and a quantization
comparison. On quantization: true QLoRA / `bitsandbytes` 4-bit is CUDA-only
and doesn't run on Apple Silicon -- CPU int8 dynamic quantization
(`torch.quantization.quantize_dynamic`, on the `qnnpack` backend, the only
quantized-kernel backend this Mac actually has) is the same-machine stand-in.

```bash
python3 engine/benchmark.py --n 30 --max-new-tokens 80
```

| Benchmark | Result |
|---|---|
| Constrained vs. unconstrained throughput | 10.33 vs. 10.57 tok/s -- effectively no overhead; the logits mask is negligible next to a full transformer forward pass |
| KV-cache memory | 17.97 MB for a prompt + ~80 generated tokens |
| fp32 vs. int8 weights (size) | 6175 MB -> 2481 MB, a **59.8%** reduction |
| fp32 vs. int8 throughput | 10.33 -> 14.12 tok/s, a **+36.6%** speedup (qnnpack's real accelerated int8 kernels on ARM, not just smaller weights) |
| **fp32 vs. int8 exact-match accuracy** | **100% -> 46.7%** |

That last row is the one that matters most, and it very nearly shipped
unnoticed: the first pass at this benchmark only measured speed and size,
which made int8 look like a clean win. A direct accuracy check tells a
different story -- LoRA fine-tuning applies small, delicate weight
adjustments, and aggressive int8 quantization of the merged weights washes
enough of that out to roughly halve tool-plan accuracy, even though JSON
validity stays at 100% either way (that's a structural guarantee from the
constrained decoder, not a statistical one -- it says nothing about whether
the *content* is right). The lesson generalizes past this one project: a
compression technique's speed and size numbers say nothing about whether the
output is still correct, and only checking the former is exactly how this
kind of regression gets shipped. The serving layer defaults to fp32 as a
result; quantization is opt-in, not on by default.

Combined with the batching result above (1.58x at batch 4, 2.37x at batch 8),
the full picture on this CPU-only machine: batching is a clean win, int8
quantization is a real speed/size win *with a real accuracy cost* that has
to be weighed deliberately rather than assumed away, and constrained
decoding's correctness guarantee is essentially free either way.

## Serving

`serve/app.py` wraps the engine in a small FastAPI service (`POST /plan`,
`GET /health`) so it can run as a standalone container instead of only via
local scripts -- the same artifact works for running in Docker locally or
deploying to a cloud instance. It defaults to fp32 for the reason directly
above; set `USE_QUANTIZED=1` only if you've separately verified acceptable
accuracy for your use case.

```bash
uvicorn serve.app:app --host 0.0.0.0 --port 8888
curl -X POST http://localhost:8888/plan -H "Content-Type: application/json" \
  -d '{"amount": 15000, "deviceId": "shared_device_42", "ipAddress": "8.8.8.8", "beneficiaryId": "suspect_99", "location": "RU", "time": 3, "userId": "user_1"}'
```

## Status

Phases 1-3 are complete: this repo is a finished, standalone piece of work
on its own -- synthetic data generation, a 99.3%-accurate LoRA fine-tune,
and a custom inference engine with manual KV-cache management, batching,
from-scratch constrained decoding, and a full benchmark suite.

A further integration step (wiring this model into the target pipeline as
an additive option alongside its existing decision paths) is documented in
PLAN.md as a possible next step, but lives in that pipeline's own repo
rather than here, and depends on that project's own review process.
