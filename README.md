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

## Next: Phase 3

Build the custom inference engine (manual KV-cache, batching, from-scratch
grammar-constrained decoding) and benchmark it. See PLAN.md.
