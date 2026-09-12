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

## Next: Phase 2

LoRA fine-tune a small instruct model (Qwen2.5-1.5B-Instruct) on this dataset.
See PLAN.md for the toolchain and hardware-specific quantization notes.
