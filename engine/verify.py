#!/usr/bin/env python3
"""Correctness check: our manual KV-cache decode loop must match `model.generate()`
greedy output exactly. If it doesn't, the cache lifecycle isn't being managed
correctly (e.g. stale positions, wrong attention mask, or truncated cache).

Usage:
    python3 engine/verify.py --n 10
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "train"))
sys.path.insert(0, str(Path(__file__).parent))
from prompt import build_messages  # noqa: E402
from generate import ManualGenerator  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, default=REPO_ROOT / "checkpoints" / "lora-adapter-v2")
    parser.add_argument("--eval-file", type=Path, default=REPO_ROOT / "data" / "eval.jsonl")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.float32)
    model = PeftModel.from_pretrained(base_model, str(args.adapter))
    model.eval()

    manual = ManualGenerator(model, tokenizer)

    rows = [json.loads(line) for line in args.eval_file.read_text().splitlines() if line.strip()][: args.n]

    n_match = 0
    manual_total_time = 0.0
    hf_total_time = 0.0

    for i, row in enumerate(rows):
        prompt_text = tokenizer.apply_chat_template(
            build_messages(row["features"]), tokenize=False, add_generation_prompt=True
        )

        t0 = time.perf_counter()
        manual_result = manual.generate(prompt_text, max_new_tokens=args.max_new_tokens)
        manual_total_time += time.perf_counter() - t0

        inputs = tokenizer(prompt_text, return_tensors="pt")
        t0 = time.perf_counter()
        with torch.no_grad():
            hf_out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        hf_total_time += time.perf_counter() - t0
        hf_new_ids = hf_out[0][inputs["input_ids"].shape[1]:].tolist()
        if hf_new_ids and hf_new_ids[-1] == tokenizer.eos_token_id:
            hf_new_ids = hf_new_ids[:-1]

        match = manual_result.token_ids == hf_new_ids
        n_match += match
        status = "OK" if match else "MISMATCH"
        print(f"[{i}] {status}  manual_tps={manual_result.tokens_per_second:.1f}")
        if not match:
            print("  manual:", manual_result.text)
            print("  hf    :", tokenizer.decode(hf_new_ids, skip_special_tokens=True))

    n = len(rows)
    print(f"\n{n_match}/{n} identical to model.generate()")
    print(f"manual total: {manual_total_time:.1f}s | hf total: {hf_total_time:.1f}s")


if __name__ == "__main__":
    main()
