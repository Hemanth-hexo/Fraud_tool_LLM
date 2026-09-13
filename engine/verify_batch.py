#!/usr/bin/env python3
"""Correctness + throughput check for batched generation.

Correctness: each row's output, decoded from a shared batch forward pass,
must match generating that same prompt alone (proves left-padding, per-row
position ids, and per-row early-stopping are all handled correctly --
get any of those wrong and a padded prompt's answer silently corrupts).

Throughput: compares wall-clock time for N prompts processed one-by-one
(ManualGenerator, from the KV-cache milestone) vs. as one batch
(BatchGenerator) -- this is the actual point of batching.

Usage:
    python3 engine/verify_batch.py --batch-size 4
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
from batch_generate import BatchGenerator  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, default=REPO_ROOT / "checkpoints" / "lora-adapter-v2")
    parser.add_argument("--eval-file", type=Path, default=REPO_ROOT / "data" / "eval.jsonl")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.float32)
    model = PeftModel.from_pretrained(base_model, str(args.adapter))
    model.eval()

    single = ManualGenerator(model, tokenizer)
    batched = BatchGenerator(model, tokenizer)

    rows = [json.loads(line) for line in args.eval_file.read_text().splitlines() if line.strip()][: args.batch_size]
    prompts = [
        tokenizer.apply_chat_template(build_messages(r["features"]), tokenize=False, add_generation_prompt=True)
        for r in rows
    ]

    print(f"Prompt lengths (tokens): {[len(tokenizer(p)['input_ids']) for p in prompts]}")

    t0 = time.perf_counter()
    single_results = [single.generate(p, max_new_tokens=args.max_new_tokens) for p in prompts]
    single_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    batch_results = batched.generate(prompts, max_new_tokens=args.max_new_tokens)
    batch_time = time.perf_counter() - t0

    n_match = 0
    for i, (s, b) in enumerate(zip(single_results, batch_results)):
        match = s.token_ids == b.token_ids
        n_match += match
        print(f"[{i}] {'OK' if match else 'MISMATCH'}")
        if not match:
            print("  single:", s.text)
            print("  batch :", b.text)

    n = len(rows)
    print(f"\n{n_match}/{n} batch output identical to single-sequence generation")
    print(f"sequential (one-by-one): {single_time:.1f}s")
    print(f"batched (all at once)  : {batch_time:.1f}s")
    print(f"speedup: {single_time / batch_time:.2f}x")


if __name__ == "__main__":
    main()
