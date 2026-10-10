"""Checks FastToolPlanner against the slow reference path on held-out data: every plan must
be identical to ManualGenerator + ConstrainedToolPlanDecoder's, and we report how often both
match the heuristic ground truth, plus latency and forward-pass counts for each.

    python3 engine/verify_fast.py --n 40
"""
import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(ROOT / "engine"))
from prompt import build_messages  # noqa: E402
from generate import ManualGenerator  # noqa: E402
from constrained import ConstrainedToolPlanDecoder  # noqa: E402
from fast_planner import FastToolPlanner, static_prefix_for  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--precision", default=os.environ.get("PRECISION", "fp16"), choices=["fp16", "fp32"])
    ap.add_argument("--adapter", default=str(ROOT / "checkpoints" / "lora-adapter-v2"))
    ap.add_argument("--skip-slow", action="store_true", help="only time the fast path")
    args = ap.parse_args()

    dtype = {"fp16": torch.float16, "fp32": torch.float32}[args.precision]
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", torch_dtype=dtype)
    model = PeftModel.from_pretrained(base, args.adapter).merge_and_unload().eval()
    vocab_size = model.get_output_embeddings().weight.shape[0]

    fast = FastToolPlanner(model, tokenizer, static_prompt_prefix=static_prefix_for(tokenizer, build_messages))
    slow = ManualGenerator(model, tokenizer)

    rows = [json.loads(line) for line in open(ROOT / "data" / "eval.jsonl")][: args.n]
    agree = fast_correct = slow_correct = 0
    fast_s, slow_s, passes, cache_hits = [], [], [], 0
    for row in rows:
        prompt = tokenizer.apply_chat_template(build_messages(row["features"]), tokenize=False, add_generation_prompt=True)
        truth = row["output"]["selectedTools"]

        r = fast.plan(prompt)
        fast_s.append(r.seconds)
        passes.append(r.forward_passes)
        cache_hits += r.used_prefix_cache
        fast_correct += r.selected_tools == truth

        if not args.skip_slow:
            prompt_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
            decoder = ConstrainedToolPlanDecoder(tokenizer, prompt_len, vocab_size, max_new_tokens=100)
            t0 = time.perf_counter()
            out = slow.generate(prompt, max_new_tokens=100, logits_processor=decoder)
            slow_s.append(time.perf_counter() - t0)
            slow_tools = json.loads(out.text)["selectedTools"]
            slow_correct += slow_tools == truth
            agree += slow_tools == r.selected_tools
            if slow_tools != r.selected_tools:
                print("DISAGREE", row["features"], "fast:", r.selected_tools, "slow:", slow_tools)

    n = len(rows)
    print(f"\nrows: {n}  precision: {args.precision}  prefix cache used: {cache_hits}/{n}")
    print(f"fast path : exact match vs heuristic {fast_correct}/{n} | median {statistics.median(fast_s):.3f}s, "
          f"p95 {sorted(fast_s)[int(0.95 * (n - 1))]:.3f}s | forward passes median {statistics.median(passes)}")
    if slow_s:
        print(f"slow path : exact match vs heuristic {slow_correct}/{n} | median {statistics.median(slow_s):.3f}s")
        print(f"fast == slow on {agree}/{n} plans | speedup {statistics.median(slow_s) / statistics.median(fast_s):.1f}x")


if __name__ == "__main__":
    main()
