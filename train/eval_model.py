#!/usr/bin/env python3
"""Evaluate a fine-tuned LoRA adapter's real generation quality on held-out data.

Loss during training is teacher-forced and doesn't guarantee the model emits
valid JSON / correct tool sets when actually generating token-by-token. This
runs real generation on the eval set and checks: JSON validity, exact
selectedTools match against the heuristic ground truth, and known-tool-name
validity.

Usage:
    python3 train/eval_model.py --adapter checkpoints/lora-adapter-v1 --n 50
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).parent))
from prompt import build_messages

REPO_ROOT = Path(__file__).parent.parent
KNOWN_TOOLS = {
    "query_fraud_history",
    "query_beneficiary_trust",
    "query_behavioral_norms",
    "query_graph_risk",
    "query_fraud_policies",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, default=REPO_ROOT / "checkpoints" / "lora-adapter-v1")
    parser.add_argument("--eval-file", type=Path, default=REPO_ROOT / "data" / "eval.jsonl")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="cpu")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.float32)
    model = PeftModel.from_pretrained(base_model, str(args.adapter))
    model = model.to(device)
    model.eval()

    rows = [json.loads(line) for line in args.eval_file.read_text().splitlines() if line.strip()][: args.n]

    n_valid_json = 0
    n_known_tools = 0
    n_exact_match = 0
    examples_shown = 0
    mismatches = []

    for row in rows:
        messages = build_messages(row["features"])
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        completion = tokenizer.decode(out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

        expected = row["output"]
        is_valid_json = False
        parsed = None
        try:
            parsed = json.loads(completion)
            is_valid_json = isinstance(parsed, dict) and "selectedTools" in parsed and "reason" in parsed
        except json.JSONDecodeError:
            pass

        if is_valid_json:
            n_valid_json += 1
            tools = parsed.get("selectedTools", [])
            if isinstance(tools, list) and all(t in KNOWN_TOOLS for t in tools):
                n_known_tools += 1
            if isinstance(tools, list) and set(tools) == set(expected["selectedTools"]):
                n_exact_match += 1
            else:
                mismatches.append((row["features"], expected, completion))
        else:
            mismatches.append((row["features"], expected, completion))

        if examples_shown < 5:
            print(f"--- example {examples_shown + 1} ---")
            print("features:", json.dumps(row["features"]))
            print("expected:", json.dumps(expected))
            print("got:     ", completion)
            print()
            examples_shown += 1

    n = len(rows)
    print(f"\n=== Results over {n} eval examples ===")
    print(f"Valid JSON matching schema : {n_valid_json}/{n} ({100 * n_valid_json / n:.1f}%)")
    print(f"All tool names known       : {n_known_tools}/{n} ({100 * n_known_tools / n:.1f}%)")
    print(f"Exact selectedTools match  : {n_exact_match}/{n} ({100 * n_exact_match / n:.1f}%)")

    if mismatches:
        print(f"\n=== Mismatches ({len(mismatches)}) ===")
        for features, expected, got in mismatches:
            print("features:", json.dumps(features))
            print("expected:", json.dumps(expected))
            print("got:     ", got)
            print()


if __name__ == "__main__":
    main()
