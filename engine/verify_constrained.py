#!/usr/bin/env python3
"""Two checks for the constrained decoder:

1. On the fine-tuned model: constrained generation should still match the
   heuristic ground truth about as well as unconstrained does (proves the
   constraint doesn't fight a model that already knows the task).

2. On the *base*, non-fine-tuned model: this is the actual proof the
   constraint is doing something. The base model has never seen this schema,
   so unconstrained greedy decoding is expected to wander off (invalid JSON,
   rambling, wrong tool names). Constrained decoding, on the same base model,
   should be 100% valid JSON with 100% known tool names regardless -- it is
   structurally incapable of anything else -- even though its actual tool
   choices are just an untrained guess.

Usage:
    python3 engine/verify_constrained.py --n 20
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "train"))
sys.path.insert(0, str(Path(__file__).parent))
from prompt import build_messages  # noqa: E402
from generate import ManualGenerator  # noqa: E402
from constrained import ConstrainedToolPlanDecoder, TOOL_NAMES  # noqa: E402

KNOWN_TOOLS = set(TOOL_NAMES)


def check_output(completion: str, expected: dict):
    try:
        parsed = json.loads(completion)
        valid_json = isinstance(parsed, dict) and "selectedTools" in parsed and "reason" in parsed
    except json.JSONDecodeError:
        return False, False, False
    if not valid_json:
        return False, False, False
    tools = parsed.get("selectedTools", [])
    known = isinstance(tools, list) and all(t in KNOWN_TOOLS for t in tools)
    exact = isinstance(tools, list) and set(tools) == set(expected["selectedTools"])
    return True, known, exact


def run(label: str, model, tokenizer, rows, constrained: bool, max_new_tokens: int):
    generator = ManualGenerator(model, tokenizer)
    n_valid = n_known = n_exact = 0
    for row in rows:
        prompt_text = tokenizer.apply_chat_template(
            build_messages(row["features"]), tokenize=False, add_generation_prompt=True
        )
        logits_processor = None
        if constrained:
            prompt_len = len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
            vocab_size = model.get_output_embeddings().weight.shape[0]
            logits_processor = ConstrainedToolPlanDecoder(
                tokenizer, prompt_len, vocab_size, max_new_tokens=max_new_tokens
            )
        result = generator.generate(prompt_text, max_new_tokens=max_new_tokens, logits_processor=logits_processor)
        valid, known, exact = check_output(result.text, row["output"])
        n_valid += valid
        n_known += known
        n_exact += exact
    n = len(rows)
    print(f"{label}: valid_json={n_valid}/{n} known_tools={n_known}/{n} exact_match={n_exact}/{n}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, default=REPO_ROOT / "checkpoints" / "lora-adapter-v2")
    parser.add_argument("--eval-file", type=Path, default=REPO_ROOT / "data" / "eval.jsonl")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = [json.loads(line) for line in args.eval_file.read_text().splitlines() if line.strip()][: args.n]

    print("=== Fine-tuned model ===")
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.float32)
    finetuned = PeftModel.from_pretrained(base_model, str(args.adapter))
    finetuned.eval()
    run("unconstrained", finetuned, tokenizer, rows, constrained=False, max_new_tokens=args.max_new_tokens)
    run("constrained  ", finetuned, tokenizer, rows, constrained=True, max_new_tokens=args.max_new_tokens)

    print("\n=== Base model (no fine-tuning -- never saw this schema) ===")
    raw_model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.float32)
    raw_model.eval()
    run("unconstrained", raw_model, tokenizer, rows, constrained=False, max_new_tokens=args.max_new_tokens)
    run("constrained  ", raw_model, tokenizer, rows, constrained=True, max_new_tokens=args.max_new_tokens)


if __name__ == "__main__":
    main()
