#!/usr/bin/env python3
"""Phase 3 benchmark suite: latency, throughput, memory, constrained vs.
unconstrained overhead, and quantized vs. fp32.

Batching throughput isn't re-measured here -- engine/verify_batch.py already
produced real numbers (1.58x at batch 4, 2.37x at batch 8) and re-running it
adds nothing new; this script covers what that one didn't.

Quantization note: PLAN.md's "quantized vs. fp16" really means QLoRA /
`bitsandbytes` 4-bit, which is CUDA-only and doesn't run on this Mac. CPU
int8 dynamic quantization (`torch.quantization.quantize_dynamic`) is used
here as the same-machine stand-in -- a different technique, but the same
underlying question (does quantizing the linear layers help or hurt on the
hardware we actually have).

Usage:
    python3 engine/benchmark.py --n 30
"""
import argparse
import io
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
from constrained import ConstrainedToolPlanDecoder  # noqa: E402


def state_dict_bytes(model) -> int:
    """Serialized size of the model's weights -- a fair size comparison
    regardless of how a given backend packs its tensors internally (plain
    fp32 tensors vs. int8 dynamic-quantization's packed params)."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes


KNOWN_TOOLS = {
    "query_fraud_history",
    "query_beneficiary_trust",
    "query_behavioral_norms",
    "query_graph_risk",
    "query_fraud_policies",
}


def check_accuracy(generator, tokenizer, rows, prompts, vocab_size, max_new_tokens):
    """Runs constrained decoding and checks exact-match tool-plan accuracy
    against the heuristic ground truth. Speed and size aren't the only things
    that can regress under a technique like quantization -- accuracy has to
    be checked directly, not assumed, however good the throughput numbers
    look; see the fp32-vs-int8 accuracy delta below for exactly why."""
    n_exact = 0
    for row, prompt_text in zip(rows, prompts):
        prompt_len = len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
        decoder = ConstrainedToolPlanDecoder(tokenizer, prompt_len, vocab_size, max_new_tokens=max_new_tokens)
        result = generator.generate(prompt_text, max_new_tokens=max_new_tokens, logits_processor=decoder)
        parsed = json.loads(result.text)  # constrained decoding guarantees this parses
        if set(parsed.get("selectedTools", [])) == set(row["output"]["selectedTools"]):
            n_exact += 1
    return n_exact / len(rows)


def run_latency(generator, prompts, max_new_tokens, logits_processor_factory=None):
    prefill_times, tokens_per_sec, cache_mb = [], [], []
    for prompt_text in prompts:
        logits_processor = logits_processor_factory(prompt_text) if logits_processor_factory else None
        result = generator.generate(prompt_text, max_new_tokens=max_new_tokens, logits_processor=logits_processor)
        prefill_times.append(result.prefill_seconds)
        if result.decode_seconds:
            tokens_per_sec.append(result.tokens_per_second)
        if result.cache_stats:
            cache_mb.append(result.cache_stats.megabytes)
    return {
        "avg_prefill_ms": 1000 * sum(prefill_times) / len(prefill_times),
        "avg_tokens_per_sec": sum(tokens_per_sec) / len(tokens_per_sec) if tokens_per_sec else 0.0,
        "avg_cache_mb": sum(cache_mb) / len(cache_mb) if cache_mb else 0.0,
    }


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

    rows = [json.loads(line) for line in args.eval_file.read_text().splitlines() if line.strip()][: args.n]
    prompts = [
        tokenizer.apply_chat_template(build_messages(r["features"]), tokenize=False, add_generation_prompt=True)
        for r in rows
    ]

    base_model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.float32)
    peft_model = PeftModel.from_pretrained(base_model, str(args.adapter))
    fp32_model = peft_model.merge_and_unload()  # plain nn.Linear layers -- required for dynamic quantization
    fp32_model.eval()

    fp32_bytes = state_dict_bytes(fp32_model)

    print("=== 1. Unconstrained vs. constrained decoding (fp32) ===")
    generator = ManualGenerator(fp32_model, tokenizer)
    unconstrained = run_latency(generator, prompts, args.max_new_tokens)
    vocab_size = fp32_model.get_output_embeddings().weight.shape[0]

    def make_constrained(prompt_text):
        prompt_len = len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
        return ConstrainedToolPlanDecoder(tokenizer, prompt_len, vocab_size, max_new_tokens=args.max_new_tokens)

    constrained = run_latency(generator, prompts, args.max_new_tokens, logits_processor_factory=make_constrained)

    print(f"unconstrained: {unconstrained['avg_tokens_per_sec']:.2f} tok/s, "
          f"prefill {unconstrained['avg_prefill_ms']:.0f}ms")
    print(f"constrained  : {constrained['avg_tokens_per_sec']:.2f} tok/s, "
          f"prefill {constrained['avg_prefill_ms']:.0f}ms")
    overhead = 100 * (1 - constrained["avg_tokens_per_sec"] / unconstrained["avg_tokens_per_sec"])
    print(f"throughput overhead from constraining: {overhead:.1f}%")

    print(f"\nKV-cache memory at end of generation: {unconstrained['avg_cache_mb']:.2f} MB avg "
          f"(prompt + ~{args.max_new_tokens} generated tokens)")

    print("\n=== 2. fp32 vs. CPU int8 dynamic quantization ===")
    torch.backends.quantized.engine = "qnnpack"  # the only quantized-CPU-kernel backend available on Apple Silicon
    int8_model = torch.quantization.quantize_dynamic(fp32_model, {torch.nn.Linear}, dtype=torch.qint8)
    int8_bytes = state_dict_bytes(int8_model)

    int8_generator = ManualGenerator(int8_model, tokenizer)
    int8_results = run_latency(int8_generator, prompts, args.max_new_tokens)

    print(f"fp32  weights: {fp32_bytes / 1e6:.1f} MB | {unconstrained['avg_tokens_per_sec']:.2f} tok/s")
    print(f"int8  weights: {int8_bytes / 1e6:.1f} MB | {int8_results['avg_tokens_per_sec']:.2f} tok/s")
    print(f"size reduction: {100 * (1 - int8_bytes / fp32_bytes):.1f}%")
    speed_delta = 100 * (int8_results["avg_tokens_per_sec"] / unconstrained["avg_tokens_per_sec"] - 1)
    print(f"speed change: {speed_delta:+.1f}%")

    print("\n=== 3. Accuracy: fp32 vs. int8 (the check speed/size numbers alone would miss) ===")
    fp32_accuracy = check_accuracy(generator, tokenizer, rows, prompts, vocab_size, args.max_new_tokens)
    int8_accuracy = check_accuracy(int8_generator, tokenizer, rows, prompts, vocab_size, args.max_new_tokens)
    print(f"fp32 exact-match accuracy: {100 * fp32_accuracy:.1f}%")
    print(f"int8 exact-match accuracy: {100 * int8_accuracy:.1f}%")
    if int8_accuracy < fp32_accuracy - 0.05:
        print("int8 quantization measurably hurts task accuracy here -- LoRA fine-tuning applies small, "
              "delicate weight adjustments that aggressive int8 quantization of the merged weights can "
              "wash out. Faster/smaller is not automatically a win; don't ship quantization on the "
              "strength of the speed/size numbers alone.")


if __name__ == "__main__":
    main()
