#!/usr/bin/env python3
"""Phase 2: LoRA fine-tune of a small instruct model on the Phase 1 dataset.

One script for both target machines -- only --precision changes:
  - Mac (16GB unified memory):      --precision fp16   (default)
  - Omen (RTX 3050, 4GB VRAM):      --precision 4bit    (QLoRA via bitsandbytes)

Usage:
    python3 train/train.py --max-train-samples 40 --epochs 1   # smoke test
    python3 train/train.py                                     # full run
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, PeftModel, get_peft_model

sys.path.insert(0, str(Path(__file__).parent))
from prompt import build_messages, build_target

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
REPO_ROOT = Path(__file__).parent.parent


class ToolPlanDataset(Dataset):
    def __init__(self, path: Path, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        row = self.examples[idx]
        messages = build_messages(row["features"])
        target = build_target(row["output"])

        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        full_text = prompt_text + target + self.tokenizer.eos_token

        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = self.tokenizer(full_text, add_special_tokens=False)["input_ids"][: self.max_length]

        labels = list(full_ids)
        for i in range(min(len(prompt_ids), len(labels))):
            labels[i] = -100  # don't train on the prompt, only the completion

        return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}


class PadCollator:
    """Pads every batch to a fixed length (not the batch's own max).

    MPS recompiles its compute kernels whenever the input shape changes, so
    dynamic per-batch padding -- where nearly every batch has a different
    length -- turned a sub-second forward/backward pass into ~60s/step here.
    A fixed length keeps every step's shape identical, so compilation happens
    once and is reused for the rest of training.
    """

    def __init__(self, pad_token_id: int, fixed_length: int):
        self.pad_token_id = pad_token_id
        self.fixed_length = fixed_length

    def __call__(self, batch):
        max_len = self.fixed_length
        input_ids, attention_mask, labels = [], [], []
        for x in batch:
            pad_len = max_len - len(x["input_ids"])
            input_ids.append(x["input_ids"] + [self.pad_token_id] * pad_len)
            attention_mask.append(x["attention_mask"] + [0] * pad_len)
            labels.append(x["labels"] + [-100] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def pick_dtype(precision: str) -> torch.dtype:
    return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[precision]


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(args, device: str):
    if args.precision == "4bit":
        if device != "cuda":
            raise SystemExit("--precision 4bit (QLoRA) requires a CUDA GPU (the Omen). Use fp16/bf16 elsewhere.")
        from transformers import BitsAndBytesConfig
        from peft import prepare_model_for_kbit_training

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=quant_config)
        return prepare_model_for_kbit_training(model)

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=pick_dtype(args.precision))
    if device != "cpu":
        model = model.to(device)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--train-file", type=Path, default=REPO_ROOT / "data" / "train.jsonl")
    parser.add_argument("--eval-file", type=Path, default=REPO_ROOT / "data" / "eval.jsonl")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "checkpoints" / "lora-adapter")
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16", "4bit"], default="fp16")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--resume-adapter", type=Path, default=None,
                         help="Continue training an existing LoRA adapter instead of starting a fresh one.")
    args = parser.parse_args()

    device = pick_device(args.device)
    print(f"Training on device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_model(args, device)

    if args.resume_adapter:
        model = PeftModel.from_pretrained(model, str(args.resume_adapter), is_trainable=True)
        print(f"Resumed LoRA adapter from {args.resume_adapter}")
    else:
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=TARGET_MODULES,
        )
        model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = ToolPlanDataset(args.train_file, tokenizer, args.max_length)
    if args.max_train_samples:
        train_dataset.examples = train_dataset.examples[: args.max_train_samples]
    eval_dataset = ToolPlanDataset(args.eval_file, tokenizer, args.max_length)
    eval_cap = args.max_eval_samples or (max(args.max_train_samples // 4, 1) if args.max_train_samples else None)
    if eval_cap:
        eval_dataset.examples = eval_dataset.examples[:eval_cap]

    longest = max(
        max(len(train_dataset[i]["input_ids"]) for i in range(len(train_dataset))),
        max(len(eval_dataset[i]["input_ids"]) for i in range(len(eval_dataset))),
    )
    fixed_length = min(args.max_length, ((longest + 7) // 8) * 8)
    print(f"Fixed sequence length for this run: {fixed_length} (longest example: {longest})")

    training_args = TrainingArguments(
        output_dir=str(args.out_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        report_to=[],
        fp16=(args.precision == "fp16" and device == "cuda"),
        bf16=(args.precision == "bf16" and device == "cuda"),
        remove_unused_columns=False,
        use_cpu=(device == "cpu"),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=PadCollator(tokenizer.pad_token_id, fixed_length),
    )

    trainer.train()
    model.save_pretrained(str(args.out_dir))
    tokenizer.save_pretrained(str(args.out_dir))
    print(f"Saved LoRA adapter to {args.out_dir}")


if __name__ == "__main__":
    main()
