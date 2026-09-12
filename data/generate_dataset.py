#!/usr/bin/env python3
"""Phase 1: synthetic training data generator for the Tool Planner LLM.

Generates realistic random transaction feature vectors and labels each one
with the exact heuristic already in aws-final's agentic_ai.service.js (ported
in heuristic.py) -- not an invented rule set. Writes JSONL: one
{"features": ..., "output": {"selectedTools": [...], "reason": "..."}}
record per line, split into train/eval with no overlap.

Usage:
    python3 generate_dataset.py --n-train 4000 --n-eval 500
"""
import argparse
import json
import random
from pathlib import Path

from heuristic import plan_tools

LOCATIONS = ["IN", "US", "UK", "AU", "SG", "RU", "KP", "FR", "DE", "BR", "CN", "ZA", "MX"]

# Keyword-bearing templates mirror the substring checks used elsewhere in
# agentic_ai.service.js (device/beneficiary/user scoring) even though only
# "shared" devices and the fixed IP affect the tool-planner heuristic itself --
# the model needs to see (and learn to ignore) these look-alike keywords too.
DEVICE_TEMPLATES = [
    "device_{n}",
    "shared_device_{n}",
    "trusted_device_{n}",
    "known_device_{n}",
    "flagged_device_{n}",
    "rooted_device_{n}",
    "emulator_{n}",
    "dev_mac_1",
]

BENEFICIARY_TEMPLATES = [
    "beneficiary_{n}",
    "flagged_beneficiary_{n}",
    "suspect_{n}",
    "mule_account_{n}",
    "blacklist_{n}",
    "sanctioned_{n}",
]

USER_TEMPLATES = [
    "user_{n}",
    "hospital_user_{n}",
    "admin_{n}",
    "officer_{n}",
    "owner_{n}",
    "compromised_user_{n}",
]


def random_ip(rng: random.Random) -> str:
    if rng.random() < 0.08:
        return "192.168.1.100"
    return ".".join(str(rng.randint(0, 255)) for _ in range(4))


def random_from_templates(rng: random.Random, templates: list[str]) -> str:
    template = rng.choice(templates)
    return template.format(n=rng.randint(1, 9999)) if "{n}" in template else template


def random_amount(rng: random.Random) -> float:
    # Weighted toward realistic small transactions, with dense mass around the
    # $1000 decision boundary and a tail of high-value extremes.
    bucket = rng.random()
    if bucket < 0.55:
        return round(rng.uniform(1, 999), 2)
    if bucket < 0.65:
        return round(rng.uniform(950, 1050), 2)
    if bucket < 0.9:
        return round(rng.uniform(1001, 15000), 2)
    return round(rng.uniform(15000, 200000), 2)


def random_features(rng: random.Random) -> dict:
    return {
        "amount": random_amount(rng),
        "deviceId": random_from_templates(rng, DEVICE_TEMPLATES),
        "ipAddress": random_ip(rng),
        "beneficiaryId": random_from_templates(rng, BENEFICIARY_TEMPLATES),
        "location": rng.choice(LOCATIONS),
        "time": rng.randint(0, 23),
        "userId": random_from_templates(rng, USER_TEMPLATES),
    }


def generate_records(n: int, rng: random.Random, seen: set) -> list[dict]:
    records = []
    while len(records) < n:
        features = random_features(rng)
        key = json.dumps(features, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        records.append({"features": features, "output": plan_tools(features)})
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-train", type=int, default=4000)
    parser.add_argument("--n-eval", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    seen: set = set()

    train = generate_records(args.n_train, rng, seen)
    eval_ = generate_records(args.n_eval, rng, seen)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "train.jsonl", train)
    write_jsonl(args.out_dir / "eval.jsonl", eval_)

    for name, records in [("train", train), ("eval", eval_)]:
        expanded = sum(1 for r in records if len(r["output"]["selectedTools"]) == 5)
        minimal = len(records) - expanded
        print(f"{name}: {len(records)} examples -> {expanded} expanded / {minimal} minimal")


if __name__ == "__main__":
    main()
