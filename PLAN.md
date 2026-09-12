# Mini LLM Tool-Planner — Project Plan

Drafted 2026-09-12. Start a fresh session in this repo and hand it this file for full context.

## Goal

Build a small, locally fine-tuned LLM with a **custom-built inference engine** to
replace OpenAI GPT-5 in the "Tool Planner" role of the `aws-final` fraud-detection
project's agentic AI service. Two payoffs from one build:

1. A standalone resume project — the inference engine internals (KV-cache,
   batching, constrained decoding) are the real differentiator, not the model itself.
2. A real integration — proof it works inside an actual deployed system, with
   real accuracy/latency/cost numbers against GPT-5.

## Constraints

- **$0 budget.** No paid APIs, no cloud GPU rental, no paid datasets. Every
  phase below is designed to cost nothing but electricity and time.
- **Hardware is uncertain.** Primary machine is an M5 MacBook Pro (16GB unified
  memory) available until roughly the first week of October 2026, after which
  it may need to be returned. Fallback machines: a MacBook Air M4 (16GB, 8-core
  GPU) or an HP Omen (RTX 3050, **4GB VRAM** — the tightest constraint of the
  three, i7 12th gen, 16GB system RAM). The plan is sized for the worst case
  (Omen) so a forced hardware switch mid-project doesn't break anything.
- **Target: done by early October 2026** (~3.5 weeks from the draft date).
  Phase 3 (the inference engine) is the long pole — front-load Phases 1-2 so
  there's real time left for it.

## Toolchain decision

**PyTorch + Hugging Face `transformers` + `peft` (LoRA)** — not MLX. MLX is
Apple-Silicon-only and won't run on the Omen's NVIDIA GPU; PyTorch runs via the
MPS backend on any Mac and via CUDA on the Omen, so one codebase survives a
hardware change. Quantization is the only thing that forks by platform:

- **On a Mac** (16GB unified memory, roomy): fp16 LoRA, no quantization needed.
- **On the Omen** (4GB VRAM, tight): 4-bit QLoRA via `bitsandbytes` — this is
  exactly the scenario QLoRA was designed for.

Same training script either way, precision picked by a config flag.

## Target model

**Qwen2.5-1.5B-Instruct** (fallback: Llama-3.2-1B-Instruct). Small enough to
comfortably QLoRA-fine-tune on the Omen's 4GB VRAM (the binding constraint),
good structured-output behavior, and the task itself uses short sequences
(a JSON feature vector in, a JSON tool-plan out) so memory pressure stays low
regardless of which machine is running it.

## The task being replaced

File: `aws-final/backend/src/services/agentic_ai.service.js`.

The "Tool Planner" phase takes transaction features (`amount`, `deviceId`,
`ipAddress`, `beneficiaryId`, `location`, `time`, `userId`) and decides which
of 5 read-only tools to run before a fraud decision is made:

```
query_fraud_history, query_beneficiary_trust, query_behavioral_norms,
query_graph_risk, query_fraud_policies
```

Output contract: `{ "selectedTools": [...], "reason": "..." }` (valid JSON,
enforced today via `response_format: { type: "json_object" }` on the OpenAI
call). Currently calls `gpt-5`; when `OPENAI_API_KEY` is unset it falls back
to a **hardcoded heuristic** already in that file (high amount / suspicious
IP / shared device → expanded tool set, else minimal set). That heuristic is
not a placeholder to replace — it's real, load-bearing logic we reuse in
Phase 1 below.

## Phase 1 — Free synthetic training data

Pure Python, no GPU, no model, no API calls. A generator that:

- Produces realistic random transaction feature vectors (varying amount,
  device/IP/beneficiary patterns including the "suspicious" keyword patterns
  the real heuristic already checks for, time of day, location).
- Labels each vector using the **same deterministic heuristic already in
  `agentic_ai.service.js`** — not an invented rule, the actual logic GPT-5 is
  currently instructed to approximate.
- Produces thousands of `(features, {selectedTools, reason})` pairs in
  minutes, for $0.

Output: a JSONL training file, plus a held-out eval split.

## Phase 2 — LoRA fine-tune, locally

- Load the base model via `transformers`.
- Fine-tune a LoRA adapter via `peft` on the Phase 1 dataset: features JSON in,
  tool-plan JSON out.
- fp16 on a Mac; 4-bit QLoRA (`bitsandbytes`) on the Omen. Same script, one
  config flag.
- Expect a few iterations to get generation reliably valid before Phase 3's
  constrained decoding is even layered on top.

## Phase 3 — Custom inference engine (the actual differentiator)

Not `model.generate()` as a black box. Build:

- **Manual KV-cache management** — own the cache lifecycle instead of letting
  a high-level API hide it.
- **Batching** — serve multiple transactions' tool-plan requests concurrently.
- **Grammar/schema-constrained decoding, implemented from scratch** — mask
  logits at each generation step so the model is *structurally incapable* of
  emitting invalid JSON or an unknown tool name. This is the standout,
  interview-defensible feature: most people fine-tune and hope; this makes
  correctness structural.
- **Benchmarks**: latency, throughput, memory footprint; constrained vs.
  unconstrained decoding; quantized vs. fp16. These numbers are the resume
  payload.

## Phase 4 — Integrate into `aws-final`

- Serve the fine-tuned model behind a small local HTTP server.
- Add it as a new service in `aws-final`'s existing `docker-compose.yml` (same
  internal network the backend/ml-service already share — no new
  infra concept, just another compose service).
- Add a third branch to `agentic_ai.service.js`'s existing
  `client ? generateToolPlan(gpt5) : heuristic` logic:
  `localModel ? generateToolPlan(local) : ...`. Additive, doesn't disturb the
  existing OpenAI/heuristic paths.
- Real before/after comparison: tool-selection accuracy against the heuristic
  ground truth, latency, and the concrete "$0 marginal cost per decision"
  argument vs. paying per-token for GPT-5.
- If time is short near the deadline, this phase can stay minimal (it needs to
  *work and be measurable*, not be production-polished) — the resume story
  doesn't depend on this being deployed to the live EC2 box, just proven to
  work.

## Repo layout

This repo holds the model/inference-engine project itself (Phases 1-3) as a
standalone, documented piece of work. Phase 4's integration is a small,
separate change made directly in the `aws-final` repo (one new compose
service, one new branch in one file) — not a merge of the two repos.

## Next step

Start Phase 1: the synthetic data generator. Everything in it is deterministic
Python with no dependencies on model choice or hardware, so it's safe to build
first regardless of which machine ends up being used.
