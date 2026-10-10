"""HTTP wrapper around the custom inference engine: a small FastAPI service
exposing the fine-tuned tool-planner model over the network, so it can run
as a standalone container instead of only via local scripts.

Env vars (all optional, sensible defaults for local/Docker use):
    BASE_MODEL      -- HF model id (default: Qwen/Qwen2.5-1.5B-Instruct)
    ADAPTER_PATH    -- path to the LoRA adapter (default: checkpoints/lora-adapter-v2)
    MAX_NEW_TOKENS  -- generation budget per request (default: 100)
    PRECISION       -- "fp16" (default) or "fp32"
    USE_QUANTIZED   -- "1" to additionally int8-quantize on top of PRECISION, "0" (default)
    FAST_DECODING   -- "1" (default): engine/fast_planner.py -- grammar-forced tokens fed in one
                       forward pass, cached system-prompt prefix, ~4-8x faster on CPU with
                       identical plans (engine/verify_fast.py). "0": the original token-by-token
                       path with a model-written `reason`.
    TORCH_THREADS   -- optional intra-op thread count for CPU inference
    TOOL_PLANNER_SHARED_SECRET -- required, no default; callers must send it back as the
                                  X-Tool-Planner-Token header on POST /plan

Precision notes, both checked directly against the eval set rather than
assumed:
  - fp16 vs. fp32: 100% exact-match accuracy either way (30/30), fp16 at
    half the weight memory (3.1GB vs. 6.2GB). No real tradeoff here, so
    fp16 is the default -- it's what makes this comfortably fit on a
    memory-constrained instance instead of running a hair away from OOM.
  - int8 dynamic quantization is a different story: it drops exact-match
    accuracy from 100% to ~47%. LoRA fine-tuning applies small, delicate
    weight adjustments that aggressive int8 quantization of the merged
    weights washes out -- fp16's gentler precision cut doesn't have this
    problem. JSON stays 100% valid regardless of precision (that's a
    structural guarantee from the constrained decoder, not a statistical
    one), but *correct* tool selection is the actual point of this model,
    so int8 stays opt-in. Only set USE_QUANTIZED=1 if you've separately
    verified acceptable accuracy for your use case.
"""
import json
import os
import secrets
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "train"))
sys.path.insert(0, str(REPO_ROOT / "engine"))
from prompt import build_messages  # noqa: E402
from generate import ManualGenerator  # noqa: E402
from constrained import ConstrainedToolPlanDecoder  # noqa: E402
from quantize import quantize_for_serving  # noqa: E402
from fast_planner import FastToolPlanner, static_prefix_for  # noqa: E402

BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
ADAPTER_PATH = os.environ.get("ADAPTER_PATH", str(REPO_ROOT / "checkpoints" / "lora-adapter-v2"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "100"))
PRECISION = os.environ.get("PRECISION", "fp16")
USE_QUANTIZED = os.environ.get("USE_QUANTIZED", "0") == "1"
FAST_DECODING = os.environ.get("FAST_DECODING", "1") == "1"
# One inference at a time: on CPU, concurrent forward passes just split the same cores and make
# every request slower (and the engine's per-request KV caches are large). A request that can't
# start within QUEUE_WAIT_S gets 503 immediately, so the caller falls back to its deterministic
# plan instead of queueing behind work it will have stopped waiting for.
INFERENCE_LOCK = threading.Lock()
QUEUE_WAIT_S = float(os.environ.get("QUEUE_WAIT_S", "1.0"))
if os.environ.get("TORCH_THREADS"):
    torch.set_num_threads(int(os.environ["TORCH_THREADS"]))
TORCH_DTYPE = {"fp16": torch.float16, "fp32": torch.float32}[PRECISION]

# SECURITY (zero-trust internal boundary, matching aws-final's ml-api pattern): a
# security group is a network-layer control, not an application-layer one -- a
# compromised container or instance anywhere permitted on that path could still
# call this endpoint directly and pull a tool plan for arbitrary attacker-chosen
# features. Fail closed if unset rather than defaulting to "no auth" or a
# guessable default -- this service must not accept unauthenticated requests.
TOOL_PLANNER_SHARED_SECRET = os.environ.get("TOOL_PLANNER_SHARED_SECRET")
if not TOOL_PLANNER_SHARED_SECRET:
    raise RuntimeError(
        "FATAL SECURITY ERROR: TOOL_PLANNER_SHARED_SECRET must be set in the environment "
        "(no default is provided) -- this service must not accept unauthenticated requests."
    )


def require_valid_service_token(x_tool_planner_token: str | None = Header(default=None)):
    if not x_tool_planner_token or not secrets.compare_digest(x_tool_planner_token, TOOL_PLANNER_SHARED_SECRET):
        raise HTTPException(status_code=401, detail="Missing or invalid X-Tool-Planner-Token")


engine_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Loading base model {BASE_MODEL} + adapter {ADAPTER_PATH} ...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=TORCH_DTYPE).to(device)
    merged = PeftModel.from_pretrained(base, ADAPTER_PATH).merge_and_unload()
    merged.eval()

    # Grab this before quantizing: a quantized linear layer's `.weight` becomes
    # a method that unpacks the packed int8 tensor on call, not a plain tensor
    # attribute -- quantization doesn't change the vocab dimension anyway.
    vocab_size = merged.get_output_embeddings().weight.shape[0]

    quant_engine = None
    if USE_QUANTIZED:
        merged, quant_engine = quantize_for_serving(merged)

    engine_state["tokenizer"] = tokenizer
    engine_state["model"] = merged
    engine_state["generator"] = ManualGenerator(merged, tokenizer)
    if FAST_DECODING:
        engine_state["fast"] = FastToolPlanner(merged, tokenizer, static_prompt_prefix=static_prefix_for(tokenizer, build_messages))
        # Warm-up: the first forward passes pay one-off allocation/kernel-selection costs; pay
        # them at startup, not on the first real transaction.
        warm = tokenizer.apply_chat_template(build_messages({"amount": 100.0}), tokenize=False, add_generation_prompt=True)
        for _ in range(2):
            engine_state["fast"].plan(warm)
    engine_state["vocab_size"] = vocab_size
    engine_state["precision"] = PRECISION
    engine_state["device"] = device
    engine_state["quant_engine"] = quant_engine or "none"
    engine_state["engine"] = "fast-constrained" if FAST_DECODING else "token-by-token-constrained"
    print(f"Ready. Device: {device}, precision: {PRECISION}, quantization: {engine_state['quant_engine']}")
    yield
    engine_state.clear()


app = FastAPI(title="Fraud Tool-Planner LLM", lifespan=lifespan)


class TransactionFeatures(BaseModel):
    amount: float
    deviceId: str | None = None
    ipAddress: str | None = None
    beneficiaryId: str | None = None
    location: str | None = None
    time: int | None = None
    userId: str | None = None


class ToolPlanResponse(BaseModel):
    selectedTools: list[str]
    reason: str
    latency_ms: float
    engine: str | None = None
    forward_passes: int | None = None


@app.get("/health")
def health():
    return {
        "status": "ok" if engine_state else "loading",
        "device": engine_state.get("device", "not loaded"),
        "precision": engine_state.get("precision", "not loaded"),
        "quantization": engine_state.get("quant_engine", "not loaded"),
        "engine": engine_state.get("engine", "not loaded"),
    }


@app.post("/plan", response_model=ToolPlanResponse)
def plan(features: TransactionFeatures, _auth=Depends(require_valid_service_token)):
    tokenizer = engine_state["tokenizer"]
    generator: ManualGenerator = engine_state["generator"]

    prompt_text = tokenizer.apply_chat_template(
        build_messages(features.model_dump(exclude_none=True)), tokenize=False, add_generation_prompt=True
    )
    if FAST_DECODING:
        if not INFERENCE_LOCK.acquire(timeout=QUEUE_WAIT_S):
            raise HTTPException(status_code=503, detail="planner busy")
        try:
            result = engine_state["fast"].plan(prompt_text)
        finally:
            INFERENCE_LOCK.release()
        tools = result.selected_tools
        return ToolPlanResponse(
            selectedTools=tools,
            # Composed, not generated: a model-written sentence cost ~15 extra forward passes
            # and the trained model only ever parroted the heuristic's two reason strings.
            reason=f"Fine-tuned local planner selected {len(tools)} tool(s): {', '.join(tools)}",
            latency_ms=1000 * result.seconds,
            engine="fast-constrained",
            forward_passes=result.forward_passes,
        )

    prompt_len = len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
    decoder = ConstrainedToolPlanDecoder(
        tokenizer, prompt_len, engine_state["vocab_size"], max_new_tokens=MAX_NEW_TOKENS
    )

    t0 = time.perf_counter()
    result = generator.generate(prompt_text, max_new_tokens=MAX_NEW_TOKENS, logits_processor=decoder)
    latency_ms = 1000 * (time.perf_counter() - t0)

    try:
        parsed = json.loads(result.text)
    except json.JSONDecodeError as exc:
        # Should be unreachable -- constrained decoding guarantees valid JSON --
        # but surfacing a 500 with the raw text beats a silent bad response.
        raise HTTPException(status_code=500, detail=f"decoder produced invalid JSON: {result.text}") from exc

    return ToolPlanResponse(selectedTools=parsed["selectedTools"], reason=parsed["reason"], latency_ms=latency_ms,
                            engine="token-by-token-constrained", forward_passes=len(result.decode_seconds) + 1)
