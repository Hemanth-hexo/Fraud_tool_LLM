"""Prompt format shared by training and inference.

Mirrors the system/user prompt aws-final's agentic_ai.service.js sends to
GPT-5 for the Tool Planner phase, so the fine-tuned model is trained on (and
later served with) the same instruction the JS code already uses.
"""
import json

SYSTEM_PROMPT = """You are the Tool Planner for a Fraud Detection Agentic AI.
Your ONLY job is to evaluate the transaction context and determine which security tools must be executed to gather necessary evidence.
You do NOT make security decisions (approve/reject). You are strictly read-only.

Available tools:
query_fraud_history, query_beneficiary_trust, query_behavioral_norms, query_graph_risk, query_fraud_policies

Guidelines:
1. LOW-COMPLEXITY / LOW-VALUE: Only select basic tools (e.g. behavior, trust).
2. HIGH-RISK / ANOMALOUS (High value, suspicious IPs): Select advanced tools (e.g. graph, RAG, fraud).

Output valid JSON matching this schema exactly:
{"selectedTools": ["tool_name_1", "tool_name_2"], "reason": "Explain why these tools were selected based on the context."}"""


def build_messages(features: dict) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Generate a tool plan for this transaction: {json.dumps(features)}"},
    ]


def build_target(output: dict) -> str:
    return json.dumps(output)
