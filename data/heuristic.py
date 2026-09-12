"""Python port of the Tool Planner heuristic from aws-final's agentic_ai.service.js.

Source: aws-final/backend/src/services/agentic_ai.service.js, the heuristic
fallback branch of augmentTransactionContext() used when OPENAI_API_KEY is
unset. This is the real, load-bearing logic GPT-5 is currently instructed to
approximate (PLAN.md, "The task being replaced") -- Phase 1's synthetic labels
come from calling this function directly, not from an invented rule set.
"""

MINIMAL_TOOLS = ["query_behavioral_norms", "query_beneficiary_trust"]
EXPANDED_TOOLS = [
    "query_behavioral_norms",
    "query_beneficiary_trust",
    "query_graph_risk",
    "query_fraud_policies",
    "query_fraud_history",
]


def plan_tools(features: dict) -> dict:
    """Mirrors the isHighRiskOrAnomalous branch in agentic_ai.service.js exactly."""
    amount = features.get("amount") or 0
    device_id = features.get("deviceId") or ""
    ip_address = features.get("ipAddress") or ""

    is_high_risk_or_anomalous = (
        amount > 1000
        or ip_address == "192.168.1.100"
        or "shared" in device_id
    )

    selected_tools = EXPANDED_TOOLS if is_high_risk_or_anomalous else MINIMAL_TOOLS
    reason = "Offline Heuristic Planner: " + (
        "High amount / suspicious anomaly detected -> expanded tools"
        if is_high_risk_or_anomalous
        else "Normal transaction -> minimal tools"
    )
    return {"selectedTools": selected_tools, "reason": reason}
