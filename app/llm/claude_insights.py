from __future__ import annotations
import json

SYSTEM = (
    "You are a senior e-commerce marketing analyst. You receive structured "
    "forecast diagnostics (already computed by a Bayesian model) and explain "
    "them. Never invent numbers; only interpret what is given. Respond with a "
    "JSON object with keys: narrative (string), risks (array of strings), "
    "recommendations (array of strings)."
)


def build_prompt(diagnostics: dict) -> tuple[str, str]:
    user = (
        "Here are the forecast diagnostics as JSON. Write a causal narrative "
        "explaining the expected revenue and blended ROAS, flag operational "
        "risks, and recommend budget actions based on per-channel saturation.\n\n"
        + json.dumps(diagnostics, indent=2)
    )
    return SYSTEM, user


def generate_insights(diagnostics: dict, provider) -> dict:
    system, user = build_prompt(diagnostics)
    raw = provider.complete(system, user)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {"narrative": raw, "risks": [], "recommendations": []}
    return {"narrative": data.get("narrative", ""),
            "risks": list(data.get("risks", [])),
            "recommendations": list(data.get("recommendations", []))}
