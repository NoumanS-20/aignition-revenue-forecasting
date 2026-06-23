from __future__ import annotations
import json
import os
from typing import Protocol


class LLMProvider(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class EchoProvider:
    """Deterministic offline provider for tests/demos without a key."""

    def complete(self, system: str, user: str) -> str:
        return json.dumps({
            "narrative": "Forecast generated from model diagnostics.",
            "risks": ["Limited history may widen intervals."],
            "recommendations": ["Shift budget toward channels with headroom."],
        })


class ClaudeProvider:
    def __init__(self, model: str = "claude-opus-4-8"):
        from anthropic import Anthropic
        self.client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.model = model

    def complete(self, system: str, user: str) -> str:
        msg = self.client.messages.create(
            model=self.model, max_tokens=1200, system=system,
            messages=[{"role": "user", "content": user}])
        return msg.content[0].text
