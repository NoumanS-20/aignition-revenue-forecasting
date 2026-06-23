from app.llm.provider import EchoProvider
from app.llm import claude_insights as ci


def test_build_prompt_includes_numbers():
    diag = {"horizon_days": 30, "total_revenue_p50": 12345.0,
            "blended_roas_p50": 2.5, "scenario": {"applied": False},
            "series": [{"channel": "google", "saturation": "has_headroom"}]}
    system, user = ci.build_prompt(diag)
    assert "12345" in user and "roas" in user.lower()


def test_generate_insights_offline():
    diag = {"horizon_days": 30, "total_revenue_p50": 100.0, "blended_roas_p50": 2.0,
            "scenario": {"applied": False}, "series": []}
    out = ci.generate_insights(diag, EchoProvider())
    assert set(out) == {"narrative", "risks", "recommendations"}
    assert isinstance(out["risks"], list)
