"""Tests for the GigaChat harness profile package."""

from __future__ import annotations

from importlib.metadata import entry_points
from typing import Any

from deepagents_gigachat import harness_profile, register_harness


def test_public_api_exports_register_harness() -> None:
    assert register_harness is harness_profile.register_harness


def test_entry_point_is_declared() -> None:
    eps = entry_points(group="deepagents.harness_profiles")

    assert any(
        ep.name == "gigachat" and ep.value == "deepagents_gigachat:register_harness"
        for ep in eps
    )


def test_register_harness_builds_expected_profile(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    def fake_register_harness_profile(provider: str, profile: Any) -> None:
        captured["provider"] = provider
        captured["profile"] = profile

    monkeypatch.setattr(
        harness_profile,
        "register_harness_profile",
        fake_register_harness_profile,
    )

    harness_profile.register_harness()

    assert captured["provider"] == "giga"

    profile = captured["profile"]
    assert "Hard Tool Rules" in profile.base_system_prompt
    assert "Refactor Workflow" in profile.base_system_prompt
    assert {"ls", "read_file", "grep", "edit_file"}.issubset(
        profile.tool_description_overrides
    )
    assert any(
        type(middleware).__name__ == "ThinkToolMiddleware"
        for middleware in profile.extra_middleware
    )
