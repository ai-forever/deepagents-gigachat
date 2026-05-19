"""Tests for the GigaChat harness profile package."""

from __future__ import annotations

from importlib.metadata import entry_points
from typing import Any

from deepagents.profiles.harness.harness_profiles import _get_harness_profile

from deepagents_gigachat import (
    ToolContractMiddleware,
    build_system_prompt,
    harness_profile,
    register_harness,
)


def test_public_api_exports_register_harness() -> None:
    assert register_harness is harness_profile.register_harness
    assert build_system_prompt("external_runtime")


def test_entry_point_is_declared() -> None:
    eps = entry_points(group="deepagents.harness_profiles")

    assert any(
        ep.name == "gigachat" and ep.value == "deepagents_gigachat:register_harness"
        for ep in eps
    )


def test_register_harness_registers_gigachat_profile() -> None:
    harness_profile.register_harness()

    profile = _get_harness_profile("gigachat:GigaChat-3-Ultra")

    assert profile is not None
    assert "write_file" in profile.tool_description_overrides
    assert "never start with '/'" in profile.tool_description_overrides["write_file"]


def test_register_harness_uses_both_provider_aliases(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    def fake_register_harness_profile(provider: str, profile: Any) -> None:
        captured[provider] = profile

    monkeypatch.setattr(
        harness_profile,
        "register_harness_profile",
        fake_register_harness_profile,
    )

    harness_profile.register_harness()

    assert {"gigachat", "giga"} == set(captured)
    profile = captured["gigachat"]
    assert captured["giga"] is profile
    assert "Never start with `/`" in profile.base_system_prompt
    assert "single-quoted heredoc" in profile.base_system_prompt
    assert "relative path" in profile.tool_description_overrides["write_file"]


def test_register_harness_can_use_external_runtime_profile(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    def fake_register_harness_profile(provider: str, profile: Any) -> None:
        captured[provider] = profile

    monkeypatch.setattr(
        harness_profile,
        "register_harness_profile",
        fake_register_harness_profile,
    )

    harness_profile.register_harness(
        profile_variant="external_runtime",
        tool_contract="Visible tools: runtime_read, runtime_write, runtime_answer.",
    )

    profile = captured["gigachat"]
    assert "External runtime tools" in profile.base_system_prompt
    assert "read_file once" not in profile.base_system_prompt
    assert "write_file" not in profile.tool_description_overrides
    assert any(isinstance(m, ToolContractMiddleware) for m in profile.extra_middleware)
