"""Tests for the GigaChat harness profile package."""

from __future__ import annotations

from importlib.metadata import entry_points
from typing import Any

from deepagents.profiles.harness.harness_profiles import _get_harness_profile

from deepagents_gigachat import harness_profile, register_harness


def test_public_api_exports_register_harness() -> None:
    assert register_harness is harness_profile.register_harness


def test_public_api_prefers_runtime_entrypoints() -> None:
    import deepagents_gigachat as package

    assert "invoke_routed" in package.__all__
    assert "build_deep_agent" in package.__all__
    assert "register_harness" in package.__all__
    assert "RoutingInput" not in package.__all__
    assert "build_routing_input" not in package.__all__
    assert "route_task" not in package.__all__
    assert not hasattr(package, "RoutingInput")
    assert not hasattr(package, "build_routing_input")
    assert not hasattr(package, "route_task")


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
    assert "NEW file only" in profile.tool_description_overrides["write_file"]
    assert "full-file rewrites" in profile.tool_description_overrides["edit_file"]


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
    assert "write_file` only for paths that do not exist yet" in profile.base_system_prompt
    assert "single-quoted heredoc" in profile.base_system_prompt
    assert "already exists" in profile.tool_description_overrides["write_file"]
    assert {type(m).__name__ for m in profile.extra_middleware} == {
        "ThinkToolMiddleware"
    }


def test_register_harness_pi_lite_profile(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    def fake_register_harness_profile(provider: str, profile: Any) -> None:
        captured[provider] = profile

    monkeypatch.setenv("DEEPAGENTS_GIGACHAT_PROFILE", "pi-lite")
    monkeypatch.setattr(
        harness_profile,
        "register_harness_profile",
        fake_register_harness_profile,
    )

    harness_profile.register_harness()

    profile = captured["gigachat"]
    assert captured["giga"] is profile
    assert "fewest reliable tool calls" in profile.base_system_prompt
    assert {type(m).__name__ for m in profile.extra_middleware} == {
        "PiLiteLoopMiddleware"
    }
    assert {"task", "write_todos"} <= profile.excluded_tools
    assert {"TodoListMiddleware", "SummarizationMiddleware"} <= (
        profile.excluded_middleware
    )
    assert profile.general_purpose_subagent.enabled is False


def test_register_harness_hybrid_profile(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    def fake_register_harness_profile(provider: str, profile: Any) -> None:
        captured[provider] = profile

    monkeypatch.setenv("DEEPAGENTS_GIGACHAT_PROFILE", "hybrid")
    monkeypatch.setattr(
        harness_profile,
        "register_harness_profile",
        fake_register_harness_profile,
    )

    harness_profile.register_harness()

    profile = captured["gigachat"]
    assert "write_file` only for paths that do not exist yet" in profile.base_system_prompt
    assert {type(m).__name__ for m in profile.extra_middleware} == {
        "HybridLoopMiddleware",
        "ThinkToolMiddleware",
    }
    assert "literal shell substitutions" in profile.tool_description_overrides[
        "execute"
    ]
    assert "read_file" not in profile.excluded_tools


def test_register_harness_adaptive_profile(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    def fake_register_harness_profile(provider: str, profile: Any) -> None:
        captured[provider] = profile

    monkeypatch.setenv("DEEPAGENTS_GIGACHAT_PROFILE", "adaptive")
    monkeypatch.setattr(
        harness_profile,
        "register_harness_profile",
        fake_register_harness_profile,
    )

    harness_profile.register_harness()

    profile = captured["gigachat"]
    assert "write_file` only for paths that do not exist yet" in profile.base_system_prompt
    assert {type(m).__name__ for m in profile.extra_middleware} == {
        "AdaptiveLoopMiddleware",
        "ThinkToolMiddleware",
    }
    assert "literal shell substitutions" in profile.tool_description_overrides[
        "execute"
    ]
    assert "read_file" not in profile.excluded_tools


def test_register_harness_adaptive_tools_profile(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    def fake_register_harness_profile(provider: str, profile: Any) -> None:
        captured[provider] = profile

    monkeypatch.setenv("DEEPAGENTS_GIGACHAT_PROFILE", "adaptive-tools")
    monkeypatch.setattr(
        harness_profile,
        "register_harness_profile",
        fake_register_harness_profile,
    )

    harness_profile.register_harness()

    profile = captured["gigachat"]
    assert {type(m).__name__ for m in profile.extra_middleware} == {
        "AdaptiveToolRoutingMiddleware",
        "ThinkToolMiddleware",
    }
    assert "literal shell substitutions" in profile.tool_description_overrides[
        "execute"
    ]
    assert "read_file" not in profile.excluded_tools


def test_adaptive_route_overrides_are_generic() -> None:
    data = harness_profile._adaptive_route_overrides(  # noqa: SLF001
        "Inner-join users.csv and orders.csv by user_id"
    )
    assert data == ()

    search = harness_profile._adaptive_route_overrides(  # noqa: SLF001
        "Find the .py file under project/ with the most lines"
    )
    assert any("search postprocessing" in override for override in search)

    filesystem = harness_profile._adaptive_route_overrides(  # noqa: SLF001
        "Move src/old/* to src/new/"
    )
    assert any("filesystem commit" in override for override in filesystem)

    code = harness_profile._adaptive_route_overrides(  # noqa: SLF001
        "Implement Stack so pytest passes"
    )
    assert code == ()


def test_filter_tools_by_name_supports_dict_tool_shapes() -> None:
    tools = [
        {"name": "read_file"},
        {"function": {"name": "execute"}},
        {"name": "write_file"},
    ]

    assert harness_profile._filter_tools_by_name(tools, {"execute"}) == [  # noqa: SLF001
        {"function": {"name": "execute"}}
    ]


def test_register_harness_pi_tools_profile(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    def fake_register_harness_profile(provider: str, profile: Any) -> None:
        captured[provider] = profile

    monkeypatch.setenv("DEEPAGENTS_GIGACHAT_PROFILE", "pi-tools")
    monkeypatch.setattr(
        harness_profile,
        "register_harness_profile",
        fake_register_harness_profile,
    )

    harness_profile.register_harness()

    profile = captured["gigachat"]
    assert "pi-like tools" in profile.base_system_prompt
    assert {type(m).__name__ for m in profile.extra_middleware} == {
        "PiToolsLoopMiddleware"
    }
    assert {"read_file", "write_file", "edit_file", "execute"} <= (
        profile.excluded_tools
    )
    assert {"task", "write_todos"} <= profile.excluded_tools
    assert profile.general_purpose_subagent.enabled is False
