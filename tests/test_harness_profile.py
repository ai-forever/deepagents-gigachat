"""Tests for the GigaChat harness profile package."""

from __future__ import annotations

from importlib.metadata import entry_points
from types import SimpleNamespace
from typing import Any

from deepagents.backends.filesystem import FilesystemBackend
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.profiles.harness.harness_profiles import _get_harness_profile

from deepagents_gigachat import harness_profile, register_harness
from deepagents_gigachat.harness_profile import VirtualFilesystemRootMiddleware


def test_public_api_exports_register_harness() -> None:
    assert register_harness is harness_profile.register_harness


def test_entry_point_is_declared() -> None:
    eps = entry_points(group="deepagents.harness_profiles")

    assert any(
        ep.name == "gigachat" and ep.value == "deepagents_gigachat:register_harness"
        for ep in eps
    )


def test_register_harness_matches_gigachat_model_specs() -> None:
    harness_profile.register_harness()

    profile = _get_harness_profile("gigachat:GigaChat-3-Ultra")

    assert profile is not None
    assert "write_file" in profile.tool_description_overrides
    assert any(
        type(middleware).__name__ == "VirtualFilesystemRootMiddleware"
        for middleware in profile.materialize_extra_middleware()
    )


def test_register_harness_builds_expected_profile(monkeypatch: Any) -> None:
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
    assert "Hard Tool Rules" in profile.base_system_prompt
    assert "Refactor Workflow" in profile.base_system_prompt
    assert "host OS filesystem root" in profile.base_system_prompt
    assert "never embed multiline text" in profile.base_system_prompt
    assert {"ls", "read_file", "write_file", "glob", "grep", "edit_file", "execute"}.issubset(
        profile.tool_description_overrides
    )
    assert "absolute virtual path" in profile.tool_description_overrides["write_file"]
    assert "'/game_of_life.py'" in profile.tool_description_overrides["write_file"]
    assert "single-quoted heredoc" in profile.tool_description_overrides["execute"]
    assert "unexpected EOF" in profile.tool_description_overrides["execute"]
    assert any(
        type(middleware).__name__ == "ThinkToolMiddleware"
        for middleware in profile.extra_middleware
    )
    assert any(
        type(middleware).__name__ == "VirtualFilesystemRootMiddleware"
        for middleware in profile.extra_middleware
    )


def test_virtual_filesystem_root_middleware_anchors_absolute_paths(tmp_path: Any) -> None:
    backend = FilesystemBackend(root_dir=tmp_path, virtual_mode=False)
    filesystem_middleware = FilesystemMiddleware(backend=backend)
    write_tool = next(tool for tool in filesystem_middleware.tools if tool.name == "write_file")
    request = SimpleNamespace(tool_call={"name": "write_file"}, tool=write_tool)

    VirtualFilesystemRootMiddleware._enable_virtual_mode_for_filesystem_tool(request)
    result = backend.write("/mandelbrot_ascii.py", "print('mandelbrot')\n")

    assert backend.virtual_mode is True
    assert result.error is None
    assert (tmp_path / "mandelbrot_ascii.py").read_text() == "print('mandelbrot')\n"
