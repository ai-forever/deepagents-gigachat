"""Tests for the pi-like benchmark tool facade."""

from __future__ import annotations

from deepagents_gigachat.pi_tools import build_pi_like_tools


def test_pi_like_write_overwrites_existing_file(tmp_path) -> None:
    (tmp_path / "a.txt").write_text("old\n", encoding="utf-8")
    tools = {tool.name: tool for tool in build_pi_like_tools(tmp_path)}

    result = tools["write"].invoke({"path": "a.txt", "content": "new\n"})

    assert "wrote" in result
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "new\n"


def test_pi_like_edit_applies_batch_against_original_file(tmp_path) -> None:
    (tmp_path / "a.txt").write_text("one\ntwo\nthree\n", encoding="utf-8")
    tools = {tool.name: tool for tool in build_pi_like_tools(tmp_path)}

    result = tools["edit"].invoke(
        {
            "path": "a.txt",
            "edits": [
                {"oldText": "one", "newText": "1"},
                {"oldText": "three", "newText": "3"},
            ],
        }
    )

    assert "replaced 2 block" in result
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "1\ntwo\n3\n"


def test_pi_like_edit_errors_are_tool_observations(tmp_path) -> None:
    (tmp_path / "a.txt").write_text("same\nsame\n", encoding="utf-8")
    tools = {tool.name: tool for tool in build_pi_like_tools(tmp_path)}

    result = tools["edit"].invoke(
        {
            "path": "a.txt",
            "edits": [{"oldText": "same", "newText": "once"}],
        }
    )

    assert result.startswith("Error:")
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "same\nsame\n"


def test_pi_like_bash_returns_output_and_status(tmp_path) -> None:
    tools = {tool.name: tool for tool in build_pi_like_tools(tmp_path)}

    result = tools["bash"].invoke({"command": "printf hi"})

    assert result == "hi\nexit=0"
