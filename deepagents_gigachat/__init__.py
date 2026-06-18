"""DeepAgents harness profile for GigaChat."""

from __future__ import annotations

from deepagents_gigachat.harness_profile import (
    LoopBreakerMiddleware,
    ShellSafetyMiddleware,
    ThinkToolMiddleware,
    ToolContractMiddleware,
    get_workspace_path,
    register_harness,
    set_workspace_path,
)
from deepagents_gigachat.prompts import build_system_prompt

__all__ = [
    "LoopBreakerMiddleware",
    "ShellSafetyMiddleware",
    "ThinkToolMiddleware",
    "ToolContractMiddleware",
    "build_system_prompt",
    "get_workspace_path",
    "register_harness",
    "set_workspace_path",
]
