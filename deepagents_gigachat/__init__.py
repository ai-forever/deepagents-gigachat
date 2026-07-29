"""DeepAgents harness profile for GigaChat."""

from __future__ import annotations

from deepagents_gigachat.harness_profile import (
    DeterministicOutputMiddleware,
    LoopBreakerMiddleware,
    ShellSafetyMiddleware,
    SpecificationAuditMiddleware,
    ThinkToolMiddleware,
    ToolContractMiddleware,
    get_initial_workspace_files,
    get_workspace_path,
    register_harness,
    set_workspace_path,
)
from deepagents_gigachat.prompts import build_system_prompt

__all__ = [
    "DeterministicOutputMiddleware",
    "LoopBreakerMiddleware",
    "ShellSafetyMiddleware",
    "SpecificationAuditMiddleware",
    "ThinkToolMiddleware",
    "ToolContractMiddleware",
    "build_system_prompt",
    "get_initial_workspace_files",
    "get_workspace_path",
    "register_harness",
    "set_workspace_path",
]
