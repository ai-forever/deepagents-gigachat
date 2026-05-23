"""DeepAgents harness profile for GigaChat."""

from __future__ import annotations

from deepagents_gigachat.harness_profile import (
    LoopBreakerMiddleware,
    ShellSafetyMiddleware,
    ThinkToolMiddleware,
    ToolContractMiddleware,
    register_harness,
)
from deepagents_gigachat.prompts import build_system_prompt

__all__ = [
    "LoopBreakerMiddleware",
    "ShellSafetyMiddleware",
    "ThinkToolMiddleware",
    "ToolContractMiddleware",
    "build_system_prompt",
    "register_harness",
]
