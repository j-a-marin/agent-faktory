"""Seed tool definitions — project-specific tools registered here.

This module is intentionally empty in the base DSL.

In your project, define tools here using ToolDefinition:

    from faktory.registry.models import (
        ToolDefinition, ToolType, DataClassification,
        ExecutionZone, ComplianceMetadata,
    )

    def get_builtin_tools(tenant_id: str = "__platform__") -> list[ToolDefinition]:
        return [
            ToolDefinition(
                tenant_id=tenant_id,
                name="my_tool",
                version="1.0.0",
                description="What this tool does and when to use it.",
                tool_type=ToolType.BUILTIN,
                execution_zone=ExecutionZone.PLATFORM,
                builtin_key="my_tool",
                compliance=ComplianceMetadata(
                    data_classification=DataClassification.INTERNAL,
                ),
            ),
        ]

Tool names here correspond to DSL names used in `compile_agent()`:
    compile_agent("# lc react(my_tool) -> hitl(my_tool) -> mem -> api")
"""

from __future__ import annotations

from .models import ToolDefinition


def get_builtin_tools(tenant_id: str = "__platform__") -> list[ToolDefinition]:
    """Return project-defined tool definitions.

    Override this in your project to register domain-specific tools.
    Returns an empty list in the base DSL.
    """
    return []
