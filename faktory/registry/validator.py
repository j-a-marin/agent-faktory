"""Compile-time compliance validator.

Called between parse and codegen. Reads tool classifications from
the registry and enforces rules before any code is generated.

If validation fails, compilation halts with a ComplianceViolation.
No code is generated. No agent is deployed.
"""

from __future__ import annotations

from faktory.parser.lc_parser import LCParsed
from faktory.parser.lg_parser import LGParsed
from .models import (
    ToolDefinition,
    DataClassification,
    ComplianceViolation,
    DataFlowViolation,
)


# Classification severity ordering for data flow checks
_CLASSIFICATION_RANK = {
    DataClassification.PUBLIC: 0,
    DataClassification.INTERNAL: 1,
    DataClassification.CONFIDENTIAL: 2,
    DataClassification.RESTRICTED: 3,
}


def validate_lc_compliance(
    parsed: LCParsed,
    tool_defs: dict[str, ToolDefinition],
) -> list[str]:
    """Validate a parsed #lc DSL against tool compliance rules.

    Args:
        parsed: Output of parse_lc_dsl()
        tool_defs: Map of tool_name → ToolDefinition (from registry lookup)

    Returns:
        List of warning strings (non-blocking).

    Raises:
        ComplianceViolation: If a hard rule is violated (blocks compilation).
        DataFlowViolation: If tools with incompatible classifications are chained.
    """
    warnings: list[str] = []

    _check_hitl_requirements(parsed.tools, parsed.hitl_tools, tool_defs)
    _check_deployment_approval(tool_defs)
    warnings.extend(_check_data_flow(parsed.tools, tool_defs))
    warnings.extend(_check_rate_limits(parsed.tools, tool_defs))

    return warnings


def validate_lg_compliance(
    parsed: LGParsed,
    tool_defs: dict[str, ToolDefinition],
) -> list[str]:
    """Validate a parsed #lg DSL against tool compliance rules.

    LG agents don't have tools in the DSL directly (tools are inside nodes),
    but we validate any tools referenced in the registry for this agent.
    """
    warnings: list[str] = []

    # LG HITL check: nodes listed in hitl() modifier
    hitl_nodes = parsed.hitl_nodes
    for tool_name, tool_def in tool_defs.items():
        if tool_def.compliance.requires_hitl and tool_name not in hitl_nodes:
            raise ComplianceViolation(
                tool_name=tool_name,
                rule="hitl_required",
                detail=(
                    f"Tool '{tool_name}' has classification "
                    f"'{tool_def.compliance.data_classification.value}' and requires "
                    f"human-in-the-loop. Add -> hitl({tool_name}) to DSL."
                ),
            )

    _check_deployment_approval(tool_defs)
    warnings.extend(_check_data_flow(list(tool_defs.keys()), tool_defs))

    return warnings


# ═══════════════════════════════════════════════════════════════════════
# Individual rule checks
# ═══════════════════════════════════════════════════════════════════════

def _check_hitl_requirements(
    tool_names: list[str],
    hitl_tools: list[str],
    tool_defs: dict[str, ToolDefinition],
) -> None:
    """HARD RULE: Tools with requires_hitl=True must be in hitl() modifier."""
    for tool_name in tool_names:
        tool_def = tool_defs.get(tool_name)
        if not tool_def:
            continue  # Unknown tool — no compliance metadata to enforce

        if tool_def.compliance.requires_hitl and tool_name not in hitl_tools:
            raise ComplianceViolation(
                tool_name=tool_name,
                rule="hitl_required",
                detail=(
                    f"Tool '{tool_name}' has classification "
                    f"'{tool_def.compliance.data_classification.value}' and requires "
                    f"human-in-the-loop approval. Add -> hitl({tool_name}) to your DSL."
                ),
            )


def _check_deployment_approval(
    tool_defs: dict[str, ToolDefinition],
) -> None:
    """HARD RULE: Tools requiring deployment approval must be signed off."""
    for tool_name, tool_def in tool_defs.items():
        if tool_def.compliance.approval_required_for_deploy and not tool_def.is_active:
            raise ComplianceViolation(
                tool_name=tool_name,
                rule="deployment_approval_required",
                detail=(
                    f"Tool '{tool_name}' requires compliance officer approval "
                    f"before deployment. Tool is currently inactive."
                ),
            )


def _check_data_flow(
    tool_names: list[str],
    tool_defs: dict[str, ToolDefinition],
) -> list[str]:
    """Check data flow compatibility between tools in the pipeline.

    HARD RULE: RESTRICTED tools with cross_tool_data_flow='isolated'
    cannot chain with any other tool.

    SOFT RULE (warning): CONFIDENTIAL tools chaining with PUBLIC tools
    may leak sensitive data — warn the tenant.
    """
    warnings: list[str] = []
    classified_tools = []

    for name in tool_names:
        td = tool_defs.get(name)
        if td:
            classified_tools.append((name, td))

    # Check isolation rules
    for name, td in classified_tools:
        if td.compliance.cross_tool_data_flow == "isolated" and len(tool_names) > 1:
            other_tools = [n for n in tool_names if n != name]
            raise DataFlowViolation(
                source_tool=name,
                target_tool=other_tools[0],
                detail=(
                    f"'{name}' has data_flow='isolated' (classification: "
                    f"{td.compliance.data_classification.value}). "
                    f"Cannot be combined with other tools in a single agent. "
                    f"Deploy as a standalone agent with -> hitl({name})."
                ),
            )

    # Check classification compatibility (warning only)
    if len(classified_tools) >= 2:
        max_rank = max(
            _CLASSIFICATION_RANK[td.compliance.data_classification]
            for _, td in classified_tools
        )
        min_rank = min(
            _CLASSIFICATION_RANK[td.compliance.data_classification]
            for _, td in classified_tools
        )

        if max_rank >= 2 and min_rank == 0:  # CONFIDENTIAL+ mixed with PUBLIC
            high_tools = [
                n for n, td in classified_tools
                if _CLASSIFICATION_RANK[td.compliance.data_classification] >= 2
            ]
            low_tools = [
                n for n, td in classified_tools
                if _CLASSIFICATION_RANK[td.compliance.data_classification] == 0
            ]
            warnings.append(
                f"Data flow warning: {high_tools} (confidential+) chained with "
                f"{low_tools} (public). LLM may leak sensitive context into "
                f"public tool calls. Consider separate agents or HITL gates."
            )

    return warnings


def _check_rate_limits(
    tool_names: list[str],
    tool_defs: dict[str, ToolDefinition],
) -> list[str]:
    """SOFT RULE: Warn if rate-limited tools are used without session limits."""
    warnings: list[str] = []

    for name in tool_names:
        td = tool_defs.get(name)
        if td and td.compliance.max_calls_per_session is not None:
            warnings.append(
                f"Tool '{name}' has max_calls_per_session="
                f"{td.compliance.max_calls_per_session}. "
                f"Runtime must enforce this limit."
            )

    return warnings
