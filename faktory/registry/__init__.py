"""Tool Registry — multi-tenant, compliance-aware tool management.

The registry wraps every tool (builtin, API, MCP, workflow) with:
  - Tenant isolation (namespace scoping)
  - Data classification (PUBLIC → RESTRICTED)
  - Compliance enforcement (HITL gates, data flow rules)
  - Audit event generation
  - Semantic search embeddings (Module 3b — coming next)
"""

from .models import (
    ToolDefinition,
    ToolType,
    DataClassification,
    ExecutionZone,
    AuthMethod,
    MCPTransport,
    AuditOutcome,
    APISpec,
    EndpointSpec,
    MCPSpec,
    WorkflowSpec,
    ComplianceMetadata,
    AuditEvent,
    ComplianceViolation,
    DataFlowViolation,
)
from .seed import get_builtin_tools
from .validator import validate_lc_compliance, validate_lg_compliance

__all__ = [
    "ToolDefinition",
    "ToolType",
    "DataClassification",
    "ExecutionZone",
    "AuthMethod",
    "MCPTransport",
    "AuditOutcome",
    "APISpec",
    "EndpointSpec",
    "MCPSpec",
    "WorkflowSpec",
    "ComplianceMetadata",
    "AuditEvent",
    "ComplianceViolation",
    "DataFlowViolation",
    "get_builtin_tools",
    "validate_lc_compliance",
    "validate_lg_compliance",
]
