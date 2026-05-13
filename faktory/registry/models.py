"""Tool Registry data models.

These are the core schemas for AGENT_FAKTORY's multi-tenant tool registry.
Every tool — builtin, API wrap, MCP server, or workflow — gets a ToolDefinition.
Every invocation produces an AuditEvent.

The registry is the compliance gate: the compiler reads classification from here
and enforces HITL, redaction, and data flow rules at compile time.

Design principles:
  - Compliance by construction, not bolted on
  - Two-zone architecture: platform (control) vs tenant (execution)
  - Tool definitions are metadata — credentials never stored here
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════════════════

class ToolType(str, Enum):
    """How the tool is implemented."""
    BUILTIN = "builtin"       # Ships with FAKTORY (the 7 seed tools)
    API = "api"               # Wraps tenant REST/GraphQL endpoint
    MCP = "mcp"               # Connects to MCP server
    WORKFLOW = "workflow"      # Compiled subgraph exposed as @tool


class DataClassification(str, Enum):
    """Data sensitivity level. Drives HITL, redaction, and data flow rules."""
    PUBLIC = "public"               # Non-sensitive, publicly available data
    INTERNAL = "internal"           # Internal docs, non-sensitive business data
    CONFIDENTIAL = "confidential"   # Customer PII, account data, sensitive records
    RESTRICTED = "restricted"       # Highest sensitivity: destructive actions, regulated data


class ExecutionZone(str, Enum):
    """Where the tool physically runs.

    Banks will never route internal traffic through our infra.
    Platform tools (builtins) can run anywhere.
    Tenant tools run inside the bank's perimeter.
    """
    PLATFORM = "platform"     # Runs on FAKTORY infra (builtins, public APIs)
    TENANT = "tenant"         # Runs inside bank's VPC / on-prem


class AuthMethod(str, Enum):
    """How the tool authenticates to its backend."""
    NONE = "none"             # Public endpoints
    API_KEY = "api_key"       # Header or query param
    OAUTH2 = "oauth2"        # OAuth 2.0 client credentials or auth code
    JWT = "jwt"              # Signed JWT bearer token
    MTLS = "mtls"            # Mutual TLS (common in banking)
    HMAC = "hmac"            # HMAC-signed requests


class MCPTransport(str, Enum):
    """MCP server transport protocol."""
    STDIO = "stdio"                    # Local subprocess
    SSE = "sse"                        # Server-sent events (legacy)
    STREAMABLE_HTTP = "streamable_http"  # Current MCP standard


class AuditOutcome(str, Enum):
    """Result of a tool invocation."""
    SUCCESS = "success"
    FAILURE = "failure"
    HITL_PENDING = "hitl_pending"
    HITL_APPROVED = "hitl_approved"
    HITL_REJECTED = "hitl_rejected"
    BLOCKED = "blocked"       # Compliance gate prevented execution


# ═══════════════════════════════════════════════════════════════════════
# API Spec (for type=api tools)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class EndpointSpec:
    """Single API endpoint within a tool."""
    method: str                        # GET, POST, PUT, DELETE
    path: str                          # /v1/orders/{order_id}
    description: str                   # Used for semantic search embedding
    params: dict[str, str] = field(default_factory=dict)    # param_name → type
    response_schema: Optional[dict] = None                   # JSON Schema
    max_response_bytes: int = 4096     # Truncation limit for LLM context


@dataclass
class APISpec:
    """REST/GraphQL API configuration for a tenant tool."""
    base_url: str                      # https://api.acme-bank.com
    auth_method: AuthMethod = AuthMethod.NONE
    credential_ref: Optional[str] = None  # Vault path (e.g. "vault://tenant-123/api-key")
    headers: dict[str, str] = field(default_factory=dict)   # Static headers
    endpoints: list[EndpointSpec] = field(default_factory=list)
    timeout_seconds: int = 10
    retry_policy: Optional[dict] = None  # {"max_retries": 3, "backoff": "exponential"}


# ═══════════════════════════════════════════════════════════════════════
# MCP Spec (for type=mcp tools)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class MCPSpec:
    """MCP server connection configuration."""
    server_url: str                    # For SSE/HTTP: https://mcp.bank.internal:8443
    transport: MCPTransport = MCPTransport.STREAMABLE_HTTP
    command: Optional[str] = None      # For stdio: the binary to exec
    args: list[str] = field(default_factory=list)  # For stdio: command args
    env: dict[str, str] = field(default_factory=dict)  # Env vars (non-secret)
    credential_ref: Optional[str] = None  # Vault path for auth


# ═══════════════════════════════════════════════════════════════════════
# Workflow Spec (for type=workflow tools — compiled subgraphs as tools)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class WorkflowSpec:
    """A mini-graph wrapped as a single tool interface.

    The DSL string gets compiled via the same pipeline:
    parse_dsl() → compile_agent() → subgraph code.
    The result is exposed as a @tool to parent agents.
    """
    dsl: str                           # Nested DSL, e.g. "# lg state(fetch, analyze, route) -> llm"
    input_schema: Optional[dict] = None   # JSON Schema for tool input
    output_schema: Optional[dict] = None  # JSON Schema for tool output


# ═══════════════════════════════════════════════════════════════════════
# Compliance Metadata
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ComplianceMetadata:
    """Regulatory and compliance constraints for a tool.

    The compiler reads these fields to enforce rules at build time.
    If a tool is RESTRICTED and the DSL omits hitl(), compilation fails.
    """
    data_classification: DataClassification = DataClassification.PUBLIC
    requires_hitl: bool = False        # Enforced at compile time
    pii_fields: list[str] = field(default_factory=list)  # Param names containing PII
    regulatory_scope: list[str] = field(default_factory=list)  # Domain-specific compliance tags
    approval_required_for_deploy: bool = False  # Needs compliance sign-off
    max_calls_per_session: Optional[int] = None  # Rate limit for sensitive ops
    data_retention_days: Optional[int] = None    # How long audit logs kept
    cross_tool_data_flow: Optional[str] = None   # "unrestricted" | "same_classification" | "isolated"

    def __post_init__(self):
        """Auto-enforce classification rules."""
        if self.data_classification == DataClassification.RESTRICTED:
            self.requires_hitl = True
            self.approval_required_for_deploy = True
            if not self.cross_tool_data_flow:
                self.cross_tool_data_flow = "isolated"
        elif self.data_classification == DataClassification.CONFIDENTIAL:
            if not self.cross_tool_data_flow:
                self.cross_tool_data_flow = "same_classification"
        else:
            if not self.cross_tool_data_flow:
                self.cross_tool_data_flow = "unrestricted"


# ═══════════════════════════════════════════════════════════════════════
# Tool Definition — the core registry record
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ToolDefinition:
    """Complete definition of a tool in the registry.

    This is the source of truth for what a tool is, how it connects,
    what compliance rules apply, and where it runs.

    The `description` field is critical — it gets embedded for semantic
    search in the two-stage K selection pipeline.
    """
    # Identity
    tenant_id: str                     # Namespace isolation
    name: str                          # Unique within tenant
    version: str = "1.0.0"            # Semver
    description: str = ""              # Embedded for semantic search

    # Implementation
    tool_type: ToolType = ToolType.BUILTIN
    execution_zone: ExecutionZone = ExecutionZone.PLATFORM

    # Type-specific specs (exactly one should be populated)
    api_spec: Optional[APISpec] = None
    mcp_spec: Optional[MCPSpec] = None
    workflow_spec: Optional[WorkflowSpec] = None
    builtin_key: Optional[str] = None  # For BUILTIN type: key used in project tool registry

    # Compliance
    compliance: ComplianceMetadata = field(default_factory=ComplianceMetadata)

    # Lifecycle
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: str = ""               # User or system that registered it
    is_active: bool = True             # Soft delete / disable
    deprecated_by: Optional[str] = None  # Version that replaces this one

    # Embedding (populated by registry on insert/update)
    embedding: Optional[list[float]] = None

    @property
    def qualified_name(self) -> str:
        """Tenant-scoped unique identifier."""
        return f"{self.tenant_id}/{self.name}"

    @property
    def qualified_versioned(self) -> str:
        """Fully qualified with version."""
        return f"{self.tenant_id}/{self.name}@{self.version}"

    def to_dict(self) -> dict:
        """Serialize for API responses / storage (excludes embedding)."""
        d = {
            "tenant_id": self.tenant_id,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "tool_type": self.tool_type.value,
            "execution_zone": self.execution_zone.value,
            "compliance": {
                "data_classification": self.compliance.data_classification.value,
                "requires_hitl": self.compliance.requires_hitl,
                "pii_fields": self.compliance.pii_fields,
                "regulatory_scope": self.compliance.regulatory_scope,
                "approval_required_for_deploy": self.compliance.approval_required_for_deploy,
                "max_calls_per_session": self.compliance.max_calls_per_session,
                "cross_tool_data_flow": self.compliance.cross_tool_data_flow,
            },
            "qualified_name": self.qualified_name,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
        if self.builtin_key:
            d["builtin_key"] = self.builtin_key
        return d


# ═══════════════════════════════════════════════════════════════════════
# Audit Event — every tool invocation produces one
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class AuditEvent:
    """Immutable audit record for a single tool invocation.

    For CONFIDENTIAL/RESTRICTED tools, inputs and outputs are
    hashed, not stored. The decision_chain provides explainability.

    These are write-once, append-only. Never mutated after creation.
    """
    # Identity
    event_id: str                      # UUID
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Context
    tenant_id: str = ""
    agent_id: str = ""
    agent_version: str = ""
    session_id: str = ""               # Thread / conversation ID

    # Tool
    tool_name: str = ""
    tool_version: str = ""
    tool_type: ToolType = ToolType.BUILTIN
    data_classification: DataClassification = DataClassification.PUBLIC

    # Execution
    outcome: AuditOutcome = AuditOutcome.SUCCESS
    latency_ms: int = 0
    error_message: Optional[str] = None

    # Content (redaction depends on classification)
    input_hash: str = ""               # SHA-256 of serialized input
    output_hash: str = ""              # SHA-256 of serialized output
    input_redacted: Optional[dict] = None   # Param names + types, values stripped
    output_summary: Optional[str] = None    # Truncated, redacted output summary

    # Explainability
    decision_chain: list[str] = field(default_factory=list)  # LLM reasoning steps
    tool_selection_reason: str = ""    # Why this tool was chosen

    # HITL
    approved_by: Optional[str] = None  # Human approver ID
    approval_timestamp: Optional[datetime] = None
    rejection_reason: Optional[str] = None

    @staticmethod
    def hash_content(content: any) -> str:
        """SHA-256 hash for audit trail without storing raw data."""
        serialized = json.dumps(content, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()

    def to_dict(self) -> dict:
        """Serialize for storage / API responses."""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "tenant_id": self.tenant_id,
            "agent_id": self.agent_id,
            "agent_version": self.agent_version,
            "session_id": self.session_id,
            "tool_name": self.tool_name,
            "tool_version": self.tool_version,
            "tool_type": self.tool_type.value,
            "data_classification": self.data_classification.value,
            "outcome": self.outcome.value,
            "latency_ms": self.latency_ms,
            "error_message": self.error_message,
            "input_hash": self.input_hash,
            "output_hash": self.output_hash,
            "input_redacted": self.input_redacted,
            "output_summary": self.output_summary,
            "decision_chain": self.decision_chain,
            "tool_selection_reason": self.tool_selection_reason,
            "approved_by": self.approved_by,
            "approval_timestamp": self.approval_timestamp.isoformat() if self.approval_timestamp else None,
            "rejection_reason": self.rejection_reason,
        }


# ═══════════════════════════════════════════════════════════════════════
# Compile-time compliance validation errors
# ═══════════════════════════════════════════════════════════════════════

class ComplianceViolation(Exception):
    """Raised when a DSL composition violates compliance rules.

    This is the enforcement mechanism: if a RESTRICTED tool appears
    in a DSL without hitl(), compilation fails here — not at runtime.
    """
    def __init__(self, tool_name: str, rule: str, detail: str = ""):
        self.tool_name = tool_name
        self.rule = rule
        self.detail = detail
        msg = f"Compliance violation for tool '{tool_name}': {rule}"
        if detail:
            msg += f" — {detail}"
        super().__init__(msg)


class DataFlowViolation(ComplianceViolation):
    """Raised when tools with incompatible classifications are chained."""
    def __init__(self, source_tool: str, target_tool: str, detail: str = ""):
        super().__init__(
            tool_name=source_tool,
            rule="data_flow_isolation",
            detail=f"Cannot chain '{source_tool}' → '{target_tool}': {detail}",
        )
