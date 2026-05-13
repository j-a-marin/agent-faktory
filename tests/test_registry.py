"""Tests for the tool registry (Module 3a — schema + compliance).

Covers:
  1. ToolDefinition construction and serialization
  2. ComplianceMetadata auto-enforcement rules
  3. AuditEvent hashing and serialization
  4. Seed data (base DSL returns empty list; project defines its own tools)
  5. Compliance validator — the compile-time gate
     - HITL enforcement for RESTRICTED tools
     - Data flow isolation
     - Mixed classification warnings
     - Rate limit warnings
  6. End-to-end compliance scenarios
"""

import pytest
from datetime import datetime, timezone

from faktory.registry import (
    ToolDefinition,
    ToolType,
    DataClassification,
    ExecutionZone,
    AuthMethod,
    MCPTransport,
    ComplianceMetadata,
    AuditEvent,
    AuditOutcome,
    APISpec,
    EndpointSpec,
    MCPSpec,
    WorkflowSpec,
    ComplianceViolation,
    DataFlowViolation,
    get_builtin_tools,
    validate_lc_compliance,
    validate_lg_compliance,
)
from faktory.parser import parse_dsl


# ═══════════════════════════════════════════════════════════════════════
# ComplianceMetadata auto-enforcement
# ═══════════════════════════════════════════════════════════════════════

class TestComplianceMetadata:

    def test_restricted_auto_enforces(self):
        """RESTRICTED classification must force HITL, approval, and isolation."""
        c = ComplianceMetadata(data_classification=DataClassification.RESTRICTED)
        assert c.requires_hitl is True
        assert c.approval_required_for_deploy is True
        assert c.cross_tool_data_flow == "isolated"

    def test_confidential_defaults(self):
        c = ComplianceMetadata(data_classification=DataClassification.CONFIDENTIAL)
        assert c.cross_tool_data_flow == "same_classification"
        assert c.requires_hitl is False

    def test_public_defaults(self):
        c = ComplianceMetadata(data_classification=DataClassification.PUBLIC)
        assert c.requires_hitl is False
        assert c.approval_required_for_deploy is False
        assert c.cross_tool_data_flow == "unrestricted"

    def test_internal_defaults(self):
        c = ComplianceMetadata(data_classification=DataClassification.INTERNAL)
        assert c.cross_tool_data_flow == "unrestricted"

    def test_restricted_cannot_override_hitl(self):
        """Even if you pass requires_hitl=False, RESTRICTED overrides it."""
        c = ComplianceMetadata(
            data_classification=DataClassification.RESTRICTED,
            requires_hitl=False,
        )
        assert c.requires_hitl is True


# ═══════════════════════════════════════════════════════════════════════
# ToolDefinition
# ═══════════════════════════════════════════════════════════════════════

class TestToolDefinition:

    def test_qualified_name(self):
        td = ToolDefinition(tenant_id="tenant-a", name="lookup_tool")
        assert td.qualified_name == "tenant-a/lookup_tool"

    def test_qualified_versioned(self):
        td = ToolDefinition(tenant_id="tenant-a", name="lookup_tool", version="2.1.0")
        assert td.qualified_versioned == "tenant-a/lookup_tool@2.1.0"

    def test_to_dict_roundtrip(self):
        td = ToolDefinition(
            tenant_id="tenant-a",
            name="sensitive_action",
            version="1.0.0",
            description="Performs a sensitive operation requiring human approval",
            tool_type=ToolType.API,
            execution_zone=ExecutionZone.TENANT,
            compliance=ComplianceMetadata(
                data_classification=DataClassification.RESTRICTED,
                pii_fields=["customer_id", "account_number"],
                regulatory_scope=["internal-policy-v2"],
            ),
        )
        d = td.to_dict()
        assert d["tenant_id"] == "tenant-a"
        assert d["compliance"]["data_classification"] == "restricted"
        assert d["compliance"]["requires_hitl"] is True
        assert "customer_id" in d["compliance"]["pii_fields"]

    def test_api_tool(self):
        td = ToolDefinition(
            tenant_id="tenant-a",
            name="core_api",
            tool_type=ToolType.API,
            api_spec=APISpec(
                base_url="https://api.internal.example.com",
                auth_method=AuthMethod.MTLS,
                credential_ref="vault://tenant-a/mtls-cert",
                endpoints=[
                    EndpointSpec(
                        method="GET",
                        path="/v1/records/{record_id}",
                        description="Fetch a record by ID",
                        params={"record_id": "string"},
                    ),
                ],
            ),
        )
        assert td.api_spec is not None
        assert td.api_spec.auth_method == AuthMethod.MTLS
        assert len(td.api_spec.endpoints) == 1

    def test_mcp_tool(self):
        td = ToolDefinition(
            tenant_id="tenant-a",
            name="crm_connector",
            tool_type=ToolType.MCP,
            mcp_spec=MCPSpec(
                server_url="https://mcp.internal:8443",
                transport=MCPTransport.STREAMABLE_HTTP,
                credential_ref="vault://tenant-a/crm-oauth",
            ),
        )
        assert td.mcp_spec.transport == MCPTransport.STREAMABLE_HTTP

    def test_workflow_tool(self):
        td = ToolDefinition(
            tenant_id="tenant-a",
            name="review_workflow",
            tool_type=ToolType.WORKFLOW,
            workflow_spec=WorkflowSpec(
                dsl="# lg state(fetch, analyze, route) -> llm",
                input_schema={"type": "object", "properties": {"record_id": {"type": "string"}}},
            ),
        )
        assert td.workflow_spec.dsl.startswith("# lg")


# ═══════════════════════════════════════════════════════════════════════
# AuditEvent
# ═══════════════════════════════════════════════════════════════════════

class TestAuditEvent:

    def test_hash_content(self):
        h1 = AuditEvent.hash_content({"query": "lookup record 123"})
        h2 = AuditEvent.hash_content({"query": "lookup record 123"})
        h3 = AuditEvent.hash_content({"query": "different query"})
        assert h1 == h2           # Deterministic
        assert h1 != h3           # Different content → different hash
        assert len(h1) == 64      # SHA-256

    def test_to_dict(self):
        evt = AuditEvent(
            event_id="evt-001",
            tenant_id="tenant-a",
            tool_name="sensitive_action",
            outcome=AuditOutcome.HITL_APPROVED,
            approved_by="approver-42",
        )
        d = evt.to_dict()
        assert d["outcome"] == "hitl_approved"
        assert d["approved_by"] == "approver-42"


# ═══════════════════════════════════════════════════════════════════════
# Seed data
# ═══════════════════════════════════════════════════════════════════════

class TestSeedData:

    def test_builtin_returns_empty_in_base_dsl(self):
        """Base DSL ships no tools — projects define their own."""
        tools = get_builtin_tools()
        assert tools == []

    def test_custom_tenant_id_accepted(self):
        """get_builtin_tools accepts any tenant_id."""
        tools = get_builtin_tools(tenant_id="my-project")
        assert isinstance(tools, list)

    def test_project_tool_registration_pattern(self):
        """Verify the ToolDefinition registration pattern used in seed.py docstring."""
        td = ToolDefinition(
            tenant_id="my-project",
            name="my_tool",
            version="1.0.0",
            description="What this tool does and when to use it.",
            tool_type=ToolType.BUILTIN,
            execution_zone=ExecutionZone.PLATFORM,
            builtin_key="my_tool",
            compliance=ComplianceMetadata(
                data_classification=DataClassification.INTERNAL,
            ),
        )
        assert td.qualified_name == "my-project/my_tool"
        assert td.compliance.cross_tool_data_flow == "unrestricted"


# ═══════════════════════════════════════════════════════════════════════
# Compliance Validator — the compile-time gate
# ═══════════════════════════════════════════════════════════════════════

def _make_tool(
    name: str,
    classification: DataClassification = DataClassification.PUBLIC,
    **kwargs,
) -> ToolDefinition:
    """Helper to create test tool definitions."""
    return ToolDefinition(
        tenant_id="test-tenant",
        name=name,
        compliance=ComplianceMetadata(
            data_classification=classification,
            **kwargs,
        ),
    )


class TestComplianceValidator:

    # ── HITL enforcement (HARD RULE) ────────────────────────────

    def test_restricted_tool_without_hitl_fails(self):
        """RESTRICTED tool in DSL without hitl() → ComplianceViolation."""
        parsed = parse_dsl("# lc react(sensitive_action) -> mem -> api")
        tool_defs = {
            "sensitive_action": _make_tool("sensitive_action", DataClassification.RESTRICTED),
        }
        with pytest.raises(ComplianceViolation, match="hitl_required"):
            validate_lc_compliance(parsed, tool_defs)

    def test_restricted_tool_with_hitl_passes(self):
        """RESTRICTED tool with hitl() → passes."""
        parsed = parse_dsl("# lc react(sensitive_action) -> hitl(sensitive_action) -> mem -> api")
        tool_defs = {
            "sensitive_action": _make_tool("sensitive_action", DataClassification.RESTRICTED),
        }
        warnings = validate_lc_compliance(parsed, tool_defs)
        assert isinstance(warnings, list)

    def test_multiple_restricted_tools_need_all_hitl(self):
        """All RESTRICTED tools must be covered by hitl()."""
        parsed = parse_dsl("# lc react(action_a, action_b) -> hitl(action_a) -> mem")
        tool_defs = {
            "action_a": _make_tool("action_a", DataClassification.RESTRICTED),
            "action_b": _make_tool("action_b", DataClassification.RESTRICTED),
        }
        with pytest.raises(ComplianceViolation, match="action_b"):
            validate_lc_compliance(parsed, tool_defs)

    def test_confidential_hitl_not_required_by_default(self):
        """CONFIDENTIAL tools don't require HITL unless explicitly set."""
        parsed = parse_dsl("# lc react(data_lookup) -> mem")
        tool_defs = {
            "data_lookup": _make_tool("data_lookup", DataClassification.CONFIDENTIAL),
        }
        warnings = validate_lc_compliance(parsed, tool_defs)
        assert isinstance(warnings, list)

    def test_confidential_with_explicit_hitl_requirement(self):
        """CONFIDENTIAL tool with requires_hitl=True manually → enforced."""
        parsed = parse_dsl("# lc react(approval_action) -> mem")
        td = ToolDefinition(
            tenant_id="test-tenant",
            name="approval_action",
            compliance=ComplianceMetadata(
                data_classification=DataClassification.CONFIDENTIAL,
                requires_hitl=True,
            ),
        )
        with pytest.raises(ComplianceViolation, match="hitl_required"):
            validate_lc_compliance(parsed, {"approval_action": td})

    # ── Data flow isolation (HARD RULE) ─────────────────────────

    def test_isolated_tool_cannot_chain(self):
        """RESTRICTED/isolated tool cannot coexist with other tools."""
        parsed = parse_dsl("# lc react(sensitive_action, public_lookup) -> hitl(sensitive_action) -> mem")
        tool_defs = {
            "sensitive_action": _make_tool("sensitive_action", DataClassification.RESTRICTED),
            "public_lookup": _make_tool("public_lookup", DataClassification.PUBLIC),
        }
        with pytest.raises(DataFlowViolation, match="data_flow_isolation"):
            validate_lc_compliance(parsed, tool_defs)

    def test_isolated_tool_standalone_passes(self):
        """RESTRICTED tool alone is fine."""
        parsed = parse_dsl("# lc react(sensitive_action) -> hitl(sensitive_action) -> mem")
        tool_defs = {
            "sensitive_action": _make_tool("sensitive_action", DataClassification.RESTRICTED),
        }
        warnings = validate_lc_compliance(parsed, tool_defs)
        assert isinstance(warnings, list)

    # ── Mixed classification warnings (SOFT RULE) ───────────────

    def test_confidential_with_public_warns(self):
        """CONFIDENTIAL + PUBLIC tools → data leak warning."""
        parsed = parse_dsl("# lc react(data_lookup, public_lookup) -> mem")
        tool_defs = {
            "data_lookup": _make_tool("data_lookup", DataClassification.CONFIDENTIAL),
            "public_lookup": _make_tool("public_lookup", DataClassification.PUBLIC),
        }
        warnings = validate_lc_compliance(parsed, tool_defs)
        assert any("data flow warning" in w.lower() for w in warnings)

    def test_same_classification_no_warning(self):
        """PUBLIC + PUBLIC → no warning."""
        parsed = parse_dsl("# lc react(tool_a, tool_b) -> mem")
        tool_defs = {
            "tool_a": _make_tool("tool_a", DataClassification.PUBLIC),
            "tool_b": _make_tool("tool_b", DataClassification.PUBLIC),
        }
        warnings = validate_lc_compliance(parsed, tool_defs)
        assert len(warnings) == 0

    # ── Rate limit warnings ─────────────────────────────────────

    def test_rate_limited_tool_warns(self):
        parsed = parse_dsl("# lc react(monitored_tool) -> hitl(monitored_tool) -> mem")
        td = ToolDefinition(
            tenant_id="test-tenant",
            name="monitored_tool",
            compliance=ComplianceMetadata(
                data_classification=DataClassification.CONFIDENTIAL,
                requires_hitl=True,
                max_calls_per_session=10,
            ),
        )
        warnings = validate_lc_compliance(parsed, {"monitored_tool": td})
        assert any("max_calls_per_session=10" in w for w in warnings)

    # ── Unknown tools (no registry entry) ───────────────────────

    def test_unknown_tool_passes(self):
        """Tools not in registry skip compliance checks (placeholder tools)."""
        parsed = parse_dsl("# lc react(my_project_tool) -> mem")
        warnings = validate_lc_compliance(parsed, {})
        assert isinstance(warnings, list)

    # ── LangGraph (#lg) validation ──────────────────────────────

    def test_lg_hitl_enforcement(self):
        parsed = parse_dsl("# lg state(action_node, report_node) -> llm -> mem")
        tool_defs = {
            "action_node": _make_tool("action_node", DataClassification.RESTRICTED),
        }
        with pytest.raises(ComplianceViolation, match="hitl_required"):
            validate_lg_compliance(parsed, tool_defs)

    def test_lg_hitl_passes_when_declared(self):
        parsed = parse_dsl("# lg state(action_node, report_node) -> hitl(action_node) -> llm -> mem")
        tool_defs = {
            "action_node": _make_tool("action_node", DataClassification.RESTRICTED),
        }
        warnings = validate_lg_compliance(parsed, tool_defs)
        assert isinstance(warnings, list)


# ═══════════════════════════════════════════════════════════════════════
# End-to-end compliance scenarios
# ═══════════════════════════════════════════════════════════════════════

class TestComplianceScenarios:
    """End-to-end scenarios demonstrating the compliance framework."""

    def test_multi_tool_public_agent(self):
        """Multiple public tools — no warnings."""
        parsed = parse_dsl("# lc react(tool_a, tool_b, tool_c) -> mem -> api")
        tool_defs = {
            "tool_a": _make_tool("tool_a", DataClassification.PUBLIC),
            "tool_b": _make_tool("tool_b", DataClassification.PUBLIC),
            "tool_c": _make_tool("tool_c", DataClassification.PUBLIC),
        }
        warnings = validate_lc_compliance(parsed, tool_defs)
        assert warnings == []

    def test_sensitive_standalone_agent(self):
        """Sensitive tool alone with HITL — passes, no warnings."""
        parsed = parse_dsl("# lc react(restricted_tool) -> hitl(restricted_tool) -> mem -> api")
        tool_defs = {
            "restricted_tool": _make_tool(
                "restricted_tool",
                DataClassification.RESTRICTED,
                pii_fields=["user_id", "record_id"],
                regulatory_scope=["internal-policy"],
            ),
        }
        warnings = validate_lc_compliance(parsed, tool_defs)
        assert isinstance(warnings, list)

    def test_restricted_plus_other_blocked(self):
        """Restricted tool cannot be combined with any other tool."""
        parsed = parse_dsl("# lc react(restricted_tool, helper_tool) -> hitl(restricted_tool) -> mem")
        tool_defs = {
            "restricted_tool": _make_tool("restricted_tool", DataClassification.RESTRICTED),
            "helper_tool": _make_tool("helper_tool", DataClassification.PUBLIC),
        }
        with pytest.raises(DataFlowViolation):
            validate_lc_compliance(parsed, tool_defs)

    def test_approval_workflow_with_rate_limit(self):
        """Approval action with rate limit — passes but warns."""
        parsed = parse_dsl(
            "# lc react(lookup_tool, approval_action) -> hitl(approval_action) -> mem -> api"
        )
        tool_defs = {
            "lookup_tool": _make_tool("lookup_tool", DataClassification.CONFIDENTIAL),
            "approval_action": ToolDefinition(
                tenant_id="test-tenant",
                name="approval_action",
                compliance=ComplianceMetadata(
                    data_classification=DataClassification.CONFIDENTIAL,
                    requires_hitl=True,
                    max_calls_per_session=5,
                ),
            ),
        }
        warnings = validate_lc_compliance(parsed, tool_defs)
        assert any("max_calls_per_session=5" in w for w in warnings)
