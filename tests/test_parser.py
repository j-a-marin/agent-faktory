"""Tests for the DSL parser.

Covers:
  - All #lc patterns: react, sql, sub, mcp
  - All #lc modifiers: mem, hitl, ctx, rt, cmd, inj, sum, trim,
    dyn, dynt, dynm, strv, strm, strc, out, ls, api, vercel
  - All #lg edge patterns: par, send, sendmap, agg, cond, cmd, sub, reduce
  - All #lg modifiers: llm, hitl, mem, ctx, ls, api, vercel
  - Serialization, edge cases, whitespace tolerance
"""

import pytest
from faktory.parser import parse_dsl, parse_lc_dsl, parse_lg_dsl
from faktory.parser.dsl_parser import DSLParseError


# ═══════════════════════════════════════════════════════════════════════
# LangChain (#lc) — Patterns
# ═══════════════════════════════════════════════════════════════════════

class TestLCPatterns:

    def test_react_pattern(self):
        ast = parse_dsl("# lc react(tool_a, tool_b) -> mem -> api")
        assert ast.pattern == "react"
        assert ast.pattern_key == "react_agent"
        assert ast.tools == ["tool_a", "tool_b"]

    def test_sql_pattern(self):
        ast = parse_dsl("# lc sql(execute_sql) -> ctx -> mem")
        assert ast.pattern == "sql"
        assert ast.pattern_key == "sql_agent"
        assert ast.tools == ["execute_sql"]

    def test_mcp_pattern(self):
        ast = parse_dsl("# lc mcp(filesystem, git) -> mem -> api")
        assert ast.pattern == "mcp"
        assert ast.pattern_key == "mcp_integration"
        assert ast.tools == ["filesystem", "git"]

    def test_sub_pattern(self):
        ast = parse_dsl("# lc sub(research, write) -> mem")
        assert ast.pattern == "sub"
        assert ast.pattern_key == "subagent"

    def test_pattern_no_tools(self):
        ast = parse_lc_dsl("react")
        assert ast.pattern == "react"
        assert ast.tools == []


# ═══════════════════════════════════════════════════════════════════════
# LangChain (#lc) — Modifiers
# ═══════════════════════════════════════════════════════════════════════

class TestLCModifiers:

    def test_mem(self):
        ast = parse_dsl("# lc react(tool_a) -> mem")
        assert ast.has_memory is True

    def test_api(self):
        ast = parse_dsl("# lc react(tool_a) -> api")
        assert ast.has_api is True

    def test_vercel(self):
        ast = parse_dsl("# lc react(tool_a) -> vercel")
        assert ast.has_vercel is True

    def test_ls(self):
        ast = parse_dsl("# lc react(tool_a) -> ls -> mem -> api")
        assert ast.has_ls is True

    def test_hitl_all(self):
        ast = parse_dsl("# lc react(tool_a) -> hitl -> mem")
        assert ast.has_hitl is True
        assert ast.hitl_tools == []

    def test_hitl_selective(self):
        ast = parse_dsl("# lc react(tool_a, tool_b) -> hitl(tool_a) -> mem")
        assert ast.has_hitl is True
        assert ast.hitl_tools == ["tool_a"]

    def test_hitl_multi(self):
        ast = parse_dsl("# lc react(tool_a, tool_b, tool_c) -> hitl(tool_a, tool_b) -> mem")
        assert ast.hitl_tools == ["tool_a", "tool_b"]

    def test_ctx(self):
        ast = parse_dsl("# lc react(tool_a) -> ctx -> mem")
        assert ast.has_ctx is True

    def test_rt_triggers_ctx(self):
        ast = parse_dsl("# lc react(tool_a) -> rt -> mem")
        assert ast.has_ctx is True

    def test_sum(self):
        ast = parse_dsl("# lc react(tool_a) -> sum -> mem")
        assert ast.has_sum is True

    def test_trim(self):
        ast = parse_dsl("# lc react(tool_a) -> trim -> mem")
        assert ast.has_trim is True

    def test_dyn(self):
        ast = parse_dsl("# lc react(tool_a) -> dyn -> mem")
        assert ast.has_dyn is True

    def test_dynt(self):
        ast = parse_dsl("# lc react(tool_a) -> dynt -> mem")
        assert ast.has_dynt is True

    def test_dynm(self):
        ast = parse_dsl("# lc react(tool_a) -> dynm -> mem")
        assert ast.has_dynm is True

    def test_cmd(self):
        ast = parse_dsl("# lc react(tool_a) -> cmd -> mem")
        assert ast.has_cmd is True

    def test_inj(self):
        ast = parse_dsl("# lc react(tool_a) -> inj -> mem")
        assert ast.has_inj is True

    def test_stream_values(self):
        ast = parse_dsl("# lc react(tool_a) -> strv -> api")
        assert ast.stream_mode == "values"

    def test_stream_messages(self):
        ast = parse_dsl("# lc react(tool_a) -> strm -> api")
        assert ast.stream_mode == "messages"

    def test_stream_custom(self):
        ast = parse_dsl("# lc react(tool_a) -> strc -> api")
        assert ast.stream_mode == "custom"

    def test_out_with_schema(self):
        ast = parse_dsl("# lc react(tool_a) -> out(DecisionReport) -> api")
        assert ast.has_out is True
        assert ast.out_schema == "DecisionReport"

    def test_out_default_schema(self):
        ast = parse_dsl("# lc react(tool_a) -> out -> api")
        assert ast.has_out is True
        assert ast.out_schema == "OutputSchema"


# ═══════════════════════════════════════════════════════════════════════
# LangChain (#lc) — Combinations + Serialization
# ═══════════════════════════════════════════════════════════════════════

class TestLCCombinations:

    def test_full_stack(self):
        ast = parse_dsl(
            "# lc react(tool_a, tool_b, tool_c) -> hitl(tool_c) -> sum -> ctx -> ls -> mem -> api -> vercel"
        )
        assert ast.tools == ["tool_a", "tool_b", "tool_c"]
        assert ast.hitl_tools == ["tool_c"]
        assert ast.has_sum is True
        assert ast.has_ctx is True
        assert ast.has_ls is True
        assert ast.has_memory is True
        assert ast.has_api is True
        assert ast.has_vercel is True

    def test_hitl_plus_dyn(self):
        """HITL with dynamic prompt — the HITL money shot pattern."""
        ast = parse_dsl("# lc react(processor) -> dyn -> hitl(processor) -> mem -> api")
        assert ast.has_hitl is True
        assert ast.has_dyn is True
        assert ast.hitl_tools == ["processor"]

    def test_to_dict(self):
        ast = parse_dsl("# lc react(tool_a) -> mem -> api")
        d = ast.to_dict()
        assert d["framework"] == "lc"
        assert d["pattern"] == "react"
        assert d["tools"] == ["tool_a"]
        assert len(d["modifiers"]) == 2


# ═══════════════════════════════════════════════════════════════════════
# LangGraph (#lg) — Edge Patterns
# ═══════════════════════════════════════════════════════════════════════

class TestLGEdgePatterns:

    def test_sequential(self):
        ast = parse_dsl("# lg state(a, b, c) -> llm -> mem")
        assert ast.graph_type == "state"
        assert ast.nodes == ["a", "b", "c"]
        assert ast.has_llm is True
        assert ast.has_memory is True

    def test_static_parallel(self):
        ast = parse_dsl("# lg state(worker_a, worker_b, aggregator) -> par(worker_a, worker_b) -> agg(aggregator)")
        assert ast.has_par is True
        assert ast.par_nodes == ["worker_a", "worker_b"]
        assert ast.aggregate_node == "aggregator"

    def test_known_fanout(self):
        ast = parse_dsl("# lg state(router, branch_a, branch_b, combiner) -> send(branch_a, branch_b) -> agg(combiner)")
        assert ast.has_send is True
        assert ast.send_targets == ["branch_a", "branch_b"]
        assert ast.aggregate_node == "combiner"

    def test_sendmap_dynamic(self):
        ast = parse_dsl("# lg state(dispatcher, worker, aggregator) -> sendmap(worker) -> agg(aggregator)")
        assert ast.has_sendmap is True
        assert ast.sendmap_target == "worker"
        assert ast.aggregate_node == "aggregator"

    def test_command_routing(self):
        ast = parse_dsl("# lg state(a, b, c) -> cmd")
        assert ast.has_cmd is True

    def test_conditional_router(self):
        ast = parse_dsl("# lg state(router, handler_a, handler_b) -> cond(route_fn)")
        assert ast.has_cond is True
        assert ast.cond_func == "route_fn"

    def test_subgraph(self):
        ast = parse_dsl("# lg state(main, sub_result) -> sub(child_graph)")
        assert ast.subgraphs == ["child_graph"]

    def test_reduce_add(self):
        ast = parse_dsl("# lg state(a, b) -> reduce(add) -> mem")
        assert ast.has_reduce is True
        assert ast.reduce_op == "add"

    def test_reduce_last(self):
        ast = parse_dsl("# lg state(a, b) -> reduce(last)")
        assert ast.reduce_op == "last"

    def test_msg_graph_type(self):
        ast = parse_dsl("# lg msg(a, b) -> llm")
        assert ast.graph_type == "msg"
        assert ast.nodes == ["a", "b"]


# ═══════════════════════════════════════════════════════════════════════
# LangGraph (#lg) — Modifiers + Combinations
# ═══════════════════════════════════════════════════════════════════════

class TestLGModifiers:

    def test_hitl_single_node(self):
        ast = parse_dsl("# lg state(ingest, process, cleanup) -> hitl(process) -> mem")
        assert ast.has_hitl is True
        assert ast.hitl_nodes == ["process"]

    def test_hitl_multi_node(self):
        ast = parse_dsl("# lg state(a, b, c) -> hitl(a, b) -> mem")
        assert ast.hitl_nodes == ["a", "b"]

    def test_full_deployment(self):
        ast = parse_dsl("# lg state(a, b) -> llm -> ls -> mem -> api -> vercel")
        assert ast.has_llm is True
        assert ast.has_ls is True
        assert ast.has_memory is True
        assert ast.has_api is True
        assert ast.has_vercel is True

    def test_parallel_with_deployment(self):
        ast = parse_dsl(
            "# lg state(fetch, analyze, report) -> par(fetch, analyze) -> agg(report) -> mem -> api"
        )
        assert ast.has_par is True
        assert ast.par_nodes == ["fetch", "analyze"]
        assert ast.aggregate_node == "report"
        assert ast.has_memory is True
        assert ast.has_api is True

    def test_to_dict(self):
        ast = parse_dsl("# lg state(a, b) -> par(a) -> agg(b) -> mem")
        d = ast.to_dict()
        assert d["framework"] == "lg"
        assert d["par_nodes"] == ["a"]
        assert d["aggregate_node"] == "b"

    def test_direct_call(self):
        ast = parse_lg_dsl("state(x, y) -> llm")
        assert ast.nodes == ["x", "y"]
        assert ast.has_llm is True


# ═══════════════════════════════════════════════════════════════════════
# Edge Cases
# ═══════════════════════════════════════════════════════════════════════

class TestEdgeCases:

    def test_whitespace_tolerance(self):
        ast = parse_dsl("#  lc  react( tool_a , tool_b )  ->  mem  ->  api")
        assert ast.pattern == "react"
        assert ast.tools == ["tool_a", "tool_b"]

    def test_no_prefix_lc(self):
        ast = parse_lc_dsl("react(tool_a) -> mem")
        assert ast.pattern == "react"
        assert ast.has_memory is True

    def test_invalid_format_raises(self):
        with pytest.raises(DSLParseError):
            parse_dsl("this is not a DSL string")

    def test_invalid_format_no_prefix_raises(self):
        with pytest.raises(DSLParseError):
            parse_dsl("react(tool_a) -> mem")
