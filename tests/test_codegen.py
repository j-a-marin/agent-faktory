"""Tests for the code generator.

compile_agent(dsl) → valid Python source string.

Covers:
  - #lc → create_agent() with correct middleware stack
  - #lg → StateGraph() with correct edge wiring
  - Placeholder tool stubs (all tools are project-defined)
  - Deployment modifiers: api, vercel, ls
  - Framework boundary invariant: #lc never uses StateGraph, #lg never uses create_agent
"""

import ast as python_ast
import pytest

from faktory.codegen import compile_agent


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def assert_valid_python(code: str):
    try:
        python_ast.parse(code)
    except SyntaxError as e:
        pytest.fail(f"Syntax error at line {e.lineno}: {e.msg}\n\n{code}")


def assert_contains(code: str, *fragments: str):
    for frag in fragments:
        assert frag in code, f"Expected {frag!r} in generated code"


def assert_not_contains(code: str, *fragments: str):
    for frag in fragments:
        assert frag not in code, f"Did not expect {frag!r} in generated code"


# ═══════════════════════════════════════════════════════════════════════
# LangChain (#lc) — create_agent() generation
# ═══════════════════════════════════════════════════════════════════════

class TestLCCodegen:

    def test_basic_react(self):
        code = compile_agent("# lc react(tool_a, tool_b) -> mem -> api")
        assert_valid_python(code)
        assert_contains(code, "create_agent", "TODO: implement tool_a", "TODO: implement tool_b")
        assert_contains(code, "FastAPI", '@app.post("/invoke")')
        assert_not_contains(code, "StateGraph")

    def test_placeholder_stubs(self):
        """All tools produce placeholder stubs — tools are project-defined."""
        code = compile_agent("# lc react(processor_a, processor_b) -> mem -> api")
        assert_valid_python(code)
        assert_contains(code, "TODO: implement processor_a")
        assert_contains(code, "TODO: implement processor_b")

    def test_hitl_selective(self):
        code = compile_agent("# lc react(tool_a) -> hitl(tool_a) -> mem -> api")
        assert_valid_python(code)
        assert_contains(code, "HumanInTheLoopMiddleware", "interrupt_on=")

    def test_hitl_plus_dyn(self):
        """HITL with dynamic prompt — the core HITL pattern."""
        code = compile_agent("# lc react(processor) -> dyn -> hitl(processor) -> mem -> api")
        assert_valid_python(code)
        assert_contains(code, "HumanInTheLoopMiddleware", "@dynamic_prompt")

    def test_summarization(self):
        code = compile_agent("# lc react(tool_a) -> sum -> mem -> api")
        assert_valid_python(code)
        assert_contains(code, "SummarizationMiddleware")

    def test_trim(self):
        code = compile_agent("# lc react(tool_a) -> trim -> mem")
        assert_valid_python(code)
        assert_contains(code, "TrimMessagesMiddleware")

    def test_context_schema(self):
        code = compile_agent("# lc react(tool_a) -> ctx -> mem -> api")
        assert_valid_python(code)
        assert_contains(code, "RuntimeContext", "context_schema=RuntimeContext")

    def test_dynamic_prompt(self):
        code = compile_agent("# lc react(tool_a) -> dyn -> mem")
        assert_valid_python(code)
        assert_contains(code, "@dynamic_prompt", "dynamic_system_prompt")

    def test_output_schema_named(self):
        code = compile_agent("# lc react(tool_a) -> out(DecisionReport) -> api")
        assert_valid_python(code)
        assert_contains(code, "class DecisionReport(BaseModel):", "response_format=DecisionReport")

    def test_output_schema_default(self):
        code = compile_agent("# lc react(tool_a) -> out -> api")
        assert_valid_python(code)
        assert_contains(code, "class OutputSchema(BaseModel):")

    def test_vercel(self):
        code = compile_agent("# lc react(tool_a) -> vercel")
        assert_valid_python(code)
        assert_contains(code, "StreamingResponse", '/api/chat')
        assert_not_contains(code, '/invoke')

    def test_langsmith(self):
        code = compile_agent("# lc react(tool_a) -> ls -> mem -> api")
        assert_valid_python(code)
        assert_contains(code, "LANGSMITH_TRACING")


# ═══════════════════════════════════════════════════════════════════════
# LangGraph (#lg) — StateGraph() generation
# ═══════════════════════════════════════════════════════════════════════

class TestLGCodegen:

    def test_sequential(self):
        code = compile_agent("# lg state(a, b, c) -> llm -> mem")
        assert_valid_python(code)
        assert_contains(code, "StateGraph", "ChatOpenAI")
        assert_not_contains(code, "create_agent")

    def test_static_parallel(self):
        code = compile_agent(
            "# lg state(fetch, analyze, report) -> par(fetch, analyze) -> agg(report) -> ls -> mem -> api"
        )
        assert_valid_python(code)
        assert_contains(code, 'graph.add_edge(START, "fetch")')
        assert_contains(code, 'graph.add_edge(START, "analyze")')
        assert_contains(code, 'graph.add_edge("fetch", "report")')
        assert_contains(code, 'graph.add_edge("report", END)')
        assert_contains(code, "LANGSMITH_TRACING", "FastAPI")

    def test_hitl_interrupt(self):
        code = compile_agent("# lg state(ingest, process, cleanup) -> hitl(process) -> mem")
        assert_valid_python(code)
        assert_contains(code, "interrupt(")
        assert_contains(code, 'interrupt_before=["process"]')

    def test_llm_nodes(self):
        code = compile_agent("# lg state(planner, writer) -> llm")
        assert_valid_python(code)
        assert_contains(code, "ChatOpenAI", "llm.invoke(", "SystemMessage")

    def test_reduce_last(self):
        code = compile_agent("# lg state(a, b) -> reduce(last)")
        assert_valid_python(code)
        assert_contains(code, "lambda a, b: b")

    def test_sendmap(self):
        code = compile_agent("# lg state(dispatcher, worker, aggregator) -> sendmap(worker) -> agg(aggregator)")
        assert_valid_python(code)
        assert_contains(code, "Send", "dispatch_to_workers")

    def test_command_routing(self):
        code = compile_agent("# lg state(a, b, c) -> cmd")
        assert_valid_python(code)
        assert_contains(code, "Command")

    def test_msg_graph_type(self):
        code = compile_agent("# lg msg(a, b) -> llm")
        assert_valid_python(code)
        assert_contains(code, "StateGraph")

    def test_fastapi(self):
        code = compile_agent("# lg state(a, b) -> llm -> api")
        assert_valid_python(code)
        assert_contains(code, "FastAPI", '@api.post("/invoke")')

    def test_vercel(self):
        code = compile_agent("# lg state(a, b) -> llm -> vercel")
        assert_valid_python(code)
        assert_contains(code, "StreamingResponse", '/api/chat')

    def test_subgraph(self):
        code = compile_agent("# lg state(main, sub_result) -> sub(child_graph)")
        assert_valid_python(code)


# ═══════════════════════════════════════════════════════════════════════
# Framework boundary — the critical architectural invariant
# ═══════════════════════════════════════════════════════════════════════

class TestFrameworkBoundary:
    """#lc → create_agent always. #lg → StateGraph always. No crossover."""

    def test_lc_never_uses_stategraph(self):
        for dsl in [
            "# lc react(tool_a) -> mem",
            "# lc react(tool_a, tool_b) -> hitl(tool_a) -> sum -> ctx -> ls -> mem -> api -> vercel",
            "# lc sql(execute_sql) -> ctx -> mem",
            "# lc mcp(fs) -> mem",
        ]:
            code = compile_agent(dsl)
            assert "StateGraph" not in code, f"#lc must not use StateGraph: {dsl}"
            assert "create_agent" in code, f"#lc must use create_agent: {dsl}"

    def test_lg_never_uses_create_agent(self):
        for dsl in [
            "# lg state(a, b) -> llm",
            "# lg state(a, b, c) -> par(a, b) -> agg(c) -> mem -> api",
            "# lg state(a, b) -> cmd",
            "# lg state(a, b) -> sendmap(b) -> agg(a)",
        ]:
            code = compile_agent(dsl)
            assert "create_agent" not in code, f"#lg must not use create_agent: {dsl}"
            assert "StateGraph" in code, f"#lg must use StateGraph: {dsl}"
