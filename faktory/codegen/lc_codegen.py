"""LangChain (#lc) code generator.

Transforms LCParsed AST into a complete runnable Python file
using LangChain v1's create_agent (high-level, no graph wiring).

"""

from __future__ import annotations

from faktory.parser.lc_parser import LCParsed


def generate_lc_code(parsed: LCParsed) -> str:
    """Generate a complete Python file from an LCParsed AST.

    Args:
        parsed: Output of parse_lc_dsl()

    Returns:
        Complete Python source code as a string.
    """
    sections: list[str] = []

    sections.append(_gen_header(parsed))
    sections.append(_gen_imports(parsed))

    if parsed.has_ls:
        sections.append(_gen_langsmith_setup())

    if parsed.has_ctx:
        sections.append(_gen_context_schema())

    if parsed.has_out:
        sections.append(_gen_output_schema(parsed.out_schema))

    if parsed.has_dyn:
        sections.append(_gen_dynamic_prompt())

    tool_var_names, tool_name_map = _gen_tools(parsed, sections)

    sections.append(_gen_create_agent(parsed, tool_var_names, tool_name_map))
    sections.append(_gen_usage(parsed))

    if parsed.has_api and not parsed.has_vercel:
        sections.append(_gen_fastapi(parsed))

    if parsed.has_vercel:
        sections.append(_gen_vercel(parsed))

    return "\n".join(sections)


# ═══════════════════════════════════════════════════════════════════════
# Section generators
# ═══════════════════════════════════════════════════════════════════════

def _gen_header(parsed: LCParsed) -> str:
    return f"# {parsed.raw}\n"


def _gen_imports(parsed: LCParsed) -> str:
    lines = [
        "from langchain.agents import create_agent",
        "from langchain.tools import tool",
    ]

    if parsed.has_hitl:
        lines.append("from langchain.agents.middleware import HumanInTheLoopMiddleware")
        lines.append("from langgraph.types import Command")

    if parsed.has_sum:
        lines.append("from langchain.agents.middleware import SummarizationMiddleware")

    if parsed.has_trim:
        lines.append("from langchain.agents.middleware import TrimMessagesMiddleware")

    if parsed.has_ctx:
        lines.append("from dataclasses import dataclass")

    if parsed.has_dyn:
        lines.append("from langchain.agents.middleware.types import ModelRequest, dynamic_prompt")

    if parsed.has_out:
        lines.append("from pydantic import BaseModel")

    if parsed.has_ls:
        lines.append("import os")

    if parsed.has_api or parsed.has_vercel:
        lines.append("from fastapi import FastAPI")
        if "from pydantic import BaseModel" not in lines:
            lines.append("from pydantic import BaseModel")

    if parsed.has_vercel:
        lines.append("from fastapi.responses import StreamingResponse")
        lines.append("from langchain_core.messages import HumanMessage")

    # Tools are project-defined — no library imports injected here.

    return "\n".join(lines) + "\n"


def _gen_langsmith_setup() -> str:
    return (
        "# LangSmith tracing\n"
        'os.environ.setdefault("LANGSMITH_TRACING", "true")\n'
        'os.environ.setdefault("LANGSMITH_PROJECT", "my-project")\n'
    )


def _gen_context_schema() -> str:
    return (
        "@dataclass\n"
        "class RuntimeContext:\n"
        '    user_id: str = "anonymous"\n'
        '    session_id: str = "default"\n'
    )


def _gen_output_schema(schema_name: str | None) -> str:
    name = schema_name or "OutputSchema"
    return (
        f"class {name}(BaseModel):\n"
        f"    # Define your output fields\n"
        f"    result: str\n"
    )


def _gen_dynamic_prompt() -> str:
    return (
        "@dynamic_prompt\n"
        "def dynamic_system_prompt(request: ModelRequest) -> str:\n"
        '    role = getattr(request.runtime.context, "role", "assistant")\n'
        '    return f"You are a {role}."\n'
    )


def _gen_tools(parsed: LCParsed, sections: list[str]) -> tuple[list[str], dict[str, str]]:
    """Generate tool definitions. Returns (var_names, name_map)."""
    tool_var_names: list[str] = []
    tool_name_map: dict[str, str] = {}  # DSL name → Python var name

    for tool_name in parsed.tools:
        # Tools are defined per-project — placeholder stubs only
        tool_var_names.append(tool_name)
        tool_name_map[tool_name] = tool_name
        placeholder = (
            f"@tool\n"
            f"def {tool_name}(query: str) -> str:\n"
            f'    """{tool_name} — implement for this project."""\n'
        )
        if parsed.has_ctx:
            placeholder += (
                f"    runtime = get_runtime(RuntimeContext)\n"
                f"    # Access runtime.context.user_id etc.\n"
            )
        placeholder += f'    return "TODO: implement {tool_name}"\n'
        sections.append(placeholder)

    return tool_var_names, tool_name_map


def _gen_create_agent(
    parsed: LCParsed,
    tool_var_names: list[str],
    tool_name_map: dict[str, str],
) -> str:
    lines = ["agent = create_agent("]
    lines.append('    model="gpt-4o-mini",')

    if tool_var_names:
        lines.append(f"    tools=[{', '.join(tool_var_names)}],")

    # System prompt
    has_rag = "rag" in parsed.tools
    sys_prompt = (
        "You are a helpful assistant. "
        "Today is {__import__('datetime').date.today().strftime('%B %d, %Y')}."
    )
    lines.append(f'    system_prompt=f"{sys_prompt}",')

    if parsed.has_ctx:
        lines.append("    context_schema=RuntimeContext,")

    if parsed.has_out:
        lines.append(f"    response_format={parsed.out_schema},")

    # Middleware stack
    middleware = _build_middleware(parsed, tool_name_map)
    if middleware:
        lines.append("    middleware=[")
        for mw in middleware:
            lines.append(f"        {mw},")
        lines.append("    ],")

    lines.append(")")
    return "\n".join(lines) + "\n"


def _build_middleware(parsed: LCParsed, tool_name_map: dict[str, str]) -> list[str]:
    """Build middleware list from parsed modifiers."""
    middleware: list[str] = []

    if parsed.has_hitl:
        hitl_tools = parsed.hitl_tools
        if hitl_tools:
            entries = []
            for t in hitl_tools:
                var = tool_name_map.get(t, t)
                entries.append(f'"{var}": {{"allowed_decisions": ["approve", "reject"]}}')
            middleware.append(
                f"HumanInTheLoopMiddleware(interrupt_on={{{', '.join(entries)}}})"
            )
        else:
            middleware.append("HumanInTheLoopMiddleware()")

    if parsed.has_sum:
        middleware.append(
            'SummarizationMiddleware(model="gpt-4o-mini", '
            'trigger=("tokens", 1000), keep=("messages", 10))'
        )

    if parsed.has_trim:
        middleware.append(
            'TrimMessagesMiddleware(max_tokens=1000, strategy="last")'
        )

    if parsed.has_dyn:
        middleware.append("dynamic_system_prompt")

    return middleware


def _gen_usage(parsed: LCParsed) -> str:
    lines = ["# Usage:"]

    if parsed.has_ctx:
        lines.append(
            '# result = agent.invoke({"messages": [{"role": "user", "content": "..."}]}, '
            'context=RuntimeContext(user_id="123"))'
        )
    else:
        lines.append(
            '# result = agent.invoke({"messages": [{"role": "user", "content": "..."}]})'
        )

    if parsed.has_memory:
        lines.append(
            '# With memory: config={"configurable": {"thread_id": "user-123"}}'
        )

    if parsed.has_hitl:
        lines.append(
            '# Resume from interrupt: agent.invoke('
            'Command(resume={"decisions": [{"type": "approve"}]}), config)'
        )

    if parsed.has_out:
        lines.append('# Structured output: result["structured_response"]')

    return "\n".join(lines) + "\n"


def _gen_fastapi(parsed: LCParsed) -> str:
    lines = [
        "",
        "# FastAPI Application",
        "app = FastAPI()",
        "",
        "class InvokeRequest(BaseModel):",
        "    message: str",
        "",
        '@app.post("/invoke")',
        "async def invoke_agent(request: InvokeRequest):",
        '    result = agent.invoke({"messages": [{"role": "user", "content": request.message}]})',
        '    return {"response": result["messages"][-1].content}',
        "",
        "# Run: uvicorn filename:app --reload",
    ]
    return "\n".join(lines) + "\n"


def _gen_vercel(parsed: LCParsed) -> str:
    lines = [
        "",
        "# Vercel AI SDK Compatible Chat Endpoint",
        "app = FastAPI()",
        "",
        '@app.post("/api/chat")',
        "async def chat(request: dict):",
        '    messages = request.get("messages", [])',
        "",
        "    async def generate():",
        '        async for chunk in agent.astream({"messages": messages}, stream_mode="messages"):',
        "            if hasattr(chunk, 'content'):",
        '                yield f"data: {chunk.content}\\n\\n"',
        '        yield "data: [DONE]\\n\\n"',
        "",
        '    return StreamingResponse(generate(), media_type="text/event-stream")',
        "",
        "# Deploy: vercel --prod",
    ]
    return "\n".join(lines) + "\n"
