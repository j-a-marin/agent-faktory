"""LangGraph (#lg) code generator.

Transforms LGParsed AST into a complete runnable Python file
using LangGraph's StateGraph API directly.

This is the "power user" path — only invoked when the DSL uses #lg,
meaning the tenant explicitly needs graph-level control (Send parallelism,
conditional routing, command-based edges, subgraphs, etc.).

"""

from __future__ import annotations

from faktory.parser.lg_parser import LGParsed


def generate_lg_code(parsed: LGParsed) -> str:
    """Generate a complete Python file from an LGParsed AST.

    Args:
        parsed: Output of parse_lg_dsl()

    Returns:
        Complete Python source code as a string.
    """
    sections: list[str] = []

    sections.append(_gen_header(parsed))
    sections.append(_gen_imports(parsed))

    if parsed.has_ls:
        sections.append(_gen_langsmith_setup())

    if parsed.has_llm:
        sections.append(_gen_llm_setup())

    sections.append(_gen_state(parsed))
    sections.append(_gen_node_functions(parsed))

    if parsed.has_send:
        sections.append(_gen_send_dispatcher(parsed))

    if parsed.has_sendmap:
        sections.append(_gen_sendmap_dispatcher(parsed))

    if parsed.has_cond:
        sections.append(_gen_conditional_router(parsed))

    sections.append(_gen_graph_build(parsed))
    sections.append(_gen_compile(parsed))
    sections.append(_gen_usage(parsed))

    if parsed.has_api and not parsed.has_vercel:
        sections.append(_gen_fastapi(parsed))

    if parsed.has_vercel:
        sections.append(_gen_vercel(parsed))

    return "\n".join(sections)


# ═══════════════════════════════════════════════════════════════════════
# Section generators
# ═══════════════════════════════════════════════════════════════════════

def _gen_header(parsed: LGParsed) -> str:
    return f"# lg {parsed.raw}\n"


def _gen_imports(parsed: LGParsed) -> str:
    lines = [
        "from langgraph.graph import StateGraph, START, END",
        "from langgraph.types import Command",
        "from typing import TypedDict, Annotated, Literal",
        "import operator",
    ]

    if parsed.has_send or parsed.has_sendmap:
        lines.append("from langgraph.types import Send")

    if parsed.has_llm:
        lines.append("from langchain_openai import ChatOpenAI")
        lines.append("from langchain_core.messages import HumanMessage, SystemMessage")

    if parsed.has_hitl:
        lines.append("from langgraph.types import interrupt")

    if parsed.has_ls:
        lines.append("import os")

    if parsed.has_api or parsed.has_vercel:
        lines.append("from fastapi import FastAPI")
        lines.append("from pydantic import BaseModel")

    if parsed.has_vercel:
        lines.append("from fastapi.responses import StreamingResponse")

    return "\n".join(lines) + "\n"


def _gen_langsmith_setup() -> str:
    return (
        "# LangSmith tracing\n"
        'os.environ.setdefault("LANGSMITH_TRACING", "true")\n'
        'os.environ.setdefault("LANGSMITH_PROJECT", "langgraph-project")\n'
    )


def _gen_llm_setup() -> str:
    return (
        "# LLM setup\n"
        'llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)\n'
    )


def _gen_state(parsed: LGParsed) -> str:
    # Determine reducer
    reducer = "operator.add"
    if parsed.has_reduce and parsed.reduce_op == "last":
        reducer = "lambda a, b: b"

    lines = [
        "# State definition",
        "class State(TypedDict):",
        f"    messages: Annotated[list, {reducer}]",
    ]

    if parsed.has_sendmap:
        lines.append("    tasks: list  # Input tasks for map-reduce fan-out")

    if parsed.has_par or parsed.has_send or parsed.has_sendmap:
        lines.append("    results: Annotated[list, operator.add]  # Aggregated parallel results")

    if parsed.has_ctx:
        lines.append('    user_id: str')
        lines.append('    session_id: str')

    return "\n".join(lines) + "\n"


def _gen_node_functions(parsed: LGParsed) -> str:
    sections: list[str] = []
    sections.append("# Node functions")

    hitl_nodes = parsed.hitl_nodes

    for i, node in enumerate(parsed.nodes):
        is_par_node = parsed.has_par and node in parsed.par_nodes

        # Signature
        if is_par_node:
            lines = [f"def {node}(state: State) -> dict:"]
        else:
            lines = [f"def {node}(state: State):"]

        lines.append(f'    """{node.replace("_", " ").title()} node logic."""')

        # HITL interrupt
        if node in hitl_nodes:
            lines.append("    # Human-in-the-loop checkpoint")
            lines.append(f'    decision = interrupt({{"node": "{node}", "state": state}})')
            lines.append('    if decision.get("approved") is False:')
            lines.append('        return {"messages": state.get("messages", []) + ["Rejected by human"]}')

        # Node body
        if parsed.has_llm:
            lines.extend([
                "    messages = state.get('messages', [])",
                f'    system_msg = """You are the {node} agent.',
                f'    Analyze the input and provide your {node} output."""',
                "",
                "    response = llm.invoke([",
                "        SystemMessage(content=system_msg),",
                '        HumanMessage(content=str(messages[-1]) if messages else "Start")',
                "    ])",
                "    result = response.content",
            ])
        else:
            lines.extend([
                f"    # TODO: Implement {node} logic",
                f'    result = f"{node} processed: {{state.get(\'messages\', [])}}"',
            ])

        lines.append("")

        # Return type depends on edge style
        if is_par_node or parsed.has_par:
            lines.append('    return {"messages": state.get("messages", []) + [result]}')
        elif parsed.has_cmd:
            lines.extend(_gen_cmd_return(parsed, node, i))
        else:
            lines.append('    return {"messages": state.get("messages", []) + [result]}')

        sections.append("\n".join(lines) + "\n")

    return "\n".join(sections)


def _gen_cmd_return(parsed: LGParsed, node: str, idx: int) -> list[str]:
    """Generate Command-based return for a node."""
    lines: list[str] = []

    if idx == 0 and len(parsed.nodes) > 2:
        lines.extend([
            "    # Conditional routing based on state",
            "    intent = 'default'  # TODO: determine from result/state",
            "",
            f"    if intent == '{parsed.nodes[1]}':",
            "        return Command(",
            '            update={"messages": state.get("messages", []) + [result]},',
            f'            goto=["{parsed.nodes[1]}"]',
            "        )",
            "    else:",
            "        return Command(",
            '            update={"messages": state.get("messages", []) + [result]},',
            f'            goto=["{parsed.nodes[2]}"]',
            "        )",
        ])
    elif idx < len(parsed.nodes) - 1:
        next_node = parsed.nodes[idx + 1]
        lines.extend([
            "    return Command(",
            '        update={"messages": state.get("messages", []) + [result]},',
            f'        goto=["{next_node}"]',
            "    )",
        ])
    else:
        lines.extend([
            "    return Command(",
            '        update={"messages": state.get("messages", []) + [result]},',
            "        goto=[END]",
            "    )",
        ])

    return lines


def _gen_send_dispatcher(parsed: LGParsed) -> str:
    targets = ", ".join(f'Send("{t}", state)' for t in parsed.send_targets)
    return (
        "# Parallel fan-out dispatcher (known targets)\n"
        "def fan_out(state: State) -> list[Send]:\n"
        '    """Fan out to known parallel nodes."""\n'
        f"    return [{targets}]\n"
    )


def _gen_sendmap_dispatcher(parsed: LGParsed) -> str:
    return (
        "# Dynamic fan-out dispatcher (map-reduce pattern)\n"
        "def dispatch_to_workers(state: State) -> list[Send]:\n"
        '    """Fan out to worker nodes based on tasks in state."""\n'
        "    tasks = state.get('tasks', [])\n"
        "    if not tasks:\n"
        "        return []\n"
        f'    return [Send("{parsed.sendmap_target}", {{"task": t, **state}}) for t in tasks]\n'
    )


def _gen_conditional_router(parsed: LGParsed) -> str:
    func_name = parsed.cond_func or "route"
    node_literals = ", ".join(f'"{n}"' for n in parsed.nodes)
    return (
        "# Conditional router\n"
        f'def {func_name}(state: State) -> Literal[{node_literals}, "__end__"]:\n'
        f'    """Route to next node based on state."""\n'
        f"    # TODO: Implement routing logic\n"
        f"    messages = state.get('messages', [])\n"
        f"    if not messages:\n"
        f'        return "{parsed.nodes[0]}"\n'
        f'    return "__end__"\n'
    )


def _gen_graph_build(parsed: LGParsed) -> str:
    lines = [
        "# Build graph",
        "graph = StateGraph(State)",
        "",
        "# Add nodes",
    ]

    for node in parsed.nodes:
        lines.append(f'graph.add_node("{node}", {node})')

    lines.append("")
    lines.append("# Add edges")

    if parsed.has_par:
        for par_node in parsed.par_nodes:
            lines.append(f'graph.add_edge(START, "{par_node}")')
        if parsed.aggregate_node:
            for par_node in parsed.par_nodes:
                lines.append(f'graph.add_edge("{par_node}", "{parsed.aggregate_node}")')
            lines.append(f'graph.add_edge("{parsed.aggregate_node}", END)')
        else:
            for par_node in parsed.par_nodes:
                lines.append(f'graph.add_edge("{par_node}", END)')

    elif parsed.has_send:
        router = parsed.nodes[0]
        target_list = ", ".join(f'"{t}"' for t in parsed.send_targets)
        lines.append(f'graph.add_edge(START, "{router}")')
        lines.append(f'graph.add_conditional_edges("{router}", fan_out, [{target_list}])')
        if parsed.aggregate_node:
            for t in parsed.send_targets:
                lines.append(f'graph.add_edge("{t}", "{parsed.aggregate_node}")')
            lines.append(f'graph.add_edge("{parsed.aggregate_node}", END)')
        else:
            for t in parsed.send_targets:
                lines.append(f'graph.add_edge("{t}", END)')

    elif parsed.has_sendmap:
        dispatcher = parsed.nodes[0]
        lines.append(f'graph.add_edge(START, "{dispatcher}")')
        lines.append(
            f'graph.add_conditional_edges("{dispatcher}", dispatch_to_workers, '
            f'["{parsed.sendmap_target}"])'
        )
        if parsed.aggregate_node:
            lines.append(f'graph.add_edge("{parsed.sendmap_target}", "{parsed.aggregate_node}")')
            lines.append(f'graph.add_edge("{parsed.aggregate_node}", END)')
        else:
            lines.append(f'graph.add_edge("{parsed.sendmap_target}", END)')

    elif parsed.has_cond:
        func_name = parsed.cond_func or "route"
        lines.append(f"graph.add_conditional_edges(START, {func_name})")

    else:
        # Sequential
        lines.append(f'graph.add_edge(START, "{parsed.nodes[0]}")')
        for i in range(len(parsed.nodes) - 1):
            lines.append(f'graph.add_edge("{parsed.nodes[i]}", "{parsed.nodes[i + 1]}")')
        lines.append(f'graph.add_edge("{parsed.nodes[-1]}", END)')

    return "\n".join(lines) + "\n"


def _gen_compile(parsed: LGParsed) -> str:
    lines = ["# Compile graph"]
    if parsed.has_hitl:
        hitl_list = ", ".join(f'"{n}"' for n in parsed.hitl_nodes)
        lines.append(f"agent = graph.compile(interrupt_before=[{hitl_list}])")
    else:
        lines.append("agent = graph.compile()")
    return "\n".join(lines) + "\n"


def _gen_usage(parsed: LGParsed) -> str:
    lines = [
        "# Usage",
        '# result = agent.invoke({"messages": ["Hello"]})',
    ]
    if parsed.has_memory:
        lines.append(
            '# With persistence: config={"configurable": {"thread_id": "1"}}'
        )
    if parsed.has_sendmap:
        lines.append(
            '# Map-reduce: agent.invoke({"tasks": ["t1", "t2", "t3"], "messages": []})'
        )
    if parsed.has_cmd:
        lines.extend([
            "",
            "# Command Patterns:",
            '# Simple route:    Command(goto=["next_node"])',
            '# Update + route:  Command(update={"key": val}, goto=["next"])',
            '# Fan-out:         Command(goto=["node_a", "node_b"])',
            '# Cross-graph:     Command(goto=["parent:node_name"])',
            '# Resume HITL:     Command(resume={"approved": True})',
        ])
    return "\n".join(lines) + "\n"


def _gen_fastapi(parsed: LGParsed) -> str:
    lines = [
        "",
        "# FastAPI Application",
        "api = FastAPI()",
        "",
        "class InvokeRequest(BaseModel):",
        "    message: str",
    ]
    if parsed.has_memory:
        lines.append('    thread_id: str = "default"')
    lines.extend([
        "",
        '@api.post("/invoke")',
        "async def invoke(request: InvokeRequest):",
    ])
    if parsed.has_memory:
        lines.extend([
            '    config = {"configurable": {"thread_id": request.thread_id}}',
            '    result = agent.invoke({"messages": [request.message]}, config=config)',
        ])
    else:
        lines.append('    result = agent.invoke({"messages": [request.message]})')
    lines.extend([
        '    return {"response": result}',
        "",
        "# Run: uvicorn filename:api --reload",
    ])
    return "\n".join(lines) + "\n"


def _gen_vercel(parsed: LGParsed) -> str:
    lines = [
        "",
        "# Vercel AI SDK Compatible Chat Endpoint",
        "api = FastAPI()",
        "",
        '@api.post("/api/chat")',
        "async def chat(request: dict):",
        '    messages = request.get("messages", [])',
        '    thread_id = request.get("thread_id", "default")',
        "",
        "    async def generate():",
        '        config = {"configurable": {"thread_id": thread_id}}',
        '        async for chunk in agent.astream({"messages": messages}, config=config):',
        "            if chunk:",
        '                yield f"data: {chunk}\\n\\n"',
        '        yield "data: [DONE]\\n\\n"',
        "",
        '    return StreamingResponse(generate(), media_type="text/event-stream")',
        "",
        "# Deploy: vercel --prod",
    ]
    return "\n".join(lines) + "\n"
