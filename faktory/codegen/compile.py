"""Unified compile entry point.

parse_dsl() → AST → compile_agent() → Python source code.

This is the main API surface for AGENT_FAKTORY.
"""

from __future__ import annotations

from typing import Union

from faktory.parser import parse_dsl, LCParsed, LGParsed
from faktory.parser.dsl_parser import DSLParseError
from .lc_codegen import generate_lc_code
from .lg_codegen import generate_lg_code


def compile_agent(dsl_line: str) -> str:
    """Compile a DSL line into a complete Python agent file.

    This is the one-call API: DSL string in → Python source out.
    Deterministic, no LLM, no IO. ~1ms.

    Args:
        dsl_line: Full DSL line, e.g. "# lc react(search, calc) -> mem -> api"

    Returns:
        Complete Python source code as a string.

    Raises:
        DSLParseError: If the DSL line cannot be parsed.

    Examples:
        >>> code = compile_agent("# lc react(search) -> mem -> api")
        >>> "create_agent" in code
        True

        >>> code = compile_agent("# lg state(a, b) -> par(a) -> agg(b)")
        >>> "StateGraph" in code
        True
    """
    ast = parse_dsl(dsl_line)

    if isinstance(ast, LCParsed):
        return generate_lc_code(ast)
    else:
        return generate_lg_code(ast)
