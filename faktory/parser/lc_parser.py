"""LangChain (#lc) DSL parser.

Deterministic parser for LangChain agent DSL strings.

Covers ALL patterns and modifiers from the Lua source:
  Patterns: react, sql, sub, mcp
  Modifiers: mem, hitl, ctx, rt, cmd, inj, sum, trim,
             dyn, dynt, dynm, strv, strm, strc, out,
             ls, api, vercel
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


# ── Registry of valid patterns ──────────────────────────────────────────
LC_PATTERNS = {
    "react": "react_agent",
    "sql": "sql_agent",
    "sub": "subagent",
    "mcp": "mcp_integration",
}

# ── Registry of valid modifiers ─────────────────────────────────────────
LC_MODIFIERS = {
    # Core middleware
    "mem": "memory",
    "hitl": "hitl_all",
    "ctx": "context_schema",
    "rt": "tool_runtime",
    "cmd": "command_tools",
    "inj": "injected_state",
    "sum": "summarize",
    "trim": "trim_messages",
    # Dynamic modifiers
    "dyn": "dynamic_prompt",
    "dynt": "dynamic_tools",
    "dynm": "dynamic_model",
    # Streaming modifiers
    "strv": "stream_values",
    "strm": "stream_messages",
    "strc": "stream_custom",
    # Output
    "out": "response_format",
    # Deployment modifiers
    "ls": "langsmith",
    "api": "fastapi",
    "vercel": "vercel_ai",
}


@dataclass
class Modifier:
    """A single modifier in the DSL pipeline."""
    name: str
    param: Optional[str] = None

    def __repr__(self) -> str:
        if self.param:
            return f"{self.name}({self.param})"
        return self.name


@dataclass
class LCParsed:
    """Result of parsing a #lc DSL string.

    This is the AST — a pure data structure with no code generation logic.
    """
    pattern: Optional[str] = None       # react | sql | sub | mcp
    pattern_key: Optional[str] = None   # internal template key
    tools: list[str] = field(default_factory=list)
    modifiers: list[Modifier] = field(default_factory=list)
    raw: str = ""                        # original DSL string

    # ── Convenience flags (derived from modifiers) ──────────────────
    @property
    def has_memory(self) -> bool:
        return any(m.name == "mem" for m in self.modifiers)

    @property
    def has_hitl(self) -> bool:
        return any(m.name == "hitl" for m in self.modifiers)

    @property
    def hitl_tools(self) -> list[str]:
        """Tools that require human approval."""
        tools = []
        for m in self.modifiers:
            if m.name == "hitl" and m.param:
                tools.extend(t.strip() for t in m.param.split(","))
        return tools

    @property
    def has_ctx(self) -> bool:
        return any(m.name in ("ctx", "rt") for m in self.modifiers)

    @property
    def has_sum(self) -> bool:
        return any(m.name == "sum" for m in self.modifiers)

    @property
    def has_trim(self) -> bool:
        return any(m.name == "trim" for m in self.modifiers)

    @property
    def has_dyn(self) -> bool:
        return any(m.name == "dyn" for m in self.modifiers)

    @property
    def has_dynt(self) -> bool:
        return any(m.name == "dynt" for m in self.modifiers)

    @property
    def has_dynm(self) -> bool:
        return any(m.name == "dynm" for m in self.modifiers)

    @property
    def has_out(self) -> bool:
        return any(m.name == "out" for m in self.modifiers)

    @property
    def out_schema(self) -> Optional[str]:
        for m in self.modifiers:
            if m.name == "out":
                return m.param or "OutputSchema"
        return None

    @property
    def has_ls(self) -> bool:
        return any(m.name == "ls" for m in self.modifiers)

    @property
    def has_api(self) -> bool:
        return any(m.name == "api" for m in self.modifiers)

    @property
    def has_vercel(self) -> bool:
        return any(m.name == "vercel" for m in self.modifiers)

    @property
    def stream_mode(self) -> Optional[str]:
        """Return stream mode if any streaming modifier is set."""
        for m in self.modifiers:
            if m.name == "strv":
                return "values"
            if m.name == "strm":
                return "messages"
            if m.name == "strc":
                return "custom"
        return None

    @property
    def has_cmd(self) -> bool:
        return any(m.name == "cmd" for m in self.modifiers)

    @property
    def has_inj(self) -> bool:
        return any(m.name == "inj" for m in self.modifiers)

    def to_dict(self) -> dict:
        """Serialize to plain dict (for JSON/API responses)."""
        return {
            "framework": "lc",
            "pattern": self.pattern,
            "pattern_key": self.pattern_key,
            "tools": self.tools,
            "modifiers": [{"name": m.name, "param": m.param} for m in self.modifiers],
            "raw": self.raw,
        }


# ── Regex patterns ──────────────────────────────────────────────────────
_PATTERN_RE = re.compile(r"^\s*(\w+)\(([^)]+)\)")
_PATTERN_NO_ARGS_RE = re.compile(r"^\s*(\w+)")
_PIPELINE_RE = re.compile(r"->\s*([^->]+)")
_MODIFIER_WITH_PARAM_RE = re.compile(r"^(\w+)\(([^)]+)\)$")


def parse_lc_dsl(dsl_str: str) -> LCParsed:
    """Parse a #lc DSL string into an LCParsed AST.


    Args:
        dsl_str: The DSL portion after "# lc " — e.g. "react(search, calc) -> mem -> api"

    Returns:
        LCParsed with pattern, tools, and modifiers populated.

    Examples:
        >>> parse_lc_dsl("react(search, calc) -> mem -> api")
        LCParsed(pattern='react', tools=['search', 'calc'], modifiers=[mem, api])

        >>> parse_lc_dsl("react(search) -> hitl(search) -> sum -> mem -> ls -> api -> vercel")
        LCParsed(pattern='react', tools=['search'], modifiers=[hitl(search), sum, mem, ls, api, vercel])
    """
    result = LCParsed(raw=dsl_str)

    # ── Extract pattern and tools: "react(search, respond)" ──────
    m = _PATTERN_RE.match(dsl_str)
    if m:
        pattern_name = m.group(1)
        tools_str = m.group(2)
        result.pattern = pattern_name
        result.pattern_key = LC_PATTERNS.get(pattern_name)
        result.tools = [t.strip() for t in tools_str.split(",") if t.strip()]
    else:
        # Pattern without tools: "react"
        m = _PATTERN_NO_ARGS_RE.match(dsl_str)
        if m:
            pattern_name = m.group(1)
            result.pattern = pattern_name
            result.pattern_key = LC_PATTERNS.get(pattern_name)

    # ── Extract modifiers after -> ───────────────────────────────
    for segment_match in _PIPELINE_RE.finditer(dsl_str):
        segment = segment_match.group(1).strip()

        # Check for parameterized modifier: hitl(send_email) or out(MySchema)
        pm = _MODIFIER_WITH_PARAM_RE.match(segment)
        if pm:
            result.modifiers.append(Modifier(name=pm.group(1), param=pm.group(2)))
        else:
            result.modifiers.append(Modifier(name=segment))

    return result
