"""LangGraph (#lg) DSL parser.

Deterministic parser for LangGraph workflow DSL strings.

Covers ALL graph types, edge patterns, and modifiers:
  Graph types: state, msg
  Edge patterns: par, send, sendmap, agg, cond, cmd, sub
  Modifiers: llm, mem, hitl, ctx, ls, api, vercel
  Reducers: reduce(add|concat|last)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LGParsed:
    """Result of parsing a #lg DSL string.

    This is the AST — a pure data structure with no code generation logic.
    """
    graph_type: str = "state"               # state | msg
    nodes: list[str] = field(default_factory=list)
    modifiers: list[dict] = field(default_factory=list)  # [{name, param}]
    raw: str = ""

    # ── Edge patterns ───────────────────────────────────────────────
    has_par: bool = False                   # static parallel from START
    par_nodes: list[str] = field(default_factory=list)

    has_send: bool = False                  # known fan-out via Send()
    send_targets: list[str] = field(default_factory=list)

    has_sendmap: bool = False               # dynamic fan-out (map-reduce)
    sendmap_target: Optional[str] = None

    aggregate_node: Optional[str] = None    # convergence point

    has_cond: bool = False                  # conditional router
    cond_func: Optional[str] = None

    has_cmd: bool = False                   # Command-based dynamic routing
    has_llm: bool = False                   # LLM-powered nodes

    has_reduce: bool = False
    reduce_op: str = "add"                  # add | concat | last

    subgraphs: list[str] = field(default_factory=list)

    # ── Convenience flags (derived from modifiers) ──────────────────
    @property
    def has_memory(self) -> bool:
        return any(m["name"] == "mem" for m in self.modifiers)

    @property
    def has_hitl(self) -> bool:
        return any(m["name"] == "hitl" for m in self.modifiers)

    @property
    def hitl_nodes(self) -> list[str]:
        nodes = []
        for m in self.modifiers:
            if m["name"] == "hitl" and m.get("param"):
                nodes.extend(n.strip() for n in m["param"].split(","))
        return nodes

    @property
    def has_ctx(self) -> bool:
        return any(m["name"] == "ctx" for m in self.modifiers)

    @property
    def has_ls(self) -> bool:
        return any(m["name"] == "ls" for m in self.modifiers)

    @property
    def has_api(self) -> bool:
        return any(m["name"] == "api" for m in self.modifiers)

    @property
    def has_vercel(self) -> bool:
        return any(m["name"] == "vercel" for m in self.modifiers)

    def to_dict(self) -> dict:
        """Serialize to plain dict (for JSON/API responses)."""
        d = {
            "framework": "lg",
            "graph_type": self.graph_type,
            "nodes": self.nodes,
            "modifiers": self.modifiers,
            "raw": self.raw,
        }
        # Edge patterns (only include if set)
        if self.has_par:
            d["par_nodes"] = self.par_nodes
        if self.has_send:
            d["send_targets"] = self.send_targets
        if self.has_sendmap:
            d["sendmap_target"] = self.sendmap_target
        if self.aggregate_node:
            d["aggregate_node"] = self.aggregate_node
        if self.has_cond:
            d["cond_func"] = self.cond_func
        if self.has_cmd:
            d["has_cmd"] = True
        if self.has_llm:
            d["has_llm"] = True
        if self.has_reduce:
            d["reduce_op"] = self.reduce_op
        if self.subgraphs:
            d["subgraphs"] = self.subgraphs
        return d


# ── Regex patterns ──────────────────────────────────────────────────────
_GRAPH_TYPE_RE = re.compile(r"^\s*(\w+)\(([^)]+)\)")
_PIPELINE_RE = re.compile(r"->\s*([^->]+)")
_FUNC_CALL_RE = re.compile(r"^(\w+)\(([^)]*)\)$")


def parse_lg_dsl(dsl_str: str) -> LGParsed:
    """Parse a #lg DSL string into an LGParsed AST.


    Args:
        dsl_str: The DSL portion after "# lg " —
                 e.g. "state(web, wiki, gen) -> par(web, wiki) -> agg(gen)"

    Returns:
        LGParsed with graph_type, nodes, edges, and modifiers populated.

    Examples:
        >>> p = parse_lg_dsl("state(a, b, c) -> llm -> mem")
        >>> p.graph_type, p.nodes, p.has_llm, p.has_memory
        ('state', ['a', 'b', 'c'], True, True)

        >>> p = parse_lg_dsl("state(web, wiki, gen) -> par(web, wiki) -> agg(gen)")
        >>> p.has_par, p.par_nodes, p.aggregate_node
        (True, ['web', 'wiki'], 'gen')

        >>> p = parse_lg_dsl("state(discover, process, decide) -> sendmap(process) -> agg(decide)")
        >>> p.has_sendmap, p.sendmap_target, p.aggregate_node
        (True, 'process', 'decide')
    """
    result = LGParsed(raw=dsl_str)

    # ── Extract graph type and nodes ─────────────────────────────
    m = _GRAPH_TYPE_RE.match(dsl_str)
    if m:
        result.graph_type = m.group(1)
        result.nodes = [n.strip() for n in m.group(2).split(",") if n.strip()]

    # ── Extract edges and modifiers after -> ─────────────────────
    for segment_match in _PIPELINE_RE.finditer(dsl_str):
        segment = segment_match.group(1).strip()

        # Try to match function-call syntax: name(args)
        fm = _FUNC_CALL_RE.match(segment)

        if fm:
            func_name = fm.group(1)
            func_args = fm.group(2).strip()

            if func_name == "par":
                # Static parallel: par(web, wiki)
                result.has_par = True
                result.par_nodes = [n.strip() for n in func_args.split(",") if n.strip()]

            elif func_name == "sendmap":
                # Dynamic fan-out: sendmap(worker)
                result.has_sendmap = True
                result.sendmap_target = func_args

            elif func_name == "agg":
                # Aggregation point: agg(combiner)
                result.aggregate_node = func_args

            elif func_name == "send":
                # Known fan-out: send(a, b)
                result.has_send = True
                result.send_targets = [t.strip() for t in func_args.split(",") if t.strip()]

            elif func_name == "cond":
                # Conditional router: cond(route_fn)
                result.has_cond = True
                result.cond_func = func_args

            elif func_name == "reduce":
                # Reducer: reduce(add) | reduce(concat) | reduce(last)
                result.has_reduce = True
                result.reduce_op = func_args or "add"

            elif func_name == "sub":
                # Subgraph: sub(graph_name)
                result.subgraphs.append(func_args)

            elif func_name == "hitl":
                # HITL on specific nodes: hitl(node_a, node_b)
                result.modifiers.append({"name": "hitl", "param": func_args})

            else:
                # Unknown function-call modifier — store as-is
                result.modifiers.append({"name": func_name, "param": func_args})

        else:
            # Simple keyword modifier: llm, mem, cmd, ctx, ls, api, vercel
            if segment == "cmd":
                result.has_cmd = True
            elif segment == "llm":
                result.has_llm = True
            else:
                result.modifiers.append({"name": segment, "param": None})

    return result
