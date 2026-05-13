"""Unified DSL parser — entry point.

Detects #lc or #lg prefix and dispatches to the appropriate parser.
"""

from __future__ import annotations

import re
from typing import Union

from .lc_parser import parse_lc_dsl, LCParsed
from .lg_parser import parse_lg_dsl, LGParsed


# Match "# lc ...", "#lc ...", "# lg ...", "#lg ..."
_DSL_PREFIX_RE = re.compile(r"^\s*#\s*(lc|lg)\s+(.+)$")


class DSLParseError(ValueError):
    """Raised when a DSL string cannot be parsed."""
    pass


def parse_dsl(line: str) -> Union[LCParsed, LGParsed]:
    """Parse a full DSL line including the # prefix.

    This is the main entry point for the parser module.

    Args:
        line: Full DSL line, e.g. "# lc react(search, calc) -> mem -> api"

    Returns:
        LCParsed or LGParsed depending on the framework prefix.

    Raises:
        DSLParseError: If the line doesn't match expected DSL format.

    Examples:
        >>> ast = parse_dsl("# lc react(search) -> mem -> api")
        >>> ast.pattern, ast.tools
        ('react', ['search'])

        >>> ast = parse_dsl("# lg state(a, b, c) -> par(a, b) -> agg(c)")
        >>> ast.graph_type, ast.has_par
        ('state', True)
    """
    m = _DSL_PREFIX_RE.match(line)
    if not m:
        raise DSLParseError(
            f"Invalid DSL format. Expected '# lc ...' or '# lg ...', got: {line!r}"
        )

    framework = m.group(1)
    dsl_body = m.group(2)

    if framework == "lc":
        return parse_lc_dsl(dsl_body)
    else:
        return parse_lg_dsl(dsl_body)
