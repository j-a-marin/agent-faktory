# faktory/parser/__init__.py
from .lc_parser import parse_lc_dsl, LCParsed
from .lg_parser import parse_lg_dsl, LGParsed
from .dsl_parser import parse_dsl

__all__ = ["parse_dsl", "parse_lc_dsl", "parse_lg_dsl", "LCParsed", "LGParsed"]
