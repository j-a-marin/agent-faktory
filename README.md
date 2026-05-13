# agent-faktory

**Deterministic DSL-to-Agent compiler. No LLM required.**

Reads a declarative description of an agent — intent, toolset,
orchestration topology, credential bindings — and emits working Python:
LangChain (`create_agent`) or LangGraph (`StateGraph`). The compile path
is fully deterministic. No model call. No nondeterminism. Same input,
same output, byte-for-byte.

## Why this exists

Agentic systems in regulated environments need two properties at once:

1. *Composability* — the same architectural patterns repeated across
   matters, tenants, workflows, with explicit type discipline at every
   boundary.
2. *Individuation* — each deployed instance bound to a specific context,
   wrapped in supervision, producing an attributable audit trail.

`agent-faktory` is the composable substrate. It compiles declarative
agent specs into working Python with TODO scaffolds at domain-logic
boundaries. The compiler produces structure; the human supplies the
domain. No LLM is invoked in the compile path.

[LiveCards](https://argentislabs.io/demo/livecards/matter-intake) is the
individuation layer. Each compiled agent gets bound per-matter, wrapped
in attorney sign-off as the architectural gate, and produces the
event-level audit trail that ABA Formal Opinion 512 (July 2024)
treats as a supervisory obligation.

Composable upstream. Individuated downstream. This repo is the upstream
half.

## Architecture

```
DSL spec
   │
   ▼
┌──────────┐    ┌──────────┐    ┌──────────┐
│  parser  │ ─▶ │ registry │ ─▶ │ codegen  │ ─▶ working Python agent
└──────────┘    └──────────┘    └──────────┘
   typed IR      type checks     LC / LG emit
```

Three subsystems, each on its own commit:

- **`faktory/parser/`** — LangChain and LangGraph DSL dialects parse to a
  typed intermediate representation.
- **`faktory/registry/`** — typed models for agents, tools, and
  credentials (API key, JWT, vault refs); seed data; a validator that
  verifies the parsed DSL satisfies registry constraints before codegen
  runs.
- **`faktory/codegen/`** — backend generators emit working Python with
  explicit TODO markers at tool-implementation boundaries.

## Reproduce

```bash
git clone https://github.com/j-a-marin/agent-faktory.git
cd agent-faktory
uv sync --extra dev
uv run pytest -q
```

Expected: **107 tests pass in under 100ms**.

## License

MIT. See `LICENSE`.

## Related

- [Argentis Labs](https://argentislabs.io) — supervision infrastructure
  for AI in regulated professions
- [LiveCards demo](https://argentislabs.io/demo/livecards/matter-intake)
  — the individuation layer this substrate composes into
- [*Locally Correct, Globally Wrong*](https://argentislabs.io/research/locally-correct)
  — working paper applying the same primitives to DeFi authorization
- Companion repo: [`argentislabs/locally-correct`](https://github.com/argentislabs/locally-correct)
