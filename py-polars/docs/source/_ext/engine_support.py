"""Sphinx directive for flagging execution-engine support in the API reference.

Usage in a docstring::

    .. engine-support:: streaming, distributed

Renders a small badge per listed engine. Engines are opt-in: methods without
the directive render exactly as before.

Every badge explains itself on hover. For a ``partially-*`` engine, pass the
reason support is partial as an option named after that engine::

    .. engine-support:: in-memory, partially-streaming
        :partially-streaming: Falls back to in-memory in a group-by context.
"""

from __future__ import annotations

from html import escape
from typing import Any

from docutils import nodes
from docutils.parsers.rst import Directive, directives

ENGINE_LABELS = {
    "in-memory": "In Memory",
    "streaming": "Streaming",
    "partially-streaming": "Partially Streaming",
    "distributed": "Distributed",
    "partially-distributed": "Partially Distributed",
    "gpu": "GPU",
}

ENGINE_TOOLTIPS = {
    "in-memory": "Supported by the in-memory engine.",
    "streaming": "Supported by the streaming engine.",
    "partially-streaming": (
        "Only partially supported by the streaming engine; some usages fall back "
        "to the in-memory engine."
    ),
    "distributed": "Supported by the distributed engine.",
    "partially-distributed": (
        "Only partially supported by the distributed engine; some usages fall back "
        "to a single node."
    ),
    "gpu": "Supported by the GPU engine.",
}


class EngineSupportDirective(Directive):  # noqa: D101
    required_arguments = 0
    optional_arguments = 1
    final_argument_whitespace = True
    has_content = False
    option_spec = dict.fromkeys(ENGINE_LABELS, directives.unchanged)

    def run(self) -> list[nodes.Node]:  # noqa: D102
        argument = self.arguments[0] if self.arguments else ""
        engines = [token.strip() for token in argument.replace(",", " ").split()]

        for engine in self.options:
            if engine not in engines:
                self.state_machine.reporter.warning(
                    f"Explanation given for engine {engine!r}, which is not listed "
                    "in the 'engine-support' directive.",
                    line=self.lineno,
                )

        badges = []
        for engine in engines:
            label = ENGINE_LABELS.get(engine)
            if label is None:
                self.state_machine.reporter.warning(
                    f"Unknown engine {engine!r} in 'engine-support' directive. "
                    f"Expected one of: {', '.join(sorted(ENGINE_LABELS))}.",
                    line=self.lineno,
                )
                continue

            tooltip = " ".join(self.options.get(engine, "").split())
            if not tooltip:
                tooltip = ENGINE_TOOLTIPS[engine]
            badges.append(
                f'<span class="engine-tag engine-tag--{engine}" tabindex="0" '
                f'aria-label="{escape(f"{label}. {tooltip}", quote=True)}" '
                f'data-engine-tooltip="{escape(tooltip, quote=True)}">{label}</span>'
            )

        if not badges:
            return []

        html = (
            '<div class="engine-tags">'
            '<span class="engine-tags-label">engine:</span>'
            f"{''.join(badges)}"
            "</div>"
        )
        return [nodes.raw("", html, format="html")]


def setup(app: Any) -> dict[str, Any]:  # noqa: D103
    app.add_directive("engine-support", EngineSupportDirective)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
