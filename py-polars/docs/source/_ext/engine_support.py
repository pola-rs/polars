"""Sphinx directive for flagging execution-engine support in the API reference.

Usage in a docstring::

    .. engine-support:: streaming, distributed

Renders a small badge per listed engine. Engines are opt-in: methods without
the directive render exactly as before.
"""

from __future__ import annotations

from typing import Any

from docutils import nodes
from docutils.parsers.rst import Directive

ENGINE_LABELS = {
    "in-memory": "In Memory",
    "streaming": "Streaming",
    "partially-streaming": "Partially Streaming",
    "distributed": "Distributed",
    "partially-distributed": "Partially Distributed",
    "gpu": "GPU",
}


class EngineSupportDirective(Directive):  # noqa: D101
    required_arguments = 0
    optional_arguments = 1
    final_argument_whitespace = True
    has_content = False

    def run(self) -> list[nodes.Node]:  # noqa: D102
        argument = self.arguments[0] if self.arguments else ""
        engines = [token.strip() for token in argument.replace(",", " ").split()]

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
            badges.append(
                f'<span class="engine-tag engine-tag--{engine}">{label}</span>'
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
