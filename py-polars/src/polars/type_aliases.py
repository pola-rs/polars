"""
Deprecated module - do not use.

Used to contain private type aliases. These are now in the `polars._typing` module.
"""
import textwrap

from typing import Any

import polars._typing as plt
from polars._utils.deprecation import issue_deprecation_warning

def __getattr__(name: str) -> Any:
    if hasattr(plt, name):
        warning_msg = textwrap.dedent("""
            The `polars.type_aliases` module was deprecated in version 1.0.0.
            The type aliases have moved to the `polars._typing` module to explicitly mark them as private.
            Please define your own type aliases, or temporarily import from the `polars._typing` module.
            A public `polars.typing` module will be added in the future.
        """)
        issue_deprecation_warning(warning_msg, stacklevel=2)
        return getattr(plt, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
