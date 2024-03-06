from __future__ import annotations

import contextlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

from polars._utils.parse_expr_input import parse_as_list_of_expressions
from polars._utils.unstable import unstable
from polars._utils.wrap import wrap_expr

if TYPE_CHECKING:
    from polars import Expr
    from polars.type_aliases import IntoExpr

with contextlib.suppress(ImportError):  # Module not available when building docs
    import polars.polars as plr

__all__ = ["register_plugin"]


@unstable()
def register_plugin(
    plugin_location: str | Path,
    function_name: str,
    inputs: IntoExpr | Iterable[IntoExpr],
    kwargs: dict[str, Any] | None = None,
    *,
    is_elementwise: bool = False,
    input_wildcard_expansion: bool = False,
    returns_scalar: bool = False,
    cast_to_supertypes: bool = False,
    pass_name_to_apply: bool = False,
    changes_length: bool = False,
) -> Expr:
    """
    Register a dynamic library as a plugin.

    .. warning::
        This functionality is unstable and may change without it
        being considered breaking.

    .. warning::
        This is highly unsafe as this will call the C function
        loaded by `lib::symbol`.

        The parameters you give dictate how polars will deal
        with the function. Make sure they are correct!

    See the `user guide <https://docs.pola.rs/user-guide/expressions/plugins/>`_
    for more information about plugins.

    Parameters
    ----------
    plugin_location
        Path to package where plugin is located. This should either be the
        directory containing the plugin module, or a path to the dynamic library file.
    function_name
        Rust function to load.
    inputs
        Arguments passed to this function. These get passed to the ``inputs``
        argument on the Rust side, and have to be of type Expression (or be
        convertible to expressions).
    kwargs
        Non-expression arguments. They must be JSON serializable.
    is_elementwise
        If the function only operates on scalars
        this will trigger fast paths.
    input_wildcard_expansion
        Expand expressions as input of this function.
    returns_scalar
        Automatically explode on unit length if it ran as final aggregation.
        this is the case for aggregations like `sum`, `min`, `covariance` etc.
    cast_to_supertypes
        Cast the input datatypes to their supertype.
    pass_name_to_apply
        if set, then the `Series` passed to the function in the group_by operation
        will ensure the name is set. This is an extra heap allocation per group.
    changes_length
        For example a `unique` or a `slice`

    Returns
    -------
    polars.Expr
    """
    pyexprs = parse_as_list_of_expressions(inputs)
    if not pyexprs:
        msg = "`inputs` must be non-empty"
        raise TypeError(msg)
    if kwargs is None:
        serialized_kwargs = b""
    else:
        import pickle

        # Choose the highest protocol supported by https://docs.rs/serde-pickle/latest/serde_pickle/
        serialized_kwargs = pickle.dumps(kwargs, protocol=5)

    lib_location = _get_dynamic_lib_location(plugin_location)

    return wrap_expr(
        plr.register_plugin(
            lib_location,
            function_name,
            pyexprs,
            serialized_kwargs,
            is_elementwise,
            input_wildcard_expansion,
            returns_scalar,
            cast_to_supertypes,
            pass_name_to_apply,
            changes_length,
        )
    )


def _get_dynamic_lib_location(plugin_location: str | Path) -> str:
    """Get location of dynamic library file."""
    if Path(plugin_location).is_file():
        return str(plugin_location)
    if not Path(plugin_location).is_dir():
        msg = f"expected file or directory, got {plugin_location!r}"
        raise TypeError(msg)
    for path in Path(plugin_location).iterdir():
        if _is_shared_lib(path):
            return str(path)
    msg = f"no dynamic library found in {plugin_location}"
    raise FileNotFoundError(msg)


def _is_shared_lib(file: Path) -> bool:
    return file.name.endswith((".so", ".dll", ".pyd"))
