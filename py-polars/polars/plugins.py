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
    *,
    plugin_location: Path | str,
    function_name: str,
    inputs: IntoExpr | Iterable[IntoExpr],
    kwargs: dict[str, Any] | None = None,
    is_elementwise: bool = False,
    input_wildcard_expansion: bool = False,
    returns_scalar: bool = False,
    cast_to_supertypes: bool = False,
    pass_name_to_apply: bool = False,
    changes_length: bool = False,
) -> Expr:
    """
    Register a plugin function.

    .. warning::
        This functionality is considered **unstable**. It may be changed
        at any point without it being considered a breaking change.

    .. warning::
        This is highly unsafe as this will call the C function
        loaded by `lib::symbol`.

        The parameters you set dictate how Polars will deal with the function.
        Make sure they are correct!

    See the `user guide <https://docs.pola.rs/user-guide/expressions/plugins/>`_
    for more information about plugins.

    Parameters
    ----------
    plugin_location
        Path to the package where plugin is located. This should either be the dynamic
        library file, or the directory containing it.
    function_name
        Name of the Rust function to register.
    inputs
        Arguments passed to this function. These get passed to the ``inputs``
        argument on the Rust side, and have to be of type Expression (or be
        convertible to expressions).
    kwargs
        Non-expression arguments. They must be JSON serializable.
    is_elementwise
        If the function only operates on scalars, this will potentially trigger fast
        paths.
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
    Expr
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
            str(lib_location),
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


def _get_dynamic_lib_location(path: Path | str) -> Path:
    """Get the file path of the dynamic library file."""
    if not isinstance(path, Path):
        path = Path(path)

    if path.is_file():
        return path

    for p in path.iterdir():
        if _is_dynamic_lib(p):
            return p
    else:
        msg = f"no dynamic library found at path: {path}"
        raise FileNotFoundError(msg)


def _is_dynamic_lib(path: Path) -> bool:
    return path.is_file() and path.suffix in (".so", ".dll", ".pyd")
