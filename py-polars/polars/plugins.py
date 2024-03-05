from __future__ import annotations

from typing import TYPE_CHECKING, Any

from polars._utils.parse_expr_input import parse_as_list_of_expressions
from polars._utils.udfs import get_dynamic_lib_location
from polars._utils.unstable import unstable
from polars._utils.wrap import wrap_expr

if TYPE_CHECKING:
    from pathlib import Path

    from polars import Expr
    from polars.type_aliases import IntoExpr

__all__ = ["register_plugin"]


@unstable()
def register_plugin(
    *inputs: IntoExpr | list[IntoExpr],
    plugin_location: str | Path,
    symbol: str,
    kwargs: dict[Any, Any] | None = None,
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
        This is highly unsafe as this will call the C function
        loaded by `lib::symbol`.

        The parameters you give dictate how polars will deal
        with the function. Make sure they are correct!

    .. note::
        This functionality is unstable and may change without it
        being considered breaking.

    See the `user guide <https://docs.pola.rs/user-guide/expressions/plugins/>`_
    for more information about plugins.

    Parameters
    ----------
    inputs
        Arguments passed to this function. These get passed to the ``inputs``
        argument on the Rust side, and have to be of type Expression (or be
        convertible to expressions).
    plugin_location
        Path to package where plugin is located. This can either be the
        ``__init__.py`` file or the directory containing the ``__init__.py`` file.
    symbol
        Function to load.
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
    Expr
    """
    if not inputs:
        msg = "`args` must be non-empty"
        raise TypeError(msg)
    pyexprs = parse_as_list_of_expressions(*inputs)
    if not pyexprs:
        msg = "`args` must be non-empty"
        raise TypeError(msg)
    if kwargs is None:
        serialized_kwargs = b""
    else:
        import pickle

        # Choose the highest protocol supported by https://docs.rs/serde-pickle/latest/serde_pickle/
        serialized_kwargs = pickle.dumps(kwargs, protocol=5)

    lib_location = get_dynamic_lib_location(plugin_location)

    return wrap_expr(
        pyexprs[0].register_plugin(
            lib_location,
            symbol,
            pyexprs[1:],
            serialized_kwargs,
            is_elementwise,
            input_wildcard_expansion,
            returns_scalar,
            cast_to_supertypes,
            pass_name_to_apply,
            changes_length,
        )
    )
