from __future__ import annotations

import contextlib
import sys
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

import polars._reexport as pl
from polars._utils.parse import parse_into_list_of_expressions
from polars._utils.wrap import wrap_expr, wrap_s

with contextlib.suppress(ImportError):  # Module not available when building docs
    import polars._plr as plr

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Callable

    from polars import DataType, Expr, Series
    from polars._typing import IntoExpr

__all__ = ["register_plugin_function", "new_v2_plugin"]


def register_plugin_function(
    *,
    plugin_path: Path | str,
    function_name: str,
    args: IntoExpr | Iterable[IntoExpr],
    kwargs: dict[str, Any] | None = None,
    is_elementwise: bool = False,
    changes_length: bool = False,
    returns_scalar: bool = False,
    cast_to_supertype: bool = False,
    input_wildcard_expansion: bool = False,
    pass_name_to_apply: bool = False,
    use_abs_path: bool = False,
) -> Expr:
    """
    Register a plugin function.

    See the `user guide <https://docs.pola.rs/user-guide/plugins/expr_plugins>`_
    for more information about plugins.

    Parameters
    ----------
    plugin_path
        Path to the plugin package. Accepts either the file path to the dynamic library
        file or the path to the directory containing it.
    function_name
        The name of the Rust function to register.
    args
        The arguments passed to this function. These get passed to the `input`
        argument on the Rust side, and have to be expressions (or be convertible
        to expressions).
    kwargs
        Non-expression arguments to the plugin function. These must be
        JSON serializable.
    is_elementwise
        Indicate that the function operates on scalars only. This will potentially
        trigger fast paths.
    changes_length
        Indicate that the function will change the length of the expression.
        For example, a `unique` or `slice` operation.
    returns_scalar
        Automatically explode on unit length if the function ran as final aggregation.
        This is the case for aggregations like `sum`, `min`, `covariance` etc.
    cast_to_supertype
        Cast the input expressions to their supertype.
    input_wildcard_expansion
        Expand wildcard expressions before executing the function.
    pass_name_to_apply
        If set to `True`, the `Series` passed to the function in a group-by operation
        will ensure the name is set. This is an extra heap allocation per group.
    use_abs_path
        If set to `True`, the path will be resolved to an absolute path.
        The path to the dynamic library is relative to the virtual environment by
        default.

    Returns
    -------
    Expr

    Warnings
    --------
    This is highly unsafe as this will call the C function loaded by
    `plugin::function_name`.

    The parameters you set dictate how Polars will handle the function.
    Make sure they are correct!
    """
    pyexprs = parse_into_list_of_expressions(args)
    serialized_kwargs = _serialize_kwargs(kwargs)
    plugin_path = _resolve_plugin_path(plugin_path, use_abs_path=use_abs_path)

    return wrap_expr(
        plr.register_plugin_function(
            plugin_path=str(plugin_path),
            function_name=function_name,
            args=pyexprs,
            kwargs=serialized_kwargs,
            is_elementwise=is_elementwise,
            input_wildcard_expansion=input_wildcard_expansion,
            returns_scalar=returns_scalar,
            cast_to_supertype=cast_to_supertype,
            pass_name_to_apply=pass_name_to_apply,
            changes_length=changes_length,
        )
    )


def _serialize_kwargs(kwargs: dict[str, Any] | None) -> bytes:
    """Serialize the function's keyword arguments."""
    if not kwargs:
        return b""

    import pickle

    # Use the highest pickle protocol supported the serde-pickle crate:
    # https://docs.rs/serde-pickle/latest/serde_pickle/
    return pickle.dumps(kwargs, protocol=5)


@lru_cache(maxsize=16)
def _resolve_plugin_path(path: Path | str, *, use_abs_path: bool = False) -> Path:
    """Get the file path of the dynamic library file."""
    if not isinstance(path, Path):
        path = Path(path)

    if path.is_file():
        return _resolve_file_path(path, use_abs_path=use_abs_path)

    for p in path.iterdir():
        if _is_dynamic_lib(p):
            return _resolve_file_path(p, use_abs_path=use_abs_path)

    msg = f"no dynamic library found at path: {path}"
    raise FileNotFoundError(msg)


def _is_dynamic_lib(path: Path) -> bool:
    return path.is_file() and path.suffix in (".so", ".dll", ".pyd")


def _resolve_file_path(path: Path, *, use_abs_path: bool = False) -> Path:
    venv_path = Path(sys.prefix)

    if use_abs_path:
        return path.resolve()
    else:
        try:
            file_path = path.relative_to(venv_path)
        except ValueError:  # Fallback
            file_path = path.resolve()

    return file_path


T = TypeVar("T")


def new_v2_plugin(
    *inputs: Expr,
    data: T,
    initialize: Callable[[T, pl.Schema], Any],
    insert: Callable[[T, Any, list[Series]], Series | None],
    finalize: Callable[[T, Any], Series | None] | None,
    combine: Callable[[T, Any, Any], Any] | None,
    new_empty: Callable[[T, Any, pl.Schema], Any],
    to_field: Callable[[T, pl.Schema], tuple[str, DataType]],
    format: Callable[[T], str],

    length_preserving: bool = False,
    row_separable: bool = False,
    returns_scalar: bool = False,
    zippable_inputs: bool = True,
    insert_has_output: bool = True,
    selection_expansion: bool = False,
) -> Expr:
    def wrap_initialize(data: T, schema: dict[str, DataType]) -> Any:
        return initialize(data, pl.Schema(schema))

    def wrap_insert(
        data: T, state: Any, inputs: list[plr.PySeries]
    ) -> plr.PySeries | None:
        inputs_s = [wrap_s(s) for s in inputs]
        out = insert(data, state, inputs_s)
        if out is None:
            return None
        else:
            return out._s

    def wrap_finalize(data: T, state: Any) -> plr.PySeries | None:
        if finalize is None:
            return None

        out = finalize(data, state)
        if out is None:
            return None
        else:
            return out._s

    def wrap_new_empty(data: T, state: Any, schema: dict[str, DataType]) -> Any:
        return new_empty(data, state, pl.Schema(schema))

    def wrap_to_field(data: T, schema: dict[str, DataType]) -> (str, DataType):
        return to_field(data, pl.Schema(schema))

    return wrap_expr(
        plr.plugin_v2_generate(
            inputs=[i._pyexpr for i in inputs],
            data=data,
            initialize=wrap_initialize,
            insert=wrap_insert,
            finalize=None if finalize is None else wrap_finalize,
            combine=combine,
            new_empty=new_empty,
            to_field=wrap_to_field,
            format=format,

            length_preserving=length_preserving,
            row_separable=row_separable,
            returns_scalar=returns_scalar,
            zippable_inputs=zippable_inputs,
            insert_has_output=insert_has_output,
            selection_expansion=selection_expansion,
        )
    )
