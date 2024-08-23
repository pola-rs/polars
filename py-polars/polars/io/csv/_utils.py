from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Sequence, cast
from warnings import warn

from polars._utils.various import find_stacklevel

if TYPE_CHECKING:
    from polars import DataFrame
Arguments = Literal["separator", "quote_char", "eol_char"]


class Hasdecode:
    """Dummy class for typing."""

    def decode(self, encoding: str) -> str:
        return ""


def _check_arg_is_1byte(
    arg_name: str, arg: str | None, *, can_be_empty: bool = False
) -> None:
    if isinstance(arg, str):
        arg_byte_length = len(arg.encode("utf-8"))
        if can_be_empty:
            if arg_byte_length > 1:
                msg = (
                    f'{arg_name}="{arg}" should be a single byte character or empty,'
                    f" but is {arg_byte_length} bytes long"
                )
                raise ValueError(msg)
        elif arg_byte_length != 1:
            msg = (
                f'{arg_name}="{arg}" should be a single byte character, but is'
                f" {arg_byte_length} bytes long"
            )
            raise ValueError(msg)
    elif hasattr(arg, "decode"):
        msg = f'{arg_name}="{arg}" should be a str not a {type(arg)}. To silence this warning please use a str'
        warn(
            msg,
            UserWarning,
            stacklevel=find_stacklevel(),
        )
        _check_arg_is_1byte(
            arg_name, cast(Hasdecode, arg).decode("utf-8"), can_be_empty=can_be_empty
        )
    elif arg is not None:
        msg = f'{arg_name}="{arg}" should be a str not a {type(arg)}'


def _check_fix_1byte_arg(
    arg_name: Arguments,
    arg: str | tuple[str, str],
    *,
    replace_map: list[tuple[str, str]],
) -> str:
    user_supplied_tuple = False
    if arg is None:
        return arg
    if isinstance(arg, tuple):
        user_supplied_tuple = True
        if len(arg) != 2:
            msg = f"If {arg_name} is a tuple, it must be 2 elements"
            raise ValueError(msg)
        arg_search, arg_replace = arg
    elif hasattr(arg, "decode"):
        return _check_fix_1byte_arg(
            arg_name, cast(Hasdecode, arg).decode("utf-8"), replace_map=replace_map
        )
    elif isinstance(arg, str):
        arg_search = arg_replace = arg

    arg_byte_length = len(arg_replace.encode("utf-8"))
    if arg_byte_length == 0:
        msg = f"{arg_name} can't be empty"
        raise ValueError(msg)

    if user_supplied_tuple and arg_byte_length > 1:
        msg = (
            f"When supplying a tuple for {arg_name} the second element must be single byte character, but is"
            f"{arg_byte_length} bytes long"
        )
        raise ValueError(msg)
    elif arg_byte_length > 1:
        if arg_name == "separator":
            arg_replace = "\x1f"
        elif arg_name == "eol_char":
            arg_replace = "\x1d"
        elif arg_name == "quote_char":
            arg_replace = "\x1e"
        else:
            msg = 'arg_name must be one of "separator","quote_char","eol_char"'
            raise ValueError
    if arg_search != arg_replace:
        replace_map.append((arg_search, arg_replace))
    return arg_replace


def _update_columns(df: DataFrame, new_columns: Sequence[str]) -> DataFrame:
    if df.width > len(new_columns):
        cols = df.columns
        for i, name in enumerate(new_columns):
            cols[i] = name
        new_columns = cols
    df.columns = list(new_columns)
    return df
