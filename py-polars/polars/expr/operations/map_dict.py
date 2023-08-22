from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable

import polars._reexport as pl
from polars import functions as F
from polars.datatypes import FLOAT_DTYPES, INTEGER_DTYPES, Categorical, Struct, Utf8
from polars.utils._parse_expr_input import parse_as_expression
from polars.utils._wrap import wrap_expr

if TYPE_CHECKING:
    from polars import Expr, Series
    from polars.type_aliases import PolarsDataType


def map_dict(
    expr: Expr,
    remapping: dict[Any, Any],
    *,
    default: Any = None,
    return_dtype: PolarsDataType | None = None,
) -> Expr:
    """Implementation of `Expr.map_dict`."""

    def _remap_key_or_value_series(
        name: str,
        values: Iterable[Any],
        dtype: PolarsDataType | None,
        dtype_if_empty: PolarsDataType | None,
        dtype_keys: PolarsDataType | None,
        is_keys: bool,
    ) -> Series:
        """
        Convert remapping keys or remapping values to `Series` with `dtype`.

        Try to convert the remapping keys or remapping values to `Series` with
        the specified dtype and check that none of the values are accidentally
        lost (replaced by nulls) during the conversion.

        Parameters
        ----------
        name
            Name of the keys or values series.
        values
            Values for the series: `remapping.keys()` or `remapping.values()`.
        dtype
            User specified dtype. If None,
        dtype_if_empty
            If no dtype is specified and values contains None, an empty list,
            or a list with only None values, set the Polars dtype of the Series
            data.
        dtype_keys
            If user set dtype is None, try to see if Series for remapping.values()
            can be converted to same dtype as the remapping.keys() Series dtype.
        is_keys
            If values contains keys or values from remapping dict.

        """
        try:
            if dtype is None:
                # If no dtype was set, which should only happen when:
                #     values = remapping.values()
                # create a Series from those values and infer the dtype.
                s = pl.Series(
                    name,
                    values,
                    dtype=None,
                    dtype_if_empty=dtype_if_empty,
                    strict=True,
                )

                if dtype_keys is not None:
                    if s.dtype == dtype_keys:
                        # Values Series has same dtype as keys Series.
                        dtype = s.dtype
                    elif (
                        (s.dtype in INTEGER_DTYPES and dtype_keys in INTEGER_DTYPES)
                        or (s.dtype in FLOAT_DTYPES and dtype_keys in FLOAT_DTYPES)
                        or (s.dtype == Utf8 and dtype_keys == Categorical)
                    ):
                        # Values Series and keys Series are of similar dtypes,
                        # that we can assume that the user wants the values Series
                        # of the same dtype as the key Series.
                        dtype = dtype_keys
                        s = pl.Series(
                            name,
                            values,
                            dtype=dtype_keys,
                            dtype_if_empty=dtype_if_empty,
                            strict=True,
                        )
                        if dtype != s.dtype:
                            raise ValueError(
                                f"remapping values for `map_dict` could not be converted to {dtype!r}: found {s.dtype!r}"
                            )
            else:
                # dtype was set, which should always be the case when:
                #     values = remapping.keys()
                # and in cases where the user set the output dtype when:
                #     values = remapping.values()
                s = pl.Series(
                    name,
                    values,
                    dtype=dtype,
                    dtype_if_empty=dtype_if_empty,
                    strict=True,
                )
                if dtype != s.dtype:
                    raise ValueError(
                        f"remapping {'keys' if is_keys else 'values'} for `map_dict` could not be converted to {dtype!r}: found {s.dtype!r}"
                    )

        except OverflowError as exc:
            if is_keys:
                raise ValueError(
                    f"remapping keys for `map_dict` could not be converted to {dtype!r}: {exc!s}"
                ) from exc
            else:
                raise ValueError(
                    f"choose a more suitable output dtype for map_dict as remapping value could not be converted to {dtype!r}: {exc!s}"
                ) from exc

        if is_keys:
            # values = remapping.keys()
            if s.null_count() == 0:  # noqa: SIM114
                pass
            elif s.null_count() == 1 and None in remapping:
                pass
            else:
                raise ValueError(
                    f"remapping keys for `map_dict` could not be converted to {dtype!r} without losing values in the conversion"
                )
        else:
            # values = remapping.values()
            if s.null_count() == 0:  # noqa: SIM114
                pass
            elif s.len() - s.null_count() == len(list(filter(None, values))):
                pass
            else:
                raise ValueError(
                    f"remapping values for `map_dict` could not be converted to {dtype!r} without losing values in the conversion"
                )

        return s

    # Use two functions to save unneeded work.
    # This factors out allocations and branches.
    def inner_with_default(s: Series) -> Series:
        # Convert Series to:
        #   - multicolumn DataFrame, if Series is a Struct.
        #   - one column DataFrame in other cases.
        df = s.to_frame().unnest(s.name) if s.dtype == Struct else s.to_frame()

        # For struct we always apply mapping to the first column.
        column = df.columns[0]
        input_dtype = df.dtypes[0]
        remap_key_column = f"__POLARS_REMAP_KEY_{column}"
        remap_value_column = f"__POLARS_REMAP_VALUE_{column}"
        is_remapped_column = f"__POLARS_REMAP_IS_REMAPPED_{column}"

        # Set output dtype:
        #  - to dtype, if specified.
        #  - to same dtype as expression specified as default value.
        #  - to None, if dtype was not specified and default was not an expression.
        return_dtype_ = (
            df.lazy().select(default).dtypes[0]
            if return_dtype is None and isinstance(default, pl.Expr)
            else return_dtype
        )

        remap_key_s = _remap_key_or_value_series(
            name=remap_key_column,
            values=remapping.keys(),
            dtype=input_dtype,
            dtype_if_empty=input_dtype,
            dtype_keys=input_dtype,
            is_keys=True,
        )

        if return_dtype_:
            # Create remap value Series with specified output dtype.
            remap_value_s = pl.Series(
                remap_value_column,
                remapping.values(),
                dtype=return_dtype_,
                dtype_if_empty=input_dtype,
            )
        else:
            # Create remap value Series with same output dtype as remap key Series,
            # if possible (if both are integers, both are floats or remap value
            # Series is pl.Utf8 and remap key Series is pl.Categorical).
            remap_value_s = _remap_key_or_value_series(
                name=remap_value_column,
                values=remapping.values(),
                dtype=None,
                dtype_if_empty=input_dtype,
                dtype_keys=input_dtype,
                is_keys=False,
            )

        default_parsed = wrap_expr(parse_as_expression(default, str_as_lit=True))
        return (
            (
                df.lazy()
                .join(
                    pl.DataFrame(
                        [
                            remap_key_s,
                            remap_value_s,
                        ]
                    )
                    .lazy()
                    .with_columns(F.lit(True).alias(is_remapped_column)),
                    how="left",
                    left_on=column,
                    right_on=remap_key_column,
                )
                .select(
                    F.when(F.col(is_remapped_column).is_not_null())
                    .then(F.col(remap_value_column))
                    .otherwise(default_parsed)
                    .alias(column)
                )
            )
            .collect(no_optimization=True)
            .to_series()
        )

    def inner(s: Series) -> Series:
        column = s.name
        input_dtype = s.dtype
        remap_key_column = f"__POLARS_REMAP_KEY_{column}"
        remap_value_column = f"__POLARS_REMAP_VALUE_{column}"
        is_remapped_column = f"__POLARS_REMAP_IS_REMAPPED_{column}"

        remap_key_s = _remap_key_or_value_series(
            name=remap_key_column,
            values=list(remapping.keys()),
            dtype=input_dtype,
            dtype_if_empty=input_dtype,
            dtype_keys=input_dtype,
            is_keys=True,
        )

        if return_dtype:
            # Create remap value Series with specified output dtype.
            remap_value_s = pl.Series(
                remap_value_column,
                remapping.values(),
                dtype=return_dtype,
                dtype_if_empty=input_dtype,
            )
        else:
            # Create remap value Series with same output dtype as remap key Series,
            # if possible (if both are integers, both are floats or remap value
            # Series is pl.Utf8 and remap key Series is pl.Categorical).
            remap_value_s = _remap_key_or_value_series(
                name=remap_value_column,
                values=remapping.values(),
                dtype=None,
                dtype_if_empty=input_dtype,
                dtype_keys=input_dtype,
                is_keys=False,
            )

        return (
            (
                s.to_frame()
                .lazy()
                .join(
                    pl.DataFrame(
                        [
                            remap_key_s,
                            remap_value_s,
                        ]
                    )
                    .lazy()
                    .with_columns(F.lit(True).alias(is_remapped_column)),
                    how="left",
                    left_on=column,
                    right_on=remap_key_column,
                )
            )
            .collect(no_optimization=True)
            .to_series(1)
        )

    func = inner_with_default if default is not None else inner
    return expr.map(func)
