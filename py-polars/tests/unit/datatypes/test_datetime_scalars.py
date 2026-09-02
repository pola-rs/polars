"""
A full coverage test of using a `pd.Timestamp` to create a series/dataframe/expr.

The test basically is a big exhaustive list of Ingestion-Paths for pd.Timestamp into
polars: For each path we check that the resulting series/dataframe does not lose
precision and has the implied precision of the pd.Timestamp
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Literal, cast

import pandas as pd
import pytest

import polars as pl
from polars.testing import assert_series_equal

if TYPE_CHECKING:
    from collections.abc import Callable

    from polars._typing import TimeUnit

# All the pd.Timestamps we are testing in our tests, cross product with each
# resolution we allow
PD_TIMESTAMPS = [
    (base.as_unit(unit), expected_time_unit)
    # annotated so that `as_unit` keeps its `Literal` argument type
    for unit, expected_time_unit in cast(
        "list[tuple[Literal['ms', 'us', 'ns', 's'], TimeUnit]]",
        [
            ("ms", "ms"),
            ("us", "us"),
            ("ns", "ns"),
            ("s", "ms"),
        ],
    )
    for base in (
        pd.Timestamp("2026-06-15 10:20:30.123456789"),
        pd.Timestamp.min,
        pd.Timestamp.max,
    )
]


@dataclass(frozen=True)
class IngestionPath:
    """
    One way (Method/expr/constructor) of getting a `pd.Timestamp` into polars.

    - `create` ingests the `pd.Timestamp` without saying anything about the target
      dtype, so polars should infer the resolution from the value itself.
    - `create_typed` ingests it while naming the resolution the value should be cast
      to, via a `dtype`/`schema` argument where the path has one and a cast on the
      result otherwise.
    - `extract` extracts a `pl.Series` holding the value out of whatever `create`
      returned (a frame, a nested column, ...), this is needed as we can't unwrap
      back to python without losing the resolution.
    """

    name: str  # Self set identifier
    create: Callable[[pd.Timestamp], Any]
    create_typed: Callable[[pd.Timestamp], Any]
    extract: Callable[[Any], pl.Series] = field(default=lambda result: result)


def _dt(value: pd.Timestamp) -> pl.Datetime:
    unit: TimeUnit = "ms" if value.unit == "s" else value.unit
    return pl.Datetime(unit)


def _cast(frame: pl.DataFrame, value: pd.Timestamp) -> pl.DataFrame:
    return frame.with_columns(pl.col("t").cast(_dt(value)))


def _null_like(value: pd.Timestamp) -> pl.Series:
    return pl.Series("t", [None], dtype=_dt(value))


def _empty_like(value: pd.Timestamp) -> pl.Series:
    return pl.Series("t", [], dtype=_dt(value))


# This is an exhaustive list of a pd/Timestamp reaching polars
# fmt: off
INGESTION_PATHS = [
    # Series constructors. The sequence is scanned for its first non-null value, whose
    # python type picks the dtype, then every element is converted one by one through
    # `py_object_to_any_value`. Nested inputs recurse into the same converter.
    IngestionPath(
        "series_list",
        lambda value: pl.Series([value]),
        lambda value: pl.Series([value], dtype=_dt(value)),
    ),
    IngestionPath(
        "series_tuple",
        lambda value: pl.Series((value,)),
        lambda value: pl.Series((value,), dtype=_dt(value)),
    ),
    IngestionPath(
        "series_generator",
        lambda value: pl.Series(item for item in [value]),
        lambda value: pl.Series((item for item in [value]), dtype=_dt(value)),
    ),
    IngestionPath(
        "series_nested_list",
        lambda value: pl.Series([[value]]),
        lambda value: pl.Series([[value]], dtype=pl.List(_dt(value))),
        lambda result: result.explode(),
    ),
    IngestionPath(
        "series_nested_struct",
        lambda value: pl.Series([{"t": value}]),
        lambda value: pl.Series([{"t": value}], dtype=pl.Struct({"t": _dt(value)})),
        lambda result: result.struct.field("t"),
    ),


    # Series methods that take a ready-made `Series` or a raw python object and builds
    # another `Series` from it. Ends up on the same path as the constructors above.
    IngestionPath(
        "series_scatter",
        lambda value: _null_like(value).scatter(0, value),
        lambda value: _null_like(value).scatter(0, value).cast(_dt(value)),
    ),
    IngestionPath(
        "series_set",
        lambda value: _null_like(value).set(pl.Series([True]), value),
        lambda value: _null_like(value).set(pl.Series([True]), value).cast(_dt(value)),
    ),
    IngestionPath(
        "series_zip_with",
        lambda value: _null_like(value).zip_with(
            pl.Series([False]), pl.Series([value])
        ),
        lambda value: (
            _null_like(value)
            .zip_with(pl.Series([False]), pl.Series([value]))
            .cast(_dt(value))
        ),
    ),
    IngestionPath(
        "series_append",
        lambda value: _empty_like(value).append(pl.Series([value])),
        lambda value: _empty_like(value).append(pl.Series([value], dtype=_dt(value))),
    ),


    # Series methods that take a scalar fill value. These wrap it in `lit()` first, so
    # they depend on how `lit` picks a resolution rather than on sequence inference.
    IngestionPath(
        "series_fill_null",
        lambda value: _null_like(value).fill_null(value),
        lambda value: _null_like(value).fill_null(value).cast(_dt(value)),
    ),
    IngestionPath(
        "series_shift_fill",
        lambda value: _null_like(value).shift(1, fill_value=value),
        lambda value: _null_like(value).shift(1, fill_value=value).cast(_dt(value)),
    ),
    IngestionPath(
        "series_extend_constant",
        lambda value: _empty_like(value).extend_constant(value, 1),
        lambda value: _empty_like(value).extend_constant(value, 1).cast(_dt(value)),
    ),
    IngestionPath(
        "series_replace",
        lambda value: _null_like(value).replace(None, value),
        lambda value: _null_like(value).replace(None, value).cast(_dt(value)),
    ),


    # Expressions built from a python scalar. All of these funnel through `lit()`, which
    # reads the resolution off the value and stamps it onto the literal.
    IngestionPath(
        "lit",
        lambda value: pl.select(pl.lit(value).alias("t")),
        lambda value: pl.select(pl.lit(value, dtype=_dt(value)).alias("t")),
        lambda result: result.get_column("t"),
    ),
    IngestionPath(
        "select_function",
        lambda value: pl.select(t=value),
        lambda value: pl.select(t=pl.lit(value, dtype=_dt(value))),
        lambda result: result.get_column("t"),
    ),
    IngestionPath(
        "repeat",
        lambda value: pl.select(pl.repeat(value, 1).alias("t")),
        lambda value: pl.select(pl.repeat(value, 1, dtype=_dt(value)).alias("t")),
        lambda result: result.get_column("t"),
    ),
    IngestionPath(
        "when_then",
        lambda value: pl.select(pl.when(True).then(pl.lit(value)).alias("t")),
        lambda value: pl.select(
            pl.when(True).then(pl.lit(value)).cast(_dt(value)).alias("t")
        ),
        lambda result: result.get_column("t"),
    ),
    IngestionPath(
        "coalesce",
        lambda value: pl.select(
            pl.coalesce(pl.lit(None, dtype=_dt(value)), pl.lit(value)).alias("t")
        ),
        lambda value: pl.select(
            pl.coalesce(pl.lit(None, dtype=_dt(value)), pl.lit(value))
            .cast(_dt(value))
            .alias("t")
        ),
        lambda result: result.get_column("t"),
    ),
    IngestionPath(
        "min_horizontal",
        lambda value: pl.select(pl.min_horizontal(pl.lit(value)).alias("t")),
        lambda value: pl.select(
            pl.min_horizontal(pl.lit(value)).cast(_dt(value)).alias("t")
        ),
        lambda result: result.get_column("t"),
    ),
    IngestionPath(
        "struct_function",
        lambda value: pl.select(pl.struct(pl.lit(value).alias("t")).alias("s")),
        lambda value: pl.select(
            pl.struct(pl.lit(value).cast(_dt(value)).alias("t")).alias("s")
        ),
        lambda result: result.get_column("s").struct.field("t"),
    ),
    IngestionPath(
        "concat_list",
        lambda value: pl.select(pl.concat_list(pl.lit(value)).alias("t")),
        lambda value: pl.select(
            pl.concat_list(pl.lit(value).cast(_dt(value))).alias("t")
        ),
        lambda result: result.get_column("t").explode(),
    ),
    IngestionPath(
        "datetime_range",
        lambda value: pl.datetime_range(value, value, eager=True),
        lambda value: pl.datetime_range(
            value, value, time_unit=_dt(value).time_unit, eager=True
        ),
    ),
    IngestionPath(
        "fold",
        lambda value: pl.select(
            pl.fold(pl.lit(value), lambda acc, _: acc, [pl.lit(value)]).alias("t")
        ),
        lambda value: pl.select(
            pl.fold(pl.lit(value), lambda acc, _: acc, [pl.lit(value)])
            .cast(_dt(value))
            .alias("t")
        ),
        lambda result: result.get_column("t"),
    ),
    IngestionPath(
        "reduce",
        lambda value: pl.select(
            pl.reduce(lambda acc, _: acc, [pl.lit(value), pl.lit(value)]).alias("t")
        ),
        lambda value: pl.select(
            pl.reduce(lambda acc, _: acc, [pl.lit(value), pl.lit(value)])
            .cast(_dt(value))
            .alias("t")
        ),
        lambda result: result.get_column("t"),
    ),
    IngestionPath(
        "list_eval",
        lambda value: pl.select(pl.lit([1]).list.eval(pl.lit(value)).alias("t")),
        lambda value: pl.select(
            pl.lit([1]).list.eval(pl.lit(value).cast(_dt(value))).alias("t")
        ),
        lambda result: result.get_column("t").explode(),
    ),
    IngestionPath(
        "over",
        lambda value: pl.DataFrame({"g": [0]}).select(
            pl.lit(value).first().over("g").alias("t")
        ),
        lambda value: pl.DataFrame({"g": [0]}).select(
            pl.lit(value).cast(_dt(value)).first().over("g").alias("t")
        ),
        lambda result: result.get_column("t"),
    ),
    IngestionPath(
        "group_by_agg",
        lambda value: (
            pl.DataFrame({"g": [0]}).group_by("g").agg(pl.lit(value).first().alias("t"))
        ),
        lambda value: (
            pl.DataFrame({"g": [0]})
            .group_by("g")
            .agg(pl.lit(value).cast(_dt(value)).first().alias("t"))
        ),
        lambda result: result.get_column("t"),
    ),


    # Frame and LazyFrame methods that only forward an expression. They add no ingestion
    # of their own, so they should fail as the `lit()` case above
    IngestionPath(
        "dataframe_with_columns",
        lambda value: pl.DataFrame({"x": [0]}).with_columns(t=pl.lit(value)),
        lambda value: pl.DataFrame({"x": [0]}).with_columns(
            t=pl.lit(value).cast(_dt(value))
        ),
        lambda result: result.get_column("t"),
    ),
    IngestionPath(
        "dataframe_with_columns_seq",
        lambda value: pl.DataFrame({"x": [0]}).with_columns_seq(t=pl.lit(value)),
        lambda value: pl.DataFrame({"x": [0]}).with_columns_seq(
            t=pl.lit(value).cast(_dt(value))
        ),
        lambda result: result.get_column("t"),
    ),
    IngestionPath(
        "dataframe_select",
        lambda value: pl.DataFrame({"x": [0]}).select(t=pl.lit(value)),
        lambda value: pl.DataFrame({"x": [0]}).select(t=pl.lit(value).cast(_dt(value))),
        lambda result: result.get_column("t"),
    ),
    IngestionPath(
        "dataframe_select_seq",
        lambda value: pl.DataFrame({"x": [0]}).select_seq(t=pl.lit(value)),
        lambda value: pl.DataFrame({"x": [0]}).select_seq(
            t=pl.lit(value).cast(_dt(value))
        ),
        lambda result: result.get_column("t"),
    ),
    IngestionPath(
        "lazyframe_select",
        lambda value: pl.LazyFrame({"x": [0]}).select(t=pl.lit(value)).collect(),
        lambda value: (
            pl.LazyFrame({"x": [0]}).select(t=pl.lit(value).cast(_dt(value))).collect()
        ),
        lambda result: result.get_column("t"),
    ),
    IngestionPath(
        "lazyframe_with_columns",
        lambda value: pl.LazyFrame({"x": [0]}).with_columns(t=pl.lit(value)).collect(),
        lambda value: (
            pl.LazyFrame({"x": [0]})
            .with_columns(t=pl.lit(value).cast(_dt(value)))
            .collect()
        ),
        lambda result: result.get_column("t"),
    ),


    # Python callbacks. Should be treated through the ordinary `Series`/`AnyValue`
    # conversion once the engine has collected the results of the python calls
    IngestionPath(
        "series_map_elements",
        lambda value: _null_like(value).map_elements(lambda _: value, skip_nulls=False),
        lambda value: _null_like(value).map_elements(
            lambda _: value, return_dtype=_dt(value), skip_nulls=False
        ),
    ),
    IngestionPath(
        "expr_map_elements",
        lambda value: pl.select(
            pl.lit(None, dtype=_dt(value))
            .map_elements(lambda _: value, skip_nulls=False)
            .alias("t")
        ),
        lambda value: pl.select(
            pl.lit(None, dtype=_dt(value))
            .map_elements(lambda _: value, return_dtype=_dt(value), skip_nulls=False)
            .alias("t")
        ),
        lambda result: result.get_column("t"),
    ),
    IngestionPath(
        "expr_map_batches",
        lambda value: pl.select(
            pl.lit(0).map_batches(lambda _: pl.Series([value])).alias("t")
        ),
        lambda value: pl.select(
            pl.lit(0)
            .map_batches(lambda _: pl.Series([value]), return_dtype=_dt(value))
            .alias("t")
        ),
        lambda result: result.get_column("t"),
    ),
    IngestionPath(
        "dataframe_map_rows",
        lambda value: pl.DataFrame({"x": [0]}).map_rows(lambda _: value),
        lambda value: pl.DataFrame({"x": [0]}).map_rows(
            lambda _: value, return_dtype=_dt(value)
        ),
        lambda result: result.to_series(0),
    ),
    IngestionPath(
        "function_map_groups",
        lambda value: (
            pl.DataFrame({"g": [0], "x": [0]})
            .group_by("g")
            .agg(pl.map_groups(["x"], lambda _: value, returns_scalar=True).alias("t"))
        ),
        lambda value: (
            pl.DataFrame({"g": [0], "x": [0]})
            .group_by("g")
            .agg(
                pl.map_groups(
                    ["x"], lambda _: value, return_dtype=_dt(value), returns_scalar=True
                ).alias("t")
            )
        ),
        lambda result: result.get_column("t"),
    ),


    # Column-oriented full dataframe constructors. Each column is handed to the `Series`
    # constructor separately, so these reduce to `sequence_to_pyseries` per column
    IngestionPath(
        "dataframe_column_sequence",
        lambda value: pl.DataFrame({"t": [value]}),
        lambda value: pl.DataFrame({"t": [value]}, schema={"t": _dt(value)}),
        lambda result: result.get_column("t"),
    ),
    IngestionPath(
        "dataframe_scalar_dict",
        lambda value: pl.DataFrame({"t": value}),
        lambda value: pl.DataFrame({"t": value}, schema={"t": _dt(value)}),
        lambda result: result.get_column("t"),
    ),
    IngestionPath(
        "lazyframe_column_sequence",
        lambda value: pl.LazyFrame({"t": [value]}).collect(),
        lambda value: pl.LazyFrame({"t": [value]}, schema={"t": _dt(value)}).collect(),
        lambda result: result.get_column("t"),
    ),
    IngestionPath(
        "from_dict",
        lambda value: pl.from_dict({"t": [value]}),
        lambda value: pl.from_dict({"t": [value]}, schema={"t": _dt(value)}),
        lambda result: result.get_column("t"),
    ),
    IngestionPath(
        "from_records_columns",
        lambda value: pl.from_records([[value]], schema=["t"], orient="col"),
        lambda value: pl.from_records(
            [[value]], schema={"t": _dt(value)}, orient="col"
        ),
        lambda result: result.get_column("t"),
    ),


    # Row-oriented full dataframe constructors. These go through `from_rows`/
    # `from_dicts` on the Rust side, which infers a dtype from the first rows and
    # appends into a buffer
    IngestionPath(
        "dataframe_rows_list",
        lambda value: pl.DataFrame([[value]], schema=["t"], orient="row"),
        lambda value: pl.DataFrame([[value]], schema={"t": _dt(value)}, orient="row"),
        lambda result: result.get_column("t"),
    ),
    IngestionPath(
        "dataframe_rows_tuple",
        lambda value: pl.DataFrame([(value,)], schema=["t"], orient="row"),
        lambda value: pl.DataFrame([(value,)], schema={"t": _dt(value)}, orient="row"),
        lambda result: result.get_column("t"),
    ),
    IngestionPath(
        "dataframe_rows_dict",
        lambda value: pl.DataFrame([{"t": value}]),
        lambda value: pl.DataFrame([{"t": value}], schema={"t": _dt(value)}),
        lambda result: result.get_column("t"),
    ),
    IngestionPath(
        "dataframe_generator_rows",
        lambda value: pl.DataFrame(
            ([item] for item in [value]), schema=["t"], orient="row"
        ),
        lambda value: pl.DataFrame(
            ([item] for item in [value]), schema={"t": _dt(value)}, orient="row"
        ),
        lambda result: result.get_column("t"),
    ),
    IngestionPath(
        "lazyframe_rows_list",
        lambda value: pl.LazyFrame([[value]], schema=["t"], orient="row").collect(),
        lambda value: pl.LazyFrame(
            [[value]], schema={"t": _dt(value)}, orient="row"
        ).collect(),
        lambda result: result.get_column("t"),
    ),
    IngestionPath(
        "from_dicts",
        lambda value: pl.from_dicts([{"t": value}]),
        lambda value: pl.from_dicts([{"t": value}], schema={"t": _dt(value)}),
        lambda result: result.get_column("t"),
    ),
    IngestionPath(
        "from_records_rows",
        lambda value: pl.from_records([[value]], schema=["t"], orient="row"),
        lambda value: pl.from_records(
            [[value]], schema={"t": _dt(value)}, orient="row"
        ),
        lambda result: result.get_column("t"),
    ),
    IngestionPath(
        "json_normalize",
        lambda value: pl.json_normalize([{"t": value}]),
        lambda value: pl.json_normalize(
            [{"t": value}], schema=pl.Schema({"t": _dt(value)})
        ),
        lambda result: result.get_column("t"),
    ),


    # Frame methods that splice in an already-built `Series`.
    # The resolution should be fixed and not changed by these methods but who knows
    IngestionPath(
        "dataframe_insert_column",
        lambda value: pl.DataFrame({"x": [0]}).insert_column(
            0, pl.Series("t", [value])
        ),
        lambda value: pl.DataFrame({"x": [0]}).insert_column(
            0, pl.Series("t", [value], dtype=_dt(value))
        ),
        lambda result: result.get_column("t"),
    ),
    IngestionPath(
        "dataframe_hstack",
        lambda value: pl.DataFrame({"x": [0]}).hstack([pl.Series("t", [value])]),
        lambda value: pl.DataFrame({"x": [0]}).hstack(
            [pl.Series("t", [value], dtype=_dt(value))]
        ),
        lambda result: result.get_column("t"),
    ),
    IngestionPath(
        "dataframe_replace_column",
        lambda value: pl.DataFrame({"t": [0]}).replace_column(
            0, pl.Series("t", [value])
        ),
        lambda value: pl.DataFrame({"t": [0]}).replace_column(
            0, pl.Series("t", [value], dtype=_dt(value))
        ),
        lambda result: result.get_column("t"),
    ),
    IngestionPath(
        "dataframe_vstack",
        lambda value: pl.DataFrame({"t": pl.Series([value])}).vstack(
            pl.DataFrame({"t": pl.Series([], dtype=_dt(value))})
        ),
        lambda value: pl.DataFrame({"t": pl.Series([value], dtype=_dt(value))}).vstack(
            pl.DataFrame({"t": pl.Series([], dtype=_dt(value))})
        ),
        lambda result: result.get_column("t"),
    ),
    IngestionPath(
        "concat",
        lambda value: pl.concat([pl.DataFrame({"t": pl.Series([value])})]),
        lambda value: _cast(
            pl.concat([pl.DataFrame({"t": pl.Series([value])})]), value
        ),
        lambda result: result.get_column("t"),
    ),
    IngestionPath(
        "lazyframe_map_batches",
        # `schema` is required here, so both variants name the resolution.
        lambda value: (
            pl.LazyFrame({"x": [0]})
            .map_batches(
                lambda _: pl.DataFrame({"t": pl.Series([value])}),
                schema={"t": _dt(value)},
            )
            .collect()
        ),
        lambda value: (
            pl.LazyFrame({"x": [0]})
            .map_batches(
                lambda _: pl.DataFrame({"t": pl.Series([value], dtype=_dt(value))}),
                schema={"t": _dt(value)},
            )
            .collect()
        ),
        lambda result: result.get_column("t"),
    ),


    # Here only for completeness, these should never break as they never do any
    # object conversion but are buffer copies
    IngestionPath(
        "series_pandas",
        lambda value: pl.Series(pd.Series([value])),
        lambda value: pl.Series(pd.Series([value]), dtype=_dt(value)),
    ),
    IngestionPath(
        "series_pandas_index",
        lambda value: pl.Series(pd.DatetimeIndex([value])),
        lambda value: pl.Series(pd.DatetimeIndex([value]), dtype=_dt(value)),
    ),
    IngestionPath(
        "dataframe_pandas",
        lambda value: pl.DataFrame(pd.DataFrame({"t": [value]})),
        lambda value: pl.DataFrame(
            pd.DataFrame({"t": [value]}), schema={"t": _dt(value)}
        ),
        lambda result: result.get_column("t"),
    ),
    IngestionPath(
        "from_pandas",
        lambda value: pl.from_pandas(pd.DataFrame({"t": [value]})),
        lambda value: _cast(pl.from_pandas(pd.DataFrame({"t": [value]})), value),
        lambda result: result.get_column("t"),
    ),
]
# fmt: on


def _equal(ingested: pl.Series, ground_truth: pl.Series) -> bool:
    try:
        assert_series_equal(
            ingested, ground_truth, check_names=False, check_dtypes=False
        )
    except AssertionError:
        return False
    return True


def _shown(series: pl.Series) -> str:
    return str(series.cast(pl.Datetime("ns")).cast(pl.String).to_list()[0])


@pytest.mark.parametrize(
    ("expected_value", "expected_time_unit"),
    PD_TIMESTAMPS,
    ids=lambda param: getattr(param, "unit", param),
)
@pytest.mark.parametrize("path", INGESTION_PATHS, ids=lambda path: path.name)
def test_pandas_timestamp_keeps_resolution(
    path: IngestionPath, expected_value: pd.Timestamp, expected_time_unit: TimeUnit
) -> None:
    # Test that using a `pd.Timestamp` in polars keeps it's value and the resolution.
    # Done by comparing path.extract(path.create()) to `pl.Series(pd.Series([value]))`,
    # which we assume always works
    ground_truth = pl.Series(pd.Series([expected_value]))
    expected_dtype = pl.Datetime(expected_time_unit)

    plain_ingestion = path.extract(path.create(expected_value))
    typed_ingestion = path.extract(path.create_typed(expected_value))

    failures = []
    for how, ingested in (
        ("inferred", plain_ingestion),
        ("requested", typed_ingestion),
    ):
        if ingested.dtype != expected_dtype:
            failures.append(
                f"{how} resolution: got {ingested.dtype}, wanted {expected_dtype}"
            )
        if not _equal(ingested, ground_truth):
            failures.append(
                f"{how} instant: got {_shown(ingested)}, wanted {_shown(ground_truth)}"
            )

    header = f"{path.name} did not keep {expected_value!r} ({expected_dtype})"
    assert not failures, "\n" + "\n".join([header, *failures])


def test_non_pandas_nanosecond_resolution_is_microseconds() -> None:
    # Assert that NON pd.Timestamps datetimes with a .nanosecond field
    # and ns resolution are not converted to ns precision 
    class NanosecondDatetime(datetime):
        resolution = timedelta(0)
        nanosecond = 789890

    value = NanosecondDatetime(2026, 6, 15, 10, 20, 30, 123456)
    series = pl.Series([value])

    assert series.dtype == pl.Datetime("us")
    assert series.to_physical().to_list() == [
        int(value.replace(tzinfo=timezone.utc).timestamp() * 1_000_000)
    ]
