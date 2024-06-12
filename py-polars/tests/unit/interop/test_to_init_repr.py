from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone

import polars as pl
from polars.testing import assert_frame_equal


def test_to_init_repr() -> None:
    # round-trip various types
    with pl.StringCache():
        df = (
            pl.LazyFrame(
                {
                    "a": [1, 2, None],
                    "b": [4.5, 5.5, 6.5],
                    "c": ["x", "y", "z"],
                    "d": [True, False, True],
                    "e": [None, "", None],
                    "f": [date(2022, 7, 5), date(2023, 2, 5), date(2023, 8, 5)],
                    "g": [time(0, 0, 0, 1), time(12, 30, 45), time(23, 59, 59, 999000)],
                    "h": [
                        datetime(2022, 7, 5, 10, 30, 45, 4560),
                        datetime(2023, 10, 12, 20, 3, 8, 11),
                        None,
                    ],
                    "null": [None, None, None],
                    "enum": ["a", "b", "c"],
                    "duration": [timedelta(days=1), timedelta(days=2), None],
                    "binary": [bytes([0]), bytes([0, 1]), bytes([0, 1, 2])],
                    "object": [timezone.utc, timezone.utc, timezone.utc],
                },
            )
            .with_columns(
                pl.col("c").cast(pl.Categorical),
                pl.col("h").cast(pl.Datetime("ns")),
                pl.col("enum").cast(pl.Enum(["a", "b", "c"])),
            )
            .collect()
        )

        result = eval(df.to_init_repr().replace("datetime.", ""))
        expected = df
        # drop "object" because it can not be compared by assert_frame_equal
        assert_frame_equal(result.drop("object"), expected.drop("object"))


def test_to_init_repr_nested_dtype() -> None:
    # round-trip nested types
    df = pl.LazyFrame(
        {
            "list": pl.Series(values=[[1], [2], [3]], dtype=pl.List(pl.Int32)),
            "list_list": pl.Series(
                values=[[[1]], [[2]], [[3]]], dtype=pl.List(pl.List(pl.Int8))
            ),
            "array": pl.Series(
                values=[[1.0], [2.0], [3.0]],
                dtype=pl.Array(pl.Float32, 1),
            ),
            "struct": pl.Series(
                values=[
                    {"x": "foo", "y": [1, 2]},
                    {"x": "bar", "y": [3, 4, 5]},
                    {"x": "foobar", "y": []},
                ],
                dtype=pl.Struct({"x": pl.String, "y": pl.List(pl.Int8)}),
            ),
        },
    ).collect()

    assert_frame_equal(eval(df.to_init_repr()), df)


def test_to_init_repr_nested_dtype_roundtrip() -> None:
    # round-trip nested types
    df = pl.LazyFrame(
        {
            "list": pl.Series(values=[[1], [2], [3]], dtype=pl.List(pl.Int32)),
            "list_list": pl.Series(
                values=[[[1]], [[2]], [[3]]], dtype=pl.List(pl.List(pl.Int8))
            ),
            "array": pl.Series(
                values=[[1.0], [2.0], [3.0]],
                dtype=pl.Array(pl.Float32, 1),
            ),
            "struct": pl.Series(
                values=[
                    {"x": "foo", "y": [1, 2]},
                    {"x": "bar", "y": [3, 4, 5]},
                    {"x": "foobar", "y": []},
                ],
                dtype=pl.Struct({"x": pl.String, "y": pl.List(pl.Int8)}),
            ),
        },
    ).collect()

    assert_frame_equal(eval(df.to_init_repr()), df)
