from __future__ import annotations

from typing import Any

import pytest

import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal


def test_replace_expr() -> None:
    country_code_dict = {
        "CA": "Canada",
        "DE": "Germany",
        "FR": "France",
        None: "Not specified",
    }
    df = pl.DataFrame(
        [
            pl.Series("int", [None, 1, None, 3], dtype=pl.Int16),
            pl.Series("country_code", ["FR", None, "ES", "DE"], dtype=pl.Utf8),
        ]
    )
    result = df.with_columns(
        pl.col("country_code").replace(country_code_dict).alias("replaced")
    )
    expected = pl.DataFrame(
        [
            pl.Series("int", [None, 1, None, 3], dtype=pl.Int16),
            pl.Series("country_code", ["FR", None, "ES", "DE"], dtype=pl.Utf8),
            pl.Series(
                "replaced",
                ["France", "Not specified", "ES", "Germany"],
                dtype=pl.Utf8,
            ),
        ]
    )
    assert_frame_equal(result, expected)

    assert_frame_equal(
        df.with_columns(
            pl.col("country_code")
            .replace(country_code_dict, default=pl.col("country_code"))
            .alias("remapped")
        ),
        pl.DataFrame(
            [
                pl.Series("int", [None, 1, None, 3], dtype=pl.Int16),
                pl.Series("country_code", ["FR", None, "ES", "DE"], dtype=pl.Utf8),
                pl.Series(
                    "remapped",
                    ["France", "Not specified", "ES", "Germany"],
                    dtype=pl.Utf8,
                ),
            ]
        ),
    )

    result = df.with_columns(
        pl.col("country_code")
        .replace(country_code_dict, default=None)
        .alias("remapped")
    )
    expected = pl.DataFrame(
        [
            pl.Series("int", [None, 1, None, 3], dtype=pl.Int16),
            pl.Series("country_code", ["FR", None, "ES", "DE"], dtype=pl.Utf8),
            pl.Series(
                "remapped",
                ["France", "Not specified", None, "Germany"],
                dtype=pl.Utf8,
            ),
        ]
    )
    assert_frame_equal(result, expected)

    assert_frame_equal(
        df.with_row_count().with_columns(
            pl.struct(pl.col(["country_code", "row_nr"]))
            .replace(
                country_code_dict,
                default=pl.col("row_nr").cast(pl.Utf8),
            )
            .alias("remapped")
        ),
        pl.DataFrame(
            [
                pl.Series("row_nr", [0, 1, 2, 3], dtype=pl.UInt32),
                pl.Series("int", [None, 1, None, 3], dtype=pl.Int16),
                pl.Series("country_code", ["FR", None, "ES", "DE"], dtype=pl.Utf8),
                pl.Series(
                    "remapped",
                    ["France", "Not specified", "2", "Germany"],
                    dtype=pl.Utf8,
                ),
            ]
        ),
    )

    with pl.StringCache():
        assert_frame_equal(
            df.with_columns(
                pl.col("country_code")
                .cast(pl.Categorical)
                .replace(country_code_dict, default=pl.col("country_code"))
                .alias("remapped")
            ),
            pl.DataFrame(
                [
                    pl.Series("int", [None, 1, None, 3], dtype=pl.Int16),
                    pl.Series("country_code", ["FR", None, "ES", "DE"], dtype=pl.Utf8),
                    pl.Series(
                        "remapped",
                        ["France", "Not specified", "ES", "Germany"],
                        dtype=pl.Categorical,
                    ),
                ]
            ),
        )

    df_categorical_lazy = df.lazy().with_columns(
        pl.col("country_code").cast(pl.Categorical)
    )

    with pl.StringCache():
        assert_frame_equal(
            df_categorical_lazy.with_columns(
                pl.col("country_code")
                .replace(country_code_dict, default=pl.col("country_code"))
                .alias("remapped")
            ).collect(),
            pl.DataFrame(
                [
                    pl.Series("int", [None, 1, None, 3], dtype=pl.Int16),
                    pl.Series(
                        "country_code", ["FR", None, "ES", "DE"], dtype=pl.Categorical
                    ),
                    pl.Series(
                        "remapped",
                        ["France", "Not specified", "ES", "Germany"],
                        dtype=pl.Categorical,
                    ),
                ]
            ),
        )

    int_to_int_dict = {1: 5, 3: 7}

    assert_frame_equal(
        df.with_columns(pl.col("int").replace(int_to_int_dict).alias("remapped")),
        pl.DataFrame(
            [
                pl.Series("int", [None, 1, None, 3], dtype=pl.Int16),
                pl.Series("country_code", ["FR", None, "ES", "DE"], dtype=pl.Utf8),
                pl.Series("remapped", [None, 5, None, 7], dtype=pl.Int16),
            ]
        ),
    )

    int_dict = {1: "b", 3: "d"}

    assert_frame_equal(
        df.with_columns(pl.col("int").replace(int_dict).alias("remapped")),
        pl.DataFrame(
            [
                pl.Series("int", [None, 1, None, 3], dtype=pl.Int16),
                pl.Series("country_code", ["FR", None, "ES", "DE"], dtype=pl.Utf8),
                pl.Series("remapped", [None, "b", None, "d"], dtype=pl.Utf8),
            ]
        ),
    )

    int_with_none_dict = {1: "b", 3: "d", None: "e"}

    assert_frame_equal(
        df.with_columns(pl.col("int").replace(int_with_none_dict).alias("remapped")),
        pl.DataFrame(
            [
                pl.Series("int", [None, 1, None, 3], dtype=pl.Int16),
                pl.Series("country_code", ["FR", None, "ES", "DE"], dtype=pl.Utf8),
                pl.Series("remapped", ["e", "b", "e", "d"], dtype=pl.Utf8),
            ]
        ),
    )

    int_with_only_none_values_dict = {3: None}

    assert_frame_equal(
        df.with_columns(
            pl.col("int")
            .replace(int_with_only_none_values_dict, default=6)
            .alias("remapped")
        ),
        pl.DataFrame(
            [
                pl.Series("int", [None, 1, None, 3], dtype=pl.Int16),
                pl.Series("country_code", ["FR", None, "ES", "DE"], dtype=pl.Utf8),
                pl.Series("remapped", [6, 6, 6, None], dtype=pl.Int16),
            ]
        ),
    )

    assert_frame_equal(
        df.with_columns(
            pl.col("int")
            .replace(int_with_only_none_values_dict, default=6, return_dtype=pl.Int32)
            .alias("remapped")
        ),
        pl.DataFrame(
            [
                pl.Series("int", [None, 1, None, 3], dtype=pl.Int16),
                pl.Series("country_code", ["FR", None, "ES", "DE"], dtype=pl.Utf8),
                pl.Series("remapped", [6, 6, 6, None], dtype=pl.Int32),
            ]
        ),
    )

    result = df.with_columns(
        pl.col("int")
        .replace(int_with_only_none_values_dict, default=None)
        .alias("remapped")
    )
    expected = pl.DataFrame(
        [
            pl.Series("int", [None, 1, None, 3], dtype=pl.Int16),
            pl.Series("country_code", ["FR", None, "ES", "DE"], dtype=pl.Utf8),
            pl.Series("remapped", [None, None, None, None], dtype=pl.Int16),
        ]
    )
    assert_frame_equal(result, expected)

    empty_dict: dict[Any, Any] = {}

    assert_frame_equal(
        df.with_columns(pl.col("int").replace(empty_dict).alias("remapped")),
        pl.DataFrame(
            [
                pl.Series("int", [None, 1, None, 3], dtype=pl.Int16),
                pl.Series("country_code", ["FR", None, "ES", "DE"], dtype=pl.Utf8),
                pl.Series("remapped", [None, 1, None, 3], dtype=pl.Int16),
            ]
        ),
    )

    float_dict = {1.0: "b", 3.0: "d"}

    with pytest.raises(
        pl.ComputeError,
        match=".*'float' object cannot be interpreted as an integer",
    ):
        df.with_columns(pl.col("int").replace(float_dict))

    df_int_as_str = df.with_columns(pl.col("int").cast(pl.Utf8))

    with pytest.raises(
        pl.ComputeError,
        match="mapping keys for `replace` could not be converted to Utf8 without losing values in the conversion",
    ):
        df_int_as_str.with_columns(pl.col("int").replace(int_dict))

    with pytest.raises(
        pl.ComputeError,
        match="mapping keys for `replace` could not be converted to Utf8 without losing values in the conversion",
    ):
        df_int_as_str.with_columns(pl.col("int").replace(int_with_none_dict))

    # 7132
    df = pl.DataFrame({"text": ["abc"]})
    mapper = {"abc": "123"}
    assert_frame_equal(
        df.select(pl.col("text").replace(mapper).str.replace_all("1", "-")),
        pl.DataFrame(
            [
                pl.Series("text", ["-23"], dtype=pl.Utf8),
            ]
        ),
    )

    result = pl.DataFrame(
        [
            pl.Series("float_to_boolean", [1.0, None]),
            pl.Series("boolean_to_int", [True, False]),
            pl.Series("boolean_to_str", [True, False]),
        ]
    ).with_columns(
        pl.col("float_to_boolean").replace({1.0: True}, default=None),
        pl.col("boolean_to_int").replace({True: 1, False: 0}, default=None),
        pl.col("boolean_to_str").replace({True: "1", False: "0"}, default=None),
    )
    expected = pl.DataFrame(
        [
            pl.Series("float_to_boolean", [True, None], dtype=pl.Boolean),
            pl.Series("boolean_to_int", [1, 0], dtype=pl.Int64),
            pl.Series("boolean_to_str", ["1", "0"], dtype=pl.Utf8),
        ]
    )
    assert_frame_equal(result, expected)

    lf = pl.LazyFrame({"a": [1, 2, 3]})
    assert_frame_equal(
        lf.select(
            pl.col("a").cast(pl.UInt8).replace({1: 11, 2: 22}, default=99)
        ).collect(),
        pl.DataFrame({"a": [11, 22, 99]}, schema_overrides={"a": pl.UInt8}),
    )

    df = (
        pl.LazyFrame({"a": ["one", "two"]})
        .with_columns(
            pl.col("a").replace({"one": 1}, default=None, return_dtype=pl.UInt32)
        )
        .fill_null(999)
        .collect()
    )
    assert_frame_equal(
        df, pl.DataFrame({"a": [1, 999]}, schema_overrides={"a": pl.UInt32})
    )


def test_replace_series() -> None:
    s = pl.Series("s", [-1, 2, None, 4, -5])
    remap = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}

    assert_series_equal(
        s.abs().replace(remap, default="?"),
        pl.Series("s", ["one", "two", "?", "four", "five"]),
    )
    assert_series_equal(
        s.replace(remap, default=s.cast(pl.Utf8)),
        pl.Series("s", ["-1", "two", None, "four", "-5"]),
    )

    remap_int = {1: 11, 2: 22, 3: 33, 4: 44, 5: 55}

    assert_series_equal(
        s.replace(remap_int),
        pl.Series("s", [-1, 22, None, 44, -5]),
    )

    assert_series_equal(
        s.cast(pl.Int16).replace(remap_int, default=None),
        pl.Series("s", [None, 22, None, 44, None], dtype=pl.Int16),
    )

    assert_series_equal(
        s.cast(pl.Int16).replace(remap_int),
        pl.Series("s", [-1, 22, None, 44, -5], dtype=pl.Int16),
    )

    assert_series_equal(
        s.cast(pl.Int16).replace(remap_int, return_dtype=pl.Float32),
        pl.Series("s", [-1.0, 22.0, None, 44.0, -5.0], dtype=pl.Float32),
    )

    assert_series_equal(
        s.cast(pl.Int16).replace(remap_int, default=9),
        pl.Series("s", [9, 22, 9, 44, 9], dtype=pl.Int16),
    )

    assert_series_equal(
        s.cast(pl.Int16).replace(remap_int, default=9, return_dtype=pl.Float32),
        pl.Series("s", [9.0, 22.0, 9.0, 44.0, 9.0], dtype=pl.Float32),
    )

    assert_series_equal(
        pl.Series("boolean_to_int", [True, False]).replace(
            {True: 1, False: 0}, default=None
        ),
        pl.Series("boolean_to_int", [1, 0]),
    )

    assert_series_equal(
        pl.Series("boolean_to_str", [True, False]).replace(
            {True: "1", False: "0"}, default=None
        ),
        pl.Series("boolean_to_str", ["1", "0"]),
    )


def test_map_dict_deprecated() -> None:
    s = pl.Series("a", [1, 2, 3])
    with pytest.deprecated_call():
        result = s.map_dict({2: 100})
    expected = pl.Series("a", [None, 100, None])
    assert_series_equal(result, expected)

    with pytest.deprecated_call():
        result = s.to_frame().select(pl.col("a").map_dict({2: 100})).to_series()
    assert_series_equal(result, expected)
