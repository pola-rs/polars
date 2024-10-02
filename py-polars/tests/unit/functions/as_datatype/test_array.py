from time import sleep

import pytest

import polars as pl
from polars.exceptions import ComputeError
from polars.testing import assert_frame_equal, assert_series_equal


def build_array_f64():
    df = pl.DataFrame(
        [
            pl.Series("f1", [1, 2]),
            pl.Series("f2", [3, 4]),
        ],
        schema={
            "f1": pl.Float64,
            "f2": pl.Float64,
        },
    )
    print(df)

    # Now we call our plugin:
    # sleep(30)
    result = print(df.with_columns(arr=mp.array(pl.all(), dtype=pl.Float64)))
    print(result)

def build_array_i32():
    df = pl.DataFrame(
        [
            pl.Series("f1", [1, 2]),
            pl.Series("f2", [3, None]),
        ],
        schema={
            "f1": pl.Int32,
            "f2": pl.Int32,
        },
    )
    print(df)

    # Now we call our plugin:
    # sleep(30)
    result = print(df.with_columns(arr=mp.array(pl.all(), dtype=pl.Int32)))
    print(result)

def build_array_i32_converted():
    df = pl.DataFrame(
        [
            pl.Series("f1", [1, None]),
            pl.Series("f2", [None, 4]),
        ],
        schema={
            "f1": pl.Int32,
            "f2": pl.Int32,
        },
    )
    print(df)

    # Now we call our plugin:
    # sleep(30)
    result = print(df.with_columns(arr=mp.array(pl.all(), dtype=pl.Float64)))
    print(result)

def test_array() -> None:
    s0 = pl.Series("a", [1.0, 2.0], dtype=pl.Float64)
    s1 = pl.Series("b", [3.0, 4.0], dtype=pl.Float64)
    expected_f64 = pl.Series("z", [[1.0, 3.0], [2.0, 4.0]], dtype=pl.Array(pl.Float64, 2))
    expected_i32 = pl.Series("z", [[1.0, 3.0], [2.0, 4.0]], dtype=pl.Array(pl.Int32, 2), strict=False)
    df = pl.DataFrame([s0, s1])
    print("\n")

    result = df.select(pl.array(["a", "b"]).alias("z"))["z"]
    print("No Cast")
    print(result)
    print()
    assert_series_equal(result, expected_f64)

    result = df.select(pl.array(["a", "b"], dtype='{"DtypeColumn":["Float64"]}').alias("z"))["z"]
    print("Cast to Float64")
    print(result)
    print()
    assert_series_equal(result, expected_f64)

    result = df.select(pl.array(["a", "b"], dtype='{"DtypeColumn":["Int32"]}').alias("z"))["z"]
    print("Cast to Int32")
    print(result)
    print()
    assert_series_equal(result, expected_i32)

def test_array_nulls() -> None:
    s0 = pl.Series("a", [1.0, None], dtype=pl.Float64)
    s1 = pl.Series("b", [None, 4.0], dtype=pl.Float64)
    expected_f64 = pl.Series("z", [[1.0, None], [None, 4.0]], dtype=pl.Array(pl.Float64, 2))
    df = pl.DataFrame([s0, s1])
    print("\n")

    result = df.select(pl.array(["a", "b"]).alias("z"))["z"]
    print("No Cast")
    print(result)
    print()
    assert_series_equal(result, expected_f64)

def test_array_string() -> None:
    s0 = pl.Series("a", ['1', '2'])
    s1 = pl.Series("b", ['3', '4'])
    expected_f64 = pl.Series("z", [[1.0, 3.0], [2.0, 4.0]], dtype=pl.Array(pl.Float64, 2))
    df = pl.DataFrame([s0, s1])
    print("\n")

    result = df.select(pl.array(["a", "b"]).alias("z"))["z"]
    print("No Cast")
    print(result)
    print()
    assert_series_equal(result, expected_f64)

    result = df.select(pl.array(["a", "b"], dtype='{"DtypeColumn":["Float64"]}').alias("z"))["z"]
    print("Cast to Float64")
    print(result)
    print()
    assert_series_equal(result, expected_f64)





