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
    expected = pl.Series("z", [[1.0, 3.0], [2.0, 4.0]], dtype=pl.Array(pl.Float64, 2))

    rem = '''
    out = s0.list.concat([s1])
    assert_series_equal(out, expected)

    out = s0.list.concat(s1)
    assert_series_equal(out, expected)
    '''

    df = pl.DataFrame([s0, s1])
    result = df.select(pl.array(["a", "b"]).alias("z"))["z"]
    print(result)
    assert_series_equal(result, expected)
