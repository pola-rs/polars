import pytest

import polars as pl
from polars.testing import assert_frame_equal


@pytest.mark.slow
def test_concat_expressions_stack_overflow() -> None:
    n = 10000
    e = pl.concat([pl.lit(x) for x in range(n)])

    df = pl.select(e)
    assert df.shape == (n, 1)


@pytest.mark.slow
def test_concat_lf_stack_overflow() -> None:
    n = 1000
    bar = pl.DataFrame({"a": 0}).lazy()

    for i in range(n):
        bar = pl.concat([bar, pl.DataFrame({"a": i}).lazy()])
    assert bar.collect().shape == (1001, 1)


def test_concat_vertically_relaxed() -> None:
    a = pl.DataFrame(
        data={"a": [1, 2, 3], "b": [True, False, None]},
        schema={"a": pl.Int8, "b": pl.Boolean},
    )
    b = pl.DataFrame(
        data={"a": [43, 2, 3], "b": [32, 1, None]},
        schema={"a": pl.Int16, "b": pl.Int64},
    )
    out = pl.concat([a, b], how="vertical_relaxed")
    assert out.schema == {"a": pl.Int16, "b": pl.Int64}
    assert out.to_dict(as_series=False) == {
        "a": [1, 2, 3, 43, 2, 3],
        "b": [1, 0, None, 32, 1, None],
    }
    out = pl.concat([b, a], how="vertical_relaxed")
    assert out.schema == {"a": pl.Int16, "b": pl.Int64}
    assert out.to_dict(as_series=False) == {
        "a": [43, 2, 3, 1, 2, 3],
        "b": [32, 1, None, 1, 0, None],
    }

    c = pl.DataFrame({"a": [1, 2], "b": [2, 1]})
    d = pl.DataFrame({"a": [1.0, 0.2], "b": [None, 0.1]})

    out = pl.concat([c, d], how="vertical_relaxed")
    assert out.schema == {"a": pl.Float64, "b": pl.Float64}
    assert out.to_dict(as_series=False) == {
        "a": [1.0, 2.0, 1.0, 0.2],
        "b": [2.0, 1.0, None, 0.1],
    }
    out = pl.concat([d, c], how="vertical_relaxed")
    assert out.schema == {"a": pl.Float64, "b": pl.Float64}
    assert out.to_dict(as_series=False) == {
        "a": [1.0, 0.2, 1.0, 2.0],
        "b": [None, 0.1, 2.0, 1.0],
    }


def test_concat_group_by() -> None:
    df = pl.DataFrame(
        {
            "g": [0, 0, 0, 0, 1, 1, 1, 1],
            "a": [0, 1, 2, 3, 4, 5, 6, 7],
            "b": [8, 9, 10, 11, 12, 13, 14, 15],
        }
    )
    out = df.group_by("g").agg(pl.concat([pl.col.a, pl.col.b]))

    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "g": [0, 1],
                "a": [[0, 1, 2, 3, 8, 9, 10, 11], [4, 5, 6, 7, 12, 13, 14, 15]],
            }
        ),
        check_row_order=False,
    )


def test_concat_19877() -> None:
    df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
    out = df.select(pl.concat([pl.col("a"), pl.col("b")]))
    assert_frame_equal(out, pl.DataFrame({"a": [1, 2, 3, 4]}))


def test_concat_zip_series_21980() -> None:
    df = pl.DataFrame({"x": 1, "y": 2})
    out = df.select(pl.concat([pl.col.x, pl.col.y]), pl.Series([3, 4]))
    assert_frame_equal(out, pl.DataFrame({"x": [1, 2], "": [3, 4]}))
