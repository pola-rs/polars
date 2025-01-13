import pytest

import polars as pl
from polars.exceptions import ShapeError
from polars.testing import assert_series_equal


def test_concat_arr() -> None:
    assert_series_equal(
        pl.select(
            pl.concat_arr(
                pl.Series([1, 3, 5]),
                pl.Series([2, 4, 6]),
            )
        ).to_series(),
        pl.Series([[1, 2], [3, 4], [5, 6]], dtype=pl.Array(pl.Int64, 2)),
    )

    assert_series_equal(
        pl.select(
            pl.concat_arr(
                pl.Series([1, 3, 5]),
                pl.Series([2, 4, None]),
            )
        ).to_series(),
        pl.Series([[1, 2], [3, 4], [5, None]], dtype=pl.Array(pl.Int64, 2)),
    )

    assert_series_equal(
        pl.select(
            pl.concat_arr(
                pl.Series([1, 3, 5]),
                pl.Series([[2], [None], None], dtype=pl.Array(pl.Int64, 1)),
            )
        ).to_series(),
        pl.Series([[1, 2], [3, None], None], dtype=pl.Array(pl.Int64, 2)),
    )


def test_concat_arr_broadcast() -> None:
    assert_series_equal(
        pl.select(
            pl.concat_arr(
                pl.Series([1, 3, 5]),
                pl.lit(None, dtype=pl.Int64),
            )
        ).to_series(),
        pl.Series([[1, None], [3, None], [5, None]], dtype=pl.Array(pl.Int64, 2)),
    )

    assert_series_equal(
        pl.select(
            pl.concat_arr(
                pl.Series([1, 3, 5]),
                pl.lit(None, dtype=pl.Array(pl.Int64, 2)),
            )
        ).to_series(),
        pl.Series([None, None, None], dtype=pl.Array(pl.Int64, 3)),
    )

    assert_series_equal(
        pl.select(
            pl.concat_arr(
                pl.Series([1, 3, 5]),
                pl.lit([0, None], dtype=pl.Array(pl.Int64, 2)),
            )
        ).to_series(),
        pl.Series(
            [[1, 0, None], [3, 0, None], [5, 0, None]], dtype=pl.Array(pl.Int64, 3)
        ),
    )

    assert_series_equal(
        pl.select(
            pl.concat_arr(pl.lit(1, dtype=pl.Int64).alias(""), pl.Series([1, 2, 3]))
        ).to_series(),
        pl.Series([[1, 1], [1, 2], [1, 3]], dtype=pl.Array(pl.Int64, 2)),
    )

    assert_series_equal(
        pl.select(
            pl.concat_arr(pl.Series([1, 2, 3]), pl.lit(1, dtype=pl.Int64))
        ).to_series(),
        pl.Series([[1, 1], [2, 1], [3, 1]], dtype=pl.Array(pl.Int64, 2)),
    )

    with pytest.raises(ShapeError, match="length of column.*did not match"):
        assert_series_equal(
            pl.select(
                pl.concat_arr(pl.Series([1, 3, 5]), pl.Series([1, 1]))
            ).to_series(),
            pl.Series([None, None, None], dtype=pl.Array(pl.Int64, 3)),
        )

    assert_series_equal(
        pl.select(
            pl.concat_arr(
                pl.Series(
                    [{"x": [1], "y": [2]}, {"x": [3], "y": None}],
                    dtype=pl.Struct({"x": pl.Array(pl.Int64, 1)}),
                ),
                pl.lit(
                    {"x": [9], "y": [11]}, dtype=pl.Struct({"x": pl.Array(pl.Int64, 1)})
                ),
            )
        ).to_series(),
        pl.Series(
            [
                [{"x": [1], "y": [2]}, {"x": [9], "y": [11]}],
                [{"x": [3], "y": [4]}, {"x": [9], "y": [11]}],
            ],
            dtype=pl.Array(pl.Struct({"x": pl.Array(pl.Int64, 1)}), 2),
        ),
    )


@pytest.mark.parametrize("inner_dtype", [pl.Int64(), pl.Null()])
def test_concat_arr_validity_combination(inner_dtype: pl.DataType) -> None:
    assert_series_equal(
        pl.select(
            pl.concat_arr(
                pl.Series([[], [], None, None], dtype=pl.Array(inner_dtype, 0)),
                pl.Series([[], [], None, None], dtype=pl.Array(inner_dtype, 0)),
                pl.Series([[None], None, [None], None], dtype=pl.Array(inner_dtype, 1)),
            ),
        ).to_series(),
        pl.Series([[None], None, None, None], dtype=pl.Array(inner_dtype, 1)),
    )

    assert_series_equal(
        pl.select(
            pl.concat_arr(
                pl.Series([None, None], dtype=inner_dtype),
                pl.Series([[None], None], dtype=pl.Array(inner_dtype, 1)),
            ),
        ).to_series(),
        pl.Series([[None, None], None], dtype=pl.Array(inner_dtype, 2)),
    )


def test_concat_arr_zero_fields() -> None:
    assert_series_equal(
        (
            pl.Series([[[]], [None]], dtype=pl.Array(pl.Array(pl.Int64, 0), 1))
            .to_frame()
            .select(pl.concat_arr(pl.first(), pl.first()))
            .to_series()
        ),
        pl.Series([[[], []], [None, None]], dtype=pl.Array(pl.Array(pl.Int64, 0), 2)),
    )

    assert_series_equal(
        (
            pl.Series([[{}], [None]], dtype=pl.Array(pl.Struct({}), 1))
            .to_frame()
            .select(pl.concat_arr(pl.first(), pl.first()))
            .to_series()
        ),
        pl.Series([[{}, {}], [None, None]], dtype=pl.Array(pl.Struct({}), 2)),
    )

    assert_series_equal(
        (
            pl.Series(
                [[{"x": []}], [{"x": None}], [None]],
                dtype=pl.Array(pl.Struct({"x": pl.Array(pl.Int64, 0)}), 1),
            )
            .to_frame()
            .select(pl.concat_arr(pl.first(), pl.first()))
            .to_series()
        ),
        pl.Series(
            [[{"x": []}, {"x": []}], [{"x": None}, {"x": None}], [None, None]],
            dtype=pl.Array(pl.Struct({"x": pl.Array(pl.Int64, 0)}), 2),
        ),
    )


@pytest.mark.may_fail_auto_streaming
def test_concat_arr_scalar() -> None:
    lit = pl.lit([b"A"], dtype=pl.Array(pl.Binary, 1))
    df = pl.select(pl.repeat(lit, 10))

    assert df._to_metadata()["repr"].to_list() == ["scalar"]

    out = df.with_columns(out=pl.concat_arr(pl.first(), pl.first()))
    assert out._to_metadata()["repr"].to_list() == ["scalar", "scalar"]
