import pytest

import polars as pl
from polars.testing import assert_frame_equal


def test_union_single_element() -> None:
    df = pl.DataFrame({"a": [1, 2, 3]})
    result = pl.union([df])
    assert result is df

    s = pl.Series("test", [1, 2, 3])
    result_s = pl.union([s])
    assert result_s is s


def test_union_group_by() -> None:
    df = pl.DataFrame(
        {
            "g": [0, 0, 0, 0, 1, 1, 1, 1],
            "a": [0, 1, 2, 3, 4, 5, 6, 7],
            "b": [8, 9, 10, 11, 12, 13, 14, 15],
        }
    )
    out = df.group_by("g").agg(pl.union([pl.col.a, pl.col.b]))

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


def test_union_basic() -> None:
    df1 = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
    df2 = pl.DataFrame({"a": [5, 6], "b": [7, 8]})

    result = pl.union([df1, df2])
    expected = pl.DataFrame({"a": [1, 2, 5, 6], "b": [3, 4, 7, 8]})

    assert_frame_equal(result, expected, check_row_order=False)


def test_union_vertical_relaxed() -> None:
    df1 = pl.DataFrame(
        {"a": [1, 2], "b": [3, 4]}, schema={"a": pl.Int32, "b": pl.Int32}
    )
    df2 = pl.DataFrame(
        {"a": [5.0, 6.0], "b": [7, 8]}, schema={"a": pl.Float64, "b": pl.Int32}
    )

    result = pl.union([df1, df2], how="vertical_relaxed")
    expected = pl.DataFrame(
        {"a": [1.0, 2.0, 5.0, 6.0], "b": [3, 4, 7, 8]},
        schema={"a": pl.Float64, "b": pl.Int32},
    )
    assert_frame_equal(result, expected, check_row_order=False)


def test_union_diagonal() -> None:
    df1 = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
    df2 = pl.DataFrame({"a": [5, 6], "c": [7, 8]})
    df3 = pl.DataFrame({"b": [9, 10], "c": [11, 12]})

    result = pl.union([df1, df2, df3], how="diagonal")
    expected = pl.DataFrame(
        {
            "a": [1, 2, 5, 6, None, None],
            "b": [3, 4, None, None, 9, 10],
            "c": [None, None, 7, 8, 11, 12],
        }
    )
    assert_frame_equal(result, expected, check_row_order=False)


def test_union_diagonal_relaxed() -> None:
    df1 = pl.DataFrame(
        {"a": [1, 2], "c": [10, 20]}, schema={"a": pl.Int32, "c": pl.Int64}
    )
    df2 = pl.DataFrame(
        {"a": [3.5, 4.5], "b": [30.1, 40.2]}, schema={"a": pl.Float64, "b": pl.Float32}
    )
    df3 = pl.DataFrame({"b": [5, 6], "c": [50, 60]})

    result = pl.union([df1, df2, df3], how="diagonal_relaxed")

    assert result.schema["a"] == pl.Float64
    assert result.schema["b"] == pl.Float64
    assert result.schema["c"] == pl.Int64

    expected = pl.DataFrame(
        {
            "a": [1.0, 2.0, 3.5, 4.5, None, None],
            "c": [10, 20, None, None, 50, 60],
            "b": [None, None, 30.1, 40.2, 5.0, 6.0],
        }
    )

    assert_frame_equal(result, expected, check_row_order=False)


def test_union_horizontal() -> None:
    df1 = pl.DataFrame({"a": [1, 2, 3]})
    df2 = pl.DataFrame({"b": [4, 5]})
    df3 = pl.DataFrame({"c": [6, 7, 8, 9]})

    result = pl.union([df1, df2, df3], how="horizontal")
    expected = pl.DataFrame(
        {"a": [1, 2, 3, None], "b": [4, 5, None, None], "c": [6, 7, 8, 9]}
    )
    assert_frame_equal(result, expected)


def test_union_align_no_common_columns() -> None:
    df1 = pl.DataFrame({"a": [1, 2]})
    df2 = pl.DataFrame({"b": [3, 4]})

    with pytest.raises(
        pl.exceptions.InvalidOperationError, match="requires at least one common column"
    ):
        pl.union([df1, df2], how="align")


def test_union_align_lazy_frames() -> None:
    lf1 = pl.DataFrame({"id": [1, 2], "x": [3, 4]}).lazy()
    lf2 = pl.DataFrame({"id": [2, 3], "y": [5, 6]}).lazy()

    result = pl.union([lf1, lf2], how="align")
    assert isinstance(result, pl.LazyFrame)

    collected = result.collect()
    expected = pl.DataFrame({"id": [1, 2, 3], "x": [3, 4, None], "y": [None, 5, 6]})
    assert_frame_equal(collected, expected, check_row_order=False)


def test_union_lazyframe_horizontal() -> None:
    lf1 = pl.DataFrame({"a": [1, 2]}).lazy()
    lf2 = pl.DataFrame({"b": [3, 4, 5]}).lazy()

    result = pl.union([lf1, lf2], how="horizontal")
    assert isinstance(result, pl.LazyFrame)

    collected = result.collect()
    expected = pl.DataFrame({"a": [1, 2, None], "b": [3, 4, 5]})
    assert_frame_equal(collected, expected)


def test_union_lazyframe_diagonal() -> None:
    lf1 = pl.DataFrame({"a": [1, 2], "b": [3, 4]}).lazy()
    lf2 = pl.DataFrame({"a": [5, 6], "c": [7, 8]}).lazy()

    result = pl.union([lf1, lf2], how="diagonal")
    assert isinstance(result, pl.LazyFrame)

    collected = result.collect()
    expected = pl.DataFrame(
        {"a": [1, 2, 5, 6], "b": [3, 4, None, None], "c": [None, None, 7, 8]}
    )
    assert_frame_equal(collected, expected, check_row_order=False)


def test_union_series_invalid_strategy() -> None:
    s1 = pl.Series("a", [1, 2, 3])
    s2 = pl.Series("b", [4, 5, 6])

    with pytest.raises(
        ValueError, match="Series only supports 'vertical' concat strategy"
    ):
        pl.union([s1, s2], how="horizontal")

    with pytest.raises(
        ValueError, match="Series only supports 'vertical' concat strategy"
    ):
        pl.union([s1, s2], how="diagonal")


def test_concat_invalid_how_parameter() -> None:
    df1 = pl.DataFrame({"a": [1, 2]})
    df2 = pl.DataFrame({"a": [3, 4]})

    with pytest.raises(ValueError, match="DataFrame `how` must be one of"):
        pl.concat([df1, df2], how="invalid_strategy")  # type: ignore[arg-type]


def test_concat_unsupported_type() -> None:
    with pytest.raises(TypeError, match="did not expect type"):
        pl.concat([1, 2, 3])  # type: ignore[type-var]


def test_union_expressions() -> None:
    expr1 = pl.col("a")
    expr2 = pl.col("b")
    union_expr = pl.union([expr1, expr2])

    df_input = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
    result = df_input.select(union_expr.alias("unioned"))

    expected = pl.DataFrame({"unioned": [1, 2, 3, 4]})
    assert_frame_equal(result, expected)


def test_union_with_empty_dataframes() -> None:
    empty_df = pl.DataFrame(schema={"a": pl.Int64, "b": pl.String})
    df_with_data = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})

    result = pl.union([empty_df, df_with_data])
    assert_frame_equal(result, df_with_data, check_row_order=False)

    result2 = pl.union([df_with_data, empty_df])
    assert_frame_equal(result2, df_with_data, check_row_order=False)
