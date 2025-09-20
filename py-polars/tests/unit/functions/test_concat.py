import io
from typing import IO

import pytest

import polars as pl
from polars.testing import assert_frame_equal


@pytest.mark.may_fail_cloud  # reason: @serialize-stack-overflow
@pytest.mark.slow
def test_concat_expressions_stack_overflow() -> None:
    n = 10000
    e = pl.concat([pl.lit(x) for x in range(n)])

    df = pl.select(e)
    assert df.shape == (n, 1)


@pytest.mark.may_fail_cloud  # reason: @serialize-stack-overflow
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


def test_concat_invalid_schema_err_20355() -> None:
    lf1 = pl.LazyFrame({"x": [1], "y": [None]})
    lf2 = pl.LazyFrame({"y": [1]})
    with pytest.raises(pl.exceptions.InvalidOperationError):
        pl.concat([lf1, lf2]).collect(engine="streaming")


def test_concat_df() -> None:
    df1 = pl.DataFrame({"a": [2, 1, 3], "b": [1, 2, 3], "c": [1, 2, 3]})
    df2 = pl.concat([df1, df1], rechunk=True)

    assert df2.shape == (6, 3)
    assert df2.n_chunks() == 1
    assert df2.rows() == df1.rows() + df1.rows()
    assert pl.concat([df1, df1], rechunk=False).n_chunks() == 2

    # concat from generator of frames
    df3 = pl.concat(items=(df1 for _ in range(2)))
    assert_frame_equal(df2, df3)

    # check that df4 is not modified following concat of itself
    df4 = pl.from_records(((1, 2), (1, 2)))
    _ = pl.concat([df4, df4, df4])

    assert df4.shape == (2, 2)
    assert df4.rows() == [(1, 1), (2, 2)]

    # misc error conditions
    with pytest.raises(ValueError):
        _ = pl.concat([])

    with pytest.raises(ValueError):
        pl.concat([df1, df1], how="rubbish")  # type: ignore[arg-type]


def test_concat_to_empty() -> None:
    assert pl.concat([pl.DataFrame([]), pl.DataFrame({"a": [1]})]).to_dict(
        as_series=False
    ) == {"a": [1]}


def test_concat_multiple_parquet_inmem() -> None:
    f = io.BytesIO()
    g = io.BytesIO()

    df1 = pl.DataFrame(
        {
            "a": [1, 2, 3],
            "b": ["xyz", "abc", "wow"],
        }
    )
    df2 = pl.DataFrame(
        {
            "a": [5, 6, 7],
            "b": ["a", "few", "entries"],
        }
    )

    dfs = pl.concat([df1, df2])

    df1.write_parquet(f)
    df2.write_parquet(g)

    f.seek(0)
    g.seek(0)

    items: list[IO[bytes]] = [f, g]
    assert_frame_equal(pl.read_parquet(items), dfs)

    f.seek(0)
    g.seek(0)

    assert_frame_equal(pl.read_parquet(items, use_pyarrow=True), dfs)

    f.seek(0)
    g.seek(0)

    fb = f.read()
    gb = g.read()

    assert_frame_equal(pl.read_parquet([fb, gb]), dfs)
    assert_frame_equal(pl.read_parquet([fb, gb], use_pyarrow=True), dfs)


def test_concat_series() -> None:
    s = pl.Series("a", [2, 1, 3])

    assert pl.concat([s, s]).len() == 6
    # check if s remains unchanged
    assert s.len() == 3


def test_concat_null_20501() -> None:
    a = pl.DataFrame({"id": [1], "value": ["foo"]})
    b = pl.DataFrame({"id": [2], "value": [None]})

    assert pl.concat([a.lazy(), b.lazy()]).collect().to_dict(as_series=False) == {
        "id": [1, 2],
        "value": ["foo", None],
    }


def test_concat_single_element() -> None:
    df = pl.DataFrame({"a": [1, 2, 3]})
    result = pl.concat([df])
    assert result is df

    s = pl.Series("test", [1, 2, 3])
    result_s = pl.concat([s])
    assert result_s is s


def test_concat_diagonal() -> None:
    df1 = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
    df2 = pl.DataFrame({"a": [5, 6], "c": [7, 8]})
    df3 = pl.DataFrame({"b": [9, 10], "c": [11, 12]})

    result = pl.concat([df1, df2, df3], how="diagonal")
    expected = pl.DataFrame(
        {
            "a": [1, 2, 5, 6, None, None],
            "b": [3, 4, None, None, 9, 10],
            "c": [None, None, 7, 8, 11, 12],
        }
    )
    assert_frame_equal(result, expected)


def test_concat_diagonal_relaxed() -> None:
    df1 = pl.DataFrame(
        {"a": [1, 2], "c": [10, 20]}, schema={"a": pl.Int32, "c": pl.Int64}
    )
    df2 = pl.DataFrame(
        {"a": [3.5, 4.5], "b": [30.1, 40.2]}, schema={"a": pl.Float64, "b": pl.Float32}
    )
    df3 = pl.DataFrame({"b": [5, 6], "c": [50, 60]})

    result = pl.concat([df1, df2, df3], how="diagonal_relaxed")

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

    assert_frame_equal(result, expected)


def test_concat_horizontal() -> None:
    df1 = pl.DataFrame({"a": [1, 2, 3]})
    df2 = pl.DataFrame({"b": [4, 5]})
    df3 = pl.DataFrame({"c": [6, 7, 8, 9]})

    result = pl.concat([df1, df2, df3], how="horizontal")
    expected = pl.DataFrame(
        {"a": [1, 2, 3, None], "b": [4, 5, None, None], "c": [6, 7, 8, 9]}
    )
    assert_frame_equal(result, expected)


def test_concat_align_no_common_columns() -> None:
    df1 = pl.DataFrame({"a": [1, 2]})
    df2 = pl.DataFrame({"b": [3, 4]})

    with pytest.raises(
        pl.exceptions.InvalidOperationError, match="requires at least one common column"
    ):
        pl.concat([df1, df2], how="align")


def test_concat_align_lazy_frames() -> None:
    lf1 = pl.DataFrame({"id": [1, 2], "x": [3, 4]}).lazy()
    lf2 = pl.DataFrame({"id": [2, 3], "y": [5, 6]}).lazy()

    result = pl.concat([lf1, lf2], how="align")
    assert isinstance(result, pl.LazyFrame)

    collected = result.collect()
    expected = pl.DataFrame({"id": [1, 2, 3], "x": [3, 4, None], "y": [None, 5, 6]})
    assert_frame_equal(collected, expected, check_row_order=False)


def test_concat_lazyframe_horizontal() -> None:
    lf1 = pl.DataFrame({"a": [1, 2]}).lazy()
    lf2 = pl.DataFrame({"b": [3, 4, 5]}).lazy()

    result = pl.concat([lf1, lf2], how="horizontal")
    assert isinstance(result, pl.LazyFrame)

    collected = result.collect()
    expected = pl.DataFrame({"a": [1, 2, None], "b": [3, 4, 5]})
    assert_frame_equal(collected, expected)


def test_concat_lazyframe_diagonal() -> None:
    lf1 = pl.DataFrame({"a": [1, 2], "b": [3, 4]}).lazy()
    lf2 = pl.DataFrame({"a": [5, 6], "c": [7, 8]}).lazy()

    result = pl.concat([lf1, lf2], how="diagonal")
    assert isinstance(result, pl.LazyFrame)

    collected = result.collect()
    expected = pl.DataFrame(
        {"a": [1, 2, 5, 6], "b": [3, 4, None, None], "c": [None, None, 7, 8]}
    )
    assert_frame_equal(collected, expected)


def test_concat_series_invalid_strategy() -> None:
    s1 = pl.Series("a", [1, 2, 3])
    s2 = pl.Series("b", [4, 5, 6])

    with pytest.raises(
        ValueError, match="Series only supports 'vertical' concat strategy"
    ):
        pl.concat([s1, s2], how="horizontal")

    with pytest.raises(
        ValueError, match="Series only supports 'vertical' concat strategy"
    ):
        pl.concat([s1, s2], how="diagonal")


def test_concat_invalid_how_parameter() -> None:
    df1 = pl.DataFrame({"a": [1, 2]})
    df2 = pl.DataFrame({"a": [3, 4]})

    with pytest.raises(ValueError, match="DataFrame `how` must be one of"):
        pl.concat([df1, df2], how="invalid_strategy")  # type: ignore[arg-type]


def test_concat_unsupported_type() -> None:
    with pytest.raises(TypeError, match="did not expect type"):
        pl.concat([1, 2, 3])  # type: ignore[type-var]


def test_concat_expressions() -> None:
    expr1 = pl.col("a")
    expr2 = pl.col("b")
    concat_expr = pl.concat([expr1, expr2])

    df_input = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
    result = df_input.select(concat_expr.alias("concatenated"))

    expected = pl.DataFrame({"concatenated": [1, 2, 3, 4]})
    assert_frame_equal(result, expected)


def test_concat_with_empty_dataframes() -> None:
    empty_df = pl.DataFrame(schema={"a": pl.Int64, "b": pl.String})
    df_with_data = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})

    result = pl.concat([empty_df, df_with_data])
    assert_frame_equal(result, df_with_data)

    result2 = pl.concat([df_with_data, empty_df])
    assert_frame_equal(result2, df_with_data)
