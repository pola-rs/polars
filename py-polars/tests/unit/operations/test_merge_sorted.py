from datetime import time

import pytest
from hypothesis import given

import polars as pl
from polars.exceptions import ComputeError
from polars.testing import assert_frame_equal, assert_series_equal
from polars.testing.parametric import series

left = pl.DataFrame({"a": [42, 13, 37], "b": [3, 8, 9]})
right = pl.DataFrame({"a": [5, 10, 1996], "b": [1, 5, 7]})
expected = pl.DataFrame(
    {
        "a": [5, 42, 10, 1996, 13, 37],
        "b": [1, 3, 5, 7, 8, 9],
    }
)

lf = left.lazy().merge_sorted(right.lazy(), "b")


@pytest.mark.parametrize("streaming", [False, True])
def test_merge_sorted(streaming: bool) -> None:
    assert_frame_equal(
        lf.collect(new_streaming=streaming),  # type: ignore[call-overload]
        expected,
    )


def test_merge_sorted_pred_pd() -> None:
    assert_frame_equal(
        lf.filter(pl.col.b > 30).collect(),
        expected.filter(pl.col.b > 30),
    )
    assert_frame_equal(
        lf.filter(pl.col.a < 6).collect(),
        expected.filter(pl.col.a < 6),
    )


def test_merge_sorted_proj_pd() -> None:
    assert_frame_equal(
        lf.select("b").collect(),
        lf.collect().select("b"),
    )
    assert_frame_equal(
        lf.select("a").collect(),
        lf.collect().select("a"),
    )


@pytest.mark.parametrize("precision", [2, 3])
def test_merge_sorted_decimal_20990(precision: int) -> None:
    dtype = pl.Decimal(precision=precision, scale=1)
    s = pl.Series("a", ["1.0", "0.1"], dtype)
    df = pl.DataFrame([s.sort()])
    result = df.lazy().merge_sorted(df.lazy(), "a").collect().get_column("a")
    expected = pl.Series("a", ["0.1", "0.1", "1.0", "1.0"], dtype)
    assert_series_equal(result, expected)


@pytest.mark.may_fail_auto_streaming
def test_merge_sorted_categorical() -> None:
    left = pl.Series("a", ["a", "b"], pl.Categorical()).sort().to_frame()
    right = pl.Series("a", ["a", "b", "b"], pl.Categorical()).sort().to_frame()
    result = left.merge_sorted(right, "a").get_column("a")
    expected = pl.Series("a", ["a", "a", "b", "b", "b"], pl.Categorical())
    assert_series_equal(result, expected)

    right = pl.Series("a", ["b", "a"], pl.Categorical()).sort().to_frame()
    with pytest.raises(
        ComputeError, match="can only merge-sort categoricals with the same categories"
    ):
        left.merge_sorted(right, "a")


@pytest.mark.parametrize(
    ("size", "ra"),
    [
        (10, [1, 7, 9]),
        (10, [0, 0, 0]),
        (10, [10, 10, 10]),
        (10, [1, None, None]),
        (10_000, [1, 2471, 6432]),
        (10_000, [777, 777, 777]),
        (10_000, [510, 1509, 1996, 2000]),
        (10_000, [None, None, None]),
        (10_000, [1, None, None]),
        (10_000, [None, None, 1]),
    ],
)
def test_merge_sorted_unbalanced(size: int, ra: list[int]) -> None:
    lhs = pl.DataFrame(
        [
            pl.Series("a", range(size), pl.Int32),
            pl.Series("b", range(size), pl.Int32),
        ]
    )
    rhs = pl.DataFrame(
        [
            pl.Series("a", ra, pl.Int32),
            pl.Series("b", [x * 7 for x in range(len(ra))], pl.Int32),
        ]
    )

    lf = lhs.lazy().merge_sorted(rhs.lazy(), "a")
    df = lf.collect(new_streaming=True)  # type: ignore[call-overload]

    nulls_last = ra[0] is not None

    assert df.height == size + len(ra)
    assert df.get_column("a").is_sorted(nulls_last=nulls_last)

    reference = (
        lhs.get_column("a").append(rhs.get_column("a")).sort(nulls_last=nulls_last)
    )
    assert_series_equal(df.get_column("a"), reference)


@given(
    lhs=series(
        name="a", allowed_dtypes=[pl.Int32], allow_null=False
    ),  # Nulls see: https://github.com/pola-rs/polars/issues/20991
    rhs=series(
        name="a", allowed_dtypes=[pl.Int32], allow_null=False
    ),  # Nulls see: https://github.com/pola-rs/polars/issues/20991
)
def test_merge_sorted_parametric_int(lhs: pl.Series, rhs: pl.Series) -> None:
    l_df = pl.DataFrame([lhs.sort()])
    r_df = pl.DataFrame([rhs.sort()])

    merge_sorted = l_df.lazy().merge_sorted(r_df.lazy(), "a").collect().get_column("a")
    append_sorted = lhs.append(rhs).sort()

    assert_series_equal(merge_sorted, append_sorted)


@given(
    lhs=series(
        name="a", allowed_dtypes=[pl.Binary], allow_null=False
    ),  # Nulls see: https://github.com/pola-rs/polars/issues/20991
    rhs=series(
        name="a", allowed_dtypes=[pl.Binary], allow_null=False
    ),  # Nulls see: https://github.com/pola-rs/polars/issues/20991
)
def test_merge_sorted_parametric_binary(lhs: pl.Series, rhs: pl.Series) -> None:
    l_df = pl.DataFrame([lhs.sort()])
    r_df = pl.DataFrame([rhs.sort()])

    merge_sorted = l_df.lazy().merge_sorted(r_df.lazy(), "a").collect().get_column("a")
    append_sorted = lhs.append(rhs).sort()

    assert_series_equal(merge_sorted, append_sorted)


@given(
    lhs=series(
        name="a", allowed_dtypes=[pl.String], allow_null=False
    ),  # Nulls see: https://github.com/pola-rs/polars/issues/20991
    rhs=series(
        name="a", allowed_dtypes=[pl.String], allow_null=False
    ),  # Nulls see: https://github.com/pola-rs/polars/issues/20991
)
def test_merge_sorted_parametric_string(lhs: pl.Series, rhs: pl.Series) -> None:
    l_df = pl.DataFrame([lhs.sort()])
    r_df = pl.DataFrame([rhs.sort()])

    merge_sorted = l_df.lazy().merge_sorted(r_df.lazy(), "a").collect().get_column("a")
    append_sorted = lhs.append(rhs).sort()

    assert_series_equal(merge_sorted, append_sorted)


@given(
    lhs=series(
        name="a",
        allowed_dtypes=[
            pl.Struct({"x": pl.Int32, "y": pl.Struct({"x": pl.Int8, "y": pl.Int8})})
        ],
        allow_null=False,
    ),  # Nulls see: https://github.com/pola-rs/polars/issues/20991
    rhs=series(
        name="a",
        allowed_dtypes=[
            pl.Struct({"x": pl.Int32, "y": pl.Struct({"x": pl.Int8, "y": pl.Int8})})
        ],
        allow_null=False,
    ),  # Nulls see: https://github.com/pola-rs/polars/issues/20991
)
def test_merge_sorted_parametric_struct(lhs: pl.Series, rhs: pl.Series) -> None:
    l_df = pl.DataFrame([lhs.sort()])
    r_df = pl.DataFrame([rhs.sort()])

    merge_sorted = l_df.lazy().merge_sorted(r_df.lazy(), "a").collect().get_column("a")
    append_sorted = lhs.append(rhs).sort()

    assert_series_equal(merge_sorted, append_sorted)


@given(
    s=series(
        name="a",
        excluded_dtypes=[
            pl.Categorical(
                ordering="lexical"
            ),  # Bug. See https://github.com/pola-rs/polars/issues/21025
        ],
        allow_null=False,  # See: https://github.com/pola-rs/polars/issues/20991
    ),
)
def test_merge_sorted_self_parametric(s: pl.Series) -> None:
    df = pl.DataFrame([s.sort()])

    merge_sorted = df.lazy().merge_sorted(df.lazy(), "a").collect().get_column("a")
    append_sorted = s.append(s).sort()

    assert_series_equal(merge_sorted, append_sorted)


# This was an encountered bug in the streaming engine, it was actually a bug
# with split_at.
def test_merge_time() -> None:
    s = pl.Series("a", [time(0, 0)], pl.Time)
    df = pl.DataFrame([s])
    assert df.merge_sorted(df, "a").get_column("a").dtype == pl.Time()
