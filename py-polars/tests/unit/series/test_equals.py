from datetime import datetime
from typing import Callable

import pytest

import polars as pl
from polars.testing import assert_series_equal


def test_equals() -> None:
    s1 = pl.Series("a", [1.0, 2.0, None], pl.Float64)
    s2 = pl.Series("a", [1, 2, None], pl.Int64)

    assert s1.equals(s2) is True
    assert s1.equals(s2, check_dtypes=True) is False
    assert s1.equals(s2, null_equal=False) is False

    df = pl.DataFrame(
        {"dtm": [datetime(2222, 2, 22, 22, 22, 22)]},
        schema_overrides={"dtm": pl.Datetime(time_zone="UTC")},
    ).with_columns(
        s3=pl.col("dtm").dt.convert_time_zone("Europe/London"),
        s4=pl.col("dtm").dt.convert_time_zone("Asia/Tokyo"),
    )
    s3 = df["s3"].rename("b")
    s4 = df["s4"].rename("b")

    assert s3.equals(s4) is False
    assert s3.equals(s4, check_dtypes=True) is False
    assert s3.equals(s4, null_equal=False) is False
    assert s3.dt.convert_time_zone("Asia/Tokyo").equals(s4) is True


def test_series_equals_check_names() -> None:
    s1 = pl.Series("foo", [1, 2, 3])
    s2 = pl.Series("bar", [1, 2, 3])
    assert s1.equals(s2) is True
    assert s1.equals(s2, check_names=True) is False


def test_eq_list_cmp_list() -> None:
    s = pl.Series([[1], [1, 2]])
    result = s == [1, 2]
    expected = pl.Series([False, True])
    assert_series_equal(result, expected)


def test_eq_list_cmp_int() -> None:
    s = pl.Series([[1], [1, 2]])
    with pytest.raises(
        TypeError, match="cannot convert Python type 'int' to List\\(Int64\\)"
    ):
        s == 1  # noqa: B015


def test_eq_array_cmp_list() -> None:
    s = pl.Series([[1, 3], [1, 2]], dtype=pl.Array(pl.Int16, 2))
    result = s == [1, 2]
    expected = pl.Series([False, True])
    assert_series_equal(result, expected)


def test_eq_array_cmp_int() -> None:
    s = pl.Series([[1, 3], [1, 2]], dtype=pl.Array(pl.Int16, 2))
    with pytest.raises(
        TypeError,
        match="cannot convert Python type 'int' to Array\\(Int16, shape=\\(2,\\)\\)",
    ):
        s == 1  # noqa: B015


def test_eq_list() -> None:
    s = pl.Series([1, 1])

    result = s == [1, 2]
    expected = pl.Series([True, False])
    assert_series_equal(result, expected)

    result = s == 1
    expected = pl.Series([True, True])
    assert_series_equal(result, expected)


def test_eq_missing_expr() -> None:
    s = pl.Series([1, None])
    result = s.eq_missing(pl.lit(1))

    assert isinstance(result, pl.Expr)
    result_evaluated = pl.select(result).to_series()
    expected = pl.Series([True, False])
    assert_series_equal(result_evaluated, expected)


def test_ne_missing_expr() -> None:
    s = pl.Series([1, None])
    result = s.ne_missing(pl.lit(1))

    assert isinstance(result, pl.Expr)
    result_evaluated = pl.select(result).to_series()
    expected = pl.Series([False, True])
    assert_series_equal(result_evaluated, expected)


def test_series_equals_strict_deprecated() -> None:
    s1 = pl.Series("a", [1.0, 2.0, None], pl.Float64)
    s2 = pl.Series("a", [1, 2, None], pl.Int64)
    with pytest.deprecated_call():
        assert not s1.equals(s2, strict=True)  # type: ignore[call-arg]


@pytest.mark.parametrize("dtype", [pl.List(pl.Int64), pl.Array(pl.Int64, 2)])
@pytest.mark.parametrize(
    ("cmp_eq", "cmp_ne"),
    [
        # We parametrize the comparison sides as the impl looks like this:
        # match (left.len(), right.len()) {
        #     (1, _) => ...,
        #     (_, 1) => ...,
        #     (_, _) => ...,
        # }
        (pl.Series.eq, pl.Series.ne),
        (
            lambda a, b: pl.Series.eq(b, a),
            lambda a, b: pl.Series.ne(b, a),
        ),
    ],
)
def test_eq_lists_arrays(
    dtype: pl.DataType,
    cmp_eq: Callable[[pl.Series, pl.Series], pl.Series],
    cmp_ne: Callable[[pl.Series, pl.Series], pl.Series],
) -> None:
    # Broadcast NULL
    assert_series_equal(
        cmp_eq(
            pl.Series([None], dtype=dtype),
            pl.Series([None, [1, None], [1, 1]], dtype=dtype),
        ),
        pl.Series([None, None, None], dtype=pl.Boolean),
    )

    assert_series_equal(
        cmp_ne(
            pl.Series([None], dtype=dtype),
            pl.Series([None, [1, None], [1, 1]], dtype=dtype),
        ),
        pl.Series([None, None, None], dtype=pl.Boolean),
    )

    # Non-broadcast full-NULL
    assert_series_equal(
        cmp_eq(
            pl.Series(3 * [None], dtype=dtype),
            pl.Series([None, [1, None], [1, 1]], dtype=dtype),
        ),
        pl.Series([None, None, None], dtype=pl.Boolean),
    )

    assert_series_equal(
        cmp_ne(
            pl.Series(3 * [None], dtype=dtype),
            pl.Series([None, [1, None], [1, 1]], dtype=dtype),
        ),
        pl.Series([None, None, None], dtype=pl.Boolean),
    )

    # Broadcast valid
    assert_series_equal(
        cmp_eq(
            pl.Series([[1, None]], dtype=dtype),
            pl.Series([None, [1, None], [1, 1]], dtype=dtype),
        ),
        pl.Series([None, True, False], dtype=pl.Boolean),
    )

    assert_series_equal(
        cmp_ne(
            pl.Series([[1, None]], dtype=dtype),
            pl.Series([None, [1, None], [1, 1]], dtype=dtype),
        ),
        pl.Series([None, False, True], dtype=pl.Boolean),
    )

    # Non-broadcast mixed
    assert_series_equal(
        cmp_eq(
            pl.Series([None, [1, 1], [1, 1]], dtype=dtype),
            pl.Series([None, [1, None], [1, 1]], dtype=dtype),
        ),
        pl.Series([None, False, True], dtype=pl.Boolean),
    )

    assert_series_equal(
        cmp_ne(
            pl.Series([None, [1, 1], [1, 1]], dtype=dtype),
            pl.Series([None, [1, None], [1, 1]], dtype=dtype),
        ),
        pl.Series([None, True, False], dtype=pl.Boolean),
    )


@pytest.mark.parametrize("dtype", [pl.List(pl.Int64), pl.Array(pl.Int64, 2)])
@pytest.mark.parametrize(
    ("cmp_eq_missing", "cmp_ne_missing"),
    [
        (pl.Series.eq_missing, pl.Series.ne_missing),
        (
            lambda a, b: pl.Series.eq_missing(b, a),
            lambda a, b: pl.Series.ne_missing(b, a),
        ),
    ],
)
def test_eq_missing_lists_arrays_19153(
    dtype: pl.DataType,
    cmp_eq_missing: Callable[[pl.Series, pl.Series], pl.Series],
    cmp_ne_missing: Callable[[pl.Series, pl.Series], pl.Series],
) -> None:
    def assert_series_equal(
        left: pl.Series,
        right: pl.Series,
        *,
        assert_series_equal_impl: Callable[[pl.Series, pl.Series], None] = globals()[
            "assert_series_equal"
        ],
    ) -> None:
        # `assert_series_equal` also uses `ne_missing` underneath so we have
        # some extra checks here to be sure.
        assert_series_equal_impl(left, right)
        assert left.to_list() == right.to_list()
        assert left.null_count() == 0
        assert right.null_count() == 0

    # Broadcast NULL
    assert_series_equal(
        cmp_eq_missing(
            pl.Series([None], dtype=dtype),
            pl.Series([None, [1, None], [1, 1]], dtype=dtype),
        ),
        pl.Series([True, False, False]),
    )

    assert_series_equal(
        cmp_ne_missing(
            pl.Series([None], dtype=dtype),
            pl.Series([None, [1, None], [1, 1]], dtype=dtype),
        ),
        pl.Series([False, True, True]),
    )

    # Non-broadcast full-NULL
    assert_series_equal(
        cmp_eq_missing(
            pl.Series(3 * [None], dtype=dtype),
            pl.Series([None, [1, None], [1, 1]], dtype=dtype),
        ),
        pl.Series([True, False, False]),
    )

    assert_series_equal(
        cmp_ne_missing(
            pl.Series(3 * [None], dtype=dtype),
            pl.Series([None, [1, None], [1, 1]], dtype=dtype),
        ),
        pl.Series([False, True, True]),
    )

    # Broadcast valid
    assert_series_equal(
        cmp_eq_missing(
            pl.Series([[1, None]], dtype=dtype),
            pl.Series([None, [1, None], [1, 1]], dtype=dtype),
        ),
        pl.Series([False, True, False]),
    )

    assert_series_equal(
        cmp_ne_missing(
            pl.Series([[1, None]], dtype=dtype),
            pl.Series([None, [1, None], [1, 1]], dtype=dtype),
        ),
        pl.Series([True, False, True]),
    )

    # Non-broadcast mixed
    assert_series_equal(
        cmp_eq_missing(
            pl.Series([None, [1, 1], [1, 1]], dtype=dtype),
            pl.Series([None, [1, None], [1, 1]], dtype=dtype),
        ),
        pl.Series([True, False, True]),
    )

    assert_series_equal(
        cmp_ne_missing(
            pl.Series([None, [1, 1], [1, 1]], dtype=dtype),
            pl.Series([None, [1, None], [1, 1]], dtype=dtype),
        ),
        pl.Series([False, True, False]),
    )


def test_equals_nested_null_categorical_14875() -> None:
    dtype = pl.List(pl.Struct({"cat": pl.Categorical}))
    s = pl.Series([[{"cat": None}]], dtype=dtype)
    assert s.equals(s)
