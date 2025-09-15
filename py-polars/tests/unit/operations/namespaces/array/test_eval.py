import pytest

import polars as pl
from polars.testing import assert_series_equal


def set_nulls(s: pl.Series, mask: list[bool]) -> pl.Series:
    return pl.select(pl.when(pl.Series([mask])).then(s).alias(s.name)).to_series()


@pytest.mark.parametrize("as_list", [False, True])
@pytest.mark.parametrize(
    "nulls",
    [
        [True] * 3,
        [False, True, True],
        [True, False, True],
        [True, True, False],
        [False, False, True],
        [True, False, False],
        [False] * 3,
    ],
)
def test_eval_basic(as_list: bool, nulls: list[bool]) -> None:
    if as_list:

        def rtdt(dt: pl.DataType) -> pl.DataType:
            return pl.List(dt)
    else:

        def rtdt(dt: pl.DataType) -> pl.DataType:
            return pl.Array(dt, 2)

    s = set_nulls(
        pl.Series("a", [[1, 4], [8, 5], [3, 2]], pl.Array(pl.Int64(), 2)), nulls
    )

    assert_series_equal(
        s.arr.eval(pl.element().rank(), as_list=as_list),
        set_nulls(
            pl.Series("a", [[1.0, 2.0], [2.0, 1.0], [2.0, 1.0]], rtdt(pl.Float64())),
            nulls,
        ),
    )
    assert_series_equal(
        s.arr.eval(pl.element() + 1, as_list=as_list),
        set_nulls(pl.Series("a", [[2, 5], [9, 6], [4, 3]], rtdt(pl.Int64())), nulls),
    )
    assert_series_equal(
        s.arr.eval(pl.element().cast(pl.String()), as_list=as_list),
        s.cast(rtdt(pl.Int64())).cast(rtdt(pl.String())),
    )

    if as_list:
        assert_series_equal(
            s.arr.eval(pl.element().unique(maintain_order=True), as_list=True),
            s.cast(rtdt(pl.Int64())),
        )


def test_eval_raises_for_non_length_preserving() -> None:
    s = pl.Series(
        "a", [["A", "B", "C"], ["C", "C", "D"], ["D", "E", "E"]], pl.Array(pl.String, 3)
    )

    with pytest.raises(pl.exceptions.InvalidOperationError, match="as_list"):
        s.arr.eval(pl.element().unique(maintain_order=True))


@pytest.mark.parametrize(
    "nulls",
    [
        [True] * 3,
        [False, True, True],
        [True, False, True],
        [True, True, False],
        [False, False, True],
        [True, False, False],
        [False] * 3,
    ],
)
def test_eval_changing_length(nulls: list[bool]) -> None:
    s = set_nulls(
        pl.Series(
            "a",
            [["A", "B", "C"], ["C", "C", "D"], ["D", "E", "E"]],
            pl.Array(pl.String, 3),
        ),
        nulls,
    )

    assert_series_equal(
        s.arr.eval(pl.element().unique(maintain_order=True), as_list=True),
        set_nulls(
            pl.Series(
                "a", [["A", "B", "C"], ["C", "D"], ["D", "E"]], pl.List(pl.String)
            ),
            nulls,
        ),
    )
