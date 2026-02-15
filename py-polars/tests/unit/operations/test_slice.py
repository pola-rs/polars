from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import polars as pl
from polars.exceptions import ComputeError
from polars.testing import assert_frame_equal, assert_frame_not_equal

if TYPE_CHECKING:
    from tests.conftest import PlMonkeyPatch


def test_tail_union() -> None:
    assert (
        pl.concat(
            [
                pl.LazyFrame({"a": [1, 2]}),
                pl.LazyFrame({"a": [3, 4]}),
                pl.LazyFrame({"a": [5, 6]}),
            ]
        )
        .tail(1)
        .collect()
    ).to_dict(as_series=False) == {"a": [6]}


def test_python_slicing_data_frame() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": ["a", "b", "c"]})
    expected = pl.DataFrame({"a": [2, 3], "b": ["b", "c"]})
    for slice_params in (
        [1, 10],  # slice > len(df)
        [1, 2],  # slice == len(df)
        [1],  # optional len
    ):
        assert_frame_equal(df.slice(*slice_params), expected)

    # Negative starting index before start of dataframe.
    expected = pl.DataFrame({"a": [1, 2], "b": ["a", "b"]})
    assert_frame_equal(df.slice(-5, 4), expected)

    for py_slice in (
        slice(1, 2),
        slice(0, 2, 2),
        slice(3, -3, -1),
        slice(1, None, -2),
        slice(-1, -3, -1),
        slice(-3, None, -3),
    ):
        # confirm frame slice matches python slice
        assert df[py_slice].rows() == df.rows()[py_slice]


def test_python_slicing_series() -> None:
    s = pl.Series(name="a", values=[0, 1, 2, 3, 4, 5], dtype=pl.UInt8)
    for srs_slice, expected in (
        [s.slice(2, 3), [2, 3, 4]],
        [s.slice(4, 1), [4]],
        [s.slice(4, None), [4, 5]],
        [s.slice(3), [3, 4, 5]],
        [s.slice(-2), [4, 5]],
        [s.slice(-7, 4), [0, 1, 2]],
        [s.slice(-700, 4), []],
    ):
        assert srs_slice.to_list() == expected  # type: ignore[attr-defined]

    for py_slice in (
        slice(1, 2),
        slice(0, 2, 2),
        slice(3, -3, -1),
        slice(1, None, -2),
        slice(-1, -3, -1),
        slice(-3, None, -3),
    ):
        # confirm series slice matches python slice
        assert s[py_slice].to_list() == s.to_list()[py_slice]


def test_python_slicing_lazy_frame() -> None:
    ldf = pl.LazyFrame({"a": [1, 2, 3, 4], "b": ["a", "b", "c", "d"]})
    expected = pl.LazyFrame({"a": [3, 4], "b": ["c", "d"]})
    for slice_params in (
        [2, 10],  # slice > len(df)
        [2, 4],  # slice == len(df)
        [2],  # optional len
    ):
        assert_frame_equal(ldf.slice(*slice_params), expected)

    for py_slice in (
        slice(1, 2),
        slice(0, 3, 2),
        slice(-3, None),
        slice(None, 2, 2),
        slice(3, None, -1),
        slice(1, None, -2),
        slice(0, None, -1),
    ):
        # confirm frame slice matches python slice
        assert ldf[py_slice].collect().rows() == ldf.collect().rows()[py_slice]

    assert_frame_equal(ldf[0::-1], ldf.head(1))
    assert_frame_equal(ldf[2::-1], ldf.head(3).reverse())
    assert_frame_equal(ldf[::-1], ldf.reverse())
    assert_frame_equal(ldf[::-2], ldf.reverse().gather_every(2))


def test_head_tail_limit() -> None:
    df = pl.DataFrame({"a": range(10), "b": range(10)})

    assert df.head(5).rows() == [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
    assert_frame_equal(df.limit(5), df.head(5))
    assert df.tail(5).rows() == [(5, 5), (6, 6), (7, 7), (8, 8), (9, 9)]
    assert_frame_not_equal(df.head(5), df.tail(5))

    # check if it doesn't fail when out of bounds
    assert df.head(100).height == 10
    assert df.limit(100).height == 10
    assert df.tail(100).height == 10

    # limit is an alias of head
    assert_frame_equal(df.head(5), df.limit(5))

    # negative values
    assert df.head(-7).rows() == [(0, 0), (1, 1), (2, 2)]
    assert len(df.head(-2)) == 8
    assert df.tail(-8).rows() == [(8, 8), (9, 9)]
    assert len(df.tail(-6)) == 4

    # negative values out of bounds
    assert len(df.head(-12)) == 0
    assert len(df.limit(-12)) == 0
    assert len(df.tail(-12)) == 0


def test_hstack_slice_pushdown() -> None:
    lf = pl.LazyFrame({f"column_{i}": [i] for i in range(2)})

    out = lf.with_columns(pl.col("column_0") * 1000).slice(0, 5)
    plan = out.explain()

    assert not plan.startswith("SLICE")


def test_hconcat_slice_pushdown() -> None:
    num_dfs = 3
    lfs = [
        pl.LazyFrame({f"column_{i}": list(range(i, i + 10))}) for i in range(num_dfs)
    ]

    out = pl.concat(lfs, how="horizontal").slice(2, 3)
    plan = out.explain()

    assert not plan.startswith("SLICE")

    expected = pl.DataFrame(
        {f"column_{i}": list(range(i + 2, i + 5)) for i in range(num_dfs)}
    )

    df_out = out.collect()
    assert_frame_equal(df_out, expected)


@pytest.mark.parametrize(
    "ref",
    [
        [0, None],  # Mixed.
        [None, None],  # Full-null.
        [0, 0],  # All-valid.
    ],
)
def test_slice_nullcount(ref: list[int | None]) -> None:
    ref *= 128  # Embiggen input.
    s = pl.Series(ref)
    assert s.null_count() == sum(x is None for x in ref)
    assert s.slice(64).null_count() == sum(x is None for x in ref[64:])
    assert s.slice(50, 60).slice(25).null_count() == sum(x is None for x in ref[75:110])


def test_slice_pushdown_set_sorted() -> None:
    ldf = pl.LazyFrame({"foo": [1, 2, 3]})
    ldf = ldf.set_sorted("foo").head(2)
    plan = ldf.explain()
    assert "SLICE" not in plan
    assert ldf.collect().height == 2


def test_slice_pushdown_literal_projection_14349() -> None:
    # use length 1 slices as length 0 has a special fast path
    lf = pl.select(a=pl.int_range(10)).lazy()
    expect = pl.DataFrame({"a": [0, 1, 2, 3, 4], "b": [10, 11, 12, 13, 14]})

    out = lf.with_columns(b=pl.int_range(10, 20, eager=True)).head(5).collect()
    assert_frame_equal(expect, out)

    out = lf.select("a", b=pl.int_range(10, 20, eager=True)).head(5).collect()
    assert_frame_equal(expect, out)

    assert pl.LazyFrame().select(x=1).head(1).collect().height == 1
    assert pl.LazyFrame().with_columns(x=1).head(1).collect().height == 1

    q = lf.select(x=1).head(1)
    assert q.collect().height == 1

    # For select, slice pushdown should happen when at least 1 input column is selected
    q = lf.select("a", x=1).head(1)
    # slice isn't in plan if it has been pushed down to the dataframe
    assert "SLICE" not in q.explain()
    assert q.collect().height == 1

    # For with_columns, slice pushdown should happen if the input has at least 1 column
    q = lf.with_columns(x=1).head(1)
    assert "SLICE" not in q.explain()
    assert q.collect().height == 1

    q = lf.with_columns(pl.col("a") + 1).head(1)
    assert "SLICE" not in q.explain()
    assert q.collect().height == 1

    # This does not project any of the original columns
    q = lf.with_columns(a=1, b=2).head(1)
    plan = q.explain()
    assert plan.index("SLICE") < plan.index("WITH_COLUMNS")
    assert q.collect().height == 1

    q = lf.with_columns(b=1, c=2).head(1)
    assert "SLICE" not in q.explain()
    assert q.collect().height == 1


@pytest.mark.parametrize(
    "input_slice",
    [
        (-1, None, -1),
        (None, 0, -1),
        (1, -1, 1),
        (None, -1, None),
        (1, 2, -1),
        (-1, 1, 1),
    ],
)
def test_slice_lazy_frame_raises_proper(input_slice: tuple[int | None]) -> None:
    ldf = pl.LazyFrame({"a": [1, 2, 3]})
    s = slice(*input_slice)
    with pytest.raises(ValueError, match="not supported"):
        ldf[s].collect()


def test_double_sort_slice_pushdown_15779() -> None:
    assert (
        pl.LazyFrame({"foo": [1, 2]}).sort("foo").head(0).sort("foo").collect()
    ).shape == (0, 1)


def test_slice_pushdown_simple_projection_18288() -> None:
    lf = pl.DataFrame({"col": ["0", "notanumber"]}).lazy()
    lf = lf.with_columns([pl.col("col").cast(pl.Int64)])
    lf = lf.with_columns([pl.col("col"), pl.lit(None)])
    assert lf.head(1).collect().to_dict(as_series=False) == {
        "col": [0],
        "literal": [None],
    }


def test_group_by_slice_all_keys() -> None:
    df = pl.DataFrame(
        {
            "a": ["Tom", "Nick", "Marry", "Krish", "Jack", None],
            "b": [
                "2020-01-01",
                "2020-01-02",
                "2020-01-03",
                "2020-01-04",
                "2020-01-05",
                None,
            ],
            "c": [5, 6, 6, 7, 8, 5],
        }
    )

    gb = df.group_by(["a", "b", "c"], maintain_order=True)
    assert_frame_equal(gb.tail(1), gb.head(1))


def test_slice_first_in_agg_18551() -> None:
    df = pl.DataFrame({"id": [1, 1, 2], "name": ["A", "B", "C"], "value": [31, 21, 32]})

    assert df.group_by("id", maintain_order=True).agg(
        sort_by=pl.col("name").sort_by("value"),
        x=pl.col("name").sort_by("value").slice(0, 1).first(),
        y=pl.col("name").sort_by("value").slice(1, 1).first(),
    ).to_dict(as_series=False) == {
        "id": [1, 2],
        "sort_by": [["B", "A"], ["C"]],
        "x": ["B", "C"],
        "y": ["A", None],
    }


def test_slice_after_sort_with_nulls_20079() -> None:
    df = pl.LazyFrame({"a": [None, 1.2, None]})
    out = df.sort("a", nulls_last=True).slice(0, 10).collect()
    expected = pl.DataFrame({"a": [1.2, None, None]})
    assert_frame_equal(out, expected)

    out = df.sort("a", nulls_last=False).slice(0, 10).collect()
    expected = pl.DataFrame({"a": [None, None, 1.2]})
    assert_frame_equal(out, expected)


def test_slice_pushdown_panic_20216() -> None:
    col = pl.col("A")

    q = pl.LazyFrame({"A": "1/1"})
    q = q.with_columns(col.str.split("/"))
    q = q.with_columns(pl.when(col.is_not_null()).then(col.list.get(0)).otherwise(None))

    assert_frame_equal(q.slice(0, 1).collect(), pl.DataFrame({"A": ["1"]}))
    assert_frame_equal(q.collect(), pl.DataFrame({"A": ["1"]}))


def test_slice_empty_morsel_input() -> None:
    q = pl.LazyFrame({"a": []})
    assert_frame_equal(q.slice(999, 3).slice(999, 3).collect(), q.collect().clear())
    assert_frame_equal(q.slice(-999, 3).slice(-999, 3).collect(), q.collect().clear())


@pytest.mark.parametrize(
    "base_query",
    [
        (
            pl.LazyFrame({"a": [[1]]})
            .select("a", BARRIER=pl.col("a").sort())
            .with_columns(MARKER=1)
            .with_columns(b=pl.col("a").list.get(1, null_on_oob=False))
        ),
        (  # Variant to ensure cluster_with_columns runs after slice pushdown.
            pl.LazyFrame({"a": [[1]]})
            .with_columns(BARRIER=pl.col("a").sort())
            .with_columns(MARKER=1)
            .with_columns(b=pl.col("a").list.get(1, null_on_oob=False))
        ),
    ],
)
def test_slice_pushdown_pushes_past_fallible(
    base_query: pl.LazyFrame, plmonkeypatch: PlMonkeyPatch
) -> None:
    # Ensure baseline fails
    with pytest.raises(ComputeError, match="index is out of bounds"):
        base_query.collect()

    q = base_query.head(1)

    plan = q.explain()
    assert plan.index("BARRIER") > plan.index("SLICE") > plan.index("MARKER")

    with pytest.raises(ComputeError, match="index is out of bounds"):
        q.collect()

    # We allow slice 0 to pass fallible
    q = base_query.head(0)
    assert_frame_equal(q.collect(), pl.DataFrame(schema=q.collect_schema()))


def slice_ref(a: list[int], offset: int, length: int) -> list[int]:
    if offset < 0:
        offset += len(a)
    if offset < 0:
        length += offset
        offset = 0
    if length < 0:
        length = 0
    return a[offset : offset + length]


@pytest.mark.slow
def test_slice_slice_pushdown() -> None:
    df = pl.DataFrame({"x": [1, 2, 3, 4]})
    for outer_offset in range(-10, 10):
        for outer_len in range(10):
            for inner_offset in range(-10, 10):
                for inner_len in range(10):
                    a = [1, 2, 3, 4]
                    a = slice_ref(a, inner_offset, inner_len)
                    a = slice_ref(a, outer_offset, outer_len)
                    assert_frame_equal(
                        pl.DataFrame({"x": a}, schema={"x": pl.Int64}),
                        df.lazy()
                        .slice(inner_offset, inner_len)
                        .slice(outer_offset, outer_len)
                        .collect(),
                    )


@pytest.mark.parametrize("groupby", [True, False])
@pytest.mark.parametrize(
    "len",
    [
        0,
        pl.lit(0),
        pl.col("b").first(),
    ],
)
@pytest.mark.parametrize(
    "offset",
    [
        0,
        pl.lit(0),
        pl.col("a").first(),
    ],
)
@pytest.mark.parametrize(
    "lit",
    [
        pl.lit(7),
        pl.lit([7]),
        pl.lit([[7]]),
        pl.lit(pl.Series([7, 8, 9])),
        pl.col("c"),  # baseline, no literal
    ],
)
def test_schema_slice_on_literal_23999(
    lit: pl.Expr, offset: pl.Expr, len: pl.Expr, groupby: bool
) -> None:
    df = pl.DataFrame(
        {
            "g": [10, 10, 10, 20, 20, 30],
            "a": [0, 0, 0, 0, 0, 0],
            "b": [1, 1, 1, 1, 1, 1],
            "c": [11, 12, 13, 21, 22, 31],
        }
    )

    # slice
    if not groupby:
        q = df.lazy().select(lit.slice(offset, len))
    else:
        q = df.lazy().group_by("g").agg(lit.slice(offset, len))
    assert q.collect_schema() == q.collect().schema


@pytest.mark.parametrize("groupby", [True, False])
@pytest.mark.parametrize(
    "idx",
    [
        0,
        pl.lit(0),
        pl.col("a").first(),
    ],
)
@pytest.mark.parametrize(
    "lit",
    [
        pl.lit(7),
        pl.lit([7]),
        pl.lit([[7]]),
        pl.lit(pl.Series([7, 8, 9])),
        pl.col("c"),  # baseline, no literal
    ],
)
def test_schema_gather_get_on_literal_24101(
    lit: pl.Expr, idx: pl.Expr, groupby: bool
) -> None:
    df = pl.DataFrame(
        {
            "g": [10, 10, 10, 20, 20, 30],
            "a": [0, 0, 0, 0, 0, 0],
            "b": [1, 1, 1, 1, 1, 1],
            "c": [11, 12, 13, 21, 22, 31],
        }
    )

    # gather
    if not groupby:
        q = df.lazy().select(lit.gather(idx))
    else:
        q = df.lazy().group_by("g").agg(lit.gather(idx))
    assert q.collect_schema() == q.collect().schema

    # get
    if not groupby:
        q = df.lazy().select(lit.get(idx))
    else:
        q = df.lazy().group_by("g").agg(lit.get(idx))
    assert q.collect_schema() == q.collect().schema


@pytest.mark.parametrize("groupby", [True, False])
@pytest.mark.parametrize(
    "len",
    [
        1,
        pl.lit(1),
        pl.col("b").first(),
    ],
)
@pytest.mark.parametrize(
    "lit",
    [
        pl.lit(7),
        pl.lit([7]),
        pl.lit([[7]]),
        pl.lit(pl.Series([7, 8, 9])),
        pl.col("c"),  # baseline, no literal
    ],
)
def test_schema_head_tail_on_literal_24102(
    lit: pl.Expr, len: pl.Expr, groupby: bool
) -> None:
    df = pl.DataFrame(
        {
            "g": [10, 10, 10, 20, 20, 30],
            "a": [0, 0, 0, 0, 0, 0],
            "b": [1, 1, 1, 1, 1, 1],
            "c": [11, 12, 13, 21, 22, 31],
        }
    )

    # head
    if not groupby:
        q = df.lazy().select(lit.head(len))
    else:
        q = df.lazy().group_by("g").agg(lit.head(len))
    assert q.collect_schema() == q.collect().schema

    # tail
    if not groupby:
        q = df.lazy().select(lit.tail(len))
    else:
        q = df.lazy().group_by("g").agg(lit.tail(len))
    assert q.collect_schema() == q.collect().schema


def test_slice_negative_offset_none_len_26150() -> None:
    df = pl.DataFrame({"c0": [1, 2, 3], "c1": [6.0, 7.0, 8.0], "c2": ["a", "b", "c"]})

    out = df.slice(-10, None)
    assert_frame_equal(out, df)
