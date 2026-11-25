from __future__ import annotations

import math

import pytest

import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal

FLOATS = [1.0, None, float("nan"), 3.0, 3.0, -0.0, 0.0, 2.0, None]


def multi_chunk(values: list[object], dtype: pl.DataType) -> pl.Series:
    mid = len(values) // 2
    return pl.concat(
        [pl.Series("a", values[:mid], dtype), pl.Series("a", values[mid:], dtype)],
        rechunk=False,
    )


@pytest.mark.parametrize("dtype", [pl.Float32, pl.Float64, pl.Int32, pl.Boolean])
def test_reverse_multi_chunk_and_nulls(dtype: pl.DataType) -> None:
    values: list[object] = (
        [True, None, False, True, None]
        if dtype == pl.Boolean
        else [1, None, 3, 4, None, 6, 7]
    )
    s = multi_chunk(values, dtype)
    assert s.n_chunks() == 2
    expected = pl.Series("a", values[::-1], dtype)
    assert_series_equal(s.reverse(), expected)
    assert_series_equal(s.slice(1, 3).reverse(), expected.slice(len(values) - 4, 3))
    no_nulls = multi_chunk([v for v in values if v is not None], dtype)
    assert no_nulls.reverse().to_list() == no_nulls.to_list()[::-1]
    assert pl.Series("a", [None, None], dtype).reverse().to_list() == [None, None]
    assert pl.Series("a", [], dtype).reverse().to_list() == []


def test_gather_every_one() -> None:
    s = pl.Series("a", [1, 2, 3, 4])
    assert s.gather_every(1).to_list() == [1, 2, 3, 4]
    assert s.gather_every(1, 2).to_list() == [3, 4]
    assert s.gather_every(1, 10).to_list() == []
    assert s.gather_every(2, 1).to_list() == [2, 4]
    assert s.slice(1).gather_every(1, 1).to_list() == [3, 4]


@pytest.mark.parametrize("dtype", [pl.Float32, pl.Float64, pl.Int64, pl.String])
def test_rle_id_matches_rle(dtype: pl.DataType) -> None:
    values: list[object] = (
        ["a", "a", None, None, "b", "a"]
        if dtype == pl.String
        else [1, 1, None, None, 2, 2, 1]
    )
    if dtype.is_float():
        values = [float("nan"), float("nan"), 0.0, -0.0, None, None, 1.0]
    s = multi_chunk(values, dtype)
    out = s.rle_id()
    assert out.dtype == pl.UInt32
    lengths = s.rle().struct.field("len").to_list()
    expected = [i for i, n in enumerate(lengths) for _ in range(n)]
    assert out.to_list() == expected
    assert out.is_sorted()
    assert pl.Series("a", [], dtype).rle_id().to_list() == []


def test_rle_id_nan_null() -> None:
    s = pl.Series([float("nan"), float("nan"), None, None, 1.0])
    assert s.rle_id().to_list() == [0, 0, 1, 1, 2]


@pytest.mark.parametrize("dtype", [pl.Float32, pl.Float64])
def test_clip_scalar_bounds_nulls(dtype: pl.DataType) -> None:
    s = multi_chunk([1.0, None, 5.0, -3.0, None, 9.0], dtype)
    out = s.clip(0, 6)
    assert out.to_list() == [1.0, None, 5.0, 0.0, None, 6.0]
    assert out.dtype == dtype


@pytest.mark.parametrize("dtype", [pl.Float32, pl.Float64])
def test_log_scalar_base(dtype: pl.DataType) -> None:
    xs = [1.0, 2.5, None, 0.0, -1.0, float("inf"), 10.0]
    s = pl.Series(xs, dtype=dtype)
    out = s.log(3.0)
    assert out.dtype == dtype
    for x, o in zip(xs, out.to_list(), strict=True):
        if x is None:
            assert o is None
        elif x < 0:
            assert math.isnan(o)
        elif x == 0:
            assert o == -math.inf
        elif dtype == pl.Float64:
            assert o == math.log(x) / math.log(3.0)
        else:
            assert o == pytest.approx(math.log(x) / math.log(3.0), rel=1e-6)
    base = pl.Series([2.0, 3.0, None, 2.0, 2.0, 2.0, 5.0], dtype=dtype)
    assert s.log(base).to_list()[2] is None


def test_pow_arctan2_columns() -> None:
    df = pl.DataFrame(
        {"b": [2.0, None, 3.0, 4.0], "e": [3.0, 2.0, None, 0.5]},
        schema={"b": pl.Float64, "e": pl.Float64},
    )
    out = df.select(p=pl.col("b").pow(pl.col("e")), a=pl.arctan2("b", "e"))
    assert out["p"].to_list() == [8.0, None, None, 2.0]
    assert out["a"][1] is None
    assert out["a"][0] == pytest.approx(math.atan2(2.0, 3.0))
    ints = pl.DataFrame({"b": [2, None, 3], "e": [3, 2, 4]})
    assert ints.select(pl.col("b").pow(pl.col("e")))["b"].to_list() == [8, None, 81]


def peaks_reference(values: list[float | None], is_max: bool) -> list[bool | None]:
    # The edges are compared against 0.
    out: list[bool | None] = []
    for i, v in enumerate(values):
        left = values[i - 1] if i > 0 else 0
        right = values[i + 1] if i + 1 < len(values) else 0
        results = [
            None if v is None or n is None else (n < v if is_max else n > v)
            for n in (left, right)
        ]
        if False in results:
            out.append(False)
        elif None in results:
            out.append(None)
        else:
            out.append(True)
    return out


@pytest.mark.parametrize("dtype", [pl.Float32, pl.Int64])
def test_peaks(dtype: pl.DataType) -> None:
    for values in (
        [1, 3, 2, 2, 5, 1, 7],
        [5],
        [],
        [1, None, 3, 2, 5],
        [None, 4, 1],
    ):
        s = pl.Series(values, dtype=dtype)
        for is_max, out in ((True, s.peak_max()), (False, s.peak_min())):
            assert out.to_list() == peaks_reference(values, is_max)  # type: ignore[arg-type]
    values = [1, 3, 2, 2, 5, 1, 7]
    chunked = multi_chunk(values, dtype)
    assert chunked.peak_max().to_list() == peaks_reference(values, True)  # type: ignore[arg-type]
    assert chunked.peak_min().to_list() == peaks_reference(values, False)  # type: ignore[arg-type]
    assert chunked.slice(1, 4).peak_max().to_list() == peaks_reference(
        values[1:5], True
    )  # type: ignore[arg-type]


def test_peaks_nan() -> None:
    assert pl.Series([1.0, float("nan"), 2.0]).peak_max().to_list() == [
        False,
        True,
        False,
    ]


def test_hist_explicit_bins_edges() -> None:
    s = pl.Series([0.0, 1.0, 1.5, 2.0, 2.0, 5.0, 5.5, -1.0, float("nan"), None, 9.0])
    out = s.hist(bins=[0.0, 1.0, 2.0, 5.0], include_category=False)
    assert out["count"].to_list() == [2, 3, 1]
    out = multi_chunk([3, 1, 1, 10, -4, None, 5], pl.Int64).hist(bins=[1, 3, 5])
    assert out["count"].to_list() == [3, 1]
    assert pl.Series([1.0, 2.0]).hist(bins=[1.0, 3.0])["count"].to_list() == [2]
    assert pl.Series([None, None], dtype=pl.Float64).hist(bins=[0.0, 1.0, 2.0])[
        "count"
    ].to_list() == [0, 0]


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_cut_qcut_labels_and_nulls() -> None:
    s = multi_chunk([1.0, None, float("nan"), 5.0, 2.0, 9.0, 5.0], pl.Float64)
    out = s.cut([2.0, 5.0], labels=["a", "b", "c"])
    assert out.dtype == pl.Enum(["a", "b", "c"])
    assert out.to_list() == ["a", None, None, "b", "a", "c", "b"]
    out = s.cut([2.0, 5.0], left_closed=True)
    assert out.to_list()[3] == "[5, inf)"
    out = s.cut([2.0, 5.0], labels=["a", "b", "c"], include_breaks=True)
    assert out.struct.field("breakpoint").to_list() == [
        2.0,
        None,
        None,
        5.0,
        2.0,
        float("inf"),
        5.0,
    ]
    assert out.struct.field("category").to_list() == [
        "a",
        None,
        None,
        "b",
        "a",
        "c",
        "b",
    ]
    q = pl.Series([1, 2, 3, 4, None, 5]).qcut(2, labels=["lo", "hi"])
    assert q.dtype == pl.Categorical
    assert q.to_list() == ["lo", "lo", "lo", "hi", None, "hi"]
    q = pl.Series([1, 2, 3, 4]).qcut(2, labels=["lo", "hi"], include_breaks=True)
    assert q.struct.field("category").to_list() == ["lo", "lo", "hi", "hi"]
    assert pl.Series([], dtype=pl.Float64).cut([1.0]).to_list() == []


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_cut_unused_labels_not_added_to_categories() -> None:
    s = pl.Series([1.0, 2.0])
    out = s.qcut([0.5], labels=["unused_lo_x", "unused_hi_x"], allow_duplicates=True)
    assert out.cast(pl.String).to_list() == ["unused_lo_x", "unused_hi_x"]


@pytest.mark.parametrize("dtype", [pl.Float32, pl.Int64])
def test_mean_horizontal_nulls(dtype: pl.DataType) -> None:
    df = pl.DataFrame(
        {"a": [1, 2, None, None], "b": [3, None, None, 4], "c": [5, 6, 7, None]},
        schema={"a": dtype, "b": dtype, "c": dtype},
    )
    out = df.select(pl.mean_horizontal("a", "b", "c"))["a"]
    assert out.dtype == (pl.Float32 if dtype == pl.Float32 else pl.Float64)
    assert out.to_list() == [3.0, 4.0, 7.0, 4.0]
    assert df.select(pl.mean_horizontal("a", "b", "c", ignore_nulls=False))[
        "a"
    ].to_list() == [3.0, None, None, None]
    full = df.drop_nulls()
    assert full.select(pl.mean_horizontal("a", "b", "c"))["a"].to_list() == [3.0]
    assert full.clear().select(pl.mean_horizontal("a", "b"))["a"].to_list() == []
    all_null = pl.DataFrame({"a": [None, None], "b": [None, None]})
    assert all_null.select(pl.mean_horizontal("a", "b"))["a"].to_list() == [None, None]
    assert_frame_equal(
        df.slice(0, 1).select(pl.mean_horizontal("a", "b")),
        pl.DataFrame(
            {"a": [2.0]},
            schema={"a": pl.Float32 if dtype == pl.Float32 else pl.Float64},
        ),
    )
