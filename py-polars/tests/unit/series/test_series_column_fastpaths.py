"""Cheap paths of `PySeries` that read a Scalar or Dictionary column unmaterialised.

Each is compared against the Series path: a plain `Column::Series` of the same values.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import pytest

import polars as pl
from polars.testing import assert_series_equal

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

LONG = [
    f"https://example.com/section-{i:03d}/item-page-{i * 7:05d}/view" for i in range(50)
]
LONG[7] = "ünïçode-" + LONG[7]  # multi-byte characters, for the truncation


def _kind(s: pl.Series) -> str:
    return s._s._column_kind()


def _coded_df(tmp_path: Path) -> pl.DataFrame:
    n = 3_000
    idx = pl.int_range(n, eager=True)
    k = pl.Series(LONG).gather(idx.hash(1) % len(LONG)).scatter([0, 1, 2, 97], None)
    path = tmp_path / "a.parquet"
    pl.DataFrame({"k": k}).write_parquet(path, row_group_size=500)
    (df,) = pl.scan_parquet(path).collect_batches(
        chunk_size=1_000_000, engine="streaming"
    )
    assert _kind(df["k"]) == "dictionary"
    return df


def _scalar(n: int = 10, value: object = 7, dtype: Any = pl.Int64) -> pl.Series:
    s = pl.DataFrame({"x": list(range(n))}).with_columns(
        pl.lit(value, dtype=dtype).alias("c")
    )["c"]
    assert _kind(s) == "scalar"
    return s


@pytest.fixture(
    params=["dictionary", "scalar", "scalar_str", "scalar_null", "scalar_f64"]
)
def column(request: pytest.FixtureRequest, tmp_path: Path) -> pl.Series:
    return {
        "dictionary": lambda: _coded_df(tmp_path)["k"],
        "scalar": lambda: _scalar(10),
        "scalar_str": lambda: _scalar(10, "zz" * 20, pl.String),
        "scalar_null": lambda: _scalar(10, None, pl.Int64),
        "scalar_f64": lambda: _scalar(10, 1.5, pl.Float64),
    }[request.param]()


@pytest.fixture
def coded(tmp_path: Path) -> pl.Series:
    return _coded_df(tmp_path)["k"]


def _plain(s: pl.Series) -> pl.Series:
    """The Series path's version of `s`: a `Column::Series` with the same chunks."""
    if _kind(s) == "scalar":
        m = pl.Series(s.name, ([s[0]] * len(s) if len(s) else []), dtype=s.dtype)
    elif len(s) <= 1:  # `from_arrow` of one row gives a scalar column
        m = pl.Series(s.name, s.to_arrow().to_pylist(), dtype=s.dtype)
    else:
        m = pl.from_arrow(s.to_arrow()).alias(s.name)  # type: ignore[union-attr]
    assert _kind(m) == "series"
    return m


def _outcome(f: Callable[[], object]) -> tuple[object, ...]:
    try:
        return ("ok", f())
    except Exception as e:
        return ("error", type(e), str(e))


def _same(result: pl.Series, expected: pl.Series) -> None:
    assert_series_equal(result, expected)
    assert result.flags == expected.flags
    assert result._s.chunk_lengths() == expected._s.chunk_lengths()


def _unmaterialised(*series: pl.Series) -> None:
    for s in series:
        assert not s._s._is_materialised()


# --- is_empty and Python slicing -----------------------------------------------------


def test_is_empty(column: pl.Series) -> None:
    m = _plain(column)
    for ignore_nulls in (False, True):
        assert column.is_empty(ignore_nulls=ignore_nulls) == m.is_empty(
            ignore_nulls=ignore_nulls
        )
    _unmaterialised(column)


def test_is_empty_all_null_dictionary(coded: pl.Series) -> None:
    nulls = coded.slice(0, 3)
    assert _kind(nulls) == "dictionary"
    assert len(nulls) == 3
    assert nulls.is_empty() is False
    assert nulls.is_empty(ignore_nulls=True) is True
    assert coded.is_empty(ignore_nulls=True) is False
    _unmaterialised(nulls, coded)


def test_python_slicing_does_not_materialise_dictionary(coded: pl.Series) -> None:
    m = _plain(coded)
    f: Callable[..., Any]
    for f in (
        lambda x: x[2:5],
        lambda x: x[-5:],
        lambda x: x[:7],
        lambda x: x[10:-10],
        lambda x: x[::2],
        lambda x: x[::-1],
        lambda x: x[5:5],
        lambda x: x[5:3],
    ):
        _same(f(coded), f(m))
        _unmaterialised(coded)
    for f in (lambda x: x[2:5], lambda x: x[-5:]):
        assert _kind(f(coded)) == "dictionary"


# --- get_fmt -------------------------------------------------------------------------


def test_get_fmt(column: pl.Series) -> None:
    m = _plain(column)
    n = len(column)
    for i in (0, 1, 7, n - 1):
        for limit in (0, 1, 5, 20, 1000):
            assert column._s.get_fmt(i, limit) == m._s.get_fmt(i, limit), (i, limit)
    assert column._s.get_fmt(97, 10) == m._s.get_fmt(97, 10) if n > 97 else True
    _unmaterialised(column)


def test_get_fmt_map_scalar() -> None:
    st = pl.List(pl.Struct({"key": pl.String, "value": pl.Int64}))
    s = pl.DataFrame({"x": [1, 2, 3]}).with_columns(
        pl.lit([{"key": "a", "value": 1}], dtype=st)
        .cast(pl.Map(pl.String, pl.Int64))
        .alias("c")
    )["c"]
    assert _kind(s) == "scalar"
    got = [s._s.get_fmt(1, limit) for limit in (0, 3, 50)]
    _unmaterialised(s)
    m = s._s._materialized()
    assert got == [m.get_fmt(1, limit) for limit in (0, 3, 50)]


def test_get_fmt_scalar_dtypes() -> None:
    cases: list[tuple[object, Any]] = [
        ("héllo wörld", pl.String),
        (None, pl.String),
        ("abc", pl.Categorical),
        ("abc", pl.Enum(["abc", "d"])),
        (b"ab", pl.Binary),
        (1, pl.UInt8),
        (-1.5, pl.Float32),
        (float("nan"), pl.Float64),
        (True, pl.Boolean),
        (Decimal("1.25"), pl.Decimal(10, 2)),
        (date(2024, 2, 29), pl.Date),
        (datetime(2024, 2, 29, 1, 2, 3), pl.Datetime("us")),
        (datetime(2024, 2, 29, 1, 2, 3, tzinfo=UTC), pl.Datetime("ns", "UTC")),
        (timedelta(hours=1), pl.Duration("ms")),
        ([1, 2], pl.List(pl.Int64)),
        ({"a": 1}, pl.Struct({"a": pl.Int64})),
    ]
    for value, dtype in cases:
        s = _scalar(3, value, dtype)
        m = _plain(s)
        for i in (0, 2):
            for limit in (0, 2, 100):
                assert s._s.get_fmt(i, limit) == m._s.get_fmt(i, limit), (value, dtype)
        _unmaterialised(s)


def test_html_repr_does_not_materialise_dictionary(coded: pl.Series) -> None:
    m = _plain(coded)
    assert coded.to_frame()._repr_html_() == m.to_frame()._repr_html_()
    _unmaterialised(coded)


# --- first / last --------------------------------------------------------------------


def _edges(s: pl.Series) -> list[object]:
    return [
        s.first(),
        s.last(),
        s.first(ignore_nulls=True),
        s.last(ignore_nulls=True),
    ]


def test_first_last(column: pl.Series) -> None:
    m = _plain(column)
    assert _edges(column) == _edges(m)
    _unmaterialised(column)


def test_first_last_dictionary_nulls(coded: pl.Series) -> None:
    # Rows 0-2 and 97 are null.
    assert coded[0] is None
    for s in (
        coded.head(98),  # null last
        coded.tail(2900),
        coded.slice(0, 3),  # all null
        coded.slice(97, 1),  # one null row
        coded.slice(3, 1),  # one valid row
        coded.slice(0, 98),
    ):
        assert _kind(s) == "dictionary"
        m = _plain(s)
        assert _edges(s) == _edges(m), s.len()
        _unmaterialised(s)
    _unmaterialised(coded)


@pytest.mark.parametrize("n", [0, 1, 3])
@pytest.mark.parametrize("value", [None, "a"])
def test_first_last_scalar_lengths(n: int, value: str | None) -> None:
    s = _scalar(n, value, pl.String)
    assert _edges(s) == _edges(_plain(s))


# --- sorted flags --------------------------------------------------------------------


def test_sorted_flags_dictionary(coded: pl.Series) -> None:
    # A dictionary column reaches Python unsorted (`set_sorted` and `sort` give a plain
    # Series), and `reverse` keeps the codes.
    one = coded.head(1)  # a one-row Series reports itself ascending
    assert (
        one._s.is_sorted_ascending_flag() is _plain(one)._s.is_sorted_ascending_flag()
    )
    for s in (coded, coded.reverse(), coded.head(5)):
        assert _kind(s) == "dictionary"
        m = _plain(s)
        assert s._s.is_sorted_ascending_flag() is False
        assert s._s.is_sorted_descending_flag() is False
        _unmaterialised(s)
        assert s.flags == m.flags


def test_sorted_flags_other_columns_unchanged(column: pl.Series) -> None:
    s = column._s._materialized()
    assert column._s.is_sorted_ascending_flag() == s.is_sorted_ascending_flag()
    assert column._s.is_sorted_descending_flag() == s.is_sorted_descending_flag()


# --- to_list -------------------------------------------------------------------------


def test_to_list_dictionary(coded: pl.Series) -> None:
    expected = coded.to_arrow().to_pylist()
    got = coded.to_list()
    assert got == expected
    assert got.count(None) == 4
    assert all(v is None or type(v) is str for v in got)
    _unmaterialised(coded)

    for s in (
        coded.head(5),
        coded[700:1700],  # across chunks
        coded.tail(1),
        coded.slice(0, 3),
        coded.reverse(),
        coded.gather([5, 1, 1, 0]),
    ):
        assert _kind(s) == "dictionary"
        assert s.to_list() == s.to_arrow().to_pylist()
        assert s.to_list() == _plain(s).to_list()
        _unmaterialised(s)
    assert coded.slice(0, 0).to_list() == []
    _unmaterialised(coded)


def test_to_list_dictionary_repeated_values(coded: pl.Series) -> None:
    both = coded.clone()
    both.append(coded.head(500))
    both.append(coded.tail(300))
    assert _kind(both) == "dictionary"
    assert both.to_list() == [
        *coded.to_arrow().to_pylist(),
        *coded.head(500).to_list(),
        *coded.tail(300).to_list(),
    ]
    _unmaterialised(both)


def test_to_list_other_columns_unchanged(column: pl.Series) -> None:
    assert column.to_list() == _plain(column).to_list()


# --- equals --------------------------------------------------------------------------


def _equals_matrix(a: pl.Series, b: pl.Series) -> list[bool]:
    return [
        a.equals(b, check_dtypes=d, check_names=n, null_equal=e)
        for d in (False, True)
        for n in (False, True)
        for e in (False, True)
    ]


def test_equals(column: pl.Series) -> None:
    m = _plain(column)
    n = len(column)
    others = [
        column,
        column.clone(),
        m,
        column.alias("other"),
        column.head(n - 1),
        column.head(0),
        column.cast(pl.String) if column.dtype != pl.String else column.cast(pl.Binary),
        column.clone().scatter(3, None),
        column.reverse(),
    ]
    for other in others:
        mo = _plain(other) if _kind(other) != "series" else other
        assert _equals_matrix(column, other) == _equals_matrix(m, mo)
        assert _equals_matrix(other, column) == _equals_matrix(mo, m)


def test_equals_early_outs_do_not_materialise(coded: pl.Series) -> None:
    other = coded.alias("b")
    assert other._s._column_kind() == "dictionary"
    assert not coded.equals(other, check_names=True)
    assert not coded.equals(other.head(10))
    assert not coded.equals(coded.head(10).cast(pl.Categorical), check_dtypes=True)
    assert not coded.equals(coded, null_equal=False)  # nulls are never equal
    assert not coded.equals(coded.gather([1, 2]))
    _unmaterialised(coded)
    # An answer that needs the values materialises.
    assert coded.equals(coded.clone())
    assert coded.equals(coded.alias("c"))


def test_equals_scalars() -> None:
    a = _scalar(5, 1, pl.Int64)
    assert a.equals(_scalar(5, 1, pl.Int64))
    assert a.equals(_scalar(5, 1, pl.Int32))
    assert not a.equals(_scalar(5, 1, pl.Int32), check_dtypes=True)
    assert not a.equals(_scalar(4, 1, pl.Int64))
    assert not a.equals(_scalar(5, 2, pl.Int64))
    nulls = _scalar(5, None, pl.Int64)
    assert nulls.equals(_scalar(5, None, pl.Int64))
    assert not nulls.equals(_scalar(5, None, pl.Int64), null_equal=False)
    assert not nulls.equals(a)
    assert not a.equals(nulls, null_equal=False)
    tz = _scalar(2, datetime(2024, 1, 1), pl.Datetime("us", "UTC"))
    assert not tz.equals(_scalar(2, datetime(2024, 1, 1), pl.Datetime("us")))
    assert not a.equals(a.alias("z"), check_names=True)
    assert a.equals(a.alias("z"))


# --- approx_n_unique -----------------------------------------------------------------


@pytest.mark.parametrize("n", [0, 1, 2, 10])
def test_approx_n_unique_scalar(n: int) -> None:
    cases: list[tuple[object, Any]] = [
        (7, pl.Int64),
        (None, pl.Int64),
        (None, pl.String),
        ("a", pl.String),
        (1.5, pl.Float64),
        (float("nan"), pl.Float64),
        (True, pl.Boolean),
        (None, pl.Null),
        ([1], pl.List(pl.Int64)),
        (Decimal("1.25"), pl.Decimal(10, 2)),
    ]
    for value, dtype in cases:
        s = _scalar(n, value, dtype)
        m = _plain(s)
        got, expected = _outcome(s.approx_n_unique), _outcome(m.approx_n_unique)
        assert got == expected, (n, value, dtype)
        if _kind(s) == "scalar":
            _unmaterialised(s)


# --- append / extend / concat keep the flag the Series path keeps ---------------------


APPEND_CASES: dict[str, Callable[[pl.Series], tuple[pl.Series, pl.Series]]] = {
    "one_plus_one": lambda k: (k.slice(5, 1), k.slice(6, 1)),
    "n_plus_empty": lambda k: (k.slice(5, 4), k.slice(5, 0)),
    "empty_plus_n": lambda k: (k.slice(5, 0), k.slice(5, 4)),
    "null_plus_null": lambda k: (k.slice(0, 3), k.slice(0, 3)),
    "null_plus_one": lambda k: (k.slice(0, 3), k.slice(5, 1)),
    "one_plus_null": lambda k: (k.slice(5, 1), k.slice(0, 3)),
    "unsorted": lambda k: (k.slice(5, 4), k.slice(5, 4)),
    "null_plus_unsorted": lambda k: (k.slice(0, 3), k.slice(5, 4)),
}


@pytest.mark.parametrize("case", APPEND_CASES)
@pytest.mark.parametrize("how", ["append", "extend", "concat"])
def test_dictionary_append_flags_match_series_path(
    tmp_path: Path, case: str, how: str
) -> None:
    k = _coded_df(tmp_path)["k"]
    a, b = APPEND_CASES[case](k)
    pa, pb = _plain(a), _plain(b)
    if how == "concat":
        result = pl.concat([a, b], rechunk=False)
        expected = pl.concat([pa, pb], rechunk=False)
    else:
        result, expected = a, pa
        getattr(result, how)(b)
        getattr(expected, how)(pb)
    _same(result, expected)
    assert result.n_chunks() == expected.n_chunks()
    assert result.to_list() == expected.to_list()


def test_to_list_short_slice_of_big_dictionary(tmp_path: Path) -> None:
    # A slice keeps the whole dictionary: the conversion must follow the rows.
    values = [f"https://example.com/item-{i:07d}" for i in range(5_000)]
    path = tmp_path / "big.parquet"
    pl.DataFrame({"b": values * 10}).write_parquet(path, row_group_size=50_000)
    (df,) = pl.scan_parquet(path).collect_batches(
        chunk_size=1_000_000, engine="streaming"
    )
    s = df["b"]
    assert _kind(s) == "dictionary"
    for sl in (s.slice(100, 10), s.slice(4_995, 10), s.slice(7, 1), s.slice(0, 6_000)):
        assert sl.to_list() == sl.to_arrow().to_pylist()
        _unmaterialised(sl)
    assert s.slice(4_995, 10).to_list() == [*values[4_995:], *values[:5]]
    assert s.slice(100, 3).to_list() == values[100:103]
    _unmaterialised(s)
