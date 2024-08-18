from datetime import date

import pytest

import polars as pl
from polars.exceptions import NoRowsReturnedError, TooManyRowsReturnedError


def test_row_tuple() -> None:
    df = pl.DataFrame({"a": ["foo", "bar", "2"], "b": [1, 2, 3], "c": [1.0, 2.0, 3.0]})

    # return row by index
    assert df.row(0) == ("foo", 1, 1.0)
    assert df.row(1) == ("bar", 2, 2.0)
    assert df.row(-1) == ("2", 3, 3.0)

    # return named row by index
    row = df.row(0, named=True)
    assert row == {"a": "foo", "b": 1, "c": 1.0}

    # return row by predicate
    assert df.row(by_predicate=pl.col("a") == "bar") == ("bar", 2, 2.0)
    assert df.row(by_predicate=pl.col("b").is_in([2, 4, 6])) == ("bar", 2, 2.0)

    # return named row by predicate
    row = df.row(by_predicate=pl.col("a") == "bar", named=True)
    assert row == {"a": "bar", "b": 2, "c": 2.0}

    # expected error conditions
    with pytest.raises(TooManyRowsReturnedError):
        df.row(by_predicate=pl.col("b").is_in([1, 3, 5]))

    with pytest.raises(NoRowsReturnedError):
        df.row(by_predicate=pl.col("a") == "???")

    # cannot set both 'index' and 'by_predicate'
    with pytest.raises(ValueError):
        df.row(0, by_predicate=pl.col("a") == "bar")

    # must call 'by_predicate' by keyword
    with pytest.raises(TypeError):
        df.row(None, pl.col("a") == "bar")  # type: ignore[call-overload]

    # cannot pass predicate into 'index'
    with pytest.raises(TypeError):
        df.row(pl.col("a") == "bar")  # type: ignore[call-overload]

    # at least one of 'index' and 'by_predicate' must be set
    with pytest.raises(ValueError):
        df.row()


def test_rows() -> None:
    df = pl.DataFrame({"a": [1, 2], "b": [1, 2]})

    # Regular rows
    assert df.rows() == [(1, 1), (2, 2)]
    assert df.reverse().rows() == [(2, 2), (1, 1)]

    # Named rows
    rows = df.rows(named=True)
    assert rows == [{"a": 1, "b": 1}, {"a": 2, "b": 2}]

    # Rows with nullarray cols
    df = df.with_columns(c=pl.lit(None))
    assert df.schema == {"a": pl.Int64, "b": pl.Int64, "c": pl.Null}
    assert df.rows() == [(1, 1, None), (2, 2, None)]
    assert df.rows(named=True) == [
        {"a": 1, "b": 1, "c": None},
        {"a": 2, "b": 2, "c": None},
    ]


def test_rows_by_key() -> None:
    df = pl.DataFrame(
        {
            "w": ["a", "b", "b", "a"],
            "x": ["q", "q", "q", "k"],
            "y": [1.0, 2.5, 3.0, 4.5],
            "z": [9, 8, 7, 6],
        }
    )

    # tuple (unnamed) rows
    assert df.rows_by_key("w") == {
        "a": [("q", 1.0, 9), ("k", 4.5, 6)],
        "b": [("q", 2.5, 8), ("q", 3.0, 7)],
    }
    assert df.rows_by_key("w", unique=True) == {
        "a": ("k", 4.5, 6),
        "b": ("q", 3.0, 7),
    }
    assert df.rows_by_key("w", include_key=True) == {
        "a": [("a", "q", 1.0, 9), ("a", "k", 4.5, 6)],
        "b": [("b", "q", 2.5, 8), ("b", "q", 3.0, 7)],
    }
    assert df.rows_by_key("w", include_key=True) == {
        key[0]: grp.rows() for key, grp in df.group_by(["w"])
    }
    assert df.rows_by_key("w", include_key=True, unique=True) == {
        "a": ("a", "k", 4.5, 6),
        "b": ("b", "q", 3.0, 7),
    }
    assert df.rows_by_key(["x", "w"]) == {
        ("a", "q"): [(1.0, 9)],
        ("b", "q"): [(2.5, 8), (3.0, 7)],
        ("a", "k"): [(4.5, 6)],
    }
    assert df.rows_by_key(["w", "x"], include_key=True) == {
        ("a", "q"): [("a", "q", 1.0, 9)],
        ("a", "k"): [("a", "k", 4.5, 6)],
        ("b", "q"): [("b", "q", 2.5, 8), ("b", "q", 3.0, 7)],
    }
    assert df.rows_by_key(["w", "x"], include_key=True, unique=True) == {
        ("a", "q"): ("a", "q", 1.0, 9),
        ("b", "q"): ("b", "q", 3.0, 7),
        ("a", "k"): ("a", "k", 4.5, 6),
    }

    # dict (named) rows
    assert df.rows_by_key("w", named=True) == {
        "a": [{"x": "q", "y": 1.0, "z": 9}, {"x": "k", "y": 4.5, "z": 6}],
        "b": [{"x": "q", "y": 2.5, "z": 8}, {"x": "q", "y": 3.0, "z": 7}],
    }
    assert df.rows_by_key("w", named=True, unique=True) == {
        "a": {"x": "k", "y": 4.5, "z": 6},
        "b": {"x": "q", "y": 3.0, "z": 7},
    }
    assert df.rows_by_key("w", named=True, include_key=True) == {
        "a": [
            {"w": "a", "x": "q", "y": 1.0, "z": 9},
            {"w": "a", "x": "k", "y": 4.5, "z": 6},
        ],
        "b": [
            {"w": "b", "x": "q", "y": 2.5, "z": 8},
            {"w": "b", "x": "q", "y": 3.0, "z": 7},
        ],
    }
    assert df.rows_by_key("w", named=True, include_key=True) == {
        key[0]: grp.rows(named=True) for key, grp in df.group_by(["w"])
    }
    assert df.rows_by_key("w", named=True, include_key=True, unique=True) == {
        "a": {"w": "a", "x": "k", "y": 4.5, "z": 6},
        "b": {"w": "b", "x": "q", "y": 3.0, "z": 7},
    }
    assert df.rows_by_key(["x", "w"], named=True) == {
        ("q", "a"): [{"y": 1.0, "z": 9}],
        ("q", "b"): [{"y": 2.5, "z": 8}, {"y": 3.0, "z": 7}],
        ("k", "a"): [{"y": 4.5, "z": 6}],
    }
    assert df.rows_by_key(["w", "x"], named=True, include_key=True) == {
        ("a", "q"): [{"w": "a", "x": "q", "y": 1.0, "z": 9}],
        ("a", "k"): [{"w": "a", "x": "k", "y": 4.5, "z": 6}],
        ("b", "q"): [
            {"w": "b", "x": "q", "y": 2.5, "z": 8},
            {"w": "b", "x": "q", "y": 3.0, "z": 7},
        ],
    }
    assert df.rows_by_key(["w", "x"], named=True, include_key=True, unique=True) == {
        ("a", "q"): {"w": "a", "x": "q", "y": 1.0, "z": 9},
        ("b", "q"): {"w": "b", "x": "q", "y": 3.0, "z": 7},
        ("a", "k"): {"w": "a", "x": "k", "y": 4.5, "z": 6},
    }


def test_iter_rows() -> None:
    df = pl.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [True, False, None],
        }
    ).with_columns(pl.Series(["a:b", "c:d", "e:f"]).str.split_exact(":", 1).alias("c"))

    # expected struct values
    c1 = {"field_0": "a", "field_1": "b"}
    c2 = {"field_0": "c", "field_1": "d"}
    c3 = {"field_0": "e", "field_1": "f"}

    # Default iter_rows behaviour
    it = df.iter_rows()
    assert next(it) == (1, True, c1)
    assert next(it) == (2, False, c2)
    assert next(it) == (3, None, c3)
    with pytest.raises(StopIteration):
        next(it)

    # Apply explicit row-buffer size
    for sz in (0, 1, 2, 3, 4):
        it = df.iter_rows(buffer_size=sz)
        assert next(it) == (1, True, c1)
        assert next(it) == (2, False, c2)
        assert next(it) == (3, None, c3)
        with pytest.raises(StopIteration):
            next(it)

        # Return named rows
        it_named = df.iter_rows(named=True, buffer_size=sz)
        row = next(it_named)
        assert row == {"a": 1, "b": True, "c": c1}
        row = next(it_named)
        assert row == {"a": 2, "b": False, "c": c2}
        row = next(it_named)
        assert row == {"a": 3, "b": None, "c": c3}

        with pytest.raises(StopIteration):
            next(it_named)

    # test over chunked frame
    df = pl.concat(
        [
            pl.DataFrame({"id": [0, 1], "values": ["a", "b"]}),
            pl.DataFrame({"id": [2, 3], "values": ["c", "d"]}),
        ],
        rechunk=False,
    )
    assert df.n_chunks() == 2
    assert df.to_dicts() == [
        {"id": 0, "values": "a"},
        {"id": 1, "values": "b"},
        {"id": 2, "values": "c"},
        {"id": 3, "values": "d"},
    ]


@pytest.mark.parametrize(
    "primitive",
    [
        pl.UInt8,
        pl.Int8,
        pl.UInt16,
        pl.Int16,
        pl.UInt32,
        pl.Int32,
        pl.UInt64,
        pl.Int64,
    ],
)
def test_row_constructor_schema(primitive: pl.DataType) -> None:
    result = pl.DataFrame(data=[[1], [2], [3]], schema={"d": primitive}, orient="row")

    assert result.dtypes == [primitive]
    assert result.to_dict(as_series=False) == {"d": [1, 2, 3]}


def test_row_constructor_uint64() -> None:
    # validate init with a valid UInt64 that exceeds Int64 upper bound
    df = pl.DataFrame(
        data=[[0], [int(2**63) + 1]], schema={"x": pl.UInt64}, orient="row"
    )
    assert df.rows() == [(0,), (9223372036854775809,)]


def test_physical_row_encoding() -> None:
    dt_str = [
        {
            "ts": date(2023, 7, 1),
            "files": "AGG_202307.xlsx",
            "period_bins": [date(2023, 7, 1), date(2024, 1, 1)],
        },
    ]

    df = pl.from_dicts(dt_str)
    df_groups = df.group_by("period_bins")
    assert df_groups.all().to_dicts() == [
        {
            "period_bins": [date(2023, 7, 1), date(2024, 1, 1)],
            "ts": [date(2023, 7, 1)],
            "files": ["AGG_202307.xlsx"],
        }
    ]
