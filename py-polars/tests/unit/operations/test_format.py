from typing import Any

import pytest

import polars as pl
from polars.testing import assert_frame_equal


def test_format_expr() -> None:
    a = [1, 2, 3]
    b = ["a", "b", None]
    df = pl.DataFrame({"a": a, "b": b})

    out = df.select(
        y=pl.format("{} abc", pl.lit("xyz")),
        z=pl.format("{} abc", pl.col.a),
        w=pl.format("{} abc {}", pl.col.a, pl.lit("xyz")),
        a=pl.format("{} abc {}", pl.lit("xyz"), pl.col.a),
        b=pl.format("abc {} {}", pl.lit("xyz"), pl.col.a),
        c=pl.format("abc {} {}", pl.lit("xyz"), pl.col.b),
        d=pl.format("abc {} {}", pl.col.a, pl.col.b),
        e=pl.format("{} abc {}", pl.col.a, pl.col.b),
        f=pl.format("{} {} abc", pl.col.a, pl.col.b),
        g=pl.format("{}{}", pl.col.a, pl.col.b),
        h=pl.format("{}", pl.col.a),
        i=pl.format("{}", pl.col.b),
    )

    b = [x if x is not None else "null" for x in b]

    expected = pl.DataFrame(
        {
            "y": ["xyz abc"] * 3,
            "z": [f"{i} abc" for i in a],
            "w": [f"{i} abc xyz" for i in a],
            "a": [f"xyz abc {i}" for i in a],
            "b": [f"abc xyz {i}" for i in a],
            "c": [f"abc xyz {i}" for i in b],
            "d": [f"abc {i} {j}" for i, j in zip(a, b)],
            "e": [f"{i} abc {j}" for i, j in zip(a, b)],
            "f": [f"{i} {j} abc" for i, j in zip(a, b)],
            "g": [f"{i}{j}" for i, j in zip(a, b)],
            "h": [f"{i}" for i in a],
            "i": [f"{i}" for i in b],
        }
    )

    assert_frame_equal(out, expected)


def test_format_fail_on_unequal() -> None:
    with pytest.raises(pl.exceptions.ShapeError):
        pl.select(pl.format("abc", pl.lit("x")))

    with pytest.raises(pl.exceptions.ShapeError):
        pl.select(pl.format("abc {}"))

    with pytest.raises(pl.exceptions.ShapeError):
        pl.select(pl.format("abc {} {}", pl.lit("x"), pl.lit("y"), pl.lit("z")))

    with pytest.raises(pl.exceptions.ShapeError):
        pl.select(pl.format("abc {}", pl.lit("x"), pl.lit("y")))


def test_format_group_by_23858() -> None:
    df = (
        pl.LazyFrame({"x": [0], "y": [0]})
        .group_by("x")
        .agg(pl.format("'{}'", pl.col("y")).alias("quoted_ys"))
        .with_columns(pl.col("quoted_ys").cast(pl.List(pl.String())).list.join(", "))
        .collect()
    )
    assert_frame_equal(df, pl.DataFrame({"x": [0], "quoted_ys": ["'0'"]}))


# Flaky - requires POLARS_MAX_THREADS=1 to trigger multiple chunks
# Only valid when run in isolation, see also GH issue #22070
def test_format_on_multiple_chunks_25159(monkeypatch: Any) -> None:
    monkeypatch.setenv("POLARS_MAX_THREADS", "1")
    df = pl.DataFrame({"group": ["A", "B"]})
    df = df.with_columns(
        pl.date_ranges(pl.date(2025, 1, 1), pl.date(2025, 1, 3))
    ).explode("date")
    out = df.group_by(pl.all()).agg(
        pl.format("{}", (pl.col("date").max()).dt.to_string()).alias("label")
    )
    assert out.shape == (6, 3)


def test_format_on_multiple_chunks_concat_25159() -> None:
    df1 = pl.DataFrame({"a": ["123"]})
    df2 = pl.DataFrame({"a": ["456"]})
    df = pl.concat([df1, df2])
    out = df.select(pl.format("{}", pl.col.a))
    assert_frame_equal(df, out)
