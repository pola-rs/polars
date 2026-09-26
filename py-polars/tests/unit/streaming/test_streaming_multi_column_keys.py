from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

import pytest

import polars as pl
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from polars._typing import JoinStrategy
    from tests.conftest import PlMonkeyPatch

pytestmark = pytest.mark.xdist_group("streaming")

LONG = "a string that is too long to be inlined"

KEYS = [
    ["i64", "i64_b"],
    ["i32", "date"],
    ["i64", "dec"],
    ["bool", "i8"],
    ["f64", "i64"],
    ["cat", "i32"],
    ["short", "short_b"],
    ["long", "short"],
    ["long", "long_b"],
    ["short", "i64"],
    ["i64", "i32", "date", "bool", "short", "long"],
    ["short", "short_b", "long"],
]


def keys_frame(n: int, offset: int = 0, *, nulls: bool) -> pl.DataFrame:
    i = pl.int_range(offset, offset + n)
    df = pl.select(
        i64=i % 7,
        i64_b=i % 3,
        i32=(i % 5).cast(pl.Int32),
        date=pl.date(2020, 1, 1) + pl.duration(days=i % 4),
        dec=((i % 6) / 4).cast(pl.Decimal(10, 2)),
        bool=i % 2 == 0,
        i8=(i % 3).cast(pl.Int8),
        f64=pl.when(i % 4 == 0)
        .then(-0.0)
        .when(i % 4 == 1)
        .then(0.0)
        .when(i % 4 == 2)
        .then(float("nan"))
        .otherwise(1.5),
        cat=(i % 3).cast(pl.String).cast(pl.Categorical),
        short=(i % 3).cast(pl.String),
        short_b=(i % 2).cast(pl.String),
        long=pl.lit(LONG) + (i % 4).cast(pl.String),
        long_b=(i % 2).cast(pl.String) + pl.lit(LONG),
        v=i,
    )
    if nulls:
        df = df.with_columns(
            pl.when(pl.col("v") % 13 != c).then(pl.col(name)).alias(name)
            for c, name in enumerate(df.columns[:-1])
        )
    return df


@pytest.fixture(params=[4, 4096], ids=["evicting", "hot"])
def hot_table_size(
    request: pytest.FixtureRequest, plmonkeypatch: PlMonkeyPatch
) -> None:
    plmonkeypatch.setenv("POLARS_HOT_TABLE_SIZE", str(request.param))


def assert_engines_equal(lf: pl.LazyFrame, *, check_row_order: bool = False) -> None:
    assert_frame_equal(
        lf.collect(engine="streaming"),
        lf.collect(engine="in-memory"),
        check_row_order=check_row_order,
    )


@pytest.mark.usefixtures("hot_table_size")
@pytest.mark.parametrize("keys", KEYS, ids="-".join)
@pytest.mark.parametrize("nulls", [False, True])
def test_group_by_multi_column_keys(keys: list[str], nulls: bool) -> None:
    lf = keys_frame(1000, nulls=nulls).lazy()
    assert_engines_equal(
        lf.group_by(keys).agg(pl.len(), pl.col("v").sum().alias("sum"))
    )
    # Order-sensitive aggregations keep every group hot.
    assert_engines_equal(
        lf.group_by(keys).agg(pl.col("v").first().alias("first"), pl.col("v").last())
    )


@pytest.mark.parametrize("keys", KEYS, ids="-".join)
@pytest.mark.parametrize("how", ["inner", "left", "full", "semi", "anti"])
@pytest.mark.parametrize("nulls_equal", [False, True])
def test_join_multi_column_keys(
    keys: list[str], how: JoinStrategy, nulls_equal: bool
) -> None:
    # The right side is built separately, so equal strings live in other buffers.
    left = keys_frame(300, nulls=True).lazy()
    right = keys_frame(200, offset=50, nulls=True).lazy()
    assert_engines_equal(left.join(right, on=keys, how=how, nulls_equal=nulls_equal))


@pytest.mark.parametrize("keys", KEYS, ids="-".join)
def test_join_multi_column_keys_sliced_build_side(keys: list[str]) -> None:
    left = keys_frame(300, nulls=True)
    right = keys_frame(400, nulls=True).slice(123, 150)
    assert_engines_equal(left.lazy().join(right.lazy(), on=keys, how="inner"))


@pytest.mark.parametrize("keys", KEYS, ids="-".join)
def test_unique_multi_column_keys(keys: list[str]) -> None:
    lf = keys_frame(500, nulls=True).lazy()
    assert_engines_equal(
        lf.unique(subset=keys, keep="first", maintain_order=True),
        check_row_order=True,
    )
    assert_engines_equal(
        lf.select(pl.struct(keys).is_first_distinct()), check_row_order=True
    )


def test_group_by_list_key_is_not_a_key_row() -> None:
    lf = pl.LazyFrame(
        {"a": [[1], [1], [2], None], "b": [1, 1, 1, 2], "v": [1, 2, 3, 4]}
    )
    assert_engines_equal(lf.group_by("a", "b").agg(pl.col("v").sum()))


def test_group_by_key_rows_output_types() -> None:
    df = pl.DataFrame(
        {
            "dec": [Decimal("1.50"), Decimal("1.50"), None],
            "s": ["x", "x", LONG],
            "b": [b"\x00", b"\x00", None],
        },
        schema={"dec": pl.Decimal(10, 2), "s": pl.String, "b": pl.Binary},
    )
    out = df.lazy().group_by("dec", "s", "b").agg(pl.len()).collect(engine="streaming")
    assert out.schema == pl.Schema(
        {"dec": pl.Decimal(10, 2), "s": pl.String, "b": pl.Binary, "len": pl.UInt32}
    )
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "dec": [Decimal("1.50"), None],
                "s": ["x", LONG],
                "b": [b"\x00", None],
                "len": [2, 1],
            },
            schema=out.schema,
        ),
        check_row_order=False,
    )
