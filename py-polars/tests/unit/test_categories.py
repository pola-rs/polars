from __future__ import annotations

import io
import subprocess
import sys

import pytest

import polars as pl
from polars.exceptions import ComputeError, SchemaError
from polars.testing import assert_frame_equal, assert_series_equal

CATS = [
    pl.Categories(),
    pl.Categories("foo"),
    pl.Categories("foo", physical=pl.UInt16),
    pl.Categories("boo"),
    pl.Categories("foo", "bar"),
    pl.Categories("foo", "baz"),
    pl.Categories("foo", "bar", physical=pl.UInt8),
    pl.Categories.random(),
]


def test_categories_eq_hash() -> None:
    left = CATS
    right = [pl.Categories(c.name(), c.namespace(), c.physical()) for c in CATS]

    for lc, rc in zip(left, right):
        assert hash(lc) == hash(rc)
        assert lc == rc

    for i in range(len(left)):
        for j in range(len(right)):
            if i != j:
                assert left[i] != right[j]


@pytest.mark.parametrize("cats", CATS)
def test_cat_parquet_roundtrip(cats: pl.Categories) -> None:
    df = pl.DataFrame({"x": ["foo", "bar", "moo"]}, schema={"x": pl.Categorical(cats)})
    f = io.BytesIO()
    df.write_parquet(f)
    del df  # Delete frame holding reference.
    f.seek(0)
    df2 = pl.scan_parquet(f).collect()
    df = pl.DataFrame({"x": ["foo", "bar", "moo"]}, schema={"x": pl.Categorical(cats)})
    assert_frame_equal(df, df2)


@pytest.mark.may_fail_cloud  # reason: these are not seen locally
def test_local_categories_gc() -> None:
    dt = pl.Categorical(pl.Categories.random())
    df = pl.DataFrame({"x": ["foo", "bar", "moo"]}, schema={"x": dt})
    assert set(df["x"].cat.get_categories()) == {"foo", "bar", "moo"}
    df2 = pl.DataFrame({"x": ["zoinks"]}, schema={"x": dt})
    assert set(df["x"].cat.get_categories()) == {"foo", "bar", "moo", "zoinks"}
    assert set(df2["x"].cat.get_categories()) == {"foo", "bar", "moo", "zoinks"}
    del df
    del df2
    df = pl.DataFrame({"x": ["a"]}, schema={"x": dt})
    assert df["x"].cat.get_categories().to_list() == ["a"]


@pytest.mark.parametrize("cats", CATS)
def test_categories_lookup(cats: pl.Categories) -> None:
    vals = ["foo", "bar", None, "moo", "bar", "moo", "foo", None]
    df = pl.DataFrame({"x": vals}, schema={"x": pl.Categorical(cats)})
    cat_vals = pl.Series("x", [cats[v] for v in vals], dtype=cats.physical())
    assert_series_equal(cat_vals, df["x"].cast(cats.physical()))
    cat_strs = pl.Series("x", [cats[v] for v in cat_vals])
    assert_series_equal(cat_strs, df["x"].cast(pl.String))


def test_concat_cat_mismatch() -> None:
    dt1 = pl.Categorical(pl.Categories.random())
    dt2 = pl.Categorical(pl.Categories.random())
    df1 = pl.DataFrame({"x": ["a", "b", "c"]}, schema={"x": dt1})
    df2 = pl.DataFrame({"x": ["d", "e", "f"]}, schema={"x": dt1})
    df12 = pl.DataFrame({"x": ["a", "b", "c", "d", "e", "f"]}, schema={"x": dt1})
    df3 = pl.DataFrame({"x": ["g", "h", "i"]}, schema={"x": dt2})
    df4 = pl.DataFrame({"x": ["j", "k", "l"]}, schema={"x": dt2})
    df34 = pl.DataFrame({"x": ["g", "h", "i", "j", "k", "l"]}, schema={"x": dt2})

    assert_frame_equal(pl.concat([df1, df2]), df12)
    assert_frame_equal(pl.concat([df3, df4]), df34)

    for left in [df1, df2]:
        for right in [df3, df4]:
            with pytest.raises(SchemaError):
                pl.concat([left, right])

    for li in range(len(CATS)):
        for ri in range(len(CATS)):
            if li == ri:
                continue

            ldf = pl.DataFrame({"x": []}, schema={"x": pl.Categorical(CATS[li])})
            rdf = pl.DataFrame({"x": []}, schema={"x": pl.Categorical(CATS[ri])})
            with pytest.raises(SchemaError):
                pl.concat([ldf, rdf])


def test_cat_overflow() -> None:
    c = pl.Categories.random(physical=pl.UInt8)
    str_s = pl.Series(range(255)).cast(pl.String)
    cat_s = str_s.cast(pl.Categorical(c))
    assert_series_equal(str_s, cat_s.cast(pl.String))
    with pytest.raises(ComputeError):
        pl.Series(["boom"], dtype=pl.Categorical(c))


def test_global_categories_gc() -> None:
    # We run this in a subprocess to ensure no other test is or has been messing
    # with the global categories.
    out = subprocess.check_output(
        [
            sys.executable,
            "-c",
            """\
import polars as pl

df = pl.DataFrame({"x": ["a", "b", "c"]}, schema={"x": pl.Categorical})
assert set(df["x"].cat.get_categories().to_list()) == {"a", "b", "c"}
df2 = pl.DataFrame({"x": ["d", "e", "f"]}, schema={"x": pl.Categorical})
assert set(df["x"].cat.get_categories().to_list()) == {"a", "b", "c", "d", "e", "f"}
del df
del df2
df3 = pl.DataFrame({"x": ["x"]}, schema={"x": pl.Categorical})
assert set(df3["x"].cat.get_categories().to_list()) == {"x"}
del df3

keep_alive = pl.DataFrame({"x": []}, schema={"x": pl.Categorical})
df = pl.DataFrame({"x": ["a", "b", "c"]}, schema={"x": pl.Categorical})
assert set(df["x"].cat.get_categories().to_list()) == {"a", "b", "c"}
df2 = pl.DataFrame({"x": ["d", "e", "f"]}, schema={"x": pl.Categorical})
assert set(df["x"].cat.get_categories().to_list()) == {"a", "b", "c", "d", "e", "f"}
del df
del df2
df3 = pl.DataFrame({"x": ["x"]}, schema={"x": pl.Categorical})
assert set(df3["x"].cat.get_categories().to_list()) == {"a", "b", "c", "d", "e", "f", "x"}

print("OK", end="")
""",
        ],
    )

    assert out == b"OK"
