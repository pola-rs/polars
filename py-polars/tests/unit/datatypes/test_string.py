import json

import polars as pl
from polars.testing import assert_series_equal


def test_series_init_string() -> None:
    s = pl.Series(["a", "b"])
    assert s.dtype == pl.String


def test_utf8_alias_eq() -> None:
    assert pl.Utf8 == pl.String
    assert pl.Utf8 == pl.String()
    assert pl.Utf8() == pl.String
    assert pl.Utf8() == pl.String()


def test_utf8_alias_hash() -> None:
    assert hash(pl.Utf8) == hash(pl.String)
    assert hash(pl.Utf8()) == hash(pl.String())


def test_utf8_alias_series_init() -> None:
    s = pl.Series(["a", "b"], dtype=pl.Utf8)
    assert s.dtype == pl.String


def test_utf8_alias_lit() -> None:
    result = pl.select(a=pl.lit(5, dtype=pl.Utf8)).to_series()
    expected = pl.Series("a", ["5"], dtype=pl.String)
    assert_series_equal(result, expected)


def test_json_decode_multiple_chunks() -> None:
    a = json.dumps({"x": None})
    b = json.dumps({"x": True})

    df_1 = pl.Series([a]).to_frame("s")
    df_2 = pl.Series([b]).to_frame("s")

    df = pl.concat([df_1, df_2])

    assert df.with_columns(pl.col("s").str.json_decode()).to_dict(as_series=False) == {
        "s": [{"x": None}, {"x": True}]
    }
