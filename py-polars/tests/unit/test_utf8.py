import pytest

import polars as pl


def test_min_max_agg_on_str() -> None:
    strings = ["b", "a", "x"]
    s = pl.Series(strings)
    assert (s.min(), s.max()) == ("a", "x")


def test_json_path_match_type_4905() -> None:
    df = pl.DataFrame({"json_val": ['{"a":"hello"}', None, '{"a":"world"}']})
    assert df.filter(
        pl.col("json_val").str.json_path_match("$.a").is_in(["hello"])
    ).to_dict(False) == {"json_val": ['{"a":"hello"}']}


def test_length_vs_nchars() -> None:
    df = pl.DataFrame({"s": ["café", "東京"]}).with_columns(
        [
            pl.col("s").str.lengths().alias("length"),
            pl.col("s").str.n_chars().alias("nchars"),
        ]
    )
    assert df.rows() == [("café", 5, 4), ("東京", 6, 2)]


def test_decode_strict() -> None:
    df = pl.DataFrame(
        {"strings": ["0IbQvTc3", "0J%2FQldCf0JA%3D", "0J%2FRgNC%2B0YHRgtC%2B"]}
    )
    assert df.select(pl.col("strings").str.decode("base64", strict=False)).to_dict(
        False
    ) == {"strings": [b"\xd0\x86\xd0\xbd77", None, None]}
    with pytest.raises(pl.ComputeError):
        df.select(pl.col("strings").str.decode("base64", strict=True))
