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
