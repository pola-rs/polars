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
