import pickle

import polars as pl


def test_pickling_simple_expression() -> None:
    e = pl.col("foo").sum()
    buf = pickle.dumps(e)
    assert str(pickle.loads(buf)) == str(e)


def serde_lazy_frame_lp() -> None:
    lf = pl.DataFrame({"a": [1, 2, 3], "b": ["a", "b", "c"]}).lazy().select(pl.col("a"))
    json = lf.write_json(to_string=True)

    assert (
        pl.LazyFrame.from_json(json)
        .collect()
        .to_series()
        .series_equal(pl.Series("a", [1, 2, 3]))
    )
