from __future__ import annotations

import polars as pl


# pickling does not work with lambda or nested functions
def my_map_fn(el: int) -> int:
    return el + 1


def test_ser_de() -> None:
    df = pl.DataFrame([pl.Series("foo", range(1, 100), dtype=pl.Int8)])
    ldf = df.lazy().select(pl.col("foo").map(my_map_fn))

    json = ldf.write_json(None, pickle_udf=True)

    print(json)

    ldf2 = pl.LazyFrame.from_json(json, pickle_udf=True)

    print(ldf2)

    assert str(ldf) == str(ldf2)
