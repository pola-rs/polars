from __future__ import annotations

import polars as pl
from typing import Callable


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
    assert ldf.collect().frame_equal(ldf2.collect())


def test_ser_de_custom() -> None:
    funcs = {
        "plus_one": lambda a: a + 1,
        "mul": lambda a, b: a * b,
    }
    rev_funcs = dict((v,k) for k,v in funcs.items())

    class CustomSer(pl.UdfSerializer):
        def serialize_udf(self, udf: Callable) -> str:
            return rev_funcs[udf]

        def deserialize_udf(self, data: str) -> Callable:
            return funcs[data]

    custom_ser = CustomSer()

    df = pl.DataFrame({ "a": range(0, 100), "b": range(1000, 1200, 2), "c": range(1000, 2000, 10) })
    ldf = df.lazy().select(
        pl.reduce(funcs["mul"], [pl.col("a").map(funcs["plus_one"]), pl.col("b"), pl.col("c")]).alias("renamed")
    )

    json = ldf.write_json(None, udf_serializer=custom_ser)

    print(json)

    ldf2 = pl.LazyFrame.from_json(json, udf_serializer=custom_ser)

    print(ldf2)

    assert str(ldf) == str(ldf2)
    assert ldf.collect().frame_equal(ldf2.collect())
