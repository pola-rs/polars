from pypolars import DataFrame, Series
from pypolars.lazy import *
from pypolars.datatypes import *
import pytest


def test_lazy():
    df = DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]})
    ldf = df.lazy().with_column(lit(1).alias("foo")).select([col("a"), col("foo")])

    print(ldf.collect())
    # test if it executes
    new = (
        df.lazy()
        .with_column(
            when(col("a").gt(lit(2))).then(lit(10)).otherwise(lit(1)).alias("new")
        )
        .collect()
    )


def test_apply():
    df = DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]})
    new = df.lazy().with_column(col("a").apply(lambda s: s * 2).alias("foo")).collect()


def test_agg():
    df = DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]})
    ldf = df.lazy().min()
    assert ldf.collect().shape == (1, 2)
