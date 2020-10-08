from pypolars import DataFrame, Series
from pypolars.lazy import *
from pypolars.datatypes import *
import pytest


def test_lazy():
    df = DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]})
    print(df)
    ldf = df.lazy().select([col("a")])

    print(ldf.collect())
