from __future__ import annotations

import io
import pickle

import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal


def test_pickle() -> None:
    a = pl.Series("a", [1, 2])
    b = pickle.dumps(a)
    out = pickle.loads(b)
    assert_series_equal(a, out)
    df = pl.DataFrame({"a": [1, 2], "b": ["a", None], "c": [True, False]})
    b = pickle.dumps(df)
    out = pickle.loads(b)
    assert_frame_equal(df, out)


def test_pickle_expr() -> None:
    for e in [pl.all(), pl.count()]:
        f = io.BytesIO()
        pickle.dump(e, f)

        f.seek(0)
        pickle.load(f)
