from __future__ import annotations

import io
import pickle

import polars as pl


def test_pickle() -> None:
    a = pl.Series("a", [1, 2])
    b = pickle.dumps(a)
    out = pickle.loads(b)
    assert a.series_equal(out)
    df = pl.DataFrame({"a": [1, 2], "b": ["a", None], "c": [True, False]})
    b = pickle.dumps(df)
    out = pickle.loads(b)
    assert df.frame_equal(out, null_equal=True)


def test_pickle_expr() -> None:
    for e in [pl.all(), pl.count()]:
        f = io.BytesIO()
        pickle.dump(e, f)

        f.seek(0)
        pickle.load(f)
