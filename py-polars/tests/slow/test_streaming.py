import time

import numpy as np

import polars as pl


def test_cross_join_stack() -> None:
    a = pl.Series(np.arange(100_000)).to_frame().lazy()
    t0 = time.time()
    # this should be instant if directly pushed into sink
    # if not the cross join will first fill the stack with all matches of a single chunk
    assert a.join(a, how="cross").head().collect(streaming=True).shape == (5, 2)
    t1 = time.time()
    assert (t1 - t0) < 0.5
