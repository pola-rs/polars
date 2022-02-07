from os import path

import numpy as np

import polars as pl


def test_invalid_utf8() -> None:
    np.random.seed(1)
    bts = bytes(np.random.randint(0, 255, 200))
    file = path.join(path.dirname(__file__), "nonutf8.csv")

    with open(file, "wb") as f:
        f.write(bts)

    a = pl.read_csv(file, has_headers=False, encoding="utf8-lossy")
    b = pl.scan_csv(file, has_headers=False, encoding="utf8-lossy").collect()
    assert a.frame_equal(b, null_equal=True)
