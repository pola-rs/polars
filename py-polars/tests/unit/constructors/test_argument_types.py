"""
Test the protocols used for Series and DataFrame argument types.

We can only test what's included in `requirements-dev.txt`, which currently does
not include pyarrow stubs.
"""

import numpy as np
import pandas as pd

from polars._typing import NumpyArray, PandasDataFrame, PandasIndex, PandasSeries


def test_protocols() -> None:
    def _func(
        df: PandasDataFrame,
        idx: PandasIndex,
        ser: PandasSeries,
        arr: NumpyArray,
    ) -> None:
        return None

    ser = pd.Series([1])
    idx = pd.Index([1])
    df = pd.DataFrame({"a": [1]})
    arr = np.array([1, 2, 3])

    # Check for no errors
    _func(df, idx, ser, arr)

    # Check for type errors
    _func(
        arr,  # type: ignore[arg-type]
        ser,  # type: ignore[arg-type]
        idx,  # type: ignore[arg-type]
        df,  # type: ignore[arg-type]
    )
