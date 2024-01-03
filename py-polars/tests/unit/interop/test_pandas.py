import pandas as pd

import polars as pl


def test_df_to_pandas_empty() -> None:
    df = pl.DataFrame()
    result = df.to_pandas()
    expected = pd.DataFrame()
    pd.testing.assert_frame_equal(result, expected)
