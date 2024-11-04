from inspect import signature
from unittest.mock import patch

import polars as pl


def test_show_signature_match() -> None:
    assert signature(pl.LazyFrame.show) == signature(pl.DataFrame.show)


def test_lf_show_calls_df_show() -> None:
    lf = pl.LazyFrame(
        {
            "foo": ["a", "b", "c", "d", "e", "f", "g"],
            "bar": [1, 2, 3, 4, 5, 6, 7],
        }
    )
    with patch.object(pl.DataFrame, "show") as df_show:
        lf.show(5)

    df_show.assert_called_once_with(
        5,
        float_precision=None,
        fmt_str_lengths=None,
        fmt_table_cell_list_len=None,
        tbl_cols=None,
    )
