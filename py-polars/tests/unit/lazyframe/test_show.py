from inspect import signature
from unittest.mock import patch

import pytest

import polars as pl
from polars.exceptions import PerformanceWarning


def test_show_signature_match() -> None:
    assert signature(pl.LazyFrame.show) == signature(pl.DataFrame.show)


def test_lf_show_calls_df_show() -> None:
    lf = pl.LazyFrame({})
    with patch.object(pl.DataFrame, "show") as df_show:
        lf.show(5)

    df_show.assert_called_once_with(
        5,
        float_precision=None,
        fmt_str_lengths=None,
        fmt_table_cell_list_len=None,
        tbl_cols=None,
    )


def test_lf_show_no_limit_issues_warning() -> None:
    lf = pl.LazyFrame({})
    with pytest.warns(PerformanceWarning):
        lf.show(limit=None)
