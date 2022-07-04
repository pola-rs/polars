# -------------------------------------------------
# Validate Series behaviour with parameteric tests
# -------------------------------------------------

# from hypothesis import given
#
# import polars as pl
# from polars.testing import (
#     series,
#     verify_series_and_expr_api,
# )
#
#
# # TODO: exclude obvious/known overflow inside the strategy before commenting back in
# @given(s=series(allowed_dtypes=_NUMERIC_COL_TYPES, name="a"))
# def test_cum_agg_extra(s: pl.Series) -> None:
#     # confirm that ops on generated Series match equivalent Expr call
#     # note: testing codepath-equivalence, not correctness.
#     for op in ("cumsum", "cummin", "cummax", "cumprod"):
#          verify_series_and_expr_api(s, None, op)
