# mypy: disable-error-code="attr-defined"
from __future__ import annotations

import polars as pl
from polars.io.bigquery import _predicate_to_row_restriction


class TestBigQueryExpressions:
    """Test coverage for `bigquery` expressions comprehension."""

    def test_is_null_expression(self) -> None:
        expr = pl.col("id").is_null()
        assert _predicate_to_row_restriction(expr) == "(`id` IS NULL)"

    def test_is_not_null_expression(self) -> None:
        expr = pl.col("id").is_not_null()
        assert _predicate_to_row_restriction(expr) == "(`id` IS NOT NULL)"

    def test_parse_combined_expression(self) -> None:
        expr = (pl.col("str") == "2") & ((pl.col("id") > 10) | (pl.col("id") < 7.0))
        assert (
            _predicate_to_row_restriction(expr)
            == "((`str` = '2') AND ((`id` > 10) OR (`id` < 7.0)))"
        )

    def test_parse_gt(self) -> None:
        expr = pl.col("ts") > "2023-08-08"
        assert _predicate_to_row_restriction(expr) == "(`ts` > '2023-08-08')"

    def test_parse_gteq(self) -> None:
        expr = pl.col("ts") >= "2023-08-08"
        assert _predicate_to_row_restriction(expr) == "(`ts` >= '2023-08-08')"

    def test_parse_eq(self) -> None:
        expr = pl.col("ts") == "2023-08-08"
        assert _predicate_to_row_restriction(expr) == "(`ts` = '2023-08-08')"

    def test_parse_lt(self) -> None:
        expr = pl.col("ts") < "2023-08-08"
        assert _predicate_to_row_restriction(expr) == "(`ts` < '2023-08-08')"

    def test_parse_lteq(self) -> None:
        expr = pl.col("ts") <= "2023-08-08"
        assert _predicate_to_row_restriction(expr) == "(`ts` <= '2023-08-08')"
