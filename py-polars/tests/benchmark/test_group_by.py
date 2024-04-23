"""
Benchmark tests for the group-by operation.

These tests are based on the H2O.ai database benchmark.

See:
https://h2oai.github.io/db-benchmark/
"""

from __future__ import annotations

import pytest

import polars as pl

pytestmark = pytest.mark.benchmark()


def test_groupby_h2oai_q1(groupby_data: pl.DataFrame) -> None:
    (
        groupby_data.lazy()
        .group_by("id1")
        .agg(
            pl.sum("v1").alias("v1_sum"),
        )
        .collect()
    )


def test_groupby_h2oai_q2(groupby_data: pl.DataFrame) -> None:
    (
        groupby_data.lazy()
        .group_by("id1", "id2")
        .agg(
            pl.sum("v1").alias("v1_sum"),
        )
        .collect()
    )


def test_groupby_h2oai_q3(groupby_data: pl.DataFrame) -> None:
    (
        groupby_data.lazy()
        .group_by("id3")
        .agg(
            pl.sum("v1").alias("v1_sum"),
            pl.mean("v3").alias("v3_mean"),
        )
        .collect()
    )


def test_groupby_h2oai_q4(groupby_data: pl.DataFrame) -> None:
    (
        groupby_data.lazy()
        .group_by("id4")
        .agg(
            pl.mean("v1").alias("v1_mean"),
            pl.mean("v2").alias("v2_mean"),
            pl.mean("v3").alias("v3_mean"),
        )
        .collect()
    )


def test_groupby_h2oai_q5(groupby_data: pl.DataFrame) -> None:
    (
        groupby_data.lazy()
        .group_by("id6")
        .agg(
            pl.sum("v1").alias("v1_sum"),
            pl.sum("v2").alias("v2_sum"),
            pl.sum("v3").alias("v3_sum"),
        )
        .collect()
    )


def test_groupby_h2oai_q6(groupby_data: pl.DataFrame) -> None:
    (
        groupby_data.lazy()
        .group_by("id4", "id5")
        .agg(
            pl.median("v3").alias("v3_median"),
            pl.std("v3").alias("v3_std"),
        )
        .collect()
    )


def test_groupby_h2oai_q7(groupby_data: pl.DataFrame) -> None:
    (
        groupby_data.lazy()
        .group_by("id3")
        .agg((pl.max("v1") - pl.min("v2")).alias("range_v1_v2"))
        .collect()
    )


def test_groupby_h2oai_q8(groupby_data: pl.DataFrame) -> None:
    (
        groupby_data.lazy()
        .drop_nulls("v3")
        .group_by("id6")
        .agg(pl.col("v3").top_k(2).alias("largest2_v3"))
        .explode("largest2_v3")
        .collect()
    )


def test_groupby_h2oai_q9(groupby_data: pl.DataFrame) -> None:
    (
        groupby_data.lazy()
        .group_by("id2", "id4")
        .agg((pl.corr("v1", "v2") ** 2).alias("r2"))
        .collect()
    )


def test_groupby_h2oai_q10(groupby_data: pl.DataFrame) -> None:
    (
        groupby_data.lazy()
        .group_by("id1", "id2", "id3", "id4", "id5", "id6")
        .agg(
            pl.sum("v3").alias("v3_sum"),
            pl.count("v1").alias("v1_count"),
        )
        .collect()
    )
