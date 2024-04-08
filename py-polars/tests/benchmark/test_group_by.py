"""
Benchmark tests for the group by operation.

These tests are based on the H2O AI database benchmark.

See:
https://h2oai.github.io/db-benchmark/
"""

from __future__ import annotations

import pytest

import polars as pl

pytestmark = pytest.mark.benchmark()


def test_h2oai_groupby_q1(h2oai_groupby_data: pl.DataFrame) -> None:
    result = (
        h2oai_groupby_data.lazy()
        .group_by("id1")
        .agg(
            pl.sum("v1").alias("v1_sum"),
        )
        .collect()
    )
    assert result.shape == (96, 2)
    assert result["v1_sum"].sum() == 28498857


def test_h2oai_groupby_q2(h2oai_groupby_data: pl.DataFrame) -> None:
    result = (
        h2oai_groupby_data.lazy()
        .group_by("id1", "id2")
        .agg(
            pl.sum("v1").alias("v1_sum"),
        )
        .collect()
    )
    assert result.shape == (9216, 3)
    assert result["v1_sum"].sum() == 28498857


def test_h2oai_groupby_q3(h2oai_groupby_data: pl.DataFrame) -> None:
    result = (
        h2oai_groupby_data.lazy()
        .group_by("id3")
        .agg(
            pl.sum("v1").alias("v1_sum"),
            pl.mean("v3").alias("v3_mean"),
        )
        .collect()
    )
    assert result.shape == (95001, 3)
    assert result["v1_sum"].sum() == 28498857
    assert result["v3_mean"].sum() == pytest.approx(4749467.631946)


def test_h2oai_groupby_q4(h2oai_groupby_data: pl.DataFrame) -> None:
    result = (
        h2oai_groupby_data.lazy()
        .group_by("id4")
        .agg(
            pl.mean("v1").alias("v1_mean"),
            pl.mean("v2").alias("v2_mean"),
            pl.mean("v3").alias("v3_mean"),
        )
        .collect()
    )
    assert result.shape == (96, 4)
    assert result["v1_mean"].sum() == pytest.approx(287.989430)
    assert result["v2_mean"].sum() == pytest.approx(767.852921)
    assert result["v3_mean"].sum() == pytest.approx(4799.873270)


def test_h2oai_groupby_q5(h2oai_groupby_data: pl.DataFrame) -> None:
    result = (
        h2oai_groupby_data.lazy()
        .group_by("id6")
        .agg(
            pl.sum("v1").alias("v1_sum"),
            pl.sum("v2").alias("v2_sum"),
            pl.sum("v3").alias("v3_sum"),
        )
        .collect()
    )
    assert result.shape == (95001, 4)
    assert result["v1_sum"].sum() == 28498857
    assert result["v2_sum"].sum() == 75988394
    assert result["v3_sum"].sum() == pytest.approx(474969574.047777)


def test_h2oai_groupby_q6(h2oai_groupby_data: pl.DataFrame) -> None:
    result = (
        h2oai_groupby_data.lazy()
        .group_by("id4", "id5")
        .agg(
            pl.median("v3").alias("v3_median"),
            pl.std("v3").alias("v3_std"),
        )
        .collect()
    )
    assert result.shape == (9216, 4)
    assert result["v3_median"].sum() == pytest.approx(460771.216444)
    assert result["v3_std"].sum() == pytest.approx(266006.904622)


def test_h2oai_groupby_q7(h2oai_groupby_data: pl.DataFrame) -> None:
    result = (
        h2oai_groupby_data.lazy()
        .group_by("id3")
        .agg((pl.max("v1") - pl.min("v2")).alias("range_v1_v2"))
        .collect()
    )
    assert result.shape == (95001, 2)
    assert result["range_v1_v2"].sum() == 379850


def test_h2oai_groupby_q8(h2oai_groupby_data: pl.DataFrame) -> None:
    result = (
        h2oai_groupby_data.lazy()
        .drop_nulls("v3")
        .group_by("id6")
        .agg(pl.col("v3").top_k(2).alias("largest2_v3"))
        .explode("largest2_v3")
        .collect()
    )
    assert result.shape == (190002, 2)
    assert result["largest2_v3"].sum() == pytest.approx(18700554.779632)


def test_h2oai_groupby_q9(h2oai_groupby_data: pl.DataFrame) -> None:
    result = (
        h2oai_groupby_data.lazy()
        .group_by("id2", "id4")
        .agg((pl.corr("v1", "v2") ** 2).alias("r2"))
        .collect()
    )
    assert result.shape == (9216, 3)
    assert result["r2"].sum() == pytest.approx(9.940515)


def test_h2oai_groupby_q10(h2oai_groupby_data: pl.DataFrame) -> None:
    result = (
        h2oai_groupby_data.lazy()
        .group_by("id1", "id2", "id3", "id4", "id5", "id6")
        .agg(
            pl.sum("v3").alias("v3_sum"),
            pl.count("v1").alias("v1_count"),
        )
        .collect()
    )
    assert result.shape == (9999993, 8)
