import sys
from datetime import date

import pytest

import polars as pl
from tests.benchmark.data import load_tpch_table

if sys.platform == "win32":
    pytest.skip("TPC-H data cannot be generated under Windows", allow_module_level=True)

pytestmark = pytest.mark.benchmark()


@pytest.fixture(scope="module")
def customer() -> pl.LazyFrame:
    return load_tpch_table("customer").lazy()


@pytest.fixture(scope="module")
def lineitem() -> pl.LazyFrame:
    return load_tpch_table("lineitem").lazy()


@pytest.fixture(scope="module")
def nation() -> pl.LazyFrame:
    return load_tpch_table("nation").lazy()


@pytest.fixture(scope="module")
def orders() -> pl.LazyFrame:
    return load_tpch_table("orders").lazy()


@pytest.fixture(scope="module")
def part() -> pl.LazyFrame:
    return load_tpch_table("part").lazy()


@pytest.fixture(scope="module")
def partsupp() -> pl.LazyFrame:
    return load_tpch_table("partsupp").lazy()


@pytest.fixture(scope="module")
def region() -> pl.LazyFrame:
    return load_tpch_table("region").lazy()


@pytest.fixture(scope="module")
def supplier() -> pl.LazyFrame:
    return load_tpch_table("supplier").lazy()


def test_tpch_q1(lineitem: pl.LazyFrame) -> None:
    var1 = date(1998, 9, 2)

    q_final = (
        lineitem.filter(pl.col("l_shipdate") <= var1)
        .group_by("l_returnflag", "l_linestatus")
        .agg(
            pl.sum("l_quantity").alias("sum_qty"),
            pl.sum("l_extendedprice").alias("sum_base_price"),
            (pl.col("l_extendedprice") * (1.0 - pl.col("l_discount")))
            .sum()
            .alias("sum_disc_price"),
            (
                pl.col("l_extendedprice")
                * (1.0 - pl.col("l_discount"))
                * (1.0 + pl.col("l_tax"))
            )
            .sum()
            .alias("sum_charge"),
            pl.mean("l_quantity").alias("avg_qty"),
            pl.mean("l_extendedprice").alias("avg_price"),
            pl.mean("l_discount").alias("avg_disc"),
            pl.len().alias("count_order"),
        )
        .sort("l_returnflag", "l_linestatus")
    )
    q_final.collect()


def test_tpch_q2(
    nation: pl.LazyFrame,
    part: pl.LazyFrame,
    partsupp: pl.LazyFrame,
    region: pl.LazyFrame,
    supplier: pl.LazyFrame,
) -> None:
    var1 = 15
    var2 = "BRASS"
    var3 = "EUROPE"

    result_q1 = (
        part.join(partsupp, left_on="p_partkey", right_on="ps_partkey")
        .join(supplier, left_on="ps_suppkey", right_on="s_suppkey")
        .join(nation, left_on="s_nationkey", right_on="n_nationkey")
        .join(region, left_on="n_regionkey", right_on="r_regionkey")
        .filter(pl.col("p_size") == var1)
        .filter(pl.col("p_type").str.ends_with(var2))
        .filter(pl.col("r_name") == var3)
    )

    q_final = (
        result_q1.group_by("p_partkey")
        .agg(pl.min("ps_supplycost"))
        .join(result_q1, on=["p_partkey", "ps_supplycost"])
        .select(
            "s_acctbal",
            "s_name",
            "n_name",
            "p_partkey",
            "p_mfgr",
            "s_address",
            "s_phone",
            "s_comment",
        )
        .sort(
            by=["s_acctbal", "n_name", "s_name", "p_partkey"],
            descending=[True, False, False, False],
        )
        .head(100)
    )
    q_final.collect()
