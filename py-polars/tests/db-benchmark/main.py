# type: ignore

import sys
import time

import numpy as np

import polars as pl

print(pl.__version__)

x = pl.read_csv(
    "G1_1e7_1e2_5_0.csv",
    dtypes={
        "id4": pl.Int32,
        "id5": pl.Int32,
        "id6": pl.Int32,
        "v1": pl.Int32,
        "v2": pl.Int32,
        "v3": pl.Float64,
    },
)
ON_STRINGS = sys.argv.pop() == "on_strings"

if not ON_STRINGS:
    x = x.with_columns([pl.col(["id1", "id2", "id3"]).cast(pl.Categorical)])
df = x.clone()
x = df.lazy()

t00 = time.time()
t0 = time.time()
print("q1")
out = x.groupby("id1").agg(pl.sum("v1").alias("v1_sum")).collect()
print(time.time() - t0)
print("out.shape", out.shape)
print('out["v1_sum"].sum()', out["v1_sum"].sum())

t0easy = time.time()
t0 = time.time()
print("q2")
out = x.groupby(["id1", "id2"]).agg(pl.sum("v1").alias("v1_sum")).collect()
print(time.time() - t0)
print("out.shape", out.shape)
print('out["v1_sum"].sum()', out["v1_sum"].sum())

t0 = time.time()
print("q3")
out = (
    x.groupby("id3")
    .agg([pl.sum("v1").alias("v1_sum"), pl.mean("v3").alias("v3_mean")])
    .collect()
)
print(time.time() - t0)
print("out.shape", out.shape)
print('out["v1_sum"].sum()', out["v1_sum"].sum())
print('out["v3_mean"].sum()', out["v3_mean"].sum())

t0 = time.time()
print("q4")
out = (
    x.groupby("id4")
    .agg(
        [
            pl.mean("v1").alias("v1_mean"),
            pl.mean("v2").alias("v2_mean"),
            pl.mean("v3").alias("v3_mean"),
        ]
    )
    .collect()
)
print(time.time() - t0)
print("out.shape", out.shape)
print('out["v1_mean"].sum()', out["v1_mean"].sum())
print('out["v2_mean"].sum()', out["v2_mean"].sum())
print('out["v3_mean"].sum()', out["v3_mean"].sum())

t0 = time.time()
print("q5")
out = (
    x.groupby("id6")
    .agg(
        [
            pl.sum("v1").alias("v1_sum"),
            pl.sum("v2").alias("v2_sum"),
            pl.sum("v3").alias("v3_sum"),
        ]
    )
    .collect()
)
print(time.time() - t0)
print("out.shape", out.shape)
print('out["v1_sum"].sum()', out["v1_sum"].sum())
print('out["v2_sum"].sum()', out["v2_sum"].sum())
easy_time = time.time() - t0easy
t0advanced = time.time()

t0 = time.time()
print("q6")
out = (
    x.groupby(["id4", "id5"])
    .agg([pl.median("v3").alias("v3_median"), pl.std("v3").alias("v3_std")])
    .collect()
)
print(time.time() - t0)
print("out.shape", out.shape)
print('out["v3_median"].sum()', out["v3_median"].sum())
print('out["v3_std"].sum()', out["v3_std"].sum())

t0 = time.time()
print("q7")
out = (
    x.groupby("id3").agg([(pl.max("v1") - pl.min("v2")).alias("range_v1_v2")]).collect()
)
print(time.time() - t0)
print("out.shape", out.shape)
print('out["range_v1_v2"].sum()', out["range_v1_v2"].sum())

t0 = time.time()
print("q8")
out = (
    x.drop_nulls("v3")
    .groupby("id6")
    .agg(pl.col("v3").top_k(2).alias("largest2_v3"))
    .explode("largest2_v3")
    .collect()
)
print(time.time() - t0)
print("out.shape", out.shape)
print('out["largest2_v3"].sum()', out["largest2_v3"].sum())

t0 = time.time()
print("q9")
out = (
    x.groupby(["id2", "id4"])
    .agg((pl.pearson_corr("v1", "v2") ** 2).alias("r2"))
    .collect()
)
print(time.time() - t0)
print("out.shape", out.shape)
print('out["r2"].sum()', out["r2"].sum())

t0 = time.time()
print("q10")
out = (
    x.groupby(["id1", "id2", "id3", "id4", "id5", "id6"])
    .agg([pl.sum("v3").alias("v3"), pl.count("v1").alias("count")])
    .collect()
)
print(time.time() - t0)
print("out.shape", out.shape)
print("easy took:", easy_time, "s")
print("advanced took:", time.time() - t0advanced, "s")
print("total took:", time.time() - t00, "s")

t00 = time.time()
t0 = time.time()
print("q1")
out = x.groupby("id1").agg(pl.sum("v1").alias("v1_sum")).collect()
print(time.time() - t0)
assert out.shape == (96, 2)
assert out["v1_sum"].sum() == 28501451

t0easy = time.time()
t0 = time.time()
print("q2")
out = x.groupby(["id1", "id2"]).agg(pl.sum("v1").alias("v1_sum")).collect()
print(time.time() - t0)
assert out.shape == (9216, 3)
assert out["v1_sum"].sum() == 28501451

t0 = time.time()
print("q3")
out = (
    x.groupby("id3")
    .agg([pl.sum("v1").alias("v1_sum"), pl.mean("v3").alias("v3_mean")])
    .collect()
)
print(time.time() - t0)
assert out.shape == (95001, 3)
assert out["v1_sum"].sum() == 28501451
assert np.isclose(out["v3_mean"].sum(), 4751358.825104358)

t0 = time.time()
print("q4")
out = (
    x.groupby("id4")
    .agg(
        [
            pl.mean("v1").alias("v1_mean"),
            pl.mean("v2").alias("v2_mean"),
            pl.mean("v3").alias("v3_mean"),
        ]
    )
    .collect()
)
print(time.time() - t0)
assert out.shape == (96, 4)
assert np.isclose(out["v1_mean"].sum(), 288.0192364601018)
assert np.isclose(out["v2_mean"].sum(), 767.9422306545811)
assert np.isclose(out["v3_mean"].sum(), 4801.784316931509)

t0 = time.time()
print("q5")
out = (
    x.groupby("id6")
    .agg(
        [
            pl.sum("v1").alias("v1_sum"),
            pl.sum("v2").alias("v2_sum"),
            pl.sum("v3").alias("v3_sum"),
        ]
    )
    .collect()
)
print(time.time() - t0)
assert out.shape == (95001, 4)
assert out["v1_sum"].sum() == 28501451
assert out["v2_sum"].sum() == 75998165
easy_time = time.time() - t0easy
t0advanced = time.time()

t0 = time.time()
print("q6")
out = (
    x.groupby(["id4", "id5"])
    .agg([pl.median("v3").alias("v3_median"), pl.std("v3").alias("v3_std")])
    .collect()
)
print(time.time() - t0)
assert out.shape == (9216, 4)
assert np.isclose(out["v3_median"].sum(), 460892.5487690001)
assert np.isclose(out["v3_std"].sum(), 266052.20492321637)

t0 = time.time()
print("q7")
out = (
    x.groupby("id3")
    .agg(
        [
            (pl.max("v1").alias("v1_max") - pl.min("v2").alias("v2_mean")).alias(
                "range_v1_v2"
            )
        ]
    )
    .collect()
)
print(time.time() - t0)
assert out.shape == (95001, 2)
assert out["range_v1_v2"].sum() == 379846

t0 = time.time()
print("q8")
out = (
    x.drop_nulls("v3")
    .sort("v3", reverse=True)
    .groupby("id6")
    .agg(pl.col("v3").head(2).alias("largest2_v3"))
    .explode("largest2_v3")
    .collect()
)
print(time.time() - t0)
assert out.shape == (190002, 2)
assert np.isclose(out["largest2_v3"].sum(), 18700642.66837202)

t0 = time.time()
print("q9")
out = (
    x.groupby(["id2", "id4"])
    .agg((pl.pearson_corr("v1", "v2") ** 2).alias("r2"))
    .collect()
)
print(time.time() - t0)
assert out.shape == (9216, 3)
assert np.isclose(out["r2"].sum(), 9.902706276948825)

t0 = time.time()
print("q10")
out = (
    x.groupby(["id1", "id2", "id3", "id4", "id5", "id6"])
    .agg([pl.sum("v3").alias("v3"), pl.count("v1").alias("count")])
    .collect()
)
print(time.time() - t0)
print("easy took:", easy_time, "s")
print("advanced took:", time.time() - t0advanced, "s")
total_time = time.time() - t00
print("total took:", total_time, "s")
assert out.shape == (9999995, 8)

# Additional tests
# the code below, does not belong to the db-benchmark
# but it triggers other code paths so the checksums assertion
# are a sort of integration tests
out = (
    x.filter(pl.col("id1") == pl.lit("id046"))
    .select([pl.sum("id6"), pl.sum("v3")])
    .collect()
)
assert out["id6"].to_list() == [430957682]
assert np.isclose(out["v3"].to_list(), 4.724150165888001e6).all()
print(out)

out = (
    x.filter(~(pl.col("id1") == pl.lit("id046")))
    .select([pl.sum("id6"), pl.sum("v3")])
    .collect()
)
print(out)

assert out["id6"].to_list() == [2137755425]
assert np.isclose(out["v3"].to_list(), 4.7040828499563754e8).all()

if not ON_STRINGS:
    if total_time > 12:
        print("query took longer than 12s, may be noise")
        exit(1)
