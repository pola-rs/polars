import polars as pl
from polars import col
import numpy as np
import time

print(pl.__version__)

x = pl.read_csv(
    "G1_1e7_1e2_5_0.csv",
    dtype={
        "id4": pl.Int32,
        "id5": pl.Int32,
        "id6": pl.Int32,
        "v1": pl.Int32,
        "v2": pl.Int32,
        "v3": pl.Float64,
    },
)
x["id1"] = x["id1"].cast(pl.Categorical)
x["id2"] = x["id2"].cast(pl.Categorical)
x["id3"] = x["id3"].cast(pl.Categorical)
df = x.clone()
x = df.lazy()

t00 = time.time()
t0 = time.time()
print("q1")
out = x.groupby("id1").agg(pl.sum("v1")).collect()
print(time.time() - t0)
assert out.shape == (96, 2)
assert out["v1_sum"].sum() == 28498857

t0easy = time.time()
t0 = time.time()
print("q2")
out = x.groupby(["id1", "id2"]).agg(pl.sum("v1")).collect()
print(time.time() - t0)
assert out.shape == (9216, 3)
assert out["v1_sum"].sum() == 28498857

t0 = time.time()
print("q3")
out = x.groupby("id3").agg([pl.sum("v1"), pl.mean("v3")]).collect()
print(time.time() - t0)
assert out.shape == (95001, 3)
assert out["v1_sum"].sum() == 28498857
assert np.isclose(out["v3_mean"].sum(), 4749467.631946711)

t0 = time.time()
print("q4")
out = x.groupby("id4").agg([pl.mean("v1"), pl.mean("v2"), pl.mean("v3")]).collect()
print(time.time() - t0)
assert out.shape == (96, 4)
assert np.isclose(out["v1_mean"].sum(), 287.9894309270617)
assert np.isclose(out["v2_mean"].sum(), 767.8529216923457)
assert np.isclose(out["v3_mean"].sum(), 4799.873270453374)

t0 = time.time()
print("q5")
out = x.groupby("id6").agg([pl.sum("v1"), pl.sum("v2"), pl.sum("v3")]).collect()
print(time.time() - t0)
assert out.shape == (95001, 4)
assert out["v1_sum"].sum() == 28498857
assert out["v2_sum"].sum() == 75988394
assert np.isclose(out["v3_sum"].sum(), 474969574.0477772)
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
assert np.isclose(out["v3_median"].sum(), 460771.21644399985)
assert np.isclose(out["v3_std"].sum(), 266006.9046221105)

t0 = time.time()
print("q7")
out = (
    x.groupby("id3").agg([(pl.max("v1") - pl.min("v2")).alias("range_v1_v2")]).collect()
)
print(time.time() - t0)
assert out.shape == (95001, 2)
assert out["range_v1_v2"].sum() == 379850

t0 = time.time()
print("q8")
out = (
    x.drop_nulls("v3")
    .sort("v3", reverse=True)
    .groupby("id6")
    .agg(col("v3").head(2).alias("largest2_v3"))
    .explode("largest2_v3")
    .collect()
)
print(time.time() - t0)
assert out.shape == (190002, 2)
assert np.isclose(out["largest2_v3"].sum(), 18700554.77963197)

t0 = time.time()
print("q9")
out = (
    x.groupby(["id2", "id4"])
    .agg((pl.pearson_corr("v1", "v2") ** 2).alias("r2"))
    .collect()
)
print(time.time() - t0)
assert out.shape == (9216, 3)
assert np.isclose(out["r2"].sum(), 9.941512305644292)

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
print("total took:", time.time() - t00, "s")
assert out.shape == (9999993, 8)
