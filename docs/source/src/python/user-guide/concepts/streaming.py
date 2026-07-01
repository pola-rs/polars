import base64

# --8<-- [start:import]
import polars as pl
# --8<-- [end:import]

# --8<-- [start:streaming]
q1 = (
    pl.scan_csv("docs/assets/data/iris.csv")
    .filter(pl.col("sepal_length") > 5)
    .group_by("species")
    .agg(pl.col("sepal_width").mean())
)
df = q1.collect(engine="streaming")
# --8<-- [end:streaming]

"""
# --8<-- [start:createplan_query]
q1 = (
    pl.scan_csv("docs/assets/data/iris.csv")
    .filter(pl.col("sepal_length") > 5)
    .group_by("species")
    .agg(
        mean_width=pl.col("sepal_width").mean(),
        mean_width2=pl.col("sepal_width").sum() / pl.col("sepal_length").count(),
    )
    .show_graph(plan_stage="physical", engine="streaming")
)
# --8<-- [end:createplan_query]
"""

# --8<-- [start:createplan]
import base64
import polars as pl

q1 = (
    pl.scan_csv("docs/assets/data/iris.csv")
    .filter(pl.col("sepal_length") > 5)
    .group_by("species")
    .agg(
        mean_width=pl.col("sepal_width").mean(),
        mean_width2=pl.col("sepal_width").sum() / pl.col("sepal_length").count(),
    )
)

q1.show_graph(
    plan_stage="physical",
    engine="streaming",
    show=False,
    output_path="docs/assets/images/query_plan.png",
)
with open("docs/assets/images/query_plan.png", "rb") as f:
    png = base64.b64encode(f.read()).decode()
    print(f'<img src="data:image/png;base64, {png}"/>')
# --8<-- [end:createplan]


# --8<-- [start:larger_than_ram]
import polars as pl

# collect(engine="streaming") keeps memory bounded -- data flows through in chunks
q = (
    pl.scan_csv("docs/assets/data/iris.csv")
    .filter(pl.col("sepal_length") > 5)
    .group_by("species")
    .agg(
        pl.col("sepal_width").mean().round(3).alias("avg_sepal_width"),
        pl.len().alias("n_rows"),
    )
    .sort("avg_sepal_width", descending=True)
)
df = q.collect(engine="streaming")
print(df)
# --8<-- [end:larger_than_ram]


# --8<-- [start:sink_parquet]
import pathlib
import tempfile

import polars as pl

with tempfile.TemporaryDirectory() as tmp:
    out = pathlib.Path(tmp) / "enriched.parquet"
    # Writes to disk in streaming batches -- full enriched result never loads into RAM
    (
        pl.scan_csv("docs/assets/data/iris.csv")
        .with_columns(
            (pl.col("sepal_length") * pl.col("sepal_width")).alias("sepal_area")
        )
        .filter(pl.col("sepal_area") > 15.0)
        .sink_parquet(out)
    )
    print(pl.read_parquet(out).head(5))
# --8<-- [end:sink_parquet]


# --8<-- [start:sink_csv]
import pathlib
import tempfile

import polars as pl

with tempfile.TemporaryDirectory() as tmp:
    out = pathlib.Path(tmp) / "summary.csv"
    (
        pl.scan_csv("docs/assets/data/iris.csv")
        .group_by("species")
        .agg(
            pl.col("sepal_length").mean().round(2).alias("avg_sepal_length"),
            pl.col("petal_length").mean().round(2).alias("avg_petal_length"),
            pl.len().alias("n"),
        )
        .sort("species")
        .sink_csv(out)
    )
    print(pl.read_csv(out))
# --8<-- [end:sink_csv]


# --8<-- [start:sink_batches]
import polars as pl

batches_seen: list[int] = []


def on_batch(batch: pl.DataFrame) -> None:
    batches_seen.append(len(batch))


pl.scan_csv("docs/assets/data/iris.csv").sink_batches(on_batch, chunk_size=50)
print(f"Processed {sum(batches_seen)} rows in {len(batches_seen)} batch(es)")
print(f"Rows per batch: {batches_seen}")
# --8<-- [end:sink_batches]


# --8<-- [start:collect_async]
import asyncio

import polars as pl


async def run_concurrent() -> None:
    lf = pl.scan_csv("docs/assets/data/iris.csv")
    # Both queries execute concurrently in Polars' thread-pool via asyncio.gather
    short, long_ = await asyncio.gather(
        lf.filter(pl.col("sepal_length") < 5.5)
        .group_by("species")
        .agg(pl.col("sepal_length").mean().round(3).alias("mean_short"))
        .collect_async(),
        lf.filter(pl.col("sepal_length") >= 5.5)
        .group_by("species")
        .agg(pl.col("sepal_length").mean().round(3).alias("mean_long"))
        .collect_async(),
    )
    print("Short sepals (< 5.5 cm):")
    print(short.sort("species"))
    print("Long sepals (>= 5.5 cm):")
    print(long_.sort("species"))


asyncio.run(run_concurrent())
# --8<-- [end:collect_async]


# --8<-- [start:partition_pruning]
import pathlib
import tempfile

import polars as pl

with tempfile.TemporaryDirectory() as tmp:
    tmp_path = pathlib.Path(tmp)
    # Write two synthetic "partitions"
    pl.DataFrame({"year": [2023] * 4, "value": [1.0, 2.0, 3.0, 4.0]}).write_parquet(
        tmp_path / "part_2023.parquet"
    )
    pl.DataFrame({"year": [2024] * 4, "value": [5.0, 6.0, 7.0, 8.0]}).write_parquet(
        tmp_path / "part_2024.parquet"
    )
    # Polars evaluates the predicate on file statistics before reading --
    # only the 2024 file is opened for this query.
    result = (
        pl.scan_parquet(tmp_path / "*.parquet")
        .filter(pl.col("year") == 2024)
        .collect(engine="streaming")
    )
    print(result)
# --8<-- [end:partition_pruning]
