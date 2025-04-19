# From local to cloud query execution

Data processing and analytics often begins small but can quickly grow beyond the capabilities of
your local machine. A typical workflow starts with exploring a sample dataset locally, developing
the analytical approach, and then scaling up to process the full dataset in the cloud.

This pattern allows you to iterate quickly during development while still handling larger datasets
in production. With Polars Cloud, you can maintain this natural workflow without rewriting your code
when moving from local to cloud execution, without requiring any migrations between local and
production tooling.

## Local exploration

For this workflow, we define the following simple mocked dataset that will act as a sample to
demonstrate the workflow. Here we will create the LazyFrame ourselves, but it could also be read as
(remote) file.

```python
import polars as pl

lf = pl.LazyFrame(
    {
        "region": [
            "Australia",
            "California",
            "Benelux",
            "Siberia",
            "Mediterranean",
            "Congo",
            "Borneo",
        ],
        "temperature": [32.1, 28.5, 30.2, 22.7, 29.3, 31.8, 33.2],
        "humidity": [40, 35, 75, 30, 45, 80, 70],
        "burn_area": [120, 85, 30, 65, 95, 25, 40],
        "vegetation_density": [0.6, 0.7, 0.9, 0.4, 0.5, 0.9, 0.8],
    }
)
```

A simple transformation will done to create a new column.

```python
(
    lf.with_columns(
            (
                (pl.col("temperature") / 10)
                * (1 - pl.col("humidity") / 100)
                * pl.col("vegetation_density")
            ).alias("fire_risk"),
    ).filter(pl.col("humidity") < 70)
    .sort(by="fire_risk", descending=True)
 .collect()
)
```

```text
shape: (4, 6)
┌───────────────┬─────────────┬──────────┬───────────┬────────────────────┬───────────┐
│ region        ┆ temperature ┆ humidity ┆ burn_area ┆ vegetation_density ┆ fire_risk │
│ ---           ┆ ---         ┆ ---      ┆ ---       ┆ ---                ┆ ---       │
│ str           ┆ f64         ┆ i64      ┆ i64       ┆ f64                ┆ f64       │
╞═══════════════╪═════════════╪══════════╪═══════════╪════════════════════╪═══════════╡
│ California    ┆ 28.5        ┆ 35       ┆ 85        ┆ 0.7                ┆ 1.29675   │
│ Australia     ┆ 32.1        ┆ 40       ┆ 120       ┆ 0.6                ┆ 1.1556    │
│ Mediterranean ┆ 29.3        ┆ 45       ┆ 95        ┆ 0.5                ┆ 0.80575   │
│ Siberia       ┆ 22.7        ┆ 30       ┆ 65        ┆ 0.4                ┆ 0.6356    │
└───────────────┴─────────────┴──────────┴───────────┴────────────────────┴───────────┘
```

## Run at scale in the cloud

Imagine that there is a larger dataset stored in a cloud provider’s storage solution. The dataset is
so large that it doesn’t fit on our local machine. However, through local analysis, we have verified
that the defined query correctly calculates the column we are looking for.

With Polars Cloud, we can easily run the same query at scale. First, we make small changes to our
query to point to our resources in the cloud.

```python
lf = pl.scan_parquet("s3://climate-data/global/*.parquet")

query = (
    lf.with_columns(
        [
            (
                (pl.col("temperature") / 10)
                * (1 - pl.col("humidity") / 100)
                * pl.col("vegetation_density")
            ).alias("fire_risk"),
        ]
    )
    .filter(pl.col("humidity") < 70)
    .sort(by="fire_risk", descending=True)
)
```

Next, we set our compute context and call `.remote(context=ctx)` on our query.

```python
import polars_cloud as pc

ctx = pc.ComputeContext(
    workspace="environmental-analysis",
    memory=32,
    cpus=8
)

query.remote(context=ctx).sink_parquet("s3://bucket/result.parquet")
```

### Continue analysis in interactive mode

Running `.sink_parquet()` will write the results to the defined bucket on S3. Alternatively, we can
take a more interactive approach by adding the parameter `interactive=True` to our compute context.

```python
ctx = pc.ComputeContext(
    workspace="environmental-analysis",
    memory=32,
    cpus=8,
    interactive=True,  # set interactive to True
)

result = query.remote(context=ctx).collect()

print(result.collect())
```

```text
shape: (4, 6)
┌───────────────┬─────────────┬──────────┬───────────┬────────────────────┬───────────┐
│ region        ┆ temperature ┆ humidity ┆ burn_area ┆ vegetation_density ┆ fire_risk │
│ ---           ┆ ---         ┆ ---      ┆ ---       ┆ ---                ┆ ---       │
│ str           ┆ f64         ┆ i64      ┆ i64       ┆ f64                ┆ f64       │
╞═══════════════╪═════════════╪══════════╪═══════════╪════════════════════╪═══════════╡
│ California    ┆ 28.5        ┆ 35       ┆ 85        ┆ 0.7                ┆ 1.29675   │
│ Australia     ┆ 32.1        ┆ 40       ┆ 120       ┆ 0.6                ┆ 1.1556    │
│ Mediterranean ┆ 29.3        ┆ 45       ┆ 95        ┆ 0.5                ┆ 0.80575   │
│ Siberia       ┆ 22.7        ┆ 30       ┆ 65        ┆ 0.4                ┆ 0.6356    │
└───────────────┴─────────────┴──────────┴───────────┴────────────────────┴───────────┘
```

We can call `.collect()` instead of `.sink_parquet()`. This will store your results to a temporary
location which can be used to further iterate upon. A LazyFrame is returned that can be used in the
next steps of the workflow.

```python
res2 = (
    result
    .filter(pl.col("fire_risk") > 1)
    .sink_parquet("s3://bucket/output-interactive.parquet")
)
```

The result of your interactive workflow can be written to S3.
