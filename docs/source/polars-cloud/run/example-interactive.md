# Interactive mode

Polars Cloud supports interactive workflows. Interactive mode allows for exploratory workflows where
a user interacts with the dataset and requires more compute resources than locally available. When
using interactive mode, the cluster stays active and intermediate state can be accessed.

In interactive mode you directly communicate with the compute nodes. Therefore, logs and metrics
stay in your environment and will not be available in the dashboard.

You can go interactive by setting `interactive=True` when
[defining your compute context](../context/compute-context.md).

```python
pc.ComputeContext(workspace="your-workspace", interactive=True)
```

## Interactive example workflow

Data processing and analytics often begins small but can quickly grow beyond the capabilities of
your local machine. A typical workflow starts with exploring a sample dataset locally, developing
the analytical approach, and then scaling up to process the full dataset in the cloud.

For this workflow, we define the following simple mocked dataset that will act as a sample to
demonstrate the workflow. Here we will create the LazyFrame ourselves, but it could also be read as
(remote) file.

{{code_block('polars-cloud/workflow','lazyframe',[])}}

We can use the Polars API to do a simple transformation and create a fire risk column:

{{code_block('polars-cloud/workflow','transformation',[])}}

```text
shape: (2, 6)
┌────────────┬─────────────┬──────────┬───────────┬────────────────────┬───────────┐
│ region     ┆ temperature ┆ humidity ┆ burn_area ┆ vegetation_density ┆ fire_risk │
│ ---        ┆ ---         ┆ ---      ┆ ---       ┆ ---                ┆ ---       │
│ str        ┆ f64         ┆ i64      ┆ i64       ┆ f64                ┆ f64       │
╞════════════╪═════════════╪══════════╪═══════════╪════════════════════╪═══════════╡
│ California ┆ 28.5        ┆ 35       ┆ 85        ┆ 0.7                ┆ 1.29675   │
│ Australia  ┆ 32.1        ┆ 40       ┆ 120       ┆ 0.6                ┆ 1.1556    │
└────────────┴─────────────┴──────────┴───────────┴────────────────────┴───────────┘
```

### Run at scale in the cloud

Once we've verified our analytical approach locally, we can scale it up to handle much larger
datasets using Polars Cloud. Imagine we have a massive dataset stored in cloud storage that doesn't
fit on our local machine.

First, we modify our query to point to our fictive dataset in our S3 bucket:

{{code_block('polars-cloud/workflow','remote-query',[])}}

Next, we set our compute context and call `.remote(context=ctx)` on our query.

{{code_block('polars-cloud/workflow','set-compute',[])}}

For more details on configuring ComputeContext parameters, see the
[compute context documentation](../context/compute-context.md).

### Explore query results

In interactive mode, calling `.collect()` on your remote query stores results in a temporary
location and returns a LazyFrame that you can continue working with:

{{code_block('polars-cloud/workflow','remote-collect',[])}}

```text
shape: (2, 6)
┌────────────┬─────────────┬──────────┬───────────┬────────────────────┬───────────┐
│ region     ┆ temperature ┆ humidity ┆ burn_area ┆ vegetation_density ┆ fire_risk │
│ ---        ┆ ---         ┆ ---      ┆ ---       ┆ ---                ┆ ---       │
│ str        ┆ f64         ┆ i64      ┆ i64       ┆ f64                ┆ f64       │
╞════════════╪═════════════╪══════════╪═══════════╪════════════════════╪═══════════╡
│ California ┆ 28.5        ┆ 35       ┆ 85        ┆ 0.7                ┆ 1.29675   │
│ Australia  ┆ 32.1        ┆ 40       ┆ 120       ┆ 0.6                ┆ 1.1556    │
└────────────┴─────────────┴──────────┴───────────┴────────────────────┴───────────┘
```

{{code_block('polars-cloud/workflow','remote-filter',[])}}

```text
shape: (1, 6)
┌───────────┬─────────────┬──────────┬───────────┬────────────────────┬───────────┐
│ region    ┆ temperature ┆ humidity ┆ burn_area ┆ vegetation_density ┆ fire_risk │
│ ---       ┆ ---         ┆ ---      ┆ ---       ┆ ---                ┆ ---       │
│ str       ┆ f64         ┆ i64      ┆ i64       ┆ f64                ┆ f64       │
╞═══════════╪═════════════╪══════════╪═══════════╪════════════════════╪═══════════╡
│ Australia ┆ 32.1        ┆ 40       ┆ 120       ┆ 0.6                ┆ 1.1556    │
└───────────┴─────────────┴──────────┴───────────┴────────────────────┴───────────┘
```

### Store intermediate and final results

You can store intermediate or final results on S3 by using `sink_parquet`:

```python
result.remote(context=ctx).sink_parquet("s3://climate-data/results/")
```
