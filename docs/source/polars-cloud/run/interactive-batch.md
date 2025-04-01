# Interactive or batch mode

In Polars Cloud, a user can define two types of compute modes: batch & interactive. Batch mode is
designed for batch job style queries. These kinds of queries are typically scheduled and run once in
a certain period. Interactive mode allows for exploratory workflows where a user interacts with the
dataset and requires more compute resources than are locally available.

The rest of this page will give examples on how to set up one or the other. More information on the
architectural differences and implications can be found on
[the infrastructure page](../providers/aws/infra.md).

Below we create a simple dataframe to use as an example to demonstrate the difference between both
modes.

{{code_block('polars-cloud/interactive-batch','example',[])}}

## Batch

Batch workflows are systematic data processing pipelines, written for scheduled execution. They
typically process large volumes of data in scheduled intervals (e.g. hourly, daily, etc.). A key
characteristic is that the executed job has a defined lifetime. A predefined compute instance should
spin up at a certain time and shut down when the job is executed.

Polars Cloud makes it easy to run your query at scale whenever your use case requires it. You can
develop your query on your local machine and define a compute context and destination to execute it
in your cloud environment.

{{code_block('polars-cloud/interactive-batch','batch',['ComputeContext'])}}

The query you execute in batch mode runs in your cloud environment. The data and results of the
query are not sent to Polars Cloud, ensuring that your data and output remain secure.

```python
lf.remote(context=ctx).sink_parquet("s3://bucket/output.parquet")
```

## Interactive

Polars Cloud also supports interactive workflows. Different from batch mode, results are being
interactively updated. Polars Cloud will not automatically close the cluster when a result has been
produced, but the cluster stays active and intermediate state can still be accessed. In interactive
mode you directly communicate with the compute nodes.

Because this mode is used for exploratory use cases and short feedback cycles, the queries are not
logged to Polars Cloud and will not be available for later inspection.

{{code_block('polars-cloud/interactive-batch','interactive',['ComputeContext'])}}

The initial query remains the same. In the compute context the parameter `interactive` should be set
to `True`.

When calling `.collect()` on your remote query execution, the output is written to a temporary
location. These intermediate result files are automatically deleted after several hours. The output
of the remote query is a LazyFrame.

```python
print(type(res1))
```

```
<class 'polars.lazyframe.frame.LazyFrame'>
```

If you want to inspect the results you can call collect again.

```python
print(res1.collect())
```

```text
shape: (4, 3)
┌────────────────┬────────────┬───────────┐
│ name           ┆ birth_year ┆ bmi       │
│ ---            ┆ ---        ┆ ---       │
│ str            ┆ i32        ┆ f64       │
╞════════════════╪════════════╪═══════════╡
│ Chloe Cooper   ┆ 1983       ┆ 19.687787 │
│ Ben Brown      ┆ 1985       ┆ 23.141498 │
│ Alice Archer   ┆ 1997       ┆ 23.791913 │
│ Daniel Donovan ┆ 1981       ┆ 27.134694 │
└────────────────┴────────────┴───────────┘
```

To continue your exploration you can use the returned LazyFrame to build another query.

{{code_block('polars-cloud/interactive-batch','interactive-next',[])}}

```python
print(res2.collect())
```

```text
shape: (2, 3)
┌──────────────┬────────────┬───────────┐
│ name         ┆ birth_year ┆ bmi       │
│ ---          ┆ ---        ┆ ---       │
│ str          ┆ i32        ┆ f64       │
╞══════════════╪════════════╪═══════════╡
│ Chloe Cooper ┆ 1983       ┆ 19.687787 │
│ Ben Brown    ┆ 1985       ┆ 23.141498 │
└──────────────┴────────────┴───────────┘
```
