# Execute remote query

Polars Cloud enables you to execute existing Polars queries on cloud infrastructure with minimal
code changes. This approach allows you to process datasets that exceed local resources or use
additional compute resources for faster execution.

!!! note "Polars Cloud is set up and connected"

    This page assumes that you have created an organization and connected a workspace to your cloud environment. If you haven't yet, follow the steps on the [Connect cloud environment](../connect-cloud.md) page.

## Define your query locally

The following example uses a query from the PDS-H benchmark suite, a derived version of the popular
TPC-H benchmark. Data generation tools and additional queries are available in the
[Polars benchmark repository](https://github.com/pola-rs/polars-benchmark).

{{code_block('polars-cloud/remote-query','local',[])}}

## Scale to the cloud

To execute your query in the cloud, you need to define a compute context. The compute context
specifies the hardware to use when executing the query in the cloud. It allows you to set the
workspace to execute your query and set compute resources. More elaborate options can be found on
the [Compute context introduction page](../context/compute-context.md).

{{code_block('polars-cloud/remote-query','context',['ComputeContext'])}}

!!! info "Run the examples yourself"

    All examples on this page can be executed using the sample datasets hosted on our S3 bucket. By including the `storage_option` parameter in your queries, you'll only incur S3 data transfer costs. No additional storage fees apply

!!! note "S3 bucket region"

    The example datasets are hosted in the `us-east-2 S3 region`. Query performance may be affected if you're running operations from a distant geographic location due to network latency.

## Working with remote query results

Once you've called `.remote(context=ctx)` on your query, you have several options for how to handle
the results, each suited to different use cases and workflows.

### Write to storage

The most straightforward approach for batch processing is to write results directly to cloud storage
using `.sink_parquet()`. This method is ideal when you want to store processed data for later use or
as part of a data pipeline:

{{code_block('polars-cloud/remote-query','sink_parquet',[])}}

Running `.sink_parquet()` will write the results to the defined bucket on S3. The query you execute
runs in your cloud environment, and both the data and results remain secure in your own
infrastructure. This approach is perfect for ETL workflows, scheduled jobs, or any time you need to
persist large datasets without transferring them to your local machine.

### Inspect results

Using `.show()` will display the first 10 rows of the result so you can inspect the structure
without having to transfer the whole dataset. This method displays the first 10 rows in your console
or notebook.

{{code_block('polars-cloud/remote-query','show',[])}}

```text
shape: (10, 4)
┌────────────┬─────────────┬─────────────┬────────────────┐
│ l_orderkey ┆ revenue     ┆ o_orderdate ┆ o_shippriority │
│ ---        ┆ ---         ┆ ---         ┆ ---            │
│ i64        ┆ f64         ┆ date        ┆ i64            │
╞════════════╪═════════════╪═════════════╪════════════════╡
│ 4791171    ┆ 440715.2185 ┆ 1995-02-23  ┆ 0              │
│ 46678469   ┆ 439855.325  ┆ 1995-01-27  ┆ 0              │
│ 23906758   ┆ 432728.5737 ┆ 1995-03-14  ┆ 0              │
│ 23861382   ┆ 428739.1368 ┆ 1995-03-09  ┆ 0              │
│ 59393639   ┆ 426036.0662 ┆ 1995-02-12  ┆ 0              │
│ 3355202    ┆ 425100.6657 ┆ 1995-03-04  ┆ 0              │
│ 9806272    ┆ 425088.0568 ┆ 1995-03-13  ┆ 0              │
│ 22810436   ┆ 423231.969  ┆ 1995-01-02  ┆ 0              │
│ 16384100   ┆ 421478.7294 ┆ 1995-03-02  ┆ 0              │
│ 52974151   ┆ 415367.1195 ┆ 1995-02-05  ┆ 0              │
└────────────┴─────────────┴─────────────┴────────────────┘
```

The `.await_and_scan()` method returns a LazyFrame pointing to intermediate results stored
temporarily in your S3 environment. These intermediate result files are automatically deleted after
several hours. For persistent storage use `sink_parquet`. The output is a LazyFrame, allowing
continued query chaining for further analysis.

{{code_block('polars-cloud/remote-query','await_scan',[])}}

```text
shape: (114_003, 4)
┌────────────┬─────────────┬─────────────┬────────────────┐
│ l_orderkey ┆ revenue     ┆ o_orderdate ┆ o_shippriority │
│ ---        ┆ ---         ┆ ---         ┆ ---            │
│ i64        ┆ f64         ┆ date        ┆ i64            │
╞════════════╪═════════════╪═════════════╪════════════════╡
│ 4791171    ┆ 440715.2185 ┆ 1995-02-23  ┆ 0              │
│ 46678469   ┆ 439855.325  ┆ 1995-01-27  ┆ 0              │
│ 23906758   ┆ 432728.5737 ┆ 1995-03-14  ┆ 0              │
│ 23861382   ┆ 428739.1368 ┆ 1995-03-09  ┆ 0              │
│ 59393639   ┆ 426036.0662 ┆ 1995-02-12  ┆ 0              │
│ …          ┆ …           ┆ …           ┆ …              │
│ 44149381   ┆ 904.3968    ┆ 1995-01-16  ┆ 0              │
│ 34297697   ┆ 897.8464    ┆ 1995-03-06  ┆ 0              │
│ 25478115   ┆ 887.2318    ┆ 1994-11-28  ┆ 0              │
│ 52204674   ┆ 860.25      ┆ 1994-12-18  ┆ 0              │
│ 47255457   ┆ 838.9381    ┆ 1994-11-18  ┆ 0              │
└────────────┴─────────────┴─────────────┴────────────────┘
```
