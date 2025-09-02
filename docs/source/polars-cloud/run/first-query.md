# Run your first query

Polars Cloud makes it simple to run your query at scale. You can take any existing Polars query and
run it on powerful cloud infrastructure with just a few additional lines of code. This allows you to
process datasets that are too large for your local machine or leverage more compute power when you
need it.

!!! note "Polars Cloud is set up and connected"

    This page assumes that you have created an organization and connected a workspace to your cloud environment. If you haven't yet, find more information on [Connect cloud environment](../connect-cloud.md) page.

## Define your query locally

If you're already familiar with Polars, you will immediately be productive with Polars Cloud. The
same lazy evaluation and API you know works exactly the same way, with the addition of defining a
ComputeContext and calling `.remote(context=ctx)` to execute your query in your cloud environment,
instead of locally.

Below we define a query that you would typically write on your local machine.

{{code_block('polars-cloud/first-query','local',[])}}

## Scale to the cloud

To execute your query in the cloud, we must define a compute context. The compute context defines
the hardware to use when executing the query in the cloud. It allows to define the workspace to
execute your query, set compute resources and define if you want to execute in job mode. More
elaborate options can be found on the
[Compute context introduction page](../context/compute-context.md)

{{code_block('polars-cloud/first-query','context',[])}}

### Job mode

Job mode is used for systematic data processing pipelines, written for scheduled execution. They
typically process large volumes of data in scheduled intervals (e.g. hourly, daily, etc.). A key
characteristic the jobs typically runs without a human in the loop.

Read more about the differences on the [compute context page](../context/compute-context.md)

### Distributed execution

To run your queries over multiple nodes, you must define your `cluster_size` in the `ComputeContext`
and call the `.distributed()` method.

{{code_block('polars-cloud/first-query','distributed',[])}}

This distributes your query execution across 10 machines in this example, providing a total of 100
cores and 100GB of RAM for processing. Find more information about executing your queries on
multiple nodes on the [distributed queries](distributed-engine.md) page.

## Working with remote query results

Once you've called `.remote(context=ctx)` on your query, you have several options for how to handle
the results, each suited to different use cases and workflows.

### Direct or intermediate storage of results

The most straightforward approach for batch processing is to write results directly to cloud storage
using `.sink_parquet()`. This method is ideal when you want to store processed data for later use or
as part of a data pipeline:

{{code_block('polars-cloud/first-query','distributed',[])}}

Running `.sink_parquet()` will write the results to the defined bucket on S3. The query you execute
in job mode runs in your cloud environment, and the data and results remain secure in your own
infrastructure. This approach is perfect for ETL workflows, scheduled jobs, or any time you need to
persist large datasets without transferring them to your local machine.

### Inspect results in your local environment

For exploratory analysis and interactive workflows, use `.collect()` or `.show()` to return query
results:

{{code_block('polars-cloud/first-query','interactive',[])}}

The `.show()` method is perfect for data exploration where you want to understand the structure and
content of your results without the overhead of transferring large datasets. It displays a sample of
rows in your console or notebook.

When calling `.collect()` on your remote query execution, the intermediate results are written to a
temporary location on S3. These intermediate result files are automatically deleted after several
hours. The output of the remote query is a LazyFrame, which means you can continue chaining
operations on the returned results for further analysis. Find a more elaborate example on the
[Example workflow page](example-workflow.md)
