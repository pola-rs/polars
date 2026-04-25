# OpenLineage

OpenLineage is an open platform for collection and analysis of data lineage. See
[openlineage.io](https://openlineage.io) for more information.

Polars acts as an OpenLineage **producer** — it emits `RunEvent` and optionally `DatasetEvent` when
a query executes. Job identity (`JobEvent`) must be provided by the caller, typically an
orchestrator such as Airflow or Dagster.

!!! note "OpenLineage is only supported for Polars on-premises"

    Note that OpenLineage is currently only supported for Polars on-premises and not for Polars Cloud. Obtain a license for Polars on-premises by [signing up here](https://w0lzyfh2w8o.typeform.com/to/zuoDgoMv).

## Setup

At a minimum, three components are required:

1. A Polars cluster configured with lineage transport
2. A Polars query annotated with job metadata via `with_lineage()`
3. A lineage collector (e.g., [Marquez](https://github.com/MarquezProject/marquez))

Optionally, an orchestrator will submit the queries and take on the responsibility for annotation.

## Cluster configuration

The cluster must be configured with a lineage transport endpoint, pointing at the collector. HTTP(S)
is the only supported transport protocol. See the dedicated cluster type pages for more information.

- [Bare metal](/polars-on-premises/bare-metal/configuration/openlineage)
- [Kubernetes Helm Chart](https://github.com/polars-inc/helm-charts/blob/main/charts/polars/README.md#lineage)

## Query annotation

Each query must be annotated with Job metadata through the `with_lineage()` API. At a minimum, both
the `job_namespace` and `job_name` must be provided.

For example:

```python
query = (
    pl.concat(
        [
            pl.scan_csv(src1),
            pl.scan_csv(src2),
        ],
        how="horizontal",
    )
    .remote(context=ctx)
    .with_lineage(
        job_namespace="prod",
        job_name="demo.job1"
    )
    .sink_parquet(dst)
)
```

Additional metadata, such as Parent Run metadata can be added, see
[with_lineage()](https://docs.cloud.pola.rs/reference/query/api/polars_cloud.LazyFrameRemote.with_lineage.html)
in the [Polars cloud API reference](https://docs.cloud.pola.rs/reference/index.html) for details.

## Running and collecting

Running the query will emit openlineage events and send them to the collector. Polars is responsible
for generating the unique `runId` for any given query. The collector will collect the events for
analysis, correlation, and visualization purposes.

For example, Marquez visualizes the above query as follows, by default at `http://localhost:3000`,

![lineage jobs](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/docs/openlineage/lineage_demo_jobs.png)

and lists the collected events.

![lineage events](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/docs/openlineage/lineage_demo_events.png)

For more information, consult the documentation of your collector.

## Dataset naming

Polars strives to align with the
[DataSet naming convention](https://openlineage.io/docs/spec/naming/). However, it may deviate when
there is a better way to capture the logical identity of the dataset, e.g. when using cloud-based
prefixes for multi-file stores, single-file datasets in cloud context, glob patterns, or hive
partitioning.

This aspect of the implementation is **unstable**.

## Supported facets

OpenLineage is extensible through the use of [facets](https://openlineage.io/docs/spec/facets/). In
addition to the core `RunEvent` and optional `DatasetEvent`, Polars supports the following facets:

- Run Facets
  - Error Message Facet
  - Parent Run Facet
- Dataset Facets
  - Schema Dataset Facet
  - Output Statistics Facet
