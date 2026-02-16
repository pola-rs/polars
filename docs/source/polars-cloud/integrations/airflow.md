# Airflow

Orchestrate Polars Cloud queries using Apache Airflow workflows while keeping your Airflow instance
lightweight and focused on orchestration. This section explains how to configure Airflow to submit
and monitor Polars Cloud workloads, delegating compute-intensive operations to remote clusters while
managing credentials securely through Airflow's built-in mechanisms.

## Prerequisites

Before integrating Polars Cloud with Airflow, ensure you have:

- The `polars` and `polars-cloud` packages installed in your Airflow environment
  ([installation guide](https://airflow.apache.org/docs/docker-stack/build.html))
- A Polars Cloud [service account](../explain/service-accounts.md) with client ID and secret

## Setting up an Airflow Connection

Configure your Polars Cloud credentials using an Airflow Connection, the recommended approach for
managing secrets in Airflow workflows.

In the Airflow UI, navigate to **Admin > Connections > Add Connection** and configure:

- **Connection ID**: `polars_cloud`
- **Connection Type**: `HTTP`
- **Login**: Your service account client ID
- **Password**: Your service account client secret

<!-- dprint-ignore-start -->

!!! info "Alternative credential storage"

    Credentials can also be stored in a
    [Secrets Backend](https://airflow.apache.org/docs/apache-airflow/stable/security/secrets/secrets-backend/index.html)
    for enhanced security. This prevents storing secrets in the Airflow database and integrates with
    external secret managers like AWS Secrets Manager, HashiCorp Vault, or Google Secret Manager.

<!-- dprint-ignore-end -->

## Authenticating

Retrieve your connection credentials and authenticate to Polars Cloud within your tasks:

```python
from airflow.sdk import BaseHook
import polars_cloud as pc

conn = BaseHook.get_connection("polars_cloud")
pc.authenticate(client_id=conn.login, client_secret=conn.password)
```

For DAGs with multiple tasks, it's recommended to create a reusable authentication decorator to
avoid repeating this code:

```python
from functools import wraps
from airflow.sdk import BaseHook
import polars_cloud as pc


def authenticate(fn):
    @wraps(fn)
    def authenticated_fn(*args, **kwargs):
        conn = BaseHook.get_connection("polars_cloud")
        pc.authenticate(client_id=conn.login, client_secret=conn.password)
        return fn(*args, **kwargs)

    return authenticated_fn
```

Apply this decorator to any task requiring Polars Cloud access:

```python
@task()
@authenticate
def my_query_task():
    # Task code with authenticated Polars Cloud access
    ...
```

## Executing a single query

Create a basic DAG that executes a single Polars query on Polars Cloud:

```python
from datetime import datetime

import polars as pl
import polars_cloud as pc

from airflow.sdk import BaseHook, dag, task


@dag(start_date=datetime(2026, 1, 1), schedule="@daily")
def single_query_dag():
    @task()
    def aggregate_sales():
        conn = BaseHook.get_connection("polars_cloud")
        pc.authenticate(client_id=conn.login, client_secret=conn.password)

        ctx = pc.ComputeContext(
            workspace="playground", cpus=8, memory=16, cluster_size=1
        )

        (
            pl.scan_parquet(
                "s3://your-bucket/input-data/*.parquet"
            )
            .group_by("status")
            .agg(pl.count())
            .remote(ctx)
            .sink_parquet("s3://your-bucket/output-data/")
        )

    aggregate_sales()


single_query_dag()
```

This example defines a
[DAG](https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/dags.html) that runs
daily, creates a [ComputeContext](../context/compute-context.md) with specific hardware
requirements, and executes a query that writes results to S3.

<!-- dprint-ignore-start -->

!!! note "Fire-and-forget execution"

    The `.sink_parquet()` method is non-blocking: it submits the query to Polars Cloud and returns
    immediately, allowing the Airflow task to complete while the query continues executing on the
    remote cluster. The cluster will shut down automatically after the configured idle timeout
    (default: 1 hour).

<!-- dprint-ignore-end -->

## Checking query status

To ensure query completion and show failures in Airflow, wait for the query result and check its
status:

```python
@task()
@authenticate
def aggregate_sales_with_check():
    ctx = pc.ComputeContext(
        workspace="playground", cpus=8, memory=16, cluster_size=1
    )

    query_result = (
        pl.scan_parquet("s3://your-bucket/input-data/*.parquet")
        .group_by("status")
        .agg(pl.count())
        .remote(ctx)
        .sink_parquet("s3://your-bucket/output-data/")
    )

    query_result.await_result()
    
    query_status = query_result.get_status()
    if query_status != pc.QueryStatus.SUCCESS:
        raise ValueError(f"Query failed with status: {query_status}")
```

The `.await_result()` method blocks until the query completes, and checking the status allows you to
raise an exception that marks the Airflow task as failed.

## Managing cluster lifecycle

There are a couple of patterns that help you manage your compute cluster lifecycle.

### Context wrapper for immediate shutdown

Use `ComputeContext` as a context manager to shut down the cluster immediately after exiting the
scope, reducing costs by avoiding idle time:

```python
@task()
@authenticate
def query_with_immediate_shutdown():
    with pc.ComputeContext(
        workspace="playground", cpus=8, memory=16, cluster_size=1
    ) as ctx:
        query_result = (
            pl.scan_parquet("s3://your-bucket/input-data/*.parquet")
            .group_by("status")
            .agg(pl.count())
            .remote(ctx)
            .sink_parquet("s3://your-bucket/output-data/")
        )

        query_result.await_result()
        if query_result.get_status() != pc.QueryStatus.SUCCESS:
            raise ValueError("Query failed")
    # Cluster shuts down here when exiting the context
```

<!-- dprint-ignore-start -->

!!! note "Context manager behavior"

    The cluster shuts down when the context exits, regardless of running queries. Always wait for
    query completion (`.await_result()`) before exiting the context to make sure your query finishes.

<!-- dprint-ignore-end -->

### Named manifests for reusable clusters

Instead of specifying hardware requirements in every task, create a named manifest that defines a
reusable cluster configuration.

You can create a manifest programmatically:

```python
pc.ComputeContext(
    workspace="playground", cpus=16, memory=32, cluster_size=3
).register("airflow-production")
```

Or you can create it through the Polars Cloud dashboard under **Compute > Manifests > Add new
manifest**.

You then reference the manifest in your tasks like so:

```python
@task()
@authenticate
def query_using_manifest():
    ctx = pc.ComputeContext(workspace="playground", name="airflow-production")

    (
        pl.scan_parquet("s3://your-bucket/input-data/*.parquet")
        .group_by("status")
        .agg(pl.count())
        .remote(ctx)
        .sink_parquet("s3://your-bucket/output-data/")
    )
```

Manifests provide several benefits:

- Hardware configuration defined once, reused across tasks and users
- Multiple queries can share the same cluster, reducing startup overhead
- Queries are automatically queued when the cluster is busy
- Simplifies DAG code by removing hardware specification details

### Manual cluster shutdown

To manually shut down a cluster immediately after all tasks complete, add a dedicated shutdown task:

```python
@dag(schedule="@daily", start_date=datetime(2026, 1, 1))
def dag_with_manual_shutdown():
    @task()
    @authenticate
    def query_1():
        ...

    @task()
    @authenticate
    def query_2():
        ...

    @task(trigger_rule=TriggerRule.ALL_DONE)
    @authenticate
    def shutdown_cluster():
        ctx = pc.ComputeContext(workspace="playground", name="airflow-production")
        ctx.stop()

    [query_1(), query_2()] >> shutdown_cluster()


dag_with_manual_shutdown()
```

The `trigger_rule=TriggerRule.ALL_DONE` ensures the shutdown task runs after all upstream tasks
complete, regardless of success or failure.

<!-- dprint-ignore-start -->

!!! tip "Configuring idle timeout"

    The `ComputeContext` accepts an `idle_timeout_mins` parameter (default: 60 minutes, minimum: 10
    minutes) that controls automatic shutdown after inactivity. For short-running workflows, reduce
    this value to minimize costs: `ComputeContext(..., idle_timeout_mins=10)`.

<!-- dprint-ignore-end -->

## Parallel query execution

Execute multiple queries concurrently on the same cluster by marking them as single-node queries.
This allows efficient resource utilization when individual queries don't require the full cluster:

```python
@dag(schedule="@daily", start_date=datetime(2026, 1, 1))
def parallel_queries_dag():
    @task()
    @authenticate
    def query_1():
        ctx = pc.ComputeContext(workspace="playground", name="airflow-production")
        (
            pl.scan_parquet("s3://your-bucket/dataset-1/*.parquet")
            .group_by("category")
            .agg(pl.sum("amount"))
            .remote(ctx)
            .single_node()
            .sink_parquet("s3://your-bucket/output-1/")
        )

    @task()
    @authenticate
    def query_2():
        ctx = pc.ComputeContext(workspace="playground", name="airflow-production")
        (
            pl.scan_parquet("s3://your-bucket/dataset-2/*.parquet")
            .group_by("region")
            .agg(pl.mean("value"))
            .remote(ctx)
            .single_node()
            .sink_parquet("s3://your-bucket/output-2/")
        )

    @task()
    @authenticate
    def query_3():
        ctx = pc.ComputeContext(workspace="playground", name="airflow-production")
        (
            pl.scan_parquet("s3://your-bucket/dataset-3/*.parquet")
            .group_by("date")
            .agg(pl.count())
            .remote(ctx)
            .single_node()
            .sink_parquet("s3://your-bucket/output-3/")
        )

    query_1()
    query_2()
    query_3()


parallel_queries_dag()
```

The `.single_node()` method indicates the query runs on a single worker, enabling the scheduler to
execute multiple queries concurrently on a multi-node cluster. Without this, queries use
`.distributed()` by default, which reserves the entire cluster and queues subsequent queries.

## Multi-stage pipelines

Build pipelines where later tasks depend on results from earlier tasks by passing intermediate
result locations between stages:

```python
@dag(schedule="@daily", start_date=datetime(2026, 1, 1))
def multistage_pipeline():
    @task()
    @authenticate
    def stage_1_transform() -> list[str]:
        ctx = pc.ComputeContext(workspace="playground", name="airflow-production")
        query_result = (
            pl.scan_parquet("s3://your-bucket/raw-data/*.parquet")
            .filter(pl.col("status") == "active")
            .with_columns(pl.col("amount").cast(pl.Float64))
            .remote(ctx)
            .execute()
            .await_result()
        )

        if query_result.location is None:
            raise ValueError("Query result location is None")

        return query_result.location

    @task()
    @authenticate
    def stage_2_aggregate(result_locations: list[str]):
        lf = pl.scan_parquet(result_locations)
        ctx = pc.ComputeContext(workspace="playground", name="airflow-production")

        (
            lf.group_by("category")
            .agg(pl.sum("amount"), pl.count())
            .remote(ctx)
            .sink_parquet("s3://your-bucket/aggregated-data/")
        )

    stage_1_results = stage_1_transform()
    stage_2_aggregate(stage_1_results)


multistage_pipeline()
```

The first stage executes a query without `.sink_parquet()`, instead using
`.execute().await_result()` to retrieve the `query_result.location`. This contains a list of
temporary S3 paths where Polars Cloud stores intermediate results. These locations are serialized
and passed to the next stage, which scans them with `pl.scan_parquet()` to continue processing.

<!-- dprint-ignore-start -->

!!! info "Intermediate result retention"

    Intermediate results stored by Polars Cloud are automatically deleted after several hours. For
    persistent storage, use `.sink_parquet()` with your own S3 path. See the
    [remote query documentation](../run/remote-query.md) for more details.

<!-- dprint-ignore-end -->

## Complete example

The following example combines all patterns: authentication decorator, named manifest, parallel
execution, multi-stage pipeline, and manual cluster shutdown:

```python
from datetime import datetime
from functools import wraps

import polars as pl
import polars_cloud as pc

from airflow.sdk import BaseHook, TriggerRule, dag, task


def authenticate(fn):
    @wraps(fn)
    def authenticated_fn(*args, **kwargs):
        conn = BaseHook.get_connection("polars_cloud")
        pc.authenticate(client_id=conn.login, client_secret=conn.password)
        return fn(*args, **kwargs)

    return authenticated_fn


WORKSPACE = "playground"
MANIFEST_NAME = "airflow-production"


@dag(schedule="@daily", start_date=datetime(2026, 1, 1))
def complete_pipeline():
    @task()
    @authenticate
    def transform_sales() -> list[str]:
        ctx = pc.ComputeContext(workspace=WORKSPACE, name=MANIFEST_NAME)
        query_result = (
            pl.scan_parquet("s3://your-bucket/sales/*.parquet")
            .with_columns((pl.col("price") * pl.col("quantity")).alias("revenue"))
            .remote(ctx)
            .single_node()
            .execute()
            .await_result()
        )
        query_status = query_result.get_status()
        if query_status != pc.QueryStatus.SUCCESS:
            raise ValueError(f"Query failed with status: {query_status}")
        if query_result.location is None:
            raise ValueError("Sales transformation failed")
        return query_result.location

    @task()
    @authenticate
    def transform_inventory() -> list[str]:
        ctx = pc.ComputeContext(workspace=WORKSPACE, name=MANIFEST_NAME)
        query_result = (
            pl.scan_parquet("s3://your-bucket/inventory/*.parquet")
            .with_columns(
                (pl.col("stock_level") < pl.col("min_threshold")).alias("low_stock")
            )
            .remote(ctx)
            .single_node()
            .execute()
            .await_result()
        )
        query_status = query_result.get_status()
        if query_status != pc.QueryStatus.SUCCESS:
            raise ValueError(f"Query failed with status: {query_status}")
        if query_result.location is None:
            raise ValueError("Inventory transformation failed")
        return query_result.location

    @task()
    @authenticate
    def transform_returns() -> list[str]:
        ctx = pc.ComputeContext(workspace=WORKSPACE, name=MANIFEST_NAME)
        query_result = (
            pl.scan_parquet("s3://your-bucket/returns/*.parquet")
            .with_columns((pl.col("return_date") - pl.col("purchase_date")).alias("return_days"))
            .remote(ctx)
            .single_node()
            .execute()
            .await_result()
        )
        query_status = query_result.get_status()
        if query_status != pc.QueryStatus.SUCCESS:
            raise ValueError(f"Query failed with status: {query_status}")
        if query_result.location is None:
            raise ValueError("Returns transformation failed")
        return query_result.location

    @task()
    @authenticate
    def generate_report(
        sales_locations: list[str],
        inventory_locations: list[str],
        returns_locations: list[str],
    ):
        sales = pl.scan_parquet(sales_locations)
        inventory = pl.scan_parquet(inventory_locations)
        returns = pl.scan_parquet(returns_locations)

        ctx = pc.ComputeContext(workspace=WORKSPACE, name=MANIFEST_NAME)
        query = (
            sales.join(inventory, on="product_id")
            .join(returns, on="product_id", how="left")
            .group_by("product_category")
            .agg(
                pl.sum("revenue").alias("total_revenue"),
                pl.sum("low_stock").alias("low_stock_count"),
                pl.count().alias("transaction_count"),
            )
            .remote(ctx)
            .distributed()
            .sink_parquet("s3://your-bucket/daily-report/")   
        )
        query_result = query.await_result()
        query_status = query_result.get_status()
        if query_status != pc.QueryStatus.SUCCESS:
            raise ValueError(f"Query failed with status: {query_status}")

    @task(trigger_rule=TriggerRule.ALL_DONE)
    @authenticate
    def shutdown_cluster():
        ctx = pc.ComputeContext(workspace=WORKSPACE, name=MANIFEST_NAME)
        ctx.stop()

    sales_results = transform_sales()
    inventory_results = transform_inventory()
    returns_results = transform_returns()
    report_task = generate_report(sales_results, inventory_results, returns_results)
    report_task >> shutdown_cluster()


complete_pipeline()
```
