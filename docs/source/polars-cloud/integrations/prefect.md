# Prefect

Integrate Polars Cloud authentication and per-task compute sizing into a Prefect flow using
`Secret` blocks for credentials.

Credentials are stored as Prefect
[`Secret` blocks](https://docs.prefect.io/v3/how-to-guides/configuration/store-secrets) so
raw values are never hard-coded in flow code. Register and populate them once with the Prefect CLI:

```sh
prefect block register -m prefect.blocks.system
prefect block create secret polars-cloud-client-id
prefect block create secret polars-cloud-client-secret
```

Authentication is performed once at the top of the flow so every task in the run shares the same
session. Each task receives its `ComputeContext` directly via `.remote(ctx)`, and the flow owns the
VM lifecycle: `start()` is called before any task runs and `stop()` is guaranteed by the `finally`
clause.

`dataset_1` and `dataset_2` share a single instance which handles both tasks sequentially
(_concurrently_ for Prefect, but _sequentially_ in Polars Cloud). `joined` runs on a larger VM,
which is started and stopped independently.

```python
import polars as pl
import polars_cloud as pc
from prefect import flow, task
from prefect.blocks.system import Secret

SMALL_CTX = pc.ComputeContext(cpus=2, memory=4)
LARGE_CTX = pc.ComputeContext(cpus=4, memory=16)


@task
def dataset_1():
    pl.scan_csv(...).remote(SMALL_CTX).sink_parquet(...)


@task
def dataset_2():
    pl.scan_ndjson(...).remote(SMALL_CTX).sink_parquet(...)


# Use a bigger machine for the join.
@task
def joined():
    pl.scan_parquet(...).remote(LARGE_CTX).sink_parquet(...)


@flow(name="Report")
def report():
    pc.authenticate(
        client_id=Secret.load("polars-cloud-client-id").get(),
        client_secret=Secret.load("polars-cloud-client-secret").get(),
    )

    SMALL_CTX.start()
    LARGE_CTX.start()
    try:
        f1 = dataset_1.submit()
        f2 = dataset_2.submit()
        joined.submit(wait_for=[f1, f2])
    finally:
        SMALL_CTX.stop()
        LARGE_CTX.stop()


report()
```
