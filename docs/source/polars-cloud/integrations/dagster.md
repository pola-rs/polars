# Dagster

Integrate Polars Cloud authentication and per-asset compute sizing into a Dagster pipeline using
typed `ConfigurableResource` subclasses.

Credentials are referenced through
[`EnvVar`](https://docs.dagster.io/guides/operate/configuration/using-environment-variables-and-secrets#dagster-envvar-class)
so the raw values are resolved at run launch and never displayed in the Dagster UI. Populate
`POLARS_CLOUD_CLIENT_ID` and `POLARS_CLOUD_CLIENT_SECRET` in the Dagster deployment environment —
either directly or by exporting from a secret manager (e.g.
[AWS Secrets Manager](https://docs.aws.amazon.com/secretsmanager/latest/userguide/retrieving-secrets-python.html)).

`PolarsCloudCompute` declares `PolarsCloudAuth` as a
[nested resource](https://docs.dagster.io/guides/build/external-resources/defining-resources) so
Dagster runs `PolarsCloudAuth.setup_for_execution` — which calls `authenticate(...)` — before any
compute resource is used. The compute resource owns the `ComputeContext` lifecycle through
`yield_for_execution`: the VM is started before each asset that uses the resource runs and stopped
on exit, including on failure. Asset bodies contain only Polars code.

Per-asset compute sizing falls out of the same pattern: declare one `PolarsCloudCompute` instance
per VM shape under distinct resource keys, and each asset takes the instance sized for it. Dagster
manages the lifecycle of each instance independently, so the asset that uses the larger VM does not
pay the cost of spinning it up while the smaller-VM assets are running.

```python
from contextlib import contextmanager

import polars as pl
import polars_cloud as pc
from dagster import asset, ConfigurableResource, Definitions, EnvVar


class PolarsCloudAuth(ConfigurableResource):
    """Service account authentication for Polars Cloud."""

    client_id: str
    client_secret: str

    def setup_for_execution(self, context) -> None:
        pc.authenticate(client_id=self.client_id, client_secret=self.client_secret)


class PolarsCloudCompute(ConfigurableResource):
    """Polars Cloud compute context with auto-managed VM lifecycle.

    Declares `PolarsCloudAuth` as a nested resource so Dagster runs `authenticate(...)`
    before this resource is used. `yield_for_execution(...)` starts a `ComputeContext`
    of the configured shape and the `finally` clause guarantees the VM is released even
    when an asset raises.
    """

    auth: PolarsCloudAuth
    cpus: int
    memory: int

    @contextmanager
    def yield_for_execution(self, context):
        ctx = pc.ComputeContext(cpus=self.cpus, memory=self.memory)
        ctx.start()
        try:
            pc.set_compute_context(ctx)
            yield self
        finally:
            ctx.stop()


@asset
def dataset_1(small_vm: PolarsCloudCompute):
    pl.scan_csv(...).remote().sink_parquet(...)


@asset
def dataset_2(small_vm: PolarsCloudCompute):
    pl.scan_ndjson(...).remote().sink_parquet(...)


# Use a bigger machine for the join.
@asset(deps=[dataset_1, dataset_2])
def joined(large_vm: PolarsCloudCompute):
    pl.scan_parquet(...).remote().sink_parquet(...)


# Share a single `PolarsCloudAuth` instance between the compute resources so the
# credentials are loaded once per run.
auth = PolarsCloudAuth(
    client_id=EnvVar("POLARS_CLOUD_CLIENT_ID"),
    client_secret=EnvVar("POLARS_CLOUD_CLIENT_SECRET"),
)

defs = Definitions(
    assets=[dataset_1, dataset_2, joined],
    resources={
        "small_vm": PolarsCloudCompute(auth=auth, cpus=2, memory=4),
        "large_vm": PolarsCloudCompute(auth=auth, cpus=4, memory=16),
    },
)
```

The Dagster resource key (the dictionary key in `Definitions.resources`) must match the parameter
name in each asset signature. The type annotation is for editor and type-checker support; Dagster
injects by key, not by type.
