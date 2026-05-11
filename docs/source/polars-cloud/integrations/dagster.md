# Dagster

Integrate Polars Cloud authentication and compute sizing into a Dagster
pipeline. Two patterns are documented below; both code blocks are complete and
runnable on their own.

- **Single compute size** — every asset uses the same VM shape. The compute
  resource auto-manages one `ComputeContext` per Dagster run via
  `yield_for_execution`; asset bodies need no per-asset setup.
- **Per-asset compute sizing** — different assets need different VM shapes.
  Each asset opens a `session()` context manager on the compute resource it
  needs.

Credentials are referenced through
[`EnvVar`](https://docs.dagster.io/guides/operate/configuration/using-environment-variables-and-secrets#dagster-envvar-class)
so the raw values are resolved at run launch and never displayed in the
Dagster UI. Populate `POLARS_CLOUD_CLIENT_ID` and `POLARS_CLOUD_CLIENT_SECRET`
in the Dagster deployment environment — either directly or by exporting from a
secret manager (e.g.
[AWS Secrets Manager](https://docs.aws.amazon.com/secretsmanager/latest/userguide/retrieving-secrets-python.html)).

In both patterns `PolarsCloudCompute` declares `PolarsCloudAuth` as a
[nested resource](https://docs.dagster.io/guides/build/external-resources/defining-resources)
so Dagster runs `PolarsCloudAuth.setup_for_execution` — which calls
`authenticate(...)` — before any compute resource is used.

## Pattern 1: single compute size

The compute resource owns the `ComputeContext` lifecycle through
`yield_for_execution`: one VM is started at run init and stopped on exit,
including on failure. Asset bodies contain only Polars code.

```python
from contextlib import contextmanager

import polars as pl
from dagster import asset, ConfigurableResource, Definitions, EnvVar

from polars_cloud import ComputeContext, authenticate, set_compute_context


class PolarsCloudAuth(ConfigurableResource):
    """Service-account authentication. Runs once per Dagster run."""

    client_id: str
    client_secret: str

    def setup_for_execution(self, context) -> None:
        authenticate(client_id=self.client_id, client_secret=self.client_secret)


class PolarsCloudCompute(ConfigurableResource):
    """Polars Cloud compute context with auto-managed VM lifecycle.

    Declares ``PolarsCloudAuth`` as a nested resource so Dagster runs
    ``authenticate(...)`` before this resource is used. The active
    ``ComputeContext`` is started in ``yield_for_execution`` and stopped on
    exit — the ``finally`` clause guarantees the VM is released even when an
    asset raises.
    """

    auth: PolarsCloudAuth
    cpus: int
    memory: int

    @contextmanager
    def yield_for_execution(self, context):
        ctx = ComputeContext(cpus=self.cpus, memory=self.memory)
        try:
            set_compute_context(ctx)
            yield self
        finally:
            ctx.stop()


@asset
def dataset_1(compute: PolarsCloudCompute):
    pl.scan_csv(...).remote().sink_parquet(...)


@asset
def dataset_2(compute: PolarsCloudCompute):
    pl.scan_ndjson(...).remote().sink_parquet(...)


@asset(deps=[dataset_1, dataset_2])
def joined(compute: PolarsCloudCompute):
    pl.scan_parquet(...).remote().sink_parquet(...)


defs = Definitions(
    assets=[dataset_1, dataset_2, joined],
    resources={
        "compute": PolarsCloudCompute(
            auth=PolarsCloudAuth(
                client_id=EnvVar("POLARS_CLOUD_CLIENT_ID"),
                client_secret=EnvVar("POLARS_CLOUD_CLIENT_SECRET"),
            ),
            cpus=4,
            memory=16,
        ),
    },
)
```

## Pattern 2: per-asset compute sizing

`set_compute_context` is process-global state, so a single auto-managed
context cannot serve multiple VM shapes within one run. When assets need
different shapes, each compute resource exposes a `session()` context manager
instead of starting its `ComputeContext` at run init. The asset opens the
session of the resource sized for it; the VM is stopped on exit, including on
failure.

A single `PolarsCloudAuth` instance is reused for both compute resources, so
the credentials are loaded once at run start.

```python
from contextlib import contextmanager

import polars as pl
from dagster import asset, ConfigurableResource, Definitions, EnvVar

from polars_cloud import ComputeContext, authenticate, set_compute_context


class PolarsCloudAuth(ConfigurableResource):
    """Service-account authentication. Runs once per Dagster run."""

    client_id: str
    client_secret: str

    def setup_for_execution(self, context) -> None:
        authenticate(client_id=self.client_id, client_secret=self.client_secret)


class PolarsCloudCompute(ConfigurableResource):
    """Polars Cloud compute context with per-asset session management.

    Each ``session()`` call starts a fresh ``ComputeContext`` of the configured
    shape and stops the VM on exit. ``PolarsCloudAuth`` is a nested resource
    so Dagster initialises it before any compute resource is used.
    """

    auth: PolarsCloudAuth
    cpus: int
    memory: int

    @contextmanager
    def session(self):
        ctx = ComputeContext(cpus=self.cpus, memory=self.memory)
        try:
            set_compute_context(ctx)
            yield ctx
        finally:
            ctx.stop()


@asset
def dataset_1(small_vm: PolarsCloudCompute):
    with small_vm.session():
        pl.scan_csv(...).remote().sink_parquet(...)


@asset
def dataset_2(small_vm: PolarsCloudCompute):
    with small_vm.session():
        pl.scan_ndjson(...).remote().sink_parquet(...)


@asset(deps=[dataset_1, dataset_2])
def joined(large_vm: PolarsCloudCompute):
    with large_vm.session():
        pl.scan_parquet(...).remote().sink_parquet(...)


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

The Dagster resource key (the dictionary key in `Definitions.resources`) must
match the parameter name in each asset signature. The type annotation is for
editor and type-checker support; Dagster injects by key, not by type.
