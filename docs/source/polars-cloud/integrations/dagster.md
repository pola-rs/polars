# Dagster

Configure Polars Cloud authentication securely within Dagster pipelines using resource based secret
management. This section details how to integrate Polars Cloud service account credentials with
Dagster's resource pattern.

Dagster implements secret management through
[Resources](https://docs.dagster.io/getting-started/concepts#resource), which provide dependency
injection for external services. The modern, type-checked way to declare a resource is
[`ConfigurableResource`](https://docs.dagster.io/guides/build/external-resources/defining-resources),
combined with
[`EnvVar`](https://docs.dagster.io/guides/operate/configuration/using-environment-variables-and-secrets#dagster-envvar-class)
for any secret you do not want to surface in the Dagster UI. Pull the secret values from one of:

1. **Secret manager** (<ins>recommended</ins>): pull the secret from a metastore (see official docs
   of your secret manager; here is
   [AWS](https://docs.aws.amazon.com/secretsmanager/latest/userguide/retrieving-secrets-python.html)'
   as an example) and expose it to the process as an environment variable, then reference it via
   `EnvVar`.
2. **Environment variables**: define the values as environment variables in your Dagster environment
   (containers or else), and reference them via `EnvVar` so they are resolved at launch time and
   never displayed in the UI.

Below a complete pipeline definition demonstrating both authentication and per-asset compute sizing.
The two `PolarsCloud*` resources own all lifecycle: `PolarsCloudAuth` calls `authenticate(...)` once
at run start (via `setup_for_execution`), and `PolarsCloudCompute` exposes a `session()` context
manager that activates a fresh `ComputeContext` for the asset body and stops the underlying VM on
exit — including on failure.

```python
import polars as pl
from contextlib import contextmanager

from dagster import asset, ConfigurableResource, Definitions, EnvVar
from polars_cloud import ComputeContext, authenticate, set_compute_context


class PolarsCloudAuth(ConfigurableResource):
    """Authenticate to Polars Cloud using a service-account secret pair.

    Source ``client_id`` and ``client_secret`` via :class:`dagster.EnvVar` so the
    raw values are resolved at run launch time and never appear in the Dagster UI.
    """

    client_id: str
    client_secret: str

    def setup_for_execution(self, context) -> None:
        authenticate(client_id=self.client_id, client_secret=self.client_secret)


class PolarsCloudCompute(ConfigurableResource):
    """A managed Polars Cloud compute context.

    Use the ``session()`` context manager from inside an asset (or op) body to
    activate a ``ComputeContext`` of the configured shape and stop the underlying
    VM on exit — including on failure.
    """

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


# Each asset declares both ``auth`` (so ``PolarsCloudAuth.setup_for_execution`` is
# invoked at run start) and the compute resource that should size its VM.

@asset
def dataset_1(auth: PolarsCloudAuth, small_vm: PolarsCloudCompute):
    with small_vm.session():
        pl.scan_csv(...).remote().sink_parquet(...)


@asset
def dataset_2(auth: PolarsCloudAuth, small_vm: PolarsCloudCompute):
    with small_vm.session():
        pl.scan_ndjson(...).remote().sink_parquet(...)


# Use a bigger machine for the join.
@asset(deps=[dataset_1, dataset_2])
def joined(auth: PolarsCloudAuth, large_vm: PolarsCloudCompute):
    with large_vm.session():
        pl.scan_parquet(...).remote().sink_parquet(...)


defs = Definitions(
    assets=[dataset_1, dataset_2, joined],
    resources={
        "auth": PolarsCloudAuth(
            client_id=EnvVar("POLARS_CLOUD_CLIENT_ID"),
            client_secret=EnvVar("POLARS_CLOUD_CLIENT_SECRET"),
        ),
        "small_vm": PolarsCloudCompute(cpus=2, memory=4),
        "large_vm": PolarsCloudCompute(cpus=4, memory=16),
    },
)
```

The Dagster resource key (the dictionary key in `Definitions.resources`) must match the parameter
name in the asset signature — the type annotation is for editor and type-checker support only.
