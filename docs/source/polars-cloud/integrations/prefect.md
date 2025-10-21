# Prefect

Configure Polars Cloud authentication securely within Prefect workflows using native secret
management patterns. This section details how to integrate Polars Cloud service account credentials
with Prefect's configuration system.

Prefect implements secure credential handling through three standard approaches:

1. **Secret manager** (<ins>recommended</ins>): pull the secret secret manager of your choice and
   use it in your workflow (see official docs; here is
   [AWS](https://docs.aws.amazon.com/secretsmanager/latest/userguide/retrieving-secrets-python.html)'
   as an example). One can also use the AWS-specific `Secret` `Block` (see below;
   [docs](https://docs.prefect.io/v3/how-to-guides/configuration/store-secrets)) to interact with
   the AWS Secret Manager.
2. **Environment variables**: load your environment variables into your running instance (container
   or else).
3. **`Block` system** ([docs](https://docs.prefect.io/v3/concepts/blocks)): Prefect defined a
   `Block` framework that can be used via the CLI
   (`prefect block register -m prefect.blocks.system`) or directly in the code
   (`from prefect.blocks.system import Secret`). A secret can be created via CLI (for instance):
   `prefect block create secret polars-cloud-client-id` and retrieved from the code as
   `Secret.load("polars-cloud-client-id").get()`.

Some code snippets for solutions **#1** and **#2** described above:

```python
# pull secrets from the aws secret manager
def service_account_from_aws(_):
    client = boto3.client("secretsmanager")
    return {
        "client_id": client.get_secret_value(SecretId="<SECRET>").get("SecretString"),
        "client_secret": client.get_secret_value(SecretId="<SECRET>").get("SecretString"),
    }
```

```python
# fetch [securely injected!] secrets from environment
@resource
def service_account_from_env(_):
    return {
        "client_id": os.getenv("POLARS_CLOUD_CLIENT_ID"),
        "client_secret": os.getenv("POLARS_CLOUD_CLIENT_SECRET"),
    }
```

Below a few lines of _pseudo-code_ to define a Prefect flow:

```python
import os
import polars as pl

from polars_cloud import ComputeContext, authenticate, set_compute_context
from prefect import flow, task

# define two compute contexts (two instance sizes)
vm_small = ComputeContext(cpus=2, memory=4)
vm_large = ComputeContext(cpus=4, memory=16)

# queries will execute on the small vm by default
set_compute_context(vm_small)

@task
def prepare_dataset_1():
    pl.scan_csv(...).remote().sink_parquet(...)

@task
def prepare_dataset_2():
    pl.scan_ndjson(...).remote().sink_parquet(...)

# use a bigger machine for this operation
@task
@set_compute_context(vm_large)
def join_datasets():
    pl.scan_parquet(...).remote().sink_parquet(...)

@flow(name="Daily report")
def report():
    # authenticate to polars cloud with the secrets created above
    authenticate(**service_account_from_env())

    prepare_dataset_1()
    prepare_dataset_2()
    join_datasets()

# run the flow
report()

# stop the instances
vm_small.stop()
vm_large.stop()
```
