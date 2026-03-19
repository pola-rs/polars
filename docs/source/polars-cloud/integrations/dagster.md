# Dagster

Configure Polars Cloud authentication securely within Dagster pipelines using resource based secret
management. This section details how to integrate Polars Cloud service account credentials with
Dagster's resource pattern.

Dagster implements secret management through
[Resources](https://docs.dagster.io/getting-started/concepts#resource)), which provide dependency
injection for external services. To configure Polars Cloud authentication, define credentials
through one of these standard approaches:

1. **Secret manager** (<ins>recommended</ins>): pull the secret from a metastore (see official docs
   of your secret manager; here is
   [AWS](https://docs.aws.amazon.com/secretsmanager/latest/userguide/retrieving-secrets-python.html)'
   as an example).
2. **Environment variables**: define the values as environment variables in your Dagster environment
   (containers or else), and pick them up from your code or Dagster configuration (via the
   `dagster.yaml` or `workspace.yaml`).

Some code snippets for the solutions described above:

```python
# pull secrets from the aws secret manager
@resource
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

Below a few lines of _pseudo-code_ to define a Dagster flow:

```python
import os
import polars as pl

from dagster import job, op, resource
from polars_cloud import ComputeContext, authenticate, set_compute_context

# define two compute contexts (two instance sizes)
vm_small = ComputeContext(cpus=2, memory=4)
vm_large = ComputeContext(cpus=4, memory=16)

# queries will execute on the small vm by default
set_compute_context(vm_small)

@op(required_resource_keys={"sa"})
def prepare_dataset_1():
    pl.scan_csv(...).remote().sink_parquet(...)

@op(required_resource_keys={"sa"})
def prepare_dataset_2():
    pl.scan_ndjson(...).remote().sink_parquet(...)

# use a bigger machine for this operation
@op(required_resource_keys={"sa"})
@set_compute_context(vm_large)
def join_datasets():
    pl.scan_parquet(...).remote().sink_parquet(...)

@job(resource_defs={"sa": service_account_from_aws})
def report():
    # authenticate to polars cloud with the secrets created above
    authenticate(**sa)

    prepare_dataset_1()
    prepare_dataset_2()
    join_datasets()

# stop the instances
vm_small.stop()
vm_large.stop()
```
