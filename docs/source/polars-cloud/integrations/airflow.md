# Airflow

Execute Polars Cloud queries remotely using Airflow workflows through secure credential management.
This section explains how to configure Airflow to submit and monitor Polars Cloud workloads using
Airflow's built-in security mechanisms, keeping service account credentials isolated from DAG code
while maintaining full workflow control.

1. **Secret manager**
   ([docs](https://airflow.apache.org/docs/apache-airflow/stable/security/secrets/secrets-backend/index.html)):
   is the <ins> Airflow-recommended way to handle secrets</ins>. It involves setting up a
   `Secret Backend` (many providers maintained by the community) in the `airflow.cfg` and let
   Airflow workers pull the given secrets via the `airflow.models.Variable` API as
   `Variable.get("<SECRET NAME>")`. Note Airflow will pull the secret in its own metastore; if this
   situation is not desirable, interacting with the cloud provider's Secret Manager (or any other
   vault accessible via API) can simply be performed as a task of your DAG; see relevant official
   docs (here is
   [AWS](https://docs.aws.amazon.com/secretsmanager/latest/userguide/retrieving-secrets-python.html)'
   as an example).
2. **Environment variables**
   ([docs](https://airflow.apache.org/docs/apache-airflow/stable/howto/variable.html#storing-variables-in-environment-variables)):
   load your environment variables into your containers after prefixing them by `AIRFLOW_VAR_`, for
   instance `AIRFLOW_VAR_POLARS_CLOUD_CLIENT_ID` and `AIRFLOW_VAR_POLARS_CLOUD_CLIENT_SECRET`. They
   should then be available through the `airflow.models.Variable` API as
   `Variable.get("POLARS_CLOUD_CLIENT_ID")`.
3. **Airflow `Variables`**
   ([docs](https://airflow.apache.org/docs/apache-airflow/stable/howto/variable.html#managing-variables)):
   in the Airflow UI > Admin > Variables tab one can add/edit key: value pairs provided to Airflow,
   which will make them accessible through the `airflow.models.Variable` API. Note these objects can
   also be defined using the Airflow CLI (if accessible):
   `airflow variables set POLARS_CLOUD_CLIENT_ID "<SECRET>"`.

Some code snippets for solutions **#1** and **#2** described above:

```python
# pull secrets from the aws secret manager
@resource
def service_account_from_aws(_):
    client = boto3.client("secretsmanager")
    Variable.set(
        "client_id",
        client.get_secret_value(SecretId="<SECRET NAME>").get("SecretString"),
    )
    Variable.set(
        "client_secret",
        client.get_secret_value(SecretId="<SECRET NAME>").get("SecretString"),
    )
```

```python
# fetch [securely injected!] secrets from environment
@resource
def service_account_from_env(_):
    Variable.set("client_id", os.getenv("POLARS_CLOUD_CLIENT_ID"))
    Variable.set("client_secret", os.getenv("POLARS_CLOUD_CLIENT_SECRET"))
```

Below a few lines of _pseudo-code_ using Airflow' `TaskFlow` API:

```python
import polars as pl

from airflow.models import Variable
from airflow.sdk import dag, task
from polars_cloud import ComputeContext, authenticate, set_compute_context

# define two compute contexts (two instance sizes)
vm_small = ComputeContext(cpus=2, memory=4)
vm_large = ComputeContext(cpus=4, memory=16)

# queries will execute on the small vm by default
set_compute_context(vm_small)

@dag(...)
def taskflow():

    @task()
    def prepare_dataset_1():
        pl.scan_csv(...).remote().sink_parquet(...)

    @task()
    def prepare_dataset_2():
        pl.scan_ndjson(...).remote().sink_parquet(...)

    # use a bigger machine for this operation
    @task()
    @set_compute_context(vm_large)
    def join_datasets():
        pl.scan_parquet(...).remote().sink_parquet(...)

    # authenticate to polars cloud with the secrets created above
    authenticate(
        client_id=Variable.get("secret_id"),
        client_secret=Variable.get("secret_secret"),
    )

    prepare_dataset_1()
    prepare_dataset_2()
    join_datasets()

taskflow()

# stop the instances
vm_small.stop()
vm_large.stop()
```
