# Google Kubernetes Engine (GKE)

!!! info "Initial configuration"

    This page expects that you've already set up a Polars cluster once through the Polars Cloud onboarding or the [getting started guide](../index.md).

## Data access using Workload Identity

Through GKE Workload Identity, you can securely access private Google Cloud Storage buckets without
needing to manage service account keys or credentials. In most scencarios, it comes down to enabling
Workload Identity Federation, creating a Kubernetes service account, and creating an IAM policy
binding.
[See the guide in the official GKE documentation](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/workload-identity).

```bash
helm upgrade --install polars polars-inc/polars \
  --set scheduler.serviceAccount.name=<YOUR_SERVICE_ACCOUNT_NAME> \
  --set worker.serviceAccount.name=<YOUR_SERVICE_ACCOUNT_NAME> \
# ...
```

Assuming you have a bucket already set up (see quick-start
[here](https://docs.cloud.google.com/storage/docs/creating-buckets)), you can then scan or sink
directly from the bucket.

```python
path = f"gs://YOUR_BUCKET_NAME/PATH/TO/DATA/"
storage_options = {
    "project": "YOUR_PROJECT_NAME",
}
q = (
    pl.scan_parquet(path, storage_options=storage_options)
# ..
)
```

You may also use Google Cloud Storage as
[an anonymous results location](https://github.com/polars-inc/helm-charts/tree/main/charts/polars#anonymous-results-data)
by configuring the values as such:

```yaml
anonymousResults:
  gcs:
    enabled: true
    endpoint: "gs://YOUR_BUCKET_NAME/PATH/TO/DATA/"
    options:
    - name: project
      value: "YOUR_PROJECT_NAME"
```
